"""Container entrypoint for training openWakeWord models on CPU."""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from datetime import timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

import datasets  # type: ignore[import-not-found]
import numpy as np
import scipy.io.wavfile  # type: ignore[import-not-found]
import yaml

DEFAULT_WORKDIR = Path(os.getenv("OPENWAKEWORD_WORKDIR", "/workspace"))
DEFAULT_PIPER_DIR = Path(os.getenv("OPENWAKEWORD_PIPER_DIR", "/opt/piper-sample-generator"))
DEFAULT_PIPER_MODEL = os.getenv("OPENWAKEWORD_PIPER_MODEL", "en_US-libritts_r-medium.pt")
DEFAULT_PIPER_MODEL_URL = os.getenv(
    "OPENWAKEWORD_PIPER_MODEL_URL",
    "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
)
DEFAULT_STEPS = int(os.getenv("OPENWAKEWORD_STEPS", "10000"))
DEFAULT_SAMPLES = int(os.getenv("OPENWAKEWORD_SAMPLES", "1000"))
DEFAULT_SAMPLES_VAL = int(os.getenv("OPENWAKEWORD_SAMPLES_VAL", "1000"))
DEFAULT_TTS_BATCH = int(os.getenv("OPENWAKEWORD_TTS_BATCH", "50"))
DEFAULT_FMA_HOURS = float(os.getenv("OPENWAKEWORD_FMA_HOURS", "1"))
SKIP_AUDIOSET = os.getenv("OPENWAKEWORD_SKIP_AUDIOSET", "0") == "1"


class UTCFormatter(logging.Formatter):
    """Formatter that forces UTC timestamps."""

    def formatTime(self, record, datefmt=None):  # noqa: D401
        dt = time.gmtime(record.created)
        if datefmt:
            return time.strftime(datefmt, dt) + "Z"
        return time.strftime("%Y-%m-%dT%H:%M:%S", dt) + "Z"


def configure_logging() -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(UTCFormatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def slugify_phrase(wakeword: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", wakeword.lower()).strip("_")
    return slug or "wakeword"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def start_http_server(serve_dir: Path) -> tuple[ThreadingHTTPServer, threading.Thread]:
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=str(serve_dir), **kwargs)
    server = ThreadingHTTPServer(("0.0.0.0", 8080), handler)
    thread = threading.Thread(target=server.serve_forever, name="http-server", daemon=True)
    thread.start()
    logging.info("http server listening on 0.0.0.0:8080")
    return server, thread


def start_progress_logger(stop_event: threading.Event) -> threading.Thread:
    def _log_status() -> None:
        if stop_event.wait(60):
            return
        logging.info("still training...")
        while not stop_event.wait(300):
            logging.info("still training...")

    thread = threading.Thread(target=_log_status, name="progress-logger", daemon=True)
    thread.start()
    return thread


def ensure_piper_assets() -> Path:
    model_path = DEFAULT_PIPER_DIR / "models" / DEFAULT_PIPER_MODEL
    if model_path.exists():
        return DEFAULT_PIPER_DIR

    logging.info("downloading Piper model from %s", DEFAULT_PIPER_MODEL_URL)
    download_file(DEFAULT_PIPER_MODEL_URL, model_path)
    return DEFAULT_PIPER_DIR


def ensure_mit_rirs(target_dir: Path) -> None:
    if any(target_dir.glob("*.wav")):
        logging.info("MIT RIR dataset already present")
        return

    logging.info("downloading MIT RIR dataset (streaming)")
    target_dir.mkdir(parents=True, exist_ok=True)
    rir_dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True
    )
    for row in rir_dataset:
        name = row["audio"]["path"].split("/")[-1]
        scipy.io.wavfile.write(
            target_dir / name,
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )


def ensure_audioset(base_dir: Path) -> Path:
    if SKIP_AUDIOSET:
        logging.info("OPENWAKEWORD_SKIP_AUDIOSET=1 set; skipping AudioSet download")
        placeholder = base_dir / "audioset_16k"
        placeholder.mkdir(parents=True, exist_ok=True)
        return placeholder

    tar_dir = base_dir / "audioset"
    tar_dir.mkdir(parents=True, exist_ok=True)
    tar_path = tar_dir / "bal_train09.tar"
    extract_dir = tar_dir / "audio"
    output_dir = base_dir / "audioset_16k"

    if not tar_path.exists():
        url = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar"
        logging.info("downloading AudioSet shard (this is large)")
        try:
            download_file(url, tar_path)
        except Exception as exc:
            logging.warning("AudioSet download failed (%s); proceeding without AudioSet background", exc)
            SKIP_AUDIOSET_TRUE = base_dir / "audioset_skipped"
            SKIP_AUDIOSET_TRUE.touch()
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir

    if not extract_dir.exists():
        logging.info("extracting AudioSet shard")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(tar_dir)

    if list(output_dir.glob("*.wav")):
        logging.info("AudioSet 16k clips already prepared")
        return output_dir

    logging.info("converting AudioSet FLAC to 16 kHz WAV")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = [str(p) for p in extract_dir.glob("**/*.flac")]
    if not audio_files:
        raise FileNotFoundError("no AudioSet FLAC files found after extraction")

    audio_dataset = datasets.Dataset.from_dict({"audio": audio_files})
    audio_dataset = audio_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    for row in audio_dataset:
        name = row["audio"]["path"].split("/")[-1].replace(".flac", ".wav")
        scipy.io.wavfile.write(
            output_dir / name,
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )

    return output_dir


def ensure_fma(base_dir: Path, hours: float) -> Path:
    output_dir = base_dir / "fma"
    if list(output_dir.glob("*.wav")):
        logging.info("FMA clips already prepared")
        return output_dir

    logging.info("downloading %.1f hours of FMA clips", hours)
    output_dir.mkdir(parents=True, exist_ok=True)
    fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
    fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))

    clip_target = int(hours * 3600 // 30)
    for idx in range(clip_target):
        row = next(fma_dataset)
        name = row["audio"]["path"].split("/")[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(
            output_dir / name,
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )

    return output_dir


def ensure_feature_files(base_dir: Path) -> tuple[Path, Path]:
    train_path = base_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    val_path = base_dir / "validation_set_features.npy"

    if not train_path.exists():
        logging.info("downloading pre-computed training features")
        download_file(
            "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            train_path,
        )

    if not val_path.exists():
        logging.info("downloading validation features")
        download_file(
            "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
            val_path,
        )

    return train_path, val_path


def write_training_config(
    *,
    session_dir: Path,
    slug: str,
    wakeword: str,
    rir_dir: Path,
    audioset_dir: Path | None,
    fma_dir: Path,
    feature_file: Path,
    val_feature_file: Path,
    steps: int,
    samples: int,
    samples_val: int,
    tts_batch: int,
) -> Path:
    config = yaml.safe_load((Path("/app/examples/custom_model.yml")).read_text())

    config["target_phrase"] = [wakeword]
    config["model_name"] = slug
    config["output_dir"] = str(session_dir)
    config["rir_paths"] = [str(rir_dir)]
    background_paths = [p for p in [audioset_dir, fma_dir] if p is not None]
    config["background_paths"] = [str(p) for p in background_paths]
    config["background_paths_duplication_rate"] = [1 for _ in background_paths]
    config["false_positive_validation_data_path"] = str(val_feature_file)
    config["feature_data_files"] = {"ACAV100M_sample": str(feature_file)}
    config["n_samples"] = samples
    config["n_samples_val"] = samples_val
    config["steps"] = steps
    config["tts_batch_size"] = tts_batch
    config["piper_sample_generator_path"] = str(DEFAULT_PIPER_DIR)

    config_path = session_dir / "training_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def run_training_process(config_path: Path, workdir: Path) -> None:
    commands: Iterable[list[str]] = (
        [sys.executable, "-m", "openwakeword.train", "--training_config", str(config_path), "--generate_clips"],
        [sys.executable, "-m", "openwakeword.train", "--training_config", str(config_path), "--augment_clips"],
        [sys.executable, "-m", "openwakeword.train", "--training_config", str(config_path), "--train_model"],
    )

    for cmd in commands:
        subprocess.run(cmd, check=True, cwd=workdir)


def locate_onnx_model(session_dir: Path, slug: str) -> Path:
    candidate = session_dir / f"{slug}.onnx"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"expected ONNX model at {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an openWakeWord model inside Docker")
    parser.add_argument("-c", "--wakeword", required=True, help="Wake word phrase to synthesise and train")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Training steps")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Synthetic training samples")
    parser.add_argument("--samples-val", type=int, default=DEFAULT_SAMPLES_VAL, help="Synthetic validation samples")
    parser.add_argument("--tts-batch", type=int, default=DEFAULT_TTS_BATCH, help="Piper TTS batch size")
    args = parser.parse_args()

    configure_logging()

    wakeword = args.wakeword.strip()
    if not wakeword:
        parser.error("Wake word must not be empty")

    slug = slugify_phrase(wakeword)
    session_dir = DEFAULT_WORKDIR / slug
    serve_dir = session_dir / "serve"
    rir_dir = session_dir / "mit_rirs"

    session_dir.mkdir(parents=True, exist_ok=True)
    serve_dir.mkdir(parents=True, exist_ok=True)

    server, server_thread = start_http_server(serve_dir)

    try:
        ensure_piper_assets()
        ensure_mit_rirs(rir_dir)
        audioset_dir = ensure_audioset(session_dir)
        fma_dir = ensure_fma(session_dir, DEFAULT_FMA_HOURS)
        feature_file, val_feature_file = ensure_feature_files(session_dir)

        config_path = write_training_config(
            session_dir=session_dir,
            slug=slug,
            wakeword=wakeword,
            rir_dir=rir_dir,
            audioset_dir=audioset_dir,
            fma_dir=fma_dir,
            feature_file=feature_file,
            val_feature_file=val_feature_file,
            steps=args.steps,
            samples=args.samples,
            samples_val=args.samples_val,
            tts_batch=args.tts_batch,
        )

        logging.info("starting training for '%s' this will take a while", wakeword)
        start_time = time.time()
        stop_event = threading.Event()
        progress_thread = start_progress_logger(stop_event)

        try:
            run_training_process(config_path, workdir=session_dir)
        finally:
            stop_event.set()
            progress_thread.join(timeout=1)

        onnx_path = locate_onnx_model(session_dir, slug)
        served_model = serve_dir / f"{slug}.onnx"
        shutil.copy2(onnx_path, served_model)

        duration = timedelta(seconds=int(time.time() - start_time))
        logging.info(
            "training complete for '%s'; download at http://0.0.0.0:8080/%s.onnx (took %s)",
            wakeword,
            slug,
            duration,
        )

        logging.info("serving trained models from %s; press Ctrl+C to stop", serve_dir)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logging.info("shutdown requested, stopping server")
    except Exception as exc:  # pragma: no cover
        logging.exception("failed to train wake word model: %s", exc)
        raise SystemExit(1) from exc
    finally:
        server.shutdown()
        server_thread.join(timeout=1)


if __name__ == "__main__":
    main()
