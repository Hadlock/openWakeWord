if you look in notebooks/ you will find a notebooks/automatic_model_training.ipynb

I already did this for microwakeword;

```
## TL;DR do this in docker locally

### TL;TL;DR

- dockebuild . -t wakeword && docker r run -it --rm -p 8080:8080 -e MICROWAKEWORD_TRAIN_BATCH=256 localhost/wakeword -c "hey snippy"

### TL;DR

- Build the image with `docker build . -t wakeword`. Docker is the easy button here, even if the CPU-only run is painfully slow. About 22 minutes on a 12th gen i5 with 2gb ram and 50gb disk
- Kick off training with `docker run -it --rm -p 8080:8080 wakeword -c "hey snippy"`. Swap the quoted phrase for your wake word; the container logs will show `starting training for 'hey snippy' this will take a while` and periodic `still training...` updates.
- Expect it to chug away for hours (or days) on a CPU, but it will finish eventually and drop the quantized model at `http://0.0.0.0:8080/hey_snippy.onnx`. Map a volume with `-v $PWD/models:/workspace` if you want to keep the outputs between runs.
- Watch the UTC timestamped logs for the final line announcing the download URL and total training time.
```

this is a different wake word than the above, but the fact remains, it is a notebook file and i'd like to also convert it to a docker container. i have included the previous Dockerfile and docker_entrypoint.py in the examples/ folder

let's convert notebooks/automatic_model_training.ipynb to a containerized system as well so i don't have to use google collab, instead of the file extension .tflite it should be .onnx as expected