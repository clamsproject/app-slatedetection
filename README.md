### Using docker

We provide a [`Dockerfile`](Dockerfile). If you want to run the slate detector as a docker container (not worrying about dependencies), build an image from the `Dockerfile` and run it with the target directory mounted to `/data`. Just MAKE SURE that target directory is writable by others (`chmod o+w $TARGET_DIR`). For example, 

```bash
chmod -R o+w $HOME/video-files && docker build . -t slate_detector && docker run --rm -v $HOME/video-files:/data slate_detector
```
