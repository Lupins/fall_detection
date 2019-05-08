sudo nvidia-docker run --rm --runtime=nvidia --name flora -it -u docker -v "/mnt:/mnt" -v "$(pwd)/Fall_detection:/home/docker/Fall_detection/" "semantix/v0.0.1" /bin/bash
