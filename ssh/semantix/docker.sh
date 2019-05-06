docker run --name milk -v /mnt:/mnt --rm -it tensorflow/tensorflow:latest-gpu bash
#docker run --name milk -v /mnt:/mnt --rm -it semantix/v0.1 bash
nvidia-docker run --rm -it tensorflow/tensorflow:1.9.0-gpu-py3 /bin/bash
