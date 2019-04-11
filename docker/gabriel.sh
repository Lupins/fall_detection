sudo nvidia-docker run --rm --name leite --runtime=nvidia -it -u docker -v "/home/leite/workspace/:/home/docker/workspace/" "semantix/v0.0.1" /bin/bash
