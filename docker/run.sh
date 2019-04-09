sudo nvidia-docker run --rm --name leite --runtime=nvidia -it -u docker -v "/home/leite/workspace/:/home/docker/workspace/" "leite:fall" /bin/bash
