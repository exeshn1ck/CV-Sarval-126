#!/bin/bash
videos_path="./videos/"
imgs_path="./imgs/"
practice_path="./practice/"


docker run --rm -it --runtime=nvidia \
           --name courses_cv \
           --network=host \
           --device=/dev/video0:/dev/video0 \
           -e DISPLAY=:0 \
           -e QT_X11_NO_MITSHM=1 \
           --privileged \
           --shm-size=4gb \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v $PWD/$practice_path:/opt/app/ \
           -v $PWD/$videos_path:/opt/videos/ \
           -v $PWD/$imgs_path:/opt/imgs/ \
           -i yolov7:latest \
           /bin/bash
