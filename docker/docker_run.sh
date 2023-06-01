DOCKER_CONTAINER_NAME=v1.1
DOCKER_IMAGE_NAME=clrnet:$DOCKER_CONTAINER_NAME
DOCKER_USER=dgist
SHARD_FOLDER_PATH=`pwd`/..
DOCKER_SHARD_FOLDER_PATH=/home/$DOCKER_USER/catkin_ws
ROS_MASTER_IP=192.168.10.20
ROS_PC_IP=192.168.10.20

nvidia-docker run \
       --runtime nvidia \
       --ipc=host \
       --gpus all \
       -it -P --rm --name $DOCKER_CONTAINER_NAME\
       --volume="$SHARD_FOLDER_PATH:$DOCKER_SHARD_FOLDER_PATH:rw" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --env="ROS_MASTER_URI=http://${ROS_MASTER_IP}:11311" \
       --env="ROS_IP=${ROS_PC_IP}" \
       --env="ROS_MARSET_URI=$ROS_MASTER_URI" \
       --env="ROS_IP=$ROS_IP" \
       -u $DOCKER_USER \
       --privileged -v /dev/bus/usb:/dev/bus/usb \
       --net=host \
       $DOCKER_IMAGE_NAME
