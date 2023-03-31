docker run -it --rm\
  --name visualizer \
  --runtime nvidia \
  --mount type=bind,source="/tmp/.X11-unix",target="/tmp/.X11-unix",readonly\
  --mount type=bind,source="$(pwd)"/shell/,target=/home/"${USER}"/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/home/"${USER}"/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/models/,target=/home/"${USER}"/project/models/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/home/"${USER}"/project/src/,readonly \
  --mount type=bind,source="$(pwd)"/tmp/,target=/home/"${USER}"/project/tmp/,readonly \
  --mount type=bind,source="/tmp/argus_socket",target="/tmp/argus_socket",readonly \
  --mount type=bind,source="/proc/device-tree/compatible",target="/proc/device-tree/compatible" \
  --mount type=bind,source="/proc/device-tree/chosen",target="/proc/device-tree/chosen" \
  --mount type=bind,source="/sys/devices",target="/sys/devices" \
  --mount type=bind,source="/sys/class/gpio",target="/sys/class/gpio" \
  --privileged \
  --env DISPLAY="$DISPLAY" \
  --device /dev/video0:/dev/video0:mwr \
  flower-face_verification_image4jetson:latest \
  /bin/bash