docker run -it --rm\
  --name test \
  --mount type=bind,source="$(pwd)"/shell/,target=/home/"${USER}"/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/home/"${USER}"/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/models/,target=/home/"${USER}"/project/models/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/home/"${USER}"/project/src/,readonly \
  --mount type=bind,source="$(pwd)"/tmp/,target=/home/"${USER}"/project/tmp/ \
  --env CID="$1" \
  flower-face_verification_image4pi:latest \
  /bin/bash