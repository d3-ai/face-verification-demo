docker run -it --rm\
  -u $UID:$GID \
  --name test \
  --mount type=bind,source="$(pwd)"/shell/,target=/home/"${USER}"/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/home/"${USER}"/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/local/,target=/home/"${USER}"/project/local/,readonly \
  --mount type=bind,source="$(pwd)"/models/,target=/home/"${USER}"/project/models/,readonly \
  --mount type=bind,source="$(pwd)"/face_verification/,target=/home/"${USER}"/project/face_verification/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/home/"${USER}"/project/src/,readonly \
  --env CID="$1" \
  flower_mockup_pi4:latest