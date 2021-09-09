#!/bin/bash
# Usage: build_push_image.sh <tag_names>
# Running the following command will build and push the base image to Docker Hub 
#        runners/build_push_image.sh base
# Running the following command to build and push two stages
#        runners/build_push_image.sh base python-req

KEYS="$@"

if [[ -z $KEYS ]]; then
    KEYS="base"
    echo "No tag found in the arguments! Building and pushing default image humancompatibleai/imitation:${KEYS}"
fi

for key in $KEYS; do
    echo "----- Building and pushing humancompatibleai/imitation:${key} ..."
    BUILD="docker build --target ${key} -t humancompatibleai/imitation:${key} ."
    PUSH="docker push humancompatibleai/imitation:${key}"
    echo "${BUILD}"
    ${BUILD}
    echo "${PUSH}"
    # ${PUSH}  # Uncomment this line to push the image to Docker Hub 
done
