#!/bin/bash
# Usage

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
    ${PUSH}
done
