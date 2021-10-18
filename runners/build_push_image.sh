#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

__usage="build_push_image.sh - Building and pushing Docker image

Usage: build_push_image.sh [options] [tags]

options:
  -h, --help                show brief help
  -p, --push                push the image to DockerHub
tags:
  base                      base stage image
  python-req                python-req stage image
"

KEYS=""
PUSH=0

while test $# -gt 0; do
  case "$1" in
  -p | --push)
    PUSH=1 # Push the image to Docker Hub
    shift
    ;;
  base)
    KEYS+="base "
    shift
    ;;
  python-req)
    KEYS+="python-req "
    shift
    ;;
  -h | --help)
    echo "${__usage}"
    exit 0
    ;;
  *)
    echo "Unrecognized flag $1" >&2
    exit 1
    ;;
  esac
done

if [[ -z $KEYS ]]; then
  KEYS="base"
  echo "No tag found in the arguments! Building default image humancompatibleai/imitation:${KEYS}"
fi

for key in $KEYS; do
  echo "----- Building humancompatibleai/imitation:${key} ..."
  BUILD_CMD="docker build --target ${key} -t humancompatibleai/imitation:${key} ."
  PUSH_CMD="docker push humancompatibleai/imitation:${key}"

  # Build image
  ${BUILD_CMD}

  # Push image if prompted
  if [[ $PUSH == 1 ]]; then
    echo "----- Pushing humancompatibleai/imitation:${key} ..."
    ${PUSH_CMD}
  fi
done
