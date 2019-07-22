#!/usr/bin/env bash

SOURCE_DIRS=(src/ tests/)

echo "pytype ${SOURCE_DIRS[@]}"
pytype "${SOURCE_DIRS[@]}"
