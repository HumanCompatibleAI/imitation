#!/usr/bin/env bash

SOURCE_DIRS=(imitation/ tests/)

echo "pytype ${SOURCE_DIRS[@]}"
pytype "${SOURCE_DIRS[@]}"
