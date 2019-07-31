#!/usr/bin/env bash

SOURCE_DIRS=(src/ tests/ experiments/)

echo "pytype ${SOURCE_DIRS[@]}"
pytype "${SOURCE_DIRS[@]}"
