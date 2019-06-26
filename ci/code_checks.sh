#!/usr/bin/env bash

SOURCE_DIRS="imitation"

RET=0

echo "flake8 --version"
flake8 --version

echo "Linting code"
flake8 ${SOURCE_DIRS}
RET=$(($RET + $?))

echo "isort --version-number"
isort --version-number

echo "Checking import order using isort"
isort --recursive --diff ${SOURCE_DIRS}
isort --recursive --check-only ${SOURCE_DIRS}
RET=$(($RET + $?))

echo "Building docs (validates docstrings)"
pushd docs/
make clean
RET=$((RET + $?))
make html
RET=$((RET + $?))
popd

if [ $RET -ne 0 ]; then
    echo "Linting failed."
fi
exit $RET
