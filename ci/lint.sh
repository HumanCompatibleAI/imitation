#!/usr/bin/env bash

SOURCE_DIRS=("imitation/" "tests/")

RET=0

echo "PEP8 compliance"
echo "flake8 --version"
flake8 --version

echo "flake8"
flake8 ${SOURCE_DIRS[@]}
RET=$(($RET + $?))

echo "Check for common typos"
echo "codespell --version"
codespell --version

echo "codespell"
codespell -I .codespell.skip --skip='*.pyc,tests/data/*' ${SOURCE_DIRS[@]}
RET=$((RET + $?))

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
