#!/usr/bin/env bash

# If you change these, also change .circleci/config.yml.
SRC_FILES=(src/ tests/ experiments/ examples/ docs/conf.py setup.py)

set -x  # echo commands
set -e  # quit immediately on error

echo "Source format checking"
flake8 ${SRC_FILES[@]}
black --check --diff ${SRC_FILES[@]}
codespell -I .codespell.skip --skip='*.pyc,tests/testdata/*,*.ipynb,*.csv' ${SRC_FILES[@]}

if [ -x "`which circleci`" ]; then
    circleci config validate
fi

echo "files changed"
echo $*

if [ "$skipexpensive" != "true" ]; then
  echo "Type checking"
  pytype -j auto ${SRC_FILES[@]}

  echo "Building docs (validates docstrings)"
  pushd docs/
  make clean
  make html
  popd

  echo "Darglint on diff"
  pushd ci/  # darglint will read config from ci/.darglint, enabling it
  git diff --cached --name-only ${against} | parallel darglint
  popd
fi
