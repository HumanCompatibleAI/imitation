#!/usr/bin/env bash

# If you change these, also change .circleci/config.yml.
SRC_FILES=(src/ tests/ experiments/ examples/ docs/conf.py setup.py)

set -x  # echo commands
set -e  # quit immediately on error

echo "Source format checking"
flake8 ${SRC_FILES[@]}
black --check --diff ${SRC_FILES[@]}
codespell -I .codespell.skip --skip='*.pyc,tests/data/*,*.ipynb,*.csv' ${SRC_FILES[@]}

if [ -x "`which circleci`" ]; then
    circleci config validate
fi

if [ "$skipexpensive" != "true" ]; then
  echo "Building docs (validates docstrings)"
  pushd docs/
  make clean
  make html
  popd

  echo "Type checking"
  pytype -j auto ${SRC_FILES[@]}
fi
