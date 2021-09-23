#!/usr/bin/env bash

# If you change these, also change .circleci/config.yml.
SRC_FILES=(src/ tests/ experiments/ examples/ docs/conf.py setup.py)

set -x  # echo commands
set -e  # quit immediately on error

echo "Source format checking"
flake8 --disable-darglint ${SRC_FILES[@]}
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
  # We run flake8 rather than darglint directly to work around:
  # https://github.com/terrencepreilly/darglint/issues/21
  # so noqa's are respected outside docstring.
  # If we got to this point, flake8 already passed, so this should
  # only find new darglint-specific errors.
  files=$(git diff --cached --name-only --diff-filter=AMR ${against} | xargs -I'{}' find '{}' -name '*.py'$)
  if [[ ${files} -ne "" ]]; then
    flake8 ${files}
  fi
fi
