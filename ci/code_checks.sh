#!/usr/bin/env bash

# If you change these, also change .circleci/config.yml.
SRC_FILES=(src/ tests/ experiments/ examples/ docs/conf.py setup.py ci/)
EXCLUDE_MYPY="(?x)(
  src/imitation/algorithms/preference_comparisons.py$
  | src/imitation/rewards/reward_nets.py$
  | src/imitation/algorithms/base.py$
  | src/imitation/scripts/train_preference_comparisons.py$
  | src/imitation/rewards/serialize.py$
  | src/imitation/scripts/common/train.py$
  | src/imitation/algorithms/mce_irl.py$
  | src/imitation/algorithms/density.py$
  | tests/algorithms/test_bc.py$
)"

set -x  # echo commands
set -e  # quit immediately on error

echo "Source format checking"
./ci/clean_notebooks.py --check ./docs/tutorials/
flake8 --darglint-ignore-regex '.*' "${SRC_FILES[@]}"
black --check --diff "${SRC_FILES[@]}"
codespell -I .codespell.skip --skip='*.pyc,tests/testdata/*,*.ipynb,*.csv' "${SRC_FILES[@]}"

if [ -x "$(which circleci)" ]; then
    circleci config validate
fi

if [ -x "$(which shellcheck)" ]; then
    find . -path ./venv -prune -o -name '*.sh' -print0 | xargs -0 shellcheck
fi

if [ "${skipexpensive:-}" != "true" ]; then
  echo "Type checking"
  pytype -j auto "${SRC_FILES[@]}"
  mypy "${SRC_FILES[@]}" --exclude "${EXCLUDE_MYPY}" --follow-imports=silent --show-error-codes

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
  files=$(git diff --cached --name-only --diff-filter=AMR | xargs -I'{}' find '{}' -name '*.py')
  if [[ ${files} != "" ]]; then
    IFS=' ' read -r -a file_array <<< "${files}"
    flake8 "${file_array[@]}"
  fi
fi

set +x

echo
echo
printf "\e[1;32mCode checks completed. No errors found\e[0m"
echo
