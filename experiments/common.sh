# shellcheck shell=bash

# Common variables for experiment scripts

export GNU_DATE=date
export GNU_GETOPT=getopt
if [[ "$OSTYPE" == "darwin"* ]]; then
  export GNU_DATE=gdate
  if [[ $(uname -m) == 'arm64' ]]; then
    export GNU_GETOPT=/opt/homebrew/opt/gnu-getopt/bin/getopt
  else
    export GNU_GETOPT=/usr/local/opt/gnu-getopt/bin/getopt
  fi
fi

TIMESTAMP=$($GNU_DATE --iso-8601=seconds)
export TIMESTAMP

# Set OMP_NUM_THREADS=2 if not yet exported.
# This is important because parallel runs of PyTorch often throttle due to
# CPU contention unless this is set to a low number.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
