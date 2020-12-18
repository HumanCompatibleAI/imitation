#!/bin/bash

# Always sync to ../data, relative to this script.
SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="$(dirname ${SCRIPT_DIR})"
DATA_DIR="${PROJECT_DIR}/data"

DRY_RUN_MODE=false
ALL_MODE=false
TEMP=$(getopt -o '' -l all,dryrun -- $@)

if [[ $? != 0 ]]; then exit 1; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    --dryrun)
      DRY_RUN_MODE=true
      shift
      ;;
    --all)
      # Download all data (by default, skips large meta-data/log files).
      # As of 2019.Nov.12, the difference in download size was 77MB vs ~800MB.
      ALL_MODE=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unrecognized flag $1" >&2
      exit 1
      ;;
  esac
done

FLAGS=()

if [[ $ALL_MODE != "true" ]]; then
  for excl_pat in '*/monitor/*' '*/parallel/*' '*/sacred/*' '*events.out.tfevents*'; do
    FLAGS=("${FLAGS[@]}" "--exclude" "${excl_pat}")
  done
fi

if [[ $DRY_RUN_MODE == "true" ]]; then
  FLAGS=("${FLAGS[@]}" "--dryrun")
elif [[ -d "${DATA_DIR}" ]]; then
  rm -r "${DATA_DIR}"
fi

aws --no-sign-request s3 sync "${FLAGS[@]}" s3://shwang-chai/public/data/ "${DATA_DIR}"
