#!/usr/bin/env bash
#
# Template script to run other benchmarks (not meant to be used directly).

set -euo pipefail

while getopts "s:d:" opt; do
  case $opt in
  s) scenario=$OPTARG ;; # name of the scenario
  d) nvme_dir=$OPTARG ;; # mount point dir for saving checkpoints (will use NVMe drive)
  *) ;;
  esac
done

shift $((OPTIND - 1)) # remove all processed positional arguments from "$@"

# Prepare NVMe drive mount
if [[ -n $nvme_dir ]]; then
  ./utils/prepare_nvme.sh "$nvme_dir"
fi

# Run benchmarks; will write to DynamoDB table, if specified in the config (in `conf/aws/dynamodb.yaml`)
python ./src/s3torchbenchmarking/"$scenario"/benchmark.py -cd conf -cn "$scenario" +path="$nvme_dir" "$@"
