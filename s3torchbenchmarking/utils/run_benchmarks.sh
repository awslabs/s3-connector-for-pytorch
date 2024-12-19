#!/usr/bin/env bash
#
# Template script to run other benchmarks (not to be used directly).

set -euo pipefail

scenario=$1 # name of the scenario
nvme_dir=$2 # mount point for saving checkpoints (will use NVMe drive)

shift 2

# Prepare NVMe drive mount
./utils/prepare_nvme.sh "$nvme_dir"

# Run benchmarks; will write to DynamoDB table, if specified in the config
python ./src/s3torchbenchmarking/"$scenario"/benchmark.py -cd conf -cn "$scenario" +path="$nvme_dir" "$@"
