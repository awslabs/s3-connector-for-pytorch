#!/usr/bin/env bash
#
# Template script to run other benchmarks (not to be used directly).

set -euo pipefail

scenario=$1        # name of the scenario
region=${2:-}      # DynamoDB region (for writing results)
table=${3:-}       # DynamoDB table name (for writing results)
nvme_dir="./nvme/" # local path for saving checkpoints

# Prepare NVMe drive mount
./utils/prepare_nvme.sh "$nvme_dir"

# Run benchmarks; will write to DynamoDB table, if provided
dynamodb_args=()
if [ -n "$region" ] && [ -n "$table" ]; then
  dynamodb_args+=(+dynamodb.region="$region" +dynamodb.table="$table")
fi

python ./src/s3torchbenchmarking/"$scenario"/benchmark.py -cd conf -cn "$scenario" path="$nvme_dir" "${dynamodb_args[@]}"
