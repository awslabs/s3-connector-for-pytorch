#!/usr/bin/env bash
#
# Template script to run other benchmarks (not meant to be used directly).

set -euo pipefail

# Parse flags and arguments
mode="default"
while [[ $# -gt 0 ]]; do
  case $1 in
    --load) mode="load"; shift ;;
    --save) mode="save"; shift ;;
    -s) scenario="$2"; shift 2 ;;
    -d) nvme_dir="$2"; shift 2 ;;
    *) break ;;
  esac
done

# Prepare NVMe drive mount
if [[ -n $nvme_dir ]]; then
  ./utils/prepare_nvme.sh "$nvme_dir"
fi


# Determine script and config
case $mode in
  load) script="load_benchmark.py"; config="${scenario}_load" ;;
  save) script="save_benchmark.py"; config="${scenario}_save" ;;
  *) 
    if [[ $scenario =~ ^dcp_(ddp|fsdp)$ ]]; then
      echo "No flags detected, running DCP save benchmarks. To run DCP load benchmarks, use the --load flag"
      script="save_benchmark.py"; config="${scenario}_save"
    else
      script="benchmark.py"; config="$scenario"
    fi ;;
esac

# Run benchmarks; will write to DynamoDB table, if specified in the config (in `conf/aws/dynamodb.yaml`)
python "./src/s3torchbenchmarking/$scenario/$script" -cd conf -cn "$config" +path="$nvme_dir" "$@"