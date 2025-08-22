#!/usr/bin/env bash
#
# Template script to run other benchmarks (not meant to be used directly).

set -euo pipefail

# Check for --save and --load flags before getopts
load_mode=false
save_mode=false
filtered_args=()
for arg in "$@"; do
  case $arg in
    --load) load_mode=true ;;
    --save) save_mode=true ;;
    *) filtered_args+=("$arg") ;;
  esac
done

# Set filtered arguments
set -- "${filtered_args[@]}"

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
# Use load_benchmark.py if -l flag is provided, otherwise use default benchmark.py
if [[ "${load_mode:-}" == "true" ]]; then
  python ./src/s3torchbenchmarking/"$scenario"/load_benchmark.py -cd conf -cn "${scenario}_load" +path="$nvme_dir" "$@"
elif  [[ "${save_mode:-}" == "true" ]]; then
  python ./src/s3torchbenchmarking/"$scenario"/save_benchmark.py -cd conf -cn "${scenario}_save" +path="$nvme_dir" "$@"
elif [[ "$scenario" == "dcp_ddp" || "$scenario" == "dcp_fsdp" ]]; then
  echo "No flags detected, running DCP save benchmarks. To run DCP load benchmarks, use the --load flag"
  python ./src/s3torchbenchmarking/"$scenario"/save_benchmark.py -cd conf -cn "${scenario}_save" +path="$nvme_dir" "$@"
else
  python ./src/s3torchbenchmarking/"$scenario"/benchmark.py -cd conf -cn "$scenario" +path="$nvme_dir" "$@"
fi

