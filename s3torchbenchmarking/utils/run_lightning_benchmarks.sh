#!/usr/bin/env bash

RESULTS_BUCKET_NAME=$1
RESULTS_PREFIX=$2

nvme_dir="./nvme/lightning/"

# Prepare NVMe drive mount
./utils/prepare_nvme.sh $nvme_dir

# Run benchmarks
s3torch-benchmark-lightning -cd conf -cn lightning_checkpointing path=$nvme_dir

# Upload results to an S3 bucket
python ./utils/upload_colated_results_to_s3.py "./multirun" "${RESULTS_BUCKET_NAME}" "${RESULTS_PREFIX}" "checkpoint"
