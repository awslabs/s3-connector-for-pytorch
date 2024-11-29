#!/usr/bin/env bash

nvme_dir="./nvme/dcp/"

# Prepare NVMe drive mount
./utils/prepare_nvme.sh $nvme_dir

# Run benchmarks
s3torch-benchmark-dcp -cd conf -cn dcp path=$nvme_dir
