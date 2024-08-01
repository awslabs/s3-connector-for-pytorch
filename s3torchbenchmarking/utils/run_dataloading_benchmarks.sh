#!/bin/bash

# s3iterabledataset
# fsspec
# mountpoint
# mountpointcache

DATALOADER=$1

# Check if the list of datasets is provided as an argument
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <DATALOADER> <dataset1> [dataset2] [dataset3] ..."
    exit 1
fi

shift 1

# Create an array from the remaining arguments (the datasets)
datasets=("$@")

# work around for PyTorch's cuda clashing with installed locally from https://github.com/pytorch/pytorch/issues/119989
unset LD_LIBRARY_PATH

for dataset in "${datasets[@]}"; do
    if [[ "$dataset" == *"shards"* ]]; then
        s3torch-benchmark -cd conf -m -cn dataloading_sharded_vit "dataset=$dataset" "dataloader=$DATALOADER"
        s3torch-benchmark -cd conf -m -cn dataloading_sharded_ent "dataset=$dataset" "dataloader=$DATALOADER"
    else
        s3torch-benchmark -cd conf -m -cn dataloading_unsharded_1epochs "dataset=$dataset" "dataloader=$DATALOADER"
        s3torch-benchmark -cd conf -m -cn dataloading_unsharded_10epochs "dataset=$dataset" "dataloader=$DATALOADER"
    fi
done