#!/bin/bash
# Check if the list of names is provided as an argument
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <PATH_TO_STORE_DATASETS> <BUCKET_NAME> <REGION_NAME> <name1> [name2] [name3] ..."
    exit 1
fi

PATH_TO_STORE_DATASETS=$1
BUCKET_NAME=$2
REGION_NAME=$3
shift 3

# Create an array from the remaining arguments (the datasets)
datasets=("$@")

mkdir -p "${PATH_TO_STORE_DATASETS}"

for dataset in "${datasets[@]}"
do
    file_name="${dataset}.yaml"
    has_shards=$(echo "${dataset}" | grep -c "shards")
    if [ "$has_shards" -gt 0 ]; then
        sharding="TAR"
    else
        sharding="null"
    fi
    
    echo "prefix_uri: s3://${BUCKET_NAME}/${dataset}/" > "${PATH_TO_STORE_DATASETS}/${file_name}"
    echo "region: ${REGION_NAME}" >> "${PATH_TO_STORE_DATASETS}/${file_name}"
    echo "sharding: ${sharding}" >> "${PATH_TO_STORE_DATASETS}/${file_name}"
done
