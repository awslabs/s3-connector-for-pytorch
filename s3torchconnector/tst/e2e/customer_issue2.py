#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from s3torchconnector import S3MapDataset, S3IterableDataset

# DATASET_URI="s3://s3torchconnector-customer-issues/100mb/"
DATASET_URI="s3://s3torchconnector-customer-issues/300mb/"
REGION = "eu-north-1"

map_dataset = S3MapDataset.from_prefix(DATASET_URI, region=REGION)
iterable_dataset = S3IterableDataset.from_prefix(DATASET_URI, region=REGION)

# Randomly access to an item in map_dataset.
object = map_dataset[0]

# Learn about bucket, key, and content of the object
bucket = object.bucket
key = object.key
content = object.read()
len(content)