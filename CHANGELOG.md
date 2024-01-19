## Unreleased

## v1.1.2 (January 19, 2024)

### New features
* Update crates and Mountpoint dependencies.
* Expose a logging method for enabling debug logs of the inner dependencies.

### Breaking changes
* No breaking changes.



## v1.1.1 (December 11, 2023)

### New features
* Update crates and Mountpoint dependencies.
* Avoid excessive memory consumption when utilizing s3map_dataset. Issue [#89](https://github.com/awslabs/s3-connector-for-pytorch/issues/89).
* Run all tests against S3 and S3 Express.

### Breaking changes
* No breaking changes.


## v1.1.0 (November 29, 2023)

### New features
* The Amazon S3 Connector for PyTorch now supports S3 Express One Zone directory buckets.

### Breaking changes
* No breaking changes.



## v1.0.0 (November 22, 2023)
* The Amazon S3 Connector for PyTorch delivers high throughput for PyTorch training jobs that access and store data in Amazon S3.

### New features
* S3IterableDataset and S3MapDataset, which allow building either an iterable-style or map-style dataset, using your S3 
stored data, by specifying an S3 URI (a bucket and optional prefix) and the region the bucket is in.
* Support for multiprocess data loading for the above datasets.
* S3Checkpoint, an interface for saving and loading model checkpoints directly to and from an S3 bucket.

### Breaking changes
* No breaking changes.
