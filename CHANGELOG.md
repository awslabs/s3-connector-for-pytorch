## v1.2.7 (October 29, 2024)

### New features
* Add support for CRT retries (awslabs/mountpoint-s3#1069).
* Add support for `CopyObject` API (#242).

## v1.2.6 (October 9, 2024)

### New features
* Add support of PyTorch Lightning checkpoints to benchmark suit (#226).

### Bug fixes
* Fix potential race condition while instantiating the `S3Client` (#237).

### Breaking changes
* No breaking changes.

## v1.2.5 (September 11, 2024)
* Enhanced error logging.
* Support tell for S3writer.
* Path-style addressing support.
* Update crates and Mountpoint dependencies.

### Breaking changes
* No breaking changes.

## v1.2.4 (July 29, 2024)

### New features
* Update crates and Mountpoint dependencies.

### Breaking changes
* No breaking changes.

## v1.2.3 (April 11, 2024)

### New features
* Update `S3ClientConfig` to pass in the configuration for allowing unsigned requests, under boolean flag `unsigned`.
* Improve the performance of `S3Reader` when utilized with `pytorch.load` by incorporating support for the `readinto` method.
* **[Experimental]** Add support for passing an optional custom endpoint to `S3LightningCheckpoint` constructor method.

### Breaking changes
* No breaking changes.

## v1.2.2 (March 22, 2024)

### New features
* Expose a new class, `S3ClientConfig`, with `throughput_target_gbps` and `part_size` parameters of the inner S3 client.

### Breaking changes
* No breaking changes.

## v1.2.1 (March 14, 2024)

### Breaking changes
* Separate completely Rust logs and Python logs. Logs from Rust components used for debugging purposes 
are configured through the following environment variables: `S3_TORCH_CONNECTOR_DEBUG_LOGS`, 
`S3_TORCH_CONNECTOR_LOGS_DIR_PATH`.

## v1.2.0 (March 13, 2024)

### New features
* Add PyTorch Lightning checkpoints support

### Bug fixes / Improvements
* Fix deadlock when enabling CRT debug logs. Removed former experimental method _enable_debug_logging().
* Refactor User-Agent setup for extensibility.
* Update lightning User-Agent prefix to `s3torchconnector/{__version__} (lightning; {lightning.__version__}`.

### Breaking changes
* No breaking changes.


## v1.1.4 (February 26, 2024)

### New features
* Support for Python 3.12.
* Additional logging when constructing Datasets, and when making requests to S3.
* Provide tooling for running benchmarks for S3 Connector for Pytorch.
* Update crates and Mountpoint dependencies.
* **[Experimental]** Allow passing in the S3 endpoint URL to Dataset constructors.

### Bug fixes

* HeadObject is no longer called when constructing datasets with `from_prefix` and seeking relative to end of file.

### Breaking changes
* No breaking changes.


## v1.1.3 (January 25, 2024)

### New features
* Update crates and Mountpoint dependencies.

### Breaking changes
* No breaking changes.


## v1.1.2 (January 19, 2024)

### New features
* Update crates and Mountpoint dependencies.
* Expose a logging method for enabling debug logs of the inner dependencies.

### Breaking changes
* No breaking changes.


## v1.1.1 (December 11, 2023)

### New features
* Update crates and Mountpoint dependencies.
* Avoid excessive memory consumption when utilizing `S3MapDataset`. 
Issue [#89](https://github.com/awslabs/s3-connector-for-pytorch/issues/89).
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
* `S3IterableDataset` and `S3MapDataset`, which allow building either an iterable-style or map-style dataset, using your S3
stored data, by specifying an S3 URI (a bucket and optional prefix) and the region the bucket is in.
* Support for multiprocess data loading for the above datasets.
* `S3Checkpoint`, an interface for saving and loading model checkpoints directly to and from an S3 bucket.

### Breaking changes
* No breaking changes.
