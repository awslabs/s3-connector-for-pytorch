## TBD

### New features
* Add support for Python 3.13 

### Bug fixes

### Other changes
* Upload image directories once per session to mitigate CI/CD S3Express boto3 400 errors (#347)

### Breaking changes

## v1.4.2 (July 14, 2025)

### New features
* Implement range-based S3 Reader for efficient partial read patterns (#339)
* Implemented AWS profile credentials support as part of S3ClientConfig (#341)

### Bug fixes
* Enable multiple CRT clients per process with different configs (#340)

### Other changes
* Expose configuration needed for testing profile credentials (#344)
* Keep usage of manylinux_2014 for wider suport of glibc (#342, #343)

### Breaking changes
* No breaking changes.

## v1.4.1 (May 20, 2025)

### New features
* Add S3 prefix strategies to prevent throttling in distributed checkpointing (#327)

### Bug fixes
* Consume mountpoint-s3-client 0.14.1 to address AWS_ERROR_HTTP_RESPONSE_FIRST_BYTE_TIMEOUT error (#332)
* Prevent S3Writer stream closure on exception to preserve original error context (#331)
* Add PyTorch 2.7.0 support (#329)

### Other changes
* Add Python version to user agent string (#333)
* Address DCP test hanging and distributed test errors (#330)
* Update GitHub Actions runners to ubuntu-24.04 (#328)

### Breaking changes
* No breaking changes.

## v1.4.0 (April 9, 2025)

### New features
* Introducing support of FSDP in DCP benchmark (#313)
* Exposing max_attempts setting through S3ClientConfig and adding support of S3ClientConfig to DCP (#322)

### Bug fixes
* Consume mountpoint-s3-client changes for race condition on GET request path, that may lead to an empty response 

### Other changes
* Update Pyo3 version (#314)

### Breaking changes
* We changed the way we handle `fork` operations. The CRT client used for communication with S3 is not stable 
during `fork` due to global state and locks held by background threads. To address this, we now clean up all 
existing CRT clients before a `fork` operation and create new CRT clients in the child process. 
This change prevents segfaults and hanging GET requests for training workloads that rely on `fork` (#320)  

## v1.3.2 (February 5, 2025)

### New features
* Consume mountpoint-s3-client changes with support for dots in bucket name for COPY operation introduced in CRT (#300)
* Escape special characters in rename operation (#297)
* Handle torch.load changes in PyTorch 2.6 (#306)
* Remove dependency on mountpoint-s3-crt (#307)

### Breaking changes
* Internal S3Client now returns `HeadObjectResult` instead of `ObjectInfo`. The only difference is that `HeadObjectResult` doesn't have `key` field. 

## v1.3.1 (January 10, 2025)

### New features
* Fix Rust build (#275)
* Update user agent for DCP (#276)
* Address Rust security issue (#279)
* Refactor(benchmarks): Overhaul Lightning Checkpointing, DCP, dataset scenarios; add DynamoDB writes and results exploitation notebook (#274, #280, #285, #286)
* Add single rank PyTorch checkpoint benchmark (#289)
* Update torch version restriction (<2.5.0) and bind torchdata to last version with DataPipes (#283)

### Breaking changes
* No breaking changes.

## v1.3.0 (November 21, 2024)

### New features
* Add support of PyTorch distributed checkpoints (#269)
* Extend benchmark framework to support distributed checkpoints (#269)
* Add support of distributed training to S3IterableDataset (#269)

### Breaking changes
* No breaking changes. 

## v1.2.7 (October 29, 2024)

### New features
* Add support for CRT retries (awslabs/mountpoint-s3#1069).
* Add support for `CopyObject` API (#242).

### Breaking changes
* No breaking changes.

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
