# Troubleshooting

If `s3torchconnector` is not working as expected, please check [Github issues](https://github.com/awslabs/s3-connector-for-pytorch/issues) to see if your issue has already been addressed. If not, feel free to [create a GitHub issue](https://github.com/awslabs/s3-connector-for-pytorch/issues/new/choose) with all the details.

For debug logging for mountpoint-s3-client and CRT logs, please refer to [Enabling Debug Logging](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/DEVELOPMENT.md#enabling-debug-logging) section in the DEVELOPMENT doc. 

### DCPOptimizedS3Reader Errors

`S3StorageReader` uses `DCPOptimizedS3Reader` (created with `S3ReaderConstructor.dcp_optimized()`) by default (v1.5.0+) for improved performance. See [PR #378](https://github.com/awslabs/s3-connector-for-pytorch/pull/378) for more details about the reader. 

If you encounter errors with the default reader, please [submit a GitHub issue](https://github.com/awslabs/s3-connector-for-pytorch/issues) describing your use case. We'd like to understand your scenario and potentially extend `DCPOptimizedS3Reader` to support it, so you can benefit from the performance improvements.

For unsupported or non-DCP access patterns, use the generic reader:

```py
from s3torchconnector import S3ReaderConstructor
from s3torchconnector.dcp import S3StorageReader

storage_reader = S3StorageReader(
    region=REGION, 
    path=CHECKPOINT_URI,
    reader_constructor=S3ReaderConstructor.default()
)
```
