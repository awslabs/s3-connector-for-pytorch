# Amazon S3 Connector for PyTorch
The Amazon S3 Connector for PyTorch delivers high throughput for PyTorch training jobs that access or store data in Amazon S3. Using the S3 Connector for PyTorch 
automatically optimizes performance when downloading training data from and writing checkpoints to Amazon S3, eliminating the need to write your own code to list S3 buckets and manage concurrent requests.


 The S3 Connector for PyTorch provides implementations of PyTorch's [dataset primitives](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) that you can use to load training data from Amazon S3.
 It supports both [map-style datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets) for random data access patterns and 
 [iterable-style datasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) for streaming sequential data access patterns. 
 The S3 Connector for PyTorch also includes a checkpointing interface to save and load checkpoints directly to Amazon S3, without first saving to local storage.
   

## Getting Started

### Prerequisites

- Python 3.8 to 3.11 is installed. 
- PyTorch >= 2.0 (TODO: Check with PyTorch 1.x)

### Installation

```shell
pip install s3torchconnector
```

### Configuration

To read objects in a bucket that is not publicly accessible, or save checkpoints to such a bucket, AWS credentials must be provided through one of the following methods:

- Install and configure `awscli` by `aws configure`.
- Set credentials in the AWS credentials profile file on the local system, located at: `~/.aws/credentials` on Unix or macOS.
- Set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables.
- If you are using this library on an EC2 instance, specify an IAM role and then give the EC2 instance access to that role.

### Examples

End to end example of how to use `s3torchconnector` can be found under the [examples](examples/) directory.

## Contributing
We welcome contributions to Amazon S3 Connector for PyTorch. Please see [CONTRIBUTING](doc/CONTRIBUTING.md) For more information on how to report bugs or submit pull requests.

### Development
See [DEVELOPMENT](doc/DEVELOPMENT.md) for information about code style,
development process, and guidelines.


### Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/).

### Code of conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). See [CODE_OF_CONDUCT.md](doc/CODE_OF_CONDUCT.md) for more details.

## License

Amazon S3 Connector for PyTorch has a BSD-style license, as found in the [LICENSE](LICENSE) file.

