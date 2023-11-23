# Amazon S3 Connector for PyTorch
The Amazon S3 Connector for PyTorch delivers high throughput for PyTorch training jobs that access or store data in Amazon S3. Using the S3 Connector for PyTorch 
automatically optimizes performance when downloading training data from and writing checkpoints to Amazon S3, eliminating the need to write your own code to list S3 buckets and manage concurrent requests.


 Amazon S3 Connector for PyTorch provides implementations of PyTorch's [dataset primitives](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) that you can use to load training data from Amazon S3.
 It supports both [map-style datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets) for random data access patterns and 
 [iterable-style datasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) for streaming sequential data access patterns. 
 The S3 Connector for PyTorch also includes a checkpointing interface to save and load checkpoints directly to Amazon S3, without first saving to local storage.
   

## Getting Started

### Prerequisites

- Python 3.8 or greater is installed (Note: Using 3.12+ is not recommended as PyTorch does not support). 
- PyTorch >= 2.0 (TODO: Check with PyTorch 1.x)

### Installation

```shell
pip install s3torchconnector
```

Amazon S3 Connector for PyTorch supports only Linux via Pip for now. For other platforms, see [DEVELOPMENT](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/doc/DEVELOPMENT.md) for build instructions.

### Configuration

To use `s3torchconnector`, AWS credentials must be provided through one of the following methods:

- If you are using this library on an EC2 instance, specify an IAM role and then give the EC2 instance access to that role.
- Install and configure [`awscli`](https://aws.amazon.com/cli/) and run `aws configure`.
- Set credentials in the AWS credentials profile file on the local system, located at: `~/.aws/credentials` on Unix or macOS.
- Set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables.

### Examples

[API docs](http://awslabs.github.io/s3-connector-for-pytorch) are showing API of the public components. 
End to end example of how to use `s3torchconnector` can be found under the [examples](https://github.com/awslabs/s3-connector-for-pytorch/tree/main/examples) directory.

#### Sample Examples

The simplest way to use the S3 Connector for PyTorch is to construct a dataset, either a map-style or iterable-style dataset, by specifying an S3 URI (a bucket and optional prefix) and the region the bucket is located in:
```shell
from s3torchconnector import S3MapDataset, S3IterableDataset

# You need to update <BUCKET> and <PREFIX>
DATASET_URI="s3://<BUCKET>/<PREFIX>"
REGION = "us-east-1"

map_dataset = S3MapDataset.from_prefix(DATASET_URI, region=REGION)

iterable_dataset = S3IterableDataset.from_prefix(DATASET_URI, region=REGION)

# Randomly access to an item in map_dataset.
object = map_dataset[0]

# Learn about bucket, key, and content of the object
bucket = object.bucket
key = object.key
content = object.read()
len(content)

# Datasets are also iterators. 
for object in iterable_dataset:
  print(object.key)

```

In addition to data loading primitives, the S3 Connector for PyTorch also provides an interface for saving and loading model checkpoints directly to and from an S3 bucket. 

```shell
from s3torchconnector import S3Checkpoint

import torchvision
import torch

CHECKPOINT_URI="s3://<BUCKET>/<KEY>"
REGION = "us-east-1"
checkpoint = S3Checkpoint(region=REGION)

model = torchvision.models.resnet18()

# Save checkpoint to S3
with checkpoint.writer(CHECKPOINT_URI + "epoch0.ckpt") as writer:
    torch.save(model.state_dict(), writer)

# Load checkpoint from S3
with checkpoint.reader(CHECKPOINT_URI + "epoch0.ckpt") as reader:
    state_dict = torch.load(reader)

model.load_state_dict(state_dict)
```

## Contributing
We welcome contributions to Amazon S3 Connector for PyTorch. Please see [CONTRIBUTING](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/doc/CONTRIBUTING.md) For more information on how to report bugs or submit pull requests.

### Development
See [DEVELOPMENT](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/doc/DEVELOPMENT.md) for information about code style,
development process, and guidelines.


### Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/).

### Code of conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). See [CODE_OF_CONDUCT.md](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/doc/CODE_OF_CONDUCT.md) for more details.

## License

Amazon S3 Connector for PyTorch has a BSD 3-Clause License, as found in the [LICENSE](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/LICENSE) file.

