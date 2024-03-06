# Amazon S3 Connector for PyTorch
The Amazon S3 Connector for PyTorch delivers high throughput for PyTorch training jobs that access or store data in 
Amazon S3. Using the S3 Connector for PyTorch 
automatically optimizes performance when downloading training data from and writing checkpoints to Amazon S3, 
eliminating the need to write your own code to list S3 buckets and manage concurrent requests.


Amazon S3 Connector for PyTorch provides implementations of PyTorch's 
[dataset primitives](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) that you can use to load 
training data from Amazon S3.
It supports both [map-style datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets) for random data 
access patterns and [iterable-style datasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) for 
streaming sequential data access patterns. 
The S3 Connector for PyTorch also includes a checkpointing interface to save and load checkpoints directly to 
Amazon S3, without first saving to local storage.
   

## Getting Started

### Prerequisites

- Python 3.8 or greater is installed (Note: Using 3.12+ is not recommended as PyTorch does not support). 
- PyTorch >= 2.0 (TODO: Check with PyTorch 1.x)

### Installation

```shell
pip install s3torchconnector
```

Amazon S3 Connector for PyTorch supports only Linux via Pip for now. For other platforms, see 
[DEVELOPMENT](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/doc/DEVELOPMENT.md) for build instructions.

### Configuration

To use `s3torchconnector`, AWS credentials must be provided through one of the following methods:

- If you are using this library on an EC2 instance, specify an IAM role and then give the EC2 instance access to 
that role.
- Install and configure [`awscli`](https://aws.amazon.com/cli/) and run `aws configure`.
- Set credentials in the AWS credentials profile file on the local system, located at: `~/.aws/credentials` 
on Unix or macOS.
- Set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables.

### Examples

[API docs](http://awslabs.github.io/s3-connector-for-pytorch) are showing API of the public components. 
End to end example of how to use `s3torchconnector` can be found under the 
[examples](https://github.com/awslabs/s3-connector-for-pytorch/tree/main/examples) directory.

#### Sample Examples

The simplest way to use the S3 Connector for PyTorch is to construct a dataset, either a map-style or iterable-style 
dataset, by specifying an S3 URI (a bucket and optional prefix) and the region the bucket is located in:
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

In addition to data loading primitives, the S3 Connector for PyTorch also provides an interface for saving and loading 
model checkpoints directly to and from an S3 bucket. 

```shell
from s3torchconnector import S3Checkpoint

import torchvision
import torch

CHECKPOINT_URI="s3://<BUCKET>/<KEY>/"
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

Using datasets or checkpoints with
[Amazon S3 Express One Zone](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-express-one-zone.html) 
directory buckets requires only to update the URI, following `base-name--azid--x-s3` bucket name format.
For example, assuming the following directory bucket name `my-test-bucket--usw2-az1--x-s3` with the Availability Zone ID
usw2-az1, then the URI used will look like: `s3://my-test-bucket--usw2-az1--x-s3/<PREFIX>` (**please note that the 
prefix for Amazon S3 Express One Zone should end with '/'**), paired with region us-west-2.


## Lightning Integration

Amazon S3 Connector for PyTorch includes an integration for PyTorch Lightning, featuring S3LightningCheckpoint, an 
implementation of Lightning's CheckpointIO. This allows users to make use of Amazon S3 Connector for PyTorch's S3 
checkpointing functionality with Pytorch Lightning.

### Getting Started

#### Installation

```sh
pip install s3torchconnector[lightning]
```

### Examples

End to end examples for the Pytorch Lightning integration can be found in the 
[examples/lightning](https://github.com/awslabs/s3-connector-for-pytorch/tree/main/examples/lightning) directory

```py
from lightning import Trainer
from s3torchconnector.lightning import S3LightningCheckpoint

...

s3_checkpoint_io = S3LightningCheckpoint("us-east-1")
trainer = Trainer(
    plugins=[s3_checkpoint_io],
    default_root_dir="s3://bucket_name/key_prefix/"
)
trainer.fit(model)
```

## Using S3 Versioning to Manage Checkpoints
When working with model checkpoints, users can accidentally overwrite or delete critical files. This can happen due to incorrect configuration, multiple hosts accessing the same storage path concurrently, or race conditions caused by usage of the async checkpoint interface.
Amazon S3 bucket versioning provides a way to protect checkpoints from inadvertent loss. With versioning enabled, S3 preserves every version of every object in the bucket.

When versioning is enabled on an S3 bucket:
* S3 stores a version history of your objects. The current version is returned by default.
* You can retrieve previous versions of your objects.
* Unintended overwrites and deletions are reversible.

To enable versioning for S3, please follow the instruction [here](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html) instruction. Please note that this incurs extra costs for storing object versions. For more details please check **How am I charged for using Versioning?** section in [Amazon S3 FAQs](https://aws.amazon.com/s3/faqs/?nc=sn&loc=7#Billing).
S3 Versioning is available only on S3 Standard, it is not supported by S3 Express One Zone.

## Contributing
We welcome contributions to Amazon S3 Connector for PyTorch. Please 
see [CONTRIBUTING](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/doc/CONTRIBUTING.md) 
For more information on how to report bugs or submit pull requests.

### Development
See [DEVELOPMENT](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/doc/DEVELOPMENT.md) for information 
about code style, development process, and guidelines.

### Compatibility with other storage services
S3 Connector for PyTorch delivers high throughput for PyTorch training jobs that access or store data in Amazon S3. 
While it may be functional against other storage services that use S3-like APIs, they may inadvertently break when we 
make changes to better support Amazon S3. We welcome contributions of minor compatibility fixes or performance 
improvements for these services if the changes can be tested against Amazon S3.

### Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS Security via our 
[vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/).

### Code of conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
See [CODE_OF_CONDUCT.md](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/doc/CODE_OF_CONDUCT.md) for 
more details.

## License

Amazon S3 Connector for PyTorch has a BSD 3-Clause License, as found in the 
[LICENSE](https://github.com/awslabs/s3-connector-for-pytorch/blob/main/LICENSE) file.

