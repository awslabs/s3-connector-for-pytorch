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

- Python 3.8-3.12 is supported. 
- PyTorch >= 2.0 (TODO: Check with PyTorch 1.x)

### Installation

```shell
pip install s3torchconnector
```

Amazon S3 Connector for PyTorch supports pre-build wheels via Pip only for Linux and MacOS for now. For other platforms,
see [DEVELOPMENT](DEVELOPMENT.md) for build instructions.

### Configuration

To use `s3torchconnector`, AWS credentials must be provided through one of the following methods:

- If you are using this library on an EC2 instance, specify an IAM role and then give the EC2 instance access to 
that role.
- Install and configure [`awscli`](https://aws.amazon.com/cli/) and run `aws configure`.
- Set credentials in the AWS credentials profile file on the local system, located at: `~/.aws/credentials` 
on Unix or macOS.
- Set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables.
- Pass the name of the desired profile that you have configured in `~/.aws/config` and `~/.aws/credentials` to the `S3ClientConfig` object.

### Examples

[API docs](http://awslabs.github.io/s3-connector-for-pytorch) are showing API of the public components. 
End to end example of how to use `s3torchconnector` can be found under the [examples](examples) directory.

#### Sample Examples

The simplest way to use the S3 Connector for PyTorch is to construct a dataset, either a map-style or iterable-style 
dataset, by specifying an S3 URI (a bucket and optional prefix) and the region the bucket is located in:
```py
from s3torchconnector import S3MapDataset, S3IterableDataset

# You need to update <BUCKET> and <PREFIX>
DATASET_URI="s3://<BUCKET>/<PREFIX>"
REGION = "us-east-1"

iterable_dataset = S3IterableDataset.from_prefix(DATASET_URI, region=REGION)

# Datasets are also iterators. 
for item in iterable_dataset:
  print(item.key)

# S3MapDataset eagerly lists all the objects under the given prefix 
# to provide support of random access.  
# S3MapDataset builds a list of all objects at the first access to its elements or 
# at the first call to get the number of elements, whichever happens first.
# This process might take some time and may give the impression of being unresponsive.
map_dataset = S3MapDataset.from_prefix(DATASET_URI, region=REGION)

# Randomly access to an item in map_dataset.
item = map_dataset[0]

# Learn about bucket, key, and content of the object
bucket = item.bucket
key = item.key
content = item.read()
len(content)
```

In addition to data loading primitives, the S3 Connector for PyTorch also provides an interface for saving and loading 
model checkpoints directly to and from an S3 bucket. 

```py
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


## Distributed checkpoints

### Overview

Amazon S3 Connector for PyTorch provides robust support for PyTorch distributed checkpoints. This feature includes:

- `S3StorageWriter`: Implementation of PyTorch's StorageWriter interface.

- `S3StorageReader`: Implementation of PyTorch's StorageReader interface. Supports configurable reading strategies via the `reader_constructor` parameter (see [Reader Configurations](#reader-configurations)).
- `S3FileSystem`: An implementation of PyTorch's FileSystemBase.

These tools enable seamless integration of Amazon S3 with 
[PyTorch Distributed Checkpoints](https://pytorch.org/docs/stable/distributed.checkpoint.html), 
allowing efficient storage and retrieval of distributed model checkpoints.

### Prerequisites and Installation

PyTorch 2.3 or newer is required.
To use the distributed checkpoints feature, install S3 Connector for PyTorch with the `dcp` extra:

```sh
pip install s3torchconnector[dcp]
```

### Sample Example

End-to-end examples for using distributed checkpoints with S3 Connector for PyTorch 
can be found in the [examples/dcp](examples/dcp) directory.

```py
from s3torchconnector.dcp import S3StorageWriter, S3StorageReader

import torchvision
import torch.distributed.checkpoint as DCP

# Configuration
CHECKPOINT_URI = "s3://<BUCKET>/<KEY>/"
REGION = "us-east-1"

model = torchvision.models.resnet18()

# Save distributed checkpoint to S3
s3_storage_writer = S3StorageWriter(region=REGION, path=CHECKPOINT_URI)
DCP.save(
    state_dict=model.state_dict(),
    storage_writer=s3_storage_writer,
)

# Load distributed checkpoint from S3
model = torchvision.models.resnet18()
model_state_dict = model.state_dict()
s3_storage_reader = S3StorageReader(region=REGION, path=CHECKPOINT_URI)
DCP.load(
    state_dict=model_state_dict,
    storage_reader=s3_storage_reader,
)
model.load_state_dict(model_state_dict)
```

## S3 Prefix Strategies for Distributed Checkpointing

S3StorageWriter implements various prefix strategies to optimize checkpoint organization in S3 buckets.
These strategies are specifically designed to prevent throttling (503 Slow Down errors) in high-throughput scenarios 
by implementing S3 key naming best practices as outlined in 
[Best practices design patterns: optimizing Amazon S3 performance](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html).

When many distributed training processes write checkpoints simultaneously, the prefixing strategies help distribute
the load across multiple S3 partitions.

### Available Strategies

#### 1. RoundRobinPrefixStrategy
Distributes checkpoints across specified prefixes in a round-robin fashion, ideal for balancing data across multiple storage locations.

```py
from s3torchconnector.dcp import RoundRobinPrefixStrategy, S3StorageWriter

model = torchvision.models.resnet18()

# Initialize with multiple prefixes and optional epoch tracking
strategy = RoundRobinPrefixStrategy(
    user_prefixes=["shard1", "shard2", "shard3"],
    epoch_num=5  # Optional: for checkpoint versioning
)

writer = S3StorageWriter(
    region=REGION,
    path="CHECKPOINT_URI",
    prefix_strategy=strategy
)

# Save checkpoint
DCP.save(
    state_dict=model.state_dict(),
    storage_writer=writer
)
```
Output Structure:
```
CHECKPOINT_URI
├── shard1/
│   └── epoch_5/
│       ├── __0_0.distcp
│       ├── __3_0.distcp
│       └── ...
├── shard2/
│   └── epoch_5/
│       ├── __1_0.distcp
│       ├── __4_0.distcp
│       └── ...
└── shard3/
    └── epoch_5/
        ├── __2_0.distcp
        ├── __5_0.distcp
        └── ...
```

#### 2. BinaryPrefixStrategy

Generates binary (base-2) prefixes for optimal partitioning in distributed environments.

```py
from s3torchconnector.dcp import BinaryPrefixStrategy

strategy = BinaryPrefixStrategy(
    epoch_num=1,          # Optional: for checkpoint versioning
    min_prefix_len=10     # Optional: minimum prefix length
)

```
Output Structure:
```
s3://my-bucket/checkpoints/
├── 0000000000/
│   └── epoch_1/
│       └── __0_0.distcp
├── 1000000000/
│   └── epoch_1/
│       └── __1_0.distcp
├── 0100000000/
│   └── epoch_1/
│       └── __2_0.distcp
└── ...
```

#### 3. HexPrefixStrategy

Uses hexadecimal (base-16) prefixes for a balance of efficiency and readability.
```py
from s3torchconnector.dcp import HexPrefixStrategy

strategy = HexPrefixStrategy(
    epoch_num=1,          # Optional: for checkpoint versioning
    min_prefix_len=4      # Optional: minimum prefix length
)
```
Output Structure:
```
s3://my-bucket/checkpoints/
├── 0000/
│   └── epoch_1/
│       └── __0_0.distcp
├── 1000/
│   └── epoch_1/
│       └── __1_0.distcp
...
├── f000/
│   └── epoch_1/
│       └── __15_0.distcp
└── ...
```

### Creating Custom Strategies

You can implement custom prefix strategies by extending the S3PrefixStrategyBase class:
```py
from s3torchconnector.dcp import S3PrefixStrategyBase

class CustomPrefixStrategy(S3PrefixStrategyBase):
    def __init__(self, custom_param):
        super().__init__()
        self.custom_param = custom_param

    def generate_prefix(self, rank: int) -> str:
        return f"custom_{self.custom_param}/{rank}/"
```

## Parallel/Distributed Training

Amazon S3 Connector for PyTorch provides support for parallel and distributed training with PyTorch, 
allowing you to leverage multiple processes and nodes for efficient data loading and training. 
Both S3IterableDataset and S3MapDataset can be used for this purpose.

### S3IterableDataset

The S3IterableDataset can be directly passed to PyTorch's DataLoader for parallel and distributed training.
By default, all worker processes will share the same list of training objects. However, 
if you need each worker to have access to a unique portion of the dataset for better parallelization, 
you can enable dataset sharding using the `enable_sharding` parameter. 
```py
dataset = S3IterableDataset.from_prefix(DATASET_URI, region=REGION, enable_sharding=True)
dataloader = DataLoader(dataset, num_workers=4)
```
When `enable_sharding` is set to True, the dataset will be automatically sharded across available number of workers. 
This sharding mechanism supports both parallel training on a single host and distributed training across multiple hosts. 
Each worker, regardless of its host, will load and process a distinct subset of the dataset.
### S3MapDataset

For the S3MapDataset, you need to pass it to DataLoader along with a [DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler) wrapped around it. 
The DistributedSampler ensures that each worker or node receives a unique subset of the dataset, 
enabling efficient parallel and distributed training.
```py
dataset = S3MapDataset.from_prefix(DATASET_URI, region=REGION)
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, num_workers=4)
```
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

End to end examples for the Pytorch Lightning integration can be found in the [examples/lightning](examples/lightning)
directory.

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
When working with model checkpoints, you can use the [S3 Versioning](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html) feature to preserve, retrieve, and restore every version of your checkpoint objects. With versioning, you can recover more easily from unintended overwrites or deletions of existing checkpoint files due to incorrect configuration or multiple hosts accessing the same storage path.

When versioning is enabled on an S3 bucket, deletions insert a delete marker instead of removing the object permanently. The delete marker becomes the current object version. If you overwrite an object, it results in a new object version in the bucket. You can always restore the previous version. See [Deleting object versions from a versioning-enabled bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/DeletingObjectVersions.html) for more details on managing object versions.

To enable versioning on an S3 bucket, see [Enabling versioning on buckets](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html). Normal Amazon S3 rates apply for every version of an object stored and transferred. To customize your data retention approach and control storage costs for earlier versions of objects, use [object versioning with S3 Lifecycle](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html).

S3 Versioning and S3 Lifecycle are not supported by S3 Express One Zone.


## Direct S3Client Usage

For advanced use cases, you can use the S3Client directly for custom streaming patterns and integration with existing pipelines.

```py
from s3torchconnector._s3client import S3Client

REGION = "us-east-1"
BUCKET_NAME = "my-bucket"
OBJECT_KEY = "large_object.bin"

s3_client = S3Client(region=REGION)

# Writing data to S3
data = b"content" * 1048576
s3writer = s3_client.put_object(bucket=BUCKET_NAME, key=OBJECT_KEY)
s3writer.write(data)
s3writer.close()

# Reading data from S3
s3reader = s3_client.get_object(bucket=BUCKET_NAME, key=OBJECT_KEY)
data = s3reader.read()
```

## Reader Configurations

Amazon S3 Connector for PyTorch supports two types of readers, configurable through `S3ReaderConstructor`.

### Reader Types

#### 1. Sequential Reader (Default)

- Downloads and buffers the entire S3 object in memory.
- Prioritizes performance over memory usage by buffering entire objects.

#### 2. Range-based Reader

- Performs byte-range requests to read specific portions of S3 objects without downloading the entire file.
- Prioritizes memory efficiency, with performance gains only for sparse partial reads.
- Features adaptive buffering with forward overlap handling:
  - **Small reads** (< `buffer_size`): Use internal buffer to reduce S3 API calls.
  - **Large reads** (≥ `buffer_size`): Bypass buffer for direct transfer.

### When to Use Each Reader

- **Sequential Reader**: For processing entire files, and when repeated access to the data is required. Best for most general use cases.
- **Range-based Reader**: For larger objects (100MB+) that require sparse partial reads, and in memory-constrained environments. 

**Note**: S3Reader instances are not thread-safe and should not be shared across threads. For multiprocessing with DataLoader, each worker process creates its own S3Reader instance automatically.

### Examples

Direct method - `S3Client` usage with range-based reader without buffer:
```py
# Direct S3Client usage for zero-copy partial reads into pre-allocated buffers, for memory efficiency and fast data transfer
from s3torchconnector._s3client import S3Client
from s3torchconnector import S3ReaderConstructor

s3_client = S3Client(region=REGION)
reader_constructor = S3ReaderConstructor.range_based(
    buffer_size=0  # No buffer, for direct transfer
)
s3reader = s3_client.get_object(
    bucket=BUCKET_NAME, 
    key=OBJECT_NAME, 
    reader_constructor=reader_constructor
)

buffer = bytearray(10 * 1024 * 1024)  # 10MB buffer
s3reader.seek(100 * 1024 * 1024)   # Skip to 100MB offset
bytes_read = s3reader.readinto(buffer)  # Direct read into buffer
```

DCP interface - `S3StorageReader` usage with range-based reader with buffer:
```py
# Load distributed checkpoint with range-based reader to optimize memory usage for large checkpoint files
from s3torchconnector.dcp import S3StorageReader
from s3torchconnector import S3ReaderConstructor

reader_constructor = S3ReaderConstructor.range_based(
    buffer_size=16*1024*1024  # 16MB buffer
)
s3_storage_reader = S3StorageReader(
    region=REGION, 
    path=CHECKPOINT_URI,
    reader_constructor=reader_constructor
)
DCP.load(
    state_dict=model_state_dict,
    storage_reader=s3_storage_reader,
)
```

Dataset interface - `S3MapDataset` usage with sequential reader:
```py
# Use sequential reader for optimal performance when reading entire objects
from s3torchconnector import S3MapDataset, S3ReaderConstructor

dataset = S3MapDataset.from_prefix(
    DATASET_URI, 
    region=REGION,
    reader_constructor=S3ReaderConstructor.sequential()
)

for item in dataset:
    content = item.read()
    ...
```

For `S3ReaderConstructor` usage details, please refer to the [`S3ReaderConstructor` documentation](https://awslabs.github.io/s3-connector-for-pytorch/autoapi/s3torchconnector/s3reader/constructor/index.html).

## Contributing

We welcome contributions to Amazon S3 Connector for PyTorch. Please see [CONTRIBUTING](CONTRIBUTING.md) for more
information on how to report bugs or submit pull requests.

### Development
See [DEVELOPMENT](DEVELOPMENT.md) for information about code style, development process, and guidelines.

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
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for more details.

## License

Amazon S3 Connector for PyTorch has a BSD 3-Clause License, as found in the [LICENSE](LICENSE) file.

