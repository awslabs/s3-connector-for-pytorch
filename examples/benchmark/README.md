# Benchmarking S3 Connector for PyTorch

This directory contains a modular component for experimental evaluation of performance of Amazon S3 Connector for
PyTorch.
The goal of this component is to create an easy to re-produce and extend performance benchmarking. This way, user can
experiment with different settings and decide to best parameters for their workloads, before committing to a setup.

By managing complex configuration space with [Hydra](https://hydra.cc/) this module splits configuration of multiple
components of a training pipeline into smaller pieces to create custom
training pipelines. This way, one can mix and match these configurations and observe the impact of different components
to
the end to end training performance. To achieve this, we split configuration to 4 pieces,
namely; `dataset`, `dataloader`, `checkpoint`, `training`.

The `dataset` configuration keeps information about where data resides. While we support sharded objects, only loading
from TAR objects supported currently. See [a](#benchmarking-s3-connector-for-pytorch)
The `dataloader` configuration contains dataloading specific setup. `kind` specify which PyTorch dataset to
use (`s3iterabledataset`, `s3mapdataset`, `fsspec`) while
`batch_size`, `num_workers` specify
relevant [PyTorch DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) parameters. `training`
configuration specify what model to learn and how many epochs to execute the training for. Currently, we support two
models `Entitlement` and [ViT](https://huggingface.co/docs/transformers/model_doc/vit).
To make it easier to add new models, and abstract the learning-sample processing logic from configuration, this module
defines a Model interface where each model expected to implement
`load_sample`, `train`, and `save` methods. Lastly, `checkpoint` configuration, defines where and how frequently to save
the checkpoints.

Once the sub-configurations are defined, one can easily create experimental configuration, that will use Hydra Sweeper
to launch multiple experiments, sequentially, and monitor performance.
For example, dataloading experiment configuration, is stored at configuration/dataloading.yaml, has the following
content:

```
defaults:
  - dataloader: ???
  - dataset: 20k_imagenet
  - training: entitlement
  - checkpoint: none
  - _self_


hydra:
  mode: MULTIRUN

  sweeper:
    params:
      dataloader: s3iterabledataset, fsspec
      dataloader.num_workers: 2,4,8,16
```

This configuration, puts a pin to dataset and training model while changing dataloader parameters to change `kind`
and `number_workers`. Once executed with this configuration, benchmark with sequentially execute 8 experiments,
with the different combinations of swept parameters. As `Entitlement` is not really performing any training, this
experiments is helpful to see upper-limit of dataloader throughput without having the backpressure from GPU.

## Getting Started

To get started, launch an EC2 instance with a GPU (we used
a [g5.2xlarge](https://aws.amazon.com/ec2/instance-types/g5/)),
choosing
the [AWS Deep Learning AMI GPU PyTorch 2.0.1 (Amazon Linux 2)](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-0-amazon-linux-2/)
as your AMI.

You will need an S3 Bucket with training data and update the `dataset` configuration with correct prefix and region.

Then from this directory, install the dependencies for the example code:

    python -m pip install -r requirements.txt

### Configuring the dataset

_Note: This is a one-time setup for each dataset configuration. The dataset configuration files, once created locally
and can be used in subsequent benchmarks, as long as the dataset on the S3 bucket is intact._

If you already have a dataset, you only need upload it to an S3 bucket and setup a YAML file under
`./configuration/dataset/` in the following format:
```yaml
# custom_dataset.yaml

prefix_uri: s3://<S3_BUCKET>/<S3_PREFIX>/
region: <AWS_REGION>
sharding: True|False # if the samples have been packed into TAR archives.
```
The benchmarking scenario will need to reference this dataset. See `./configuration/dataloading.yaml` or `./configuration/sharding.yaml` for reference.

_Note: Ensure the bucket is in the same region as the EC2 instance to eliminate network latency effects in your measurements._

Alternatively, you can use `datagen.py` to procedurally generate an image dataset and upload it to Amazon S3. The script
also creates a Hydra configuration file at the appropriate path.

```
$ python datagen.py --help
Usage: datagen.py [OPTIONS]

  Synthesizes a dataset that will be used for benchmarking and uploads it to
  an S3 bucket.

Options:
  -n, --num-samples FLOAT  Number of samples to generate.  [default: 1k]
  --resolution TEXT        Resolution written in 'widthxheight' format
                           [default: 496x387]
  --shard-size TEXT        If supplied, the images are grouped into tar files
                           of the given size. Size can be supplied as an IEC
                           or SI prefix. Eg: 16Mib, 4Kb, 1Gib.Note: these are
                           case-sensitive notations.
  --s3-bucket TEXT         S3 Bucket name. Note: Ensure the credentials are
                           made available either through environment variables
                           or a shared credentials file.  [required]
  --s3-prefix TEXT         Optional S3 Key prefix where the dataset will be
                           uploaded. Note: a prefix will be autogenerated. eg:
                           s3://<BUCKET>/1k_256x256_16Mib_sharded/
  --region TEXT            Region where the S3 bucket is hosted.  [default:
                           us-east-1]
  --help                   Show this message and exit.

```

Here are some sample dataset configurations that we ran our benchmarks against:

- `n: 20k, resolution: 496x387`
- `n: 20k, resolution: 496x387, shard-size: {4, 8, 16, 32, 64}MB`

Example:

```
# Configure AWS Credentials

$ python datagen.py -n 20k \
   --resolution 496x387 \
   --shard-size 4MB \
   --s3-bucket swift-benchmark-dataset \
   --region eu-west-2

Generating data: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 1243.50it/s]
Uploading to S3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 3378.87it/s]
Dataset uploaded to: s3://swift-benchmark-dataset/20k_496x387_images_4MB_shards/
Dataset Configuration created at: ./configuration/dataset/20k_496x387_images_4MB_shards.yaml
Configure your experiment by setting the entry:
    dataset: 20k_496x387_images_4MB_shards
Alternatively, you can run specify it on the cmd-line when running the benchmark like so:
    python benchmark.py -m -cn <CONFIG-NAME> 'dataset=20k_496x387_images_4MB_shards'
```

---

Next, once updating the configuration of dataset accordingly, or other configurations as needed you need to run:

    python benchmark.py -m -cn YOUR-TEST-CONFIGURATION # dataloading OR checkpointing

_Note: For overriding any other benchmark parameters, see [Hydra Overrides](https://hydra.cc/docs/advanced/override_grammar/basic/)._

Experiments will report total training time, number of training samples as well as host-level metrics like CPU
Utilisation, GPU Utilisation (if available) etc.

## Next Steps

- Use [Hydra Callbacks](https://hydra.cc/docs/experimental/callbacks/) to aggregate and plot benchmark results.
- Add more models (LLMs?) to monitor training performance.