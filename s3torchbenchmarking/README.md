# Benchmarking the S3 Connector for PyTorch

This directory contains a modular component for the experimental evaluation of the performance of the Amazon S3 Connector for
PyTorch.
The goal of this component is to be able to run performance benchmarks for PyTorch connectors in an easy-to-reproduce and
extensible fashion. This way, users can experiment with different settings and arrive at the optimal configuration for their workloads,
before committing to a setup.

By managing complex configuration space with [Hydra](https://hydra.cc/) we are able to define modular configuration pieces mapped to various
stages of the training pipeline. This approach allows one to mix and match configurations and measure the performance 
impact to the end-to-end training process. To achieve this, we split configuration to 4 pieces. Namely:

**dataset**: The `dataset` configuration keeps information about where the data resides. While we support sharded objects, only loading
from TAR objects is supported currently.

**dataloader**: Used to configure the [PyTorch DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). Parameters like
`batch_size` and `num_workers` are self-explanatory. `kind` is used to specify which PyTorch dataset to use(`s3iterabledataset`, `s3mapdataset`, `fsspec`).

**training**: Specify what model to learn and how many epochs to execute the training for. Currently, we implemented two
models `Entitlement` and [ViT](https://huggingface.co/docs/transformers/model_doc/vit). To make it easier to add new models, and abstract the learning-sample processing logic from configuration, this module
defines a Model interface where each model expected to implement `load_sample`, `train`, and `save` methods.

**checkpoint**: Defines where(`disk`, `s3`) and how frequently checkpoints are to be saved.

Once the sub-configurations are defined, one can easily create an experimental configuration that will use the Hydra Sweeper
to launch multiple experiments sequentially.

For example, the `dataloading` experiment stored at `./conf/dataloading.yaml` has the following
content:

```
defaults:
  - dataloader: ???
  - dataset: unsharded_dataset
  - training: entitlement
  - checkpoint: none


hydra:
  mode: MULTIRUN

  sweeper:
    params:
      dataloader: s3iterabledataset, fsspec
      dataloader.num_workers: 2,4,8,16
```

This configuration pins the `dataset` and `training` model while overriding the `dataloader` to change `kind`
and `num_workers`. Running this benchmark will result in sequentially running 8 different scenarios,
each with the different combinations of swept parameters. As `Entitlement` is not really performing any training, this
experiment is helpful to see upper-limit of dataloader throughput without being susceptible to GPU backpressure.

## Getting Started

The benchmarking code is available within the `s3torchbenchmarking`. First navigate into the directory:

    cd s3torchbenchmkaring

The tests can be run locally, or you can launch an EC2 instance with a GPU(we used a [g5.2xlarge](https://aws.amazon.com/ec2/instance-types/g5/)), choosing 
the [AWS Deep Learning AMI GPU PyTorch 2.0.1 (Amazon Linux 2)](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-0-amazon-linux-2/) as your AMI. Activate the venv within this machine
by running:

    source pytorch activate

If running locally you can optionally configure a Python virtualenv:

    python -m venv <ENV-NAME>
    python <PATH-TO-VENV>/bin/activate


Then from this directory, install the dependencies:

    python -m pip install .

This would make the `s3torch-benchmark` and `s3torch-datagen` commands available to you. Note: the installation would
recommend $PATH modifications if necessary, allowing you to use the commands directly.

**(Optional) Install Mountpoint**

Required only if you're running benchmarks using PyTorch with Mountpoint for S3.

    wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.rpm
    sudo yum install ./mount-s3.rpm # For an RHEL system

For other distros see [Installing Mountpoint for Amazon S3](https://github.com/awslabs/mountpoint-s3/blob/main/doc/INSTALL.md).

_Note: Mountpoint benchmarks are currently only supported on *nix-based systems and rely on `sudo` capabilities._  

### (Pre-requisite) Configure AWS Credentials

The commands provided below(`datagen.py`, `benchmark.py`) rely on the standard [AWS credential discovery mechanism](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html). 
Supplement the command as necessary to ensure the AWS credentials are made available to the process. For eg: by setting
the `AWS_PROFILE` environment variable.

### Configuring the dataset

_Note: This is a one-time setup for each dataset configuration. The dataset configuration files, once created locally
and can be used in subsequent benchmarks, as long as the dataset on the S3 bucket is intact._

If you already have a dataset, you only need upload it to an S3 bucket and setup a YAML file under
`./conf/dataset/` in the following format:

```yaml
# custom_dataset.yaml

prefix_uri: s3://<S3_BUCKET>/<S3_PREFIX>/
region: <AWS_REGION>
sharding: TAR|null # if the samples have been packed into TAR archives.
```

This dataset can then be referenced in an experiment with an entry like `dataset: custom_dataset` (note that we're 
omitting the *.yaml extension). This will result in running the benchmarks against this dataset. Some experiments have 
already been defined for reference - see `./conf/dataloading.yaml` or `./conf/sharding.yaml`.

_Note: Ensure the bucket is in the same region as the EC2 instance to eliminate network latency effects in your
measurements._

Alternatively, you can use the `s3torch-datagen` command to procedurally generate an image dataset and upload it to 
Amazon S3. The script also creates a Hydra configuration file at the appropriate path.

```
$ s3torch-datagen --help
Usage: s3torch-datagen [OPTIONS]

  Synthesizes a dataset that will be used for benchmarking and uploads it to
  an S3 bucket.

Options:
  -n, --num-samples FLOAT  Number of samples to generate.  Can be supplied as
                           an IEC or SI prefix. Eg: 1k, 2M. Note: these are
                           case-sensitive notations. [default: 1k]
  --resolution TEXT        Resolution written in 'widthxheight' format
                           [default: 496x387]
  --shard-size TEXT        If supplied, the images are grouped into tar files
                           of the given size. Size can be supplied as an IEC
                           or SI prefix. Eg: 16Mib, 4Kb, 1Gib. Note: these are
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

- `-n 20k --resolution 496x387`
- `-n 20k --resolution 496x387 --shard-size {4, 8, 16, 32, 64}MiB`

Example:

```
$ s3torch-datagen -n 20k \
   --resolution 496x387 \
   --shard-size 4MB \
   --s3-bucket swift-benchmark-dataset \
   --region eu-west-2

Generating data: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 1243.50it/s]
Uploading to S3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 3378.87it/s]
Dataset uploaded to: s3://swift-benchmark-dataset/20k_496x387_images_4MB_shards/
Dataset Configuration created at: ./conf/dataset/20k_496x387_images_4MB_shards.yaml
Configure your experiment by setting the entry:
    dataset: 20k_496x387_images_4MB_shards
Alternatively, you can run specify it on the cmd-line when running the benchmark like so:
    s3torch-benchmark -cd conf  -m -cn <CONFIG-NAME> 'dataset=20k_496x387_images_4MB_shards'
```

---

Finally, once the dataset and other configuration modules have been defined, you can kick off the benchmark by running:

    $ cd s3torchbenchmarking/

    $ s3torch-benchmark -cd conf -m -cn YOUR-TEST-CONFIGURATION 
    
    # Example-1:
    $ s3torch-benchmark -cd conf -m -cn dataloading 'dataset.prefix_uri=<S3-PREFIX>' 'dataset.region=eu-west-2'

    # Example-2:
    $ s3torch-benchmark -cd conf -m -cn checkpointing 'dataset.prefix_uri=<S3-PREFIX>' 'dataset.region=eu-west-2'

    # Example with suppressed warnings (*nix only) - makes logs less noisy.
    $ s3torch-benchmark -cd conf -m -cn checkpointing 'dataset.prefix_uri=<S3-PREFIX>' 'dataset.region=eu-west-2' 2>/dev/null

_Note: For overriding any other benchmark parameters,
see [Hydra Overrides](https://hydra.cc/docs/advanced/override_grammar/basic/). You can also run `s3torch-benchmark --hydra-help` to learn more._


Experiments will report total training time, number of training samples as well as host-level metrics like CPU
Utilisation, GPU Utilisation (if available) etc. The results for individual jobs will be written out to dedicated
`result.json` files within their corresponding [output dirs](https://hydra.cc/docs/configure_hydra/intro/#hydraruntime).
When using Multirun mode, a `collated_results.json` will be written out to the [common sweep dir](https://hydra.cc/docs/configure_hydra/intro/#hydrasweep). 

## Next Steps

- Use [Hydra Callbacks](https://hydra.cc/docs/experimental/callbacks/) to aggregate and plot benchmark results.
- Add more models (LLMs?) to monitor training performance.
- Support plugging in user-defined models and automatic discovery of the same.