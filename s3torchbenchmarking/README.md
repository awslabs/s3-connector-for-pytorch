# s3torchbenchmarking

This Python package houses a set of benchmarks for experimentally evaluating the performance of
the **Amazon S3 Connector for PyTorch** library.

With the use of the [Hydra](https://hydra.cc/) framework, we are able to define modular configuration pieces mapped to
various stages of the training pipeline. This approach allows one to mix and match configurations and measure the
performance impact to the end-to-end training process.

**Four scenarios** are available:

1. **Dataset benchmarks**
    - Compare our connector against other Dataset classes
    - All scenarios save data to S3
    - Measure performance in data fetching and indexing
2. **PyTorch's Distributed Checkpointing (DCP) benchmarks**
    - Assess our connector's performance versus PyTorch's default distributed checkpointing mechanism
    - For detailed information, refer to [DCP using DDP `README`](src/s3torchbenchmarking/dcp_ddp/README.md)
or [DCP using FSDP `README`](src/s3torchbenchmarking/dcp_fsdp/README.md)
3. **PyTorch Lightning Checkpointing benchmarks**
    - Evaluate our connector within the PyTorch Lightning framework
    - Compare against PyTorch Lightning's default checkpointing implementation
4. **PyTorch Checkpointing benchmarks**
    - TODO!

## Getting started

The benchmarking code is located in the `src/s3torchbenchmarking` module. The scenarios are designed to be run on an EC2
instance with one (or many) GPU(s).

### EC2 instance setup (recommended)

From your EC2 AWS Console, launch an instance with one (or many) GPU(s) (e.g., G5 instance type); we recommend using
an [AWS Deep Learning AMI (DLAMI)][dlami], such
as [AWS Deep Learning AMI GPU PyTorch 2.5 (Amazon Linux 2023)][dlami-pytorch].

> [!NOTE]
> Some benchmarks can be long-running. To avoid the shortcomings around expired AWS tokens, we recommend attaching a
> role to your EC2 instance with:
>
> - Full access to S3
> - (Optional) Full access to DynamoDB — for writing run results
>
> See the [Running the benchmarks](#running-the-benchmarks) section for more details.

For optimal results, it is recommended to run the benchmarks on a dedicated EC2 instance _without_ other
resource-intensive processes.

### Creating a new Conda environment (env)

> [!WARNING]
> While some DLAMIs provide a pre-configured Conda env (`source activate pytorch`), we have observed compatibility
> issues with the latest PyTorch versions (2.5.X) at the time of writing. We recommend creating a new one from scratch
> as detailed below.

Once your instance is running, `ssh` into it, and create a new Conda env:

```shell
conda create -n pytorch-benchmarks python=3.12
conda init
```

Then, activate it (_you will need to log out and in again in the meantime, as signaled by `conda init`_):

```shell
source activate pytorch-benchmarks
```

Finally, from within this directory, install the `s3torchbenchmarking` module:

```shell
# `-e` so local modifications get picked up, if any
pip install -e .
```

> [!NOTE]
> For some scenarios, you may be required to install the [Mountpoint for Amazon S3][mountpoint-s3] file client: please
> refer to their README for instructions.

### (Pre-requisite) Configure AWS Credentials

The benchmarks and other commands provided below rely on the standard [AWS credential discovery mechanism][credentials].
Supplement the command as necessary to ensure the AWS credentials are made available to the process, e.g., by setting
the `AWS_PROFILE` environment variable.

### Creating a dataset (optional; for "dataset" benchmarks only)

You can use your own dataset for the benchmarks, or you can generate one on-the-fly using the `s3torch-datagen` command.

Here are some sample dataset configurations that we ran our benchmarks against:

```shell
s3torch-datagen -n 100k --shard-size 128MiB --s3-bucket my-bucket --region us-east-1
```

## Running the benchmarks

You can run the different benchmarks by editing their corresponding config files, then running one of those shell
scripts (specifically, you must provide a value for all keys marked with `???`):

```shell
# Dataset benchmarks
vim ./conf/dataset.yaml           # 1. edit config
./utils/run_dataset_benchmarks.sh # 2. run scenario

# PyTorch Checkpointing benchmarks
vim ./conf/pytorch_checkpointing.yaml # 1. edit config
./utils/run_checkpoint_benchmarks.sh # 2. run scenario

# PyTorch Lightning Checkpointing benchmarks
vim ./conf/lightning_checkpointing.yaml # 1. edit config
./utils/run_lightning_benchmarks.sh      # 2. run scenario

# PyTorch’s Distributed Checkpointing (DCP) benchmarks
vim ./conf/dcp_ddp.yaml           # 1. edit config
vim ./conf/dcp_fsdp.yaml
./utils/run_dcp_ddp_benchmarks.sh # 2. run scenario
./utils/run_dcp_fsdp_benchmarks.sh
```

> [!NOTE]
> Ensure the bucket is in the same region as the EC2 instance, to eliminate network latency effects in your
> measurements.

Each of those scripts relies on Hydra config files, located under the [`conf`](conf) directory. You may edit those as you
see fit to configure the runs: in particular, parameters under the `hydra.sweeper.params` path will create as many jobs
as the cartesian product of those.

Also, as the scripts pass the inline parameters you give them to Hydra, you may override their behaviors this way:

```shell
./utils/run_dataset_benchmarks.sh +disambiguator=some_key
```

## Getting the results

### Scenario organization

Benchmark results are organized as follows, inside a default `./multirun` directory (e.g.):

```
./multirun
└── dataset
    └── 2024-12-20_13-42-27
        ├── 0
        │   ├── benchmark.log
        │   └── job_results.json
        ├── 1
        │   ├── benchmark.log
        │   └── job_results.json
        ├── multirun.yaml
        └── run_results.json
```

Scenarios are organized at the top level, each in its own directory named after the scenario (e.g., `dataset`). Within
each scenario directory, you'll find individual run directories, automatically named by Hydra using the creation
timestamp (e.g., `2024-12-20_13-42-27`).

Each run directory contains job subdirectories (e.g., `0`, `1`, etc.), corresponding to a specific subset of parameters.

### Experiment reporting

Experiments will report various metrics, such as throughput and processed time — the exact types vary per scenario.
Results are stored in two locations:

1. In the job subdirectories:
    - `benchmark.log`: Individual job logs (collected by Hydra)
    - `job_results.json`: Individual job results
2. In the run directory:
    - `multirun.yaml`: Global Hydra configuration for the run
    - `run_results.json`: Comprehensive run results, including additional metadata

If a DynamoDB table is defined in the [`conf/aws/dynamodb.yaml`](conf/aws/dynamodb.yaml) configuration file, results
will also be written to the specified table.

[dlami]: https://docs.aws.amazon.com/dlami/

[dlami-pytorch]: https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-5-amazon-linux-2023/

[mountpoint-s3]: https://github.com/awslabs/mountpoint-s3/tree/main

[credentials]: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html

[hydra-overrides]: https://hydra.cc/docs/advanced/override_grammar/basic/
