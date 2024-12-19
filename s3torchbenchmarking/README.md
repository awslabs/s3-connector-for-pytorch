# s3torchbenchmarking

This Python package houses a set of benchmarks for experimentally evaluating the performance of the Amazon S3 Connector
for PyTorch.

Our primary objective is to facilitate reproducible and extensible performance benchmarks for PyTorch connectors. This
empowers users to experiment with various configurations and identify optimal settings for their specific workloads
before finalizing their setup.

By managing complex configuration space with [Hydra](https://hydra.cc/), we are able to define modular configuration
pieces mapped to various stages of the training pipeline. This approach allows one to mix and match configurations and
measure the performance impact to the end-to-end training process.

**Three scenarios** are available:

1. **Dataset Benchmarks**
    - Compare our connector against other Dataset classes
    - All scenarios save data to S3
    - Measure performance in data fetching and indexing
2. **PyTorch Lightning Checkpointing Benchmarks**
    - Evaluate our connector within the PyTorch Lightning framework
    - Compare against PyTorch Lightning's default checkpointing implementation
3. **PyTorch's Distributed Checkpointing (DCP) Benchmarks**
    - Assess our connector's performance versus PyTorch's default distributed checkpointing mechanism
    - For detailed information, refer to the [dedicated DCP `README`](src/s3torchbenchmarking/dcp/README.md)

## Getting started

The benchmarking code is located in the `src/s3torchbenchmarking` module. You can run the tests either locally or on an
EC2 instance with one (or many) GPU(s).

### EC2 instance setup (recommended)

From your EC2 AWS Console, launch an instance with one (or many) GPU(s) (e.g., G5 instance type); we recommend using
an [AWS Deep Learning AMI (DLAMI)](https://docs.aws.amazon.com/dlami/), such as
the [AWS Deep Learning AMI GPU PyTorch 2.5 (Amazon Linux 2023)][dlami-pytorch]. This simplifies the PyTorch and
environment setup.

> [!NOTE]
> Some benchmarks can be long-running. We recommend attaching a role to your EC2 instance with:
>
> - Full access to S3
> - (Optional) Full access to DynamoDB for writing run results
>
> See the [Running the benchmarks](#running-the-benchmarks) section for more details.

### Creating a new Conda environment (env)

> [!WARNING]
> While some DLAMIs provide a pre-configured Conda env (`source activate pytorch`), we've observed compatibility issues
> with the latest PyTorch versions (2.5.X) at the time of writing. We recommend creating a new one from scratch as
> detailed below.

Once your instance is running, `ssh` into it, and create a new Conda env:

```shell
conda create -n pytorch-benchmarks python=3.12
conda init
```

Then, activate it (_you may need to log out and log in again in the meantime, as signaled by `conda init`_):

```shell
source activate pytorch-benchmarks
```

Finally, from within this directory, install the `s3torchbenchmarking` module:

```shell
pip install .
```

> [!NOTE]
> For some scenarios, you may be required to install the [Mountpoint for Amazon S3][mountpoint-s3] file client: please
> refer to their README for installation instructions.

### Configure AWS credentials

Some parts of the benchmarks rely on the standard [AWS credential discovery mechanism][credentials]. Supplement the
command as necessary to ensure the AWS credentials are made available to the process, e.g., by setting the `AWS_PROFILE`
environment variable.

### Configuring the dataset

_Note: This is a one-time setup for each dataset configuration. The dataset configuration files, once created locally
and can be used in subsequent benchmarks, as long as the dataset on the S3 bucket is intact._

If you already have a dataset, you only need upload it to an S3 bucket and set up a YAML file under
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

## Running the benchmarks

You can run the different benchmarks by running one of those shell script:

```shell
# For "dataset" benchmarks":
./utils/run_dataset_benchmarks.sh 

# For "PyTorch Lightning Checkpointing benchmarks":
./utils/run_lighning_benchmarks.sh

# For "PyTorch’s Distributed Checkpointing (DCP) benchmarks":
./utils/run_dcp_benchmarks.sh
```

Each of those scripts rely on Hydra config files, located under the [`conf`](conf) directory. You may edit those as you
see fit to configure the runs: in particular, parameters under the `hydra.sweeper.params` path will create as many jobs
as the cartesian product of those.

Also, as the scripts pass the inline parameters you give them to Hydra, so you may override their behaviors this way:

```shell
./utils/run_dataset_benchmarks.sh +disambiguator=some_key
```

## Getting the results

Experiments will report various metrics, like throughput, processed time, etc. The results for individual jobs and runs
(one run will contain 1 to N jobs) will be written out to dedicated files, respectively `job_results.json` and
`run_results.json`, within their corresponding output directory (see the YAML config files).

If a DynamoDB table was defined in the respective config file ([`conf/aws/dynamodb.yaml`](conf/aws/dynamodb.yaml)), the
results will be written in such table.

[dlami-pytorch]: https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-5-amazon-linux-2023/

[mountpoint-s3]: https://github.com/awslabs/mountpoint-s3/tree/main

[credentials]: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html

[hydra-overrides]: https://hydra.cc/docs/advanced/override_grammar/basic/
