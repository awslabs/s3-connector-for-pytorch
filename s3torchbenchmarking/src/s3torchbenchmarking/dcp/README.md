## PyTorch's Distributed Checkpoint (DCP) benchmarks

The `dcp` Python package holds all the logic to execute benchmarks for [PyTorch's Distributed Checkpointing][DCP]
feature against the `s3torchconnector` library.

### Purpose

The benchmarks are designed to test the "save" and "load" mechanisms of PyTorch
(`torch.distributed.checkpoint[save,load]`), and compare the s3torchconnector library performance vs. other libraries,
like fsspec.

### Usage

> [!IMPORTANT]
> The benchmarks are designed to be run on a EC2 instance.

Install the `s3torchbenchmarking` package with `pip` (see the [root README](../../../README.md) for instructions); once
installed, the DCP benchmarks can be run with:

```shell
$ s3torch-benchmark-dcp
```

The command can be executed from any directory; it will create a `./multirun/` directory (at the location of execution),
and store all benchmark results there.

#### Potential caveats

While installing the package, one may run into an error like:

```
TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
```

This error can be addressed by running:

```shell
$ pip install "setuptools<71"
```

### Configuration

The benchmark runs can be customized using the [`config.yaml`](config.yaml) file. This section outlines the key
configuration options and their impacts.

#### Configuration Requirements

All keys in the `config.yaml` file must be defined for a run to execute successfully.

#### Key Configuration Options

`epochs`

- Specifies the number of iterations for the "save" and "load" operations.
- Note: This does not affect model training, as no actual training occurs in these benchmarks.

`path`

- Designates the directory for benchmark operations.
- If the specified directory doesn't exist, it will be created automatically.
- For optimal performance using an SSD filesystem, refer to the [`prepare_nvme.sh`](../../../utils/prepare_nvme.sh)
  script.

`hydra.sweeper.params`

This section allows for multiple benchmark configurations:

- The benchmark will run sequential jobs for each combination of the specified parameters.
- Available options include:
    - `+model`: Choose from pre-trained models listed in [`models.py`](models.py).
    - `+backend`: Select `nccl`, `gloo`, or both.
    - `+world_size`: Defines the number of processes. Note: Values exceeding the node's capacity will be automatically
      capped.
    - `+thread_count`: Defines the number of threads to use for saving the checkpoints.
    - `+checkpoint.storage`: Choose `s3`, `disk`, or both.

#### Example Configuration

```yaml
s3:
  region: eu-west-1
  uri: s3://my-benchmark-bucket
epochs: 3
path: nvme

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      +model: small,medium
      +backend: nccl,gloo
      +world_size: 2,4
      +thread_count: 1
      +checkpoint.storage: s3,disk
```

This configuration will run benchmarks for all combinations of the specified models, backends, world sizes, and storage
options, totaling 16 (2×2×2×1×2) different benchmark scenarios.

### Important notes

- The benchmarks may take some time to complete, depending on the hardware and network configuration.
- For optimal results, it is recommended to run the benchmarks on a dedicated EC2 instance without other
  resource-intensive processes.
- When specifying an S3 bucket in the `config.yaml` file, make sure that 1/ the bucket already exists in the specified
  region, and 2/ the EC2 user/role used has write-permissions to it.

### Results

The benchmark results are organized in the following structure:

```shell
multirun/
└── YYYY-MM-DD
    └── HH-MM-SS
        ├── 0
        │   ├── benchmark.log
        │   └── results_small_nccl_2_2_s3.json
        ├── 1
        │   ├── benchmark.log
        │   └── results_small_nccl_2_2_disk.json
        ├── 2
        │   ├── benchmark.log
        │   └── results_small_nccl_4_2_s3.json
        ├── 3
        │   ├── benchmark.log
        │   └── results_small_nccl_4_2_disk.json
        └── multirun.yaml
```

Each run creates a new subdirectory with a timestamp (e.g., `2024-11-11/10-21-16`). The `./multirun/` directory
structure is managed by [Hydra](https://hydra.cc/), the underlying configuration framework.


[DCP]: https://pytorch.org/docs/stable/distributed.checkpoint.html
