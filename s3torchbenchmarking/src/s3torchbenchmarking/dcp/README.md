## PyTorch's Distributed Checkpoint (DCP) benchmarks

The `dcp` Python package holds all the logic to execute benchmarks for [PyTorch's Distributed Checkpointing][DCP]
feature against the `s3torchconnector` library.

### Purpose

These benchmarks are designed to:

1. Test the "save" mechanism of PyTorch DCP (`torch.distributed.checkpoint.save`);
2. Compare the performance of the s3torchconnector library against other libraries and local storage;
3. Measure throughput (in MiB/s) and save times (in seconds).

### Usage

> [!IMPORTANT]
> The benchmarks are designed to be run on a EC2 instance.

Install the `s3torchbenchmarking` package with `pip` (see the [root README](../../../README.md) for instructions),
along with the `s3torchconnector[dcp]` extra; once installed, the DCP benchmarks can be run with:

```shell
$ s3torch-benchmark-dcp -cd conf -cn dcp
```

The command must be executed from the package's root, where it can read from the `config/` directory; it will create a
`./multirun/` directory (at the location of execution), and store all benchmark results there.

> [!WARNING]
> When saving on local disk, consider clearing the `path` specified in your config between runs to prevent disk space
> issues.

#### Potential caveats

If you encounter the following errors during installation, try the associated command:

**Error**:

```
RuntimeError: Failed to import transformers.models.vit.modeling_vit because of the following error (look up to see its traceback):
operator torchvision::nms does not exist
```

**Try**:

```shell
$ conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

**Error**:

```
TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
```

**Try**:

```shell
$ pip install "setuptools<71"
```

### Configuration

The benchmark runs can be customized using the [`dcp.yaml`](../../../conf/dcp.yaml) file. This section outlines the key
configuration options and their impacts.

#### Configuration Requirements

All keys in the `dcp.yaml` file must be defined for a run to execute successfully.

#### Key Configuration Options

`epochs`

- Specifies the number of iterations for "saving" a model's checkpoint.
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
    - `+world_size`: Defines the number of workers.
    - `+thread_count`: Defines the number of threads to use for saving the checkpoints.
    - `+checkpoint.storage`: Choose `s3`, `disk`, or both.

#### Example Configuration

```yaml
s3:
  region: eu-west-1
  uri: s3://my-bucket
epochs: 3
path: ./nvme/

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      +model: vit-base,T0_3B
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
- Ensure the specified S3 bucket exists in the given region and the EC2 user/role has read+write permissions.

### Results

Benchmark results are organized as follows:

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

Each run creates a timestamped subdirectory. The `./multirun/` directory is managed by [Hydra](https://hydra.cc/).

Result file names reflect the parameter combinations, e.g.,

```
+model: vit-base
+backend: nccl
+world_size: 2
+thread_count: 1
+checkpoint.storage: s3
```

will produce the file `results_vit-base_nccl_2_1_s3.json` (respecting parameters declaration order).

### References

- https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
- https://pytorch.org/docs/stable/elastic/run.html
- https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

[DCP]: https://pytorch.org/docs/stable/distributed.checkpoint.html
