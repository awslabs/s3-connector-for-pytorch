## PyTorch's Distributed Checkpoint (DCP) benchmarks using Fully Sharded Data Parallel (FSDP) training

The `dcp_fsdp` package provides a comprehensive suite of benchmarks and utilities designed to evaluate and measure the performance of [PyTorch's Distributed Checkpointing (DCP)][DCP] feature using the `s3torchconnector` library.

These benchmarks specifically use Fully Sharded Data Parallel (FSDP), which is PyTorch's memory-efficient distributed training approach where model parameters are sharded across GPUs/processes. Unlike DDP, FSDP distributes model parameters across processes, making it particularly suitable for training large models that wouldn't fit in a single GPU's memory.

## Files and Scripts Overview

### Core Python Scripts

#### Individual Operation Scripts
- **[`save.py`](save.py)** - Performs repeated checkpoint saving operations to S3 with stress testing capabilities
- **[`load.py`](load.py)** - Loads existing checkpoints from S3 with specified suffix
- **[`newload.py`](newload.py)** - Alternative checkpoint loading implementation with enhanced error handling
- **[`save_and_load.py`](save_and_load.py)** - Combined save and load operations with performance metrics and timing analysis

#### Benchmark and Utility Scripts
- **[`benchmark.py`](benchmark.py)** - Main benchmarking framework using Hydra configuration for comprehensive performance testing
- **[`wait.py`](wait.py)** - Utility script to keep cluster nodes busy with minimal resource usage during testing
- **[`llama_model_config.py`](llama_model_config.py)** - Model configuration definitions for LLaMA models used in benchmarks

### Slurm Batch Scripts

#### Primary Batch Scripts
- **[`savebatch.sh`](savebatch.sh)** - Slurm batch script for distributed checkpoint saving operations (pairs with `save.py`)
- **[`loadbatch.sh`](loadbatch.sh)** - Slurm batch script for distributed checkpoint loading operations (pairs with `newload.py`)
- **[`batch.sh`](batch.sh)** - General-purpose Slurm batch script for combined save/load operations (pairs with `save_and_load.py`)
- **[`waitbatch.sh`](waitbatch.sh)** - Slurm batch script to maintain cluster nodes in idle state (pairs with `wait.py`)

## Usage Guide

### Prerequisites
Before running the batch scripts, ensure you have:
1. AWS ParallelCluster set up with Slurm scheduler
2. S3 bucket configured with appropriate permissions
3. Required Python packages installed (`s3torchconnector`, `torch`, `transformers`)
4. Conda environment activated with Python 3.12+

### Script Execution Order

#### Option 1: Individual Operations (Recommended for Testing)

1. **Save Checkpoints First:**
   ```bash
   # Edit savebatch.sh to configure:
   # - Number of nodes (--nodes=X)
   # - S3 URI (--uri s3://your-bucket/path)
   # - AWS region (--region us-east-2)
   
   chmod +x savebatch.sh
   sbatch savebatch.sh
   ```
   
2. **Load Checkpoints After Saving:**
   ```bash
   # Edit loadbatch.sh to configure:
   # - Same S3 URI as save operation,
   # - Checkpoint suffix from save operation output,  use cat slurm-<jobid>.out to figure that out
   
   chmod +x loadbatch.sh
   sbatch loadbatch.sh
   ```

#### Option 2: Combined Operations (For Comprehensive Testing)

```bash
# Edit batch.sh to configure cluster and S3 settings
chmod +x batch.sh
sbatch batch.sh
```

#### Option 3: Single Script Execution (For Development/Debugging)

```bash
# Direct execution with torchrun
torchrun --nnodes=2 --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=<master_ip>:12356 \
  save.py --backend gloo --uri s3://your-bucket --region us-east-2
```

### Key Configuration Parameters

#### Slurm Configuration
- `--nodes`: Number of compute nodes (adjust based on model size and testing requirements)
- `--ntasks-per-node`: Typically 1 (torchrun handles multi-GPU coordination)
- `--cpus-per-task`: Usually 16 for adequate data loading support
- `--partition`: Use "train" queue as defined in cluster config

#### Script Parameters
- `--backend`: Choose "gloo" for CPU or "nccl" for GPU operations
- `--uri`: S3 bucket URI (format: `s3://bucket-name/path`)
- `--region`: AWS region where S3 bucket is located
- `--model`: Model size (default: "L7b" for LLaMA 7B)
- `--iterations`: Number of save/load cycles for stress testing
- `--suffix`: Checkpoint identifier (auto-generated for saves, required for loads)

### Monitoring and Output

- **Job Status:** Use `squeue` to monitor running jobs
- **Output Logs:** Check `slurm-<job_id>.out` files for detailed execution logs
- **Performance Metrics:** Scripts output timing data, throughput measurements, and error rates

### Cluster Management Utilities

#### Keep Nodes Active
Use `waitbatch.sh` to maintain cluster nodes in an active state between benchmark runs, this runs `wait.py` and holds off for 2 minutes which you can configure:
```bash
chmod +x waitbatch.sh
sbatch waitbatch.sh
```

#### Common Slurm Commands
- `sbatch <script.sh>` - Submit batch job
- `squeue` - View job queue status
- `scancel <job_id>` - Cancel specific job
- `sinfo` - View cluster node information

## Performance Metrics

The benchmarks measure:
- **Checkpoint saving throughput** (MiB/s)
- **Checkpoint loading throughput** (MiB/s) 
- **Save/load durations** (seconds, excluding model initialization)
- **S3 request rates and success/failure ratios**
- **Multi-node coordination overhead**

## Configuration

Benchmark runs can be customized through:
- **Script arguments:** Direct command-line parameters
- **Slurm directives:** Resource allocation and job scheduling
- **[`dcp_fsdp.yaml`](../../../conf/dcp_fsdp.yaml):** Hydra configuration file for `benchmark.py`


## Troubleshooting

- **Insufficient Capacity:** Gradually increase node count or use multiple instance types
- **Network Issues:** Verify VPC/subnet configuration and security groups
- **S3 Access:** Confirm IAM roles and bucket permissions
- **Memory Issues:** Adjust model size or sharding strategy


## References

- [PyTorch Distributed Checkpoint Documentation][DCP]
- [PyTorch Distributed Checkpoint Recipe](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
- [PyTorch Elastic Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [AWS ParallelCluster User Guide](https://docs.aws.amazon.com/parallelcluster/)

[DCP]: https://pytorch.org/docs/stable/distributed.checkpoint.html
