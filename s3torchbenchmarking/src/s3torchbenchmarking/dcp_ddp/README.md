## PyTorch's Distributed Checkpoint (DCP) benchmarks using DistributedDataParallel (DDP) training

The `dcp` Python package provides a suite of benchmarks designed to evaluate and measure the performance
of [PyTorch's Distributed Checkpointing (DCP)][DCP] feature in comparison to the `s3torchconnector` library.

These benchmarks specifically use DistributedDataParallel (DDP), which is PyTorch's standard approach 
for distributed training where the model is replicated across multiple GPUs/processes. 
With DDP, each process maintains a complete copy of the model parameters, making it suitable for scenarios 
where memory requirements per GPU are manageable.

### Purpose

These benchmarks focus on testing the "save" mechanism of PyTorch DCP (`torch.distributed.checkpoint.save`). The primary
objectives are to evaluate the `s3torchconnector` library's performance against other libraries and local storage
options, by measuring the following metrics:

- Checkpoint saving throughput (in MiB/s);
- Checkpoint "corrected" save durations (in seconds), which exclude the influence of model load duration on the device.

### Configuration

The benchmark runs can be customized through the [`dcp_ddp.yaml`](../../../conf/dcp_ddp.yaml) file.

> [!IMPORTANT]
> A `+path` option is passed to the running script ([`run_dcp_ddp_benchmarks.sh`](../../../utils/run_dcp_ddp_benchmarks.sh)),
> and will be used only if `checkpoint.storage` key includes `disk`.

### References

- https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
- https://pytorch.org/docs/stable/elastic/run.html
- https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

[DCP]: https://pytorch.org/docs/stable/distributed.checkpoint.html

[multirun]: https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/
