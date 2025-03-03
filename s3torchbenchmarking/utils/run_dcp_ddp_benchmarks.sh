#!/usr/bin/env bash
#
# Run PyTorch’s Distributed Checkpointing (DCP) benchmarks using DistributedDataParallel (DDP) training.

./utils/run_benchmarks.sh -s dcp_ddp -d ./nvme/ "$@"
