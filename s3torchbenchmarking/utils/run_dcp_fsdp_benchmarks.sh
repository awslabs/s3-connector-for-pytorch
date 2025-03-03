#!/usr/bin/env bash
#
# Run PyTorch’s Distributed Checkpointing (DCP) benchmarks using Fully Sharded Data Parallel (FSDP) training.

./utils/run_benchmarks.sh -s dcp_fsdp -d ./nvme/ "$@"
