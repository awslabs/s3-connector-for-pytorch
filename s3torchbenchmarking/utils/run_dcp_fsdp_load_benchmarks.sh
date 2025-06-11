#!/usr/bin/env bash
#
# Run PyTorch’s Distributed Checkpointing (DCP) Load benchmarks using Fully Sharded Data Parallel (FSDP) training.

./utils/run_benchmarks.sh -s dcp_fsdp_load -d ./nvme/ "$@"
