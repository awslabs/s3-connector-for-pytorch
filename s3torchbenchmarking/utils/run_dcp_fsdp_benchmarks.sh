#!/usr/bin/env bash
#
# Run PyTorch’s Distributed Checkpointing (DCP) benchmarks using Fully Sharded Data Parallel (FSDP) training.
# Usage:
#   ./run_dcp_fsdp_benchmarks.sh        # Run save benchmarks (default)
#   ./run_dcp_fsdp_benchmarks.sh --save  # Run save benchmarks (explicit)
#   ./run_dcp_fsdp_benchmarks.sh --load  # Run load benchmarks
./utils/run_benchmarks.sh -s dcp_fsdp -d ./nvme/ "$@"

