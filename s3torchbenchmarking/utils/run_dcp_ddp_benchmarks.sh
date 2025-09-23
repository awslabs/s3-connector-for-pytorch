#!/usr/bin/env bash
#
# Run PyTorch's Distributed Checkpointing (DCP) benchmarks using DistributedDataParallel (DDP) training.
# Usage:
#   ./run_dcp_ddp_benchmarks.sh        # Run save benchmarks (default)
#   ./run_dcp_ddp_benchmarks.sh --save  # Run save benchmarks (explicit)
#   ./run_dcp_ddp_benchmarks.sh --load  # Run load benchmarks

./utils/run_benchmarks.sh -s dcp_ddp -d ./nvme/ "$@"
