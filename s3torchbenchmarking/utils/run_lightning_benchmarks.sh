#!/usr/bin/env bash
#
# Run PyTorch Lightning Checkpointing benchmarks.

./utils/run_benchmarks.sh lightning_checkpointing ./nvme/ "$@"
