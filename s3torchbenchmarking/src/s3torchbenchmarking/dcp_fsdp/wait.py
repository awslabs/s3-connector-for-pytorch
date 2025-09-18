#!/usr/bin/env python3
import time
import argparse
import os
import torch.distributed as dist


def keep_busy(duration_minutes=60):
    """Keep nodes busy with minimal resource usage."""
    end_time = time.time() + (duration_minutes * 60)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("gloo")

    print(f"Rank {rank}: Starting idle loop for {duration_minutes} minutes")

    while time.time() < end_time:
        # Minimal work to stay alive
        time.sleep(10)

        # Periodic sync to keep distributed group alive
        if world_size > 1 and time.time() % 60 < 10:
            try:
                dist.barrier()
            except:
                pass

    print(f"Rank {rank}: Idle period complete")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duration", type=int, default=60, help="Duration in minutes to stay busy"
    )
    args = parser.parse_args()

    keep_busy(args.duration)
