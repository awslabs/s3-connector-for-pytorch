#!/bin/bash
#SBATCH --job-name=fsdp-2x2
#SBATCH --partition=train
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=1    # ONE torchrun per node
#SBATCH --cpus-per-task=16
#-------------------------------------------------
# Network interface detection - try multiple common interfaces
for iface in eth0 ens5 ib0; do
    if ip link show $iface &>/dev/null; then
        export NCCL_SOCKET_IFNAME=$iface
        export GLOO_SOCKET_IFNAME=$iface
        echo "Using network interface: $iface"
        break
    fi
done

# First node becomes the rendez-vous server
head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
         hostname -I | awk '{print $1}')  # first IPv4 only
export MASTER_PORT=12356              # any free port

# For CPU-only training with gloo backend
export WORLD_SIZE=$SLURM_JOB_NUM_NODES   # 1 process per node for CPU
echo "MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE"

# Add timeout and debugging
export NCCL_DEBUG=INFO
export GLOO_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# Add network connectivity test
echo "Testing connectivity from head node..."
srun --nodes=1 --ntasks=1 -w "$head_node" ping -c 1 $MASTER_ADDR
#-------------------------------------------------
# Launch - CPU only with 1 process per node
srun torchrun \
   --nnodes=$SLURM_JOB_NUM_NODES \
   --nproc_per_node=1 \
   --rdzv_backend=c10d \
   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
   --rdzv_id=$SLURM_JOB_ID \
   --max_restarts=3 \
   --start_method=spawn \
  load3.py \
   --backend gloo \
   --uri s3://shadow-copies-fsdp/ \
   --region us-east-2gg
