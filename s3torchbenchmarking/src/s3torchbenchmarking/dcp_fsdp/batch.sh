#!/bin/bash
#SBATCH --job-name=fsdp-2x2
#SBATCH --partition=train
#SBATCH --nodes=100
#SBATCH --ntasks-per-node=1    # ONE torchrun per node
#SBATCH --cpus-per-task=16
#-------------------------------------------------
# Pick an interface that all nodes can see.
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0   # gloo/C10d uses this, too
# First node becomes the rendez-vous server
head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
         hostname -I | awk '{print $1}')  # first IPv4 only
export MASTER_PORT=12356              # any free port
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * 2))   # 2 GPUs each
echo "MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE"
#-------------------------------------------------
# Launch
srun torchrun \
   --nnodes=$SLURM_JOB_NUM_NODES \
   --nproc_per_node=2 \
   --rdzv_backend=c10d \
   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
   --rdzv_id=$SLURM_JOB_ID \
  load3.py \
   --backend gloo \
   --uri <bucket> \
   --region us-east-2 \
