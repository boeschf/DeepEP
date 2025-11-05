#!/bin/bash
#SBATCH --account=csstaff
#SBATCH --job-name=deepep-test-internode
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_02.log
#SBATCH --uenv=pytorch/v2.8.0:rc24:/user-environment
#SBATCH --view=default

srun -ul --mpi=pmi2 bash -c "
PYTHONPATH=\$(pwd) \
NVSHMEM_DISABLE_CUDA_VMM=1 \
NVSHMEM_SYMMETRIC_SIZE=4G \
NVSHMEM_REMOTE_TRANSPORT=libfabric \
MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \
MASTER_PORT=29500 \
RANK=\${SLURM_NODEID} \
WORLD_SIZE=\${SLURM_NTASKS} \
.venv/bin/python tests/test_internode.py --num-processes=4
"

#FI_MR_CACHE_MONITOR=userfaultfd NCCL_NET_GDR_LEVEL=PHB NCCL_NET="AWS Libfabric" LD_LIBRARY_PATH=/user-environment/env/default/lib64:/user-environment/env/default/lib/python3.12/site-packages/torch/lib  python tests/test_intranode.py --num-processes=4

