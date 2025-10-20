#!/bin/bash

#SBATCH --job-name=code_problems
#SBATCH --partition=main
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=256G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8      # number of gpus per node
#SBATCH --output=/mnt/weka/home/anzexie/code_agent/code_questions/generate_code_problems/slurm_logs/slurm_%j.out
#SBATCH --error=/mnt/weka/home/anzexie/code_agent/code_questions/generate_code_problems/slurm_logs/slurm_%j.errs
#SBATCH --time=200:00:00
# SBATCH --qos=lowprio
# SBATCH --partition=lowprio
    #SBATCH --partition=main
    #SBATCH --reservation=moe

#module load cuda/12.4
# ldconfig /.singularity.d/lib
#source /lustre/scratch/users/guowei.he/scripts/load-apptainer.sh


source  /mnt/weka/home/anzexie/miniconda3/etc/profile.d/conda.sh
conda activate vllm


srun --nodes=9 --ntasks-per-node=1 \
    ray start --address='10.24.0.143:6379' --num-gpus=8 --block


