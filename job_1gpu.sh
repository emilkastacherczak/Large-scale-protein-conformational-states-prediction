#!/bin/bash
#SBATCH --job-name=bioemu_1gpu
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load Miniforge3
conda activate /net/tscratch/people/plgestacherczak/conda_envs/bioemu

export HF_HOME=$SCRATCH/hf_cache
export CUDA_VISIBLE_DEVICES=0

echo "=== JOB: $SLURM_JOB_ID | NODE: $SLURMD_NODENAME | GPUs: 1 ==="
python ~/bioemu_benchmark.py