#!/bin/bash
#SBATCH --job-name=bioemu_8gpu
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load Miniforge3
conda activate bioemu

export HF_HOME=$SCRATCH/hf_cache

# Każdy task dostaje swoje GPU
srun --ntasks=8 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  python ~/bioemu_benchmark.py
'