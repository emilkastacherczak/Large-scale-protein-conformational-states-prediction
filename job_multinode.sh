#!/bin/bash
#SBATCH --job-name=bioemu_multinode
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --output=bioemu_outputs/%x_%j.out
#SBATCH --error=bioemu_outputs/%x_%j.err

module load Miniconda3
conda init bash
conda activate /net/tscratch/people/plgestacherczak/conda_envs/bioemu

export HF_HOME=$SCRATCH/hf_cache

cd "$SLURM_SUBMIT_DIR"

srun bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  python "$SLURM_SUBMIT_DIR/bioemu_benchmark.py"
'