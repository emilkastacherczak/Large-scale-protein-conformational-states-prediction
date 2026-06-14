#!/bin/bash
#SBATCH --job-name=bioemu_numsamples_1gpu
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate /net/tscratch/people/plgestacherczak/conda_envs/bioemu

export HF_HOME=$SCRATCH/hf_cache
export CUDA_VISIBLE_DEVICES=0

cd "$SLURM_SUBMIT_DIR"
echo "=== JOB: $SLURM_JOB_ID | NODE: $SLURMD_NODENAME | GPUs: 1 ==="
python "$SLURM_SUBMIT_DIR/bioemu_benchmark_numsamples.py"
