# Benchmarking Large-Scale Protein Conformational Sampling with BioEmu

This repository contains a reproducible benchmark for large-scale protein
conformational sampling using the `bioemu` sampler. The benchmark measures
wall-clock time and peak GPU memory (VRAM) for many protein sequences across
different batch sizes and compute configurations (single-GPU, multi-GPU,
multinode). Experiments were executed on the Athena supercomputer.

## Main components
- `bioemu_benchmark.py` - main benchmark driver (warmup + timed runs, CSV/log
	output, SLURM-aware partitioning).
- `job_1gpu.sh`, `job_8gpu.sh`, `job_multinode.sh` - job scripts.
- `100x50aa-100aa/`, `100x101aa-500aa/`, `100x501aa-1000aa/` - FASTA datasets used for the runs.
- `analyze.ipynb` - notebook for aggregating results and plotting figures.

