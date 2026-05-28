import os, sys, time, csv, glob, subprocess, socket, logging
from pathlib import Path

import torch
from bioemu.sample import main as bioemu_sample

NUM_SAMPLES    = 100
BATCH_SIZES    = [10, 20, 50]
FASTA_DIRS     = {
    "50-100aa":   os.path.expandvars("$HOME/fasta/50-100aa"),
    "101-500aa":  os.path.expandvars("$HOME/fasta/101-500aa"),
    "501-1000aa": os.path.expandvars("$HOME/fasta/501-1000aa"),
}

OUTPUT_CSV  = os.path.expandvars("$SCRATCH/bioemu_results.csv")
OUTPUT_DIR  = os.path.expandvars("$SCRATCH/bioemu_outputs")
LOG_DIR     = os.path.expandvars("$SCRATCH/bioemu_logs")


def get_protein_logger(seq_range, stem):
    log_subdir = os.path.join(LOG_DIR, seq_range)
    os.makedirs(log_subdir, exist_ok=True)
    log_path = os.path.join(log_subdir, f"{stem}.log")

    logger = logging.getLogger(f"bioemu.{seq_range}.{stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Avoid duplicate handlers if the same protein is logged again.
    if not any(isinstance(h, logging.FileHandler) and
               getattr(h, "baseFilename", None) == os.path.abspath(log_path)
               for h in logger.handlers):
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
    return logger

def get_gpu_info():
    gpus = []
    for i in range(torch.cuda.device_count()):
        gpus.append({
            "index": i,
            "name":  torch.cuda.get_device_name(i),
            "total_mem_MB": torch.cuda.get_device_properties(i).total_memory // 1024**2,
        })
    return gpus

def measure_vram_mb():
    total = 0
    for i in range(torch.cuda.device_count()):
        total += torch.cuda.memory_allocated(i)
    return total / 1024**2

def peak_vram_mb():
    total = 0
    for i in range(torch.cuda.device_count()):
        total += torch.cuda.max_memory_allocated(i)
    return total / 1024**2


def run_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    node       = socket.gethostname()
    n_gpus     = torch.cuda.device_count()
    gpu_info   = get_gpu_info()
    job_id     = os.environ.get("SLURM_JOB_ID", "local")
    slurm_node = os.environ.get("SLURM_JOB_NODELIST", node)
    
    print(f"Host: {node} | GPUs: {n_gpus} | Job: {job_id}")
    for g in gpu_info:
        print(f"  GPU {g['index']}: {g['name']} ({g['total_mem_MB']} MB)")

    fieldnames = [
        "job_id", "node", "n_gpus", "gpu_name",
        "seq_range", "fasta_file", "seq_len",
        "num_samples", "batch_size",
        "wall_time_s", "peak_vram_mb", "status"
    ]

    write_header = not Path(OUTPUT_CSV).exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for seq_range, fasta_dir in FASTA_DIRS.items():
            fastas = sorted(glob.glob(f"{fasta_dir}/*.fasta") +
                            glob.glob(f"{fasta_dir}/*.fa"))
            if not fastas:
                print(f"BRAK plików w {fasta_dir}, pomijam.")
                continue

            for batch_size in BATCH_SIZES:
                for fasta_path in fastas:
                    # Odczytaj długość sekwencji
                    seq = ""
                    with open(fasta_path) as fp:
                        for line in fp:
                            if not line.startswith(">"):
                                seq += line.strip()
                    seq_len = len(seq)

                    stem = Path(fasta_path).stem
                    out_subdir = (f"{OUTPUT_DIR}/{seq_range}/"
                                  f"bs{batch_size}/{stem}")
                    os.makedirs(out_subdir, exist_ok=True)

                    logger = get_protein_logger(seq_range, stem)
                    logger.info(
                        "START %s | seq_range=%s | seq_len=%d | "
                        "num_samples=%d | batch_size=%d | node=%s | job=%s | "
                        "output_dir=%s",
                        Path(fasta_path).name, seq_range, seq_len,
                        NUM_SAMPLES, batch_size, slurm_node, job_id, out_subdir)

                    # Wyczyść statystyki VRAM
                    torch.cuda.reset_peak_memory_stats()

                    status = "ok"
                    t0 = time.perf_counter()
                    try:
                        bioemu_sample(
                            sequence=fasta_path,
                            num_samples=NUM_SAMPLES,
                            output_dir=out_subdir,
                            batch_size_100=batch_size,
                        )
                    except Exception as e:
                        status = f"ERROR: {e}"
                        print(f"  BŁĄD: {e}")
                        logger.exception("BŁĄD podczas bioemu_sample: %s", e)
                    wall = time.perf_counter() - t0

                    pvram = peak_vram_mb()

                    logger.info(
                        "END   %s | batch_size=%d | wall_time_s=%.2f | "
                        "peak_vram_mb=%.1f | status=%s",
                        Path(fasta_path).name, batch_size, wall, pvram, status)

                    row = dict(
                        job_id=job_id, node=slurm_node, n_gpus=n_gpus,
                        gpu_name=gpu_info[0]["name"] if gpu_info else "?",
                        seq_range=seq_range, fasta_file=Path(fasta_path).name,
                        seq_len=seq_len, num_samples=NUM_SAMPLES,
                        batch_size=batch_size,
                        wall_time_s=round(wall, 2),
                        peak_vram_mb=round(pvram, 1),
                        status=status,
                    )
                    writer.writerow(row)
                    f.flush()

                    print(f"  [{seq_range}] {Path(fasta_path).name} | "
                          f"bs={batch_size} | "
                          f"t={wall:.1f}s | VRAM={pvram:.0f}MB | {status}")

    print(f"\nWyniki zapisane: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()