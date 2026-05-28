import os, sys, time, csv, glob, socket, logging
from pathlib import Path

import torch
from bioemu.sample import main as bioemu_sample

NUM_SAMPLES        = 100
BATCH_SIZES        = [10, 20, 50]
WARMUP_NUM_SAMPLES = 1           
WARMUP_BATCH_SIZE  = BATCH_SIZES[0]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FASTA_DIRS = {
    "50-100aa":   os.path.join(BASE_DIR, "100x50aa-100aa"),
    "101-500aa":  os.path.join(BASE_DIR, "100x101aa-500aa"),
    "501-1000aa": os.path.join(BASE_DIR, "100x501aa-1000aa"),
}

RANK   = int(os.environ.get("SLURM_PROCID", "0"))
NRANKS = int(os.environ.get("SLURM_NTASKS", "1"))

_RANK_SUFFIX = f".rank{RANK}" if NRANKS > 1 else ""
OUTPUT_CSV = os.path.join(BASE_DIR, f"bioemu_results{_RANK_SUFFIX}.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "bioemu_outputs")
LOG_DIR    = (os.path.join(BASE_DIR, "bioemu_logs", f"rank{RANK}")
              if NRANKS > 1 else os.path.join(BASE_DIR, "bioemu_logs"))

_SCRATCH    = os.environ.get("SCRATCH", BASE_DIR)
CACHE_BASE  = os.path.join(_SCRATCH, "bioemu_caches")
CACHE_EMBED = os.path.join(CACHE_BASE, "embeds")
CACHE_SO3   = os.path.join(CACHE_BASE, "so3")


def get_protein_logger(seq_range, stem):
    log_subdir = os.path.join(LOG_DIR, seq_range)
    os.makedirs(log_subdir, exist_ok=True)
    log_path = os.path.join(log_subdir, f"{stem}.log")

    logger = logging.getLogger(f"bioemu.{seq_range}.{stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
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


def _read_seq_len(fasta_path):
    seq = ""
    with open(fasta_path) as fp:
        for line in fp:
            if not line.startswith(">"):
                seq += line.strip()
    return len(seq)


def _timed_sample(fasta_path, num_samples, batch_size, out_subdir, logger,
                  phase, cache_warm):

    os.makedirs(out_subdir, exist_ok=True)
    logger.info(
        "START phase=%s | %s | num_samples=%d | batch_size=%d | "
        "cache_warm=%s | output_dir=%s",
        phase, Path(fasta_path).name, num_samples, batch_size,
        cache_warm, out_subdir)

    torch.cuda.reset_peak_memory_stats()
    status = "ok"
    t0 = time.perf_counter()
    try:
        bioemu_sample(
            sequence=fasta_path,
            num_samples=num_samples,
            output_dir=out_subdir,
            batch_size_100=batch_size,
            cache_embeds_dir=CACHE_EMBED,
            cache_so3_dir=CACHE_SO3,
        )
    except Exception as e:
        status = f"ERROR: {e}"
        print(f"  BŁĄD ({phase}): {e}")
        logger.exception("BŁĄD podczas bioemu_sample (%s): %s", phase, e)
    wall  = time.perf_counter() - t0
    pvram = peak_vram_mb()

    logger.info(
        "END   phase=%s | %s | num_samples=%d | batch_size=%d | "
        "wall_time_s=%.2f | peak_vram_mb=%.1f | status=%s",
        phase, Path(fasta_path).name, num_samples, batch_size,
        wall, pvram, status)
    return wall, pvram, status


def run_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_EMBED, exist_ok=True)
    os.makedirs(CACHE_SO3,   exist_ok=True)

    node       = socket.gethostname()
    n_gpus     = torch.cuda.device_count()
    gpu_info   = get_gpu_info()
    job_id     = os.environ.get("SLURM_JOB_ID", "local")
    slurm_node = os.environ.get("SLURM_JOB_NODELIST", node)

    print(f"Host: {node} | visible GPUs: {n_gpus} | "
          f"rank: {RANK}/{NRANKS} | job: {job_id}")
    print(f"  cache_embeds_dir: {CACHE_EMBED}")
    print(f"  cache_so3_dir   : {CACHE_SO3}")
    for g in gpu_info:
        print(f"  GPU {g['index']}: {g['name']} ({g['total_mem_MB']} MB)")


    all_proteins = []
    for seq_range, fasta_dir in FASTA_DIRS.items():
        fastas = sorted(glob.glob(f"{fasta_dir}/*.fasta") +
                        glob.glob(f"{fasta_dir}/*.fa"))
        if not fastas:
            print(f"BRAK plików w {fasta_dir}, pomijam.")
            continue
        for fasta_path in fastas:
            all_proteins.append((seq_range, fasta_path))

    my_proteins = all_proteins[RANK::NRANKS]
    print(f"Rank {RANK}: {len(my_proteins)}/{len(all_proteins)} proteins "
          f"(each → 1 warmup + {len(BATCH_SIZES)} timed runs)")

    fieldnames = [
        "job_id", "node", "rank", "n_gpus", "gpu_name",
        "seq_range", "fasta_file", "seq_len",
        "phase",                 # "warmup" or "sample"
        "cache_warm",            # False on warmup, True on timed runs
        "num_samples", "batch_size",
        "wall_time_s", "peak_vram_mb", "status",
    ]

    write_header = not Path(OUTPUT_CSV).exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        def _emit(seq_range, fasta_path, seq_len, phase, cache_warm,
                  num_samples, batch_size, wall, pvram, status):
            row = dict(
                job_id=job_id, node=slurm_node, rank=RANK, n_gpus=n_gpus,
                gpu_name=gpu_info[0]["name"] if gpu_info else "?",
                seq_range=seq_range, fasta_file=Path(fasta_path).name,
                seq_len=seq_len,
                phase=phase, cache_warm=cache_warm,
                num_samples=num_samples, batch_size=batch_size,
                wall_time_s=round(wall, 2),
                peak_vram_mb=round(pvram, 1),
                status=status,
            )
            writer.writerow(row)
            f.flush()
            print(f"  [rank{RANK}] [{seq_range}] {Path(fasta_path).name} | "
                  f"phase={phase} | bs={batch_size} | ns={num_samples} | "
                  f"t={wall:.1f}s | VRAM={pvram:.0f}MB | {status}")

        for seq_range, fasta_path in my_proteins:
            seq_len = _read_seq_len(fasta_path)
            stem    = Path(fasta_path).stem
            logger  = get_protein_logger(seq_range, stem)

            warmup_dir = f"{OUTPUT_DIR}/warmup/{seq_range}/{stem}"
            w_wall, w_pvram, w_status = _timed_sample(
                fasta_path=fasta_path,
                num_samples=WARMUP_NUM_SAMPLES,
                batch_size=WARMUP_BATCH_SIZE,
                out_subdir=warmup_dir,
                logger=logger,
                phase="warmup",
                cache_warm=False,
            )
            _emit(seq_range, fasta_path, seq_len, "warmup", False,
                  WARMUP_NUM_SAMPLES, WARMUP_BATCH_SIZE,
                  w_wall, w_pvram, w_status)

            if w_status != "ok":
                logger.warning(
                    "Pomijam timed runs dla %s — warmup zwrócił %s",
                    Path(fasta_path).name, w_status)
                continue

            for batch_size in BATCH_SIZES:
                out_subdir = (f"{OUTPUT_DIR}/{seq_range}/"
                              f"bs{batch_size}/{stem}")
                wall, pvram, status = _timed_sample(
                    fasta_path=fasta_path,
                    num_samples=NUM_SAMPLES,
                    batch_size=batch_size,
                    out_subdir=out_subdir,
                    logger=logger,
                    phase="sample",
                    cache_warm=True,
                )
                _emit(seq_range, fasta_path, seq_len, "sample", True,
                      NUM_SAMPLES, batch_size, wall, pvram, status)

    print(f"\nRank {RANK} wyniki zapisane: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()
