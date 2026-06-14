"""bioemu benchmark — num_samples sweep.

Companion to bioemu_benchmark.py. Same structure (SLURM rank sharding,
per-protein logs, per-rank CSV, $SCRATCH cache redirect, warmup phase
recorded separately) but the inner sweep is num_samples at a fixed
batch_size, not batch_size at a fixed num_samples.

SWEEP is a list of (num_samples, batch_size) tuples — edit it to turn
this into a joint sweep if needed; the script handles either case.

Protein subset: PROTEINS_PER_BUCKET fastas per seq_range, picked
alphabetically. Edit PROTEIN_ALLOWLIST below to use a hand-picked list
(e.g. only proteins that completed cleanly in the prior experiment).
"""
import os, time, csv, glob, socket, logging
from pathlib import Path

import torch
from bioemu.sample import main as bioemu_sample

# --- Sweep definition ----------------------------------------------------
# Each entry is (num_samples, batch_size). Fixed-bs sweep by default;
# convert to a joint sweep by adding more bs values per row.
SWEEP = [
    (  10, 50),
    (  25, 50),
    (  50, 50),
    ( 100, 50),
    ( 200, 50),
    ( 500, 50),
    (1000, 50),
]

WARMUP_NUM_SAMPLES = 1
WARMUP_BATCH_SIZE  = 50           # match the sweep's bs so the cache state
                                  # matches what timed runs see

# How many fastas to test per seq_range bucket. Keep small — 6 sweep
# values × this count × 3 buckets = total runs.
PROTEINS_PER_BUCKET = 10

# Filenames (without dir) chosen for this experiment. Picked from proteins
# that completed all 3 batch sizes cleanly in the prior batch-size run,
# spread evenly across each bucket's length range. The 501-1000aa entries
# are smoke tests — bioemu's `range() arg 3 must not be zero` hit 100%
# of them last time, so they'll fail in warmup (<1 s each) and the
# script will skip their sample sweep. Kept in the list so we'd notice
# if a bioemu update fixes the bug.
# Leave empty to fall back to the first PROTEINS_PER_BUCKET alphabetically.
PROTEIN_ALLOWLIST = [
    # --- 50-100aa (len 50–100) ---
    "MGYG000139077_02576.fasta",
    "MGYG000169671_00821.fasta",
    "MGYG000002467_02821.fasta",
    "MGYG000078498_00408.fasta",
    "MGYG000001648_00738.fasta",
    "MGYG000003606_00776.fasta",
    "MGYG000190757_00999.fasta",
    "MGYG000251045_00692.fasta",
    "MGYG000189504_00958.fasta",
    "MGYG000051780_00386.fasta",
    # --- 101-500aa (len 102–314) ---
    "MGYG000065599_00399.fasta",
    "MGYG000252235_00511.fasta",
    "MGYG000176001_00991.fasta",
    "MGYG000083600_00853.fasta",
    "MGYG000018992_02079.fasta",
    "MGYG000000389_02024.fasta",
    "MGYG000002818_01096.fasta",
    "MGYG000002683_01443.fasta",
    "MGYG000000574_02893.fasta",
    "MGYG000004730_00424.fasta",
    # --- 501-1000aa (len 540–814, smoke tests; expected to fail) ---
    "MGYG000000477_01569.fasta",
    "MGYG000000619_00976.fasta",
    "MGYG000001250_01095.fasta",
    "MGYG000001451_04612.fasta",
    "MGYG000001479_01649.fasta",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FASTA_DIRS = {
    "50-100aa":   os.path.join(BASE_DIR, "100x50aa-100aa"),
    "101-500aa":  os.path.join(BASE_DIR, "100x101aa-500aa"),
    "501-1000aa": os.path.join(BASE_DIR, "100x501aa-1000aa"),
}

# --- Identity / paths ----------------------------------------------------
RANK     = int(os.environ.get("SLURM_PROCID", "0"))
NRANKS   = int(os.environ.get("SLURM_NTASKS", "1"))
JOB_NAME = os.environ.get("SLURM_JOB_NAME", "local")

# Distinct prefix so this experiment's CSVs don't mix with the
# batch-size benchmark output. (Same naming scheme: rank suffix only
# when nranks > 1; SLURM job name included to keep 1gpu/8gpu/multinode
# results in their own files.)
_EXP_TAG     = "numsamples"
_RANK_SUFFIX = f".rank{RANK}" if NRANKS > 1 else ""
OUTPUT_CSV   = os.path.join(BASE_DIR,
                f"bioemu_results.{_EXP_TAG}.{JOB_NAME}{_RANK_SUFFIX}.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "bioemu_outputs", _EXP_TAG)
LOG_DIR      = (os.path.join(BASE_DIR, "bioemu_logs", _EXP_TAG,
                             JOB_NAME, f"rank{RANK}")
                if NRANKS > 1 else
                os.path.join(BASE_DIR, "bioemu_logs", _EXP_TAG, JOB_NAME))

_SCRATCH    = os.environ.get("SCRATCH", BASE_DIR)
CACHE_BASE  = os.path.join(_SCRATCH, "bioemu_caches")
CACHE_EMBED = os.path.join(CACHE_BASE, "embeds")
CACHE_SO3   = os.path.join(CACHE_BASE, "so3")


def get_protein_logger(seq_range, stem):
    log_subdir = os.path.join(LOG_DIR, seq_range)
    os.makedirs(log_subdir, exist_ok=True)
    log_path = os.path.join(log_subdir, f"{stem}.log")

    logger = logging.getLogger(f"bioemu.{_EXP_TAG}.{seq_range}.{stem}")
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
    return [{
        "index": i,
        "name":  torch.cuda.get_device_name(i),
        "total_mem_MB": torch.cuda.get_device_properties(i).total_memory // 1024**2,
    } for i in range(torch.cuda.device_count())]

def peak_vram_mb():
    return sum(torch.cuda.max_memory_allocated(i)
               for i in range(torch.cuda.device_count())) / 1024**2


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


def select_proteins():
    """Return list of (seq_range, fasta_path). Uses PROTEIN_ALLOWLIST if
    set, else the first PROTEINS_PER_BUCKET fastas alphabetically per
    bucket."""
    out = []
    for seq_range, fasta_dir in FASTA_DIRS.items():
        fastas = sorted(glob.glob(f"{fasta_dir}/*.fasta") +
                        glob.glob(f"{fasta_dir}/*.fa"))
        if PROTEIN_ALLOWLIST:
            chosen = [p for p in fastas
                      if Path(p).name in PROTEIN_ALLOWLIST]
        else:
            chosen = fastas[:PROTEINS_PER_BUCKET]
        if not chosen:
            print(f"BRAK plików w {fasta_dir}, pomijam.")
            continue
        for p in chosen:
            out.append((seq_range, p))
    return out


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
          f"rank: {RANK}/{NRANKS} | job: {job_id} | exp: {_EXP_TAG}")
    print(f"  sweep: {SWEEP}")
    print(f"  cache_embeds_dir: {CACHE_EMBED}")
    print(f"  cache_so3_dir   : {CACHE_SO3}")
    for g in gpu_info:
        print(f"  GPU {g['index']}: {g['name']} ({g['total_mem_MB']} MB)")

    all_proteins = select_proteins()
    my_proteins  = all_proteins[RANK::NRANKS]
    print(f"Rank {RANK}: {len(my_proteins)}/{len(all_proteins)} proteins "
          f"(each → 1 warmup + {len(SWEEP)} timed runs)")

    fieldnames = [
        "experiment", "job_id", "node", "rank", "n_gpus", "gpu_name",
        "seq_range", "fasta_file", "seq_len",
        "phase", "cache_warm",
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
                experiment=_EXP_TAG,
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
                  f"phase={phase} | ns={num_samples} | bs={batch_size} | "
                  f"t={wall:.1f}s | VRAM={pvram:.0f}MB | {status}")

        for seq_range, fasta_path in my_proteins:
            seq_len = _read_seq_len(fasta_path)
            stem    = Path(fasta_path).stem
            logger  = get_protein_logger(seq_range, stem)

            # Warmup: populate MSA + embedding caches once per protein.
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

            # Timed runs: the num_samples sweep with warm caches.
            for num_samples, batch_size in SWEEP:
                out_subdir = (f"{OUTPUT_DIR}/{seq_range}/"
                              f"ns{num_samples}_bs{batch_size}/{stem}")
                wall, pvram, status = _timed_sample(
                    fasta_path=fasta_path,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    out_subdir=out_subdir,
                    logger=logger,
                    phase="sample",
                    cache_warm=True,
                )
                _emit(seq_range, fasta_path, seq_len, "sample", True,
                      num_samples, batch_size, wall, pvram, status)

    print(f"\nRank {RANK} wyniki zapisane: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()
