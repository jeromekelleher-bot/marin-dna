"""Export H100 complete-variant rates from the recorded scoring intervals (CPU only)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

HERE = Path(__file__).resolve().parent
ROWS = 9490
TEST_SHA = "658dd3c49b2c171d7c5c973e014dd3d2e8d425e25187eb30513df64d9ef58883"
MARIN = "mix-v0.9-p1B-i24-exp135-m5.1-step-59158"
MODELS = [
    ("fp8", MARIN, "Optimized FP8", 256, 128),
    ("bf16", MARIN, "Compiled BF16", 256, 128),
    ("evo1b", "evo2_1b_base", "Evo 2 1B base", 8192, 16),
    ("evo7b", "evo2_7b", "Evo 2 7B", 8192, 8),
    ("evo40b", "evo2_40b", "Evo 2 40B", 8192, 1),
]


def extract_shard(log):
    """Extract both strand clocks at batch 1 and the final batch; tqdm prints whole seconds."""
    pattern = r"strand=([+-]):[^\r\n]*?\|\s*(\d+)/(\d+)\s+\[([\d:]+)<"
    endpoints = {"+": {}, "-": {}}
    counts = set()
    for strand, completed, total, elapsed in re.findall(pattern, log):
        total, completed = int(total), int(completed)
        counts.add(total)
        if completed not in (1, total):
            continue
        seconds = 0
        for component in elapsed.split(":"):
            seconds = seconds * 60 + int(component)
        endpoints[strand]["first" if completed == 1 else "last"] = seconds
    if len(counts) != 1 or any(set(pair) != {"first", "last"} for pair in endpoints.values()):
        raise ValueError("Missing or inconsistent strand timing endpoints")
    if any(pair["last"] <= pair["first"] for pair in endpoints.values()):
        raise ValueError("Non-positive strand duration")
    return {"rows": counts.pop(), "strand_elapsed_seconds": endpoints}


def calculate(kind, measured, length, batch):
    if kind == "evo40b":
        shards = measured["partitions"]
        if len(shards) != 4 or sum(p["rows"] for p in shards) != ROWS:
            raise ValueError("Expected four disjoint partitions covering the release")
        seconds = sum(pair["last"] - pair["first"] for shard in shards
                      for pair in shard["strand_elapsed_seconds"].values())
        return ROWS - len(shards), seconds, 2, 4, 1
    if (measured["n_variants"], measured["context_tokens"], measured["batch_size"],
        measured["gpu_count"], measured["variant_dataset_sha256"]) != (ROWS, length, batch, 1, TEST_SHA):
        raise ValueError(f"Unexpected workload in {kind}")
    if kind in {"fp8", "bf16"}:
        if measured["quantization"] != ("none" if kind == "bf16" else "te-fp8-delayed"):
            raise ValueError(f"Unexpected quantization in {kind}")
        if not measured["torch_compile"] or measured["n_strand_rows"] != 2 * ROWS:
            raise ValueError("Expected compiled, paired-strand Marin run")
        if kind == "fp8" and not all(measured[k] > 0 for k in ("quantized_linear_count", "fused_mlp_count", "fused_qkv_count")):
            raise ValueError("FP8 conversion incomplete")
        repeats = [a + b for a, b in zip(measured["forward_seconds"], measured["reverse_seconds"], strict=True)]
        if len(repeats) != 3 or repeats != measured["repeat_seconds"]:
            raise ValueError("Expected three paired Marin timing repetitions")
        seconds = median(repeats)
        repetitions = len(repeats)
    else:
        if measured["warmup_batches_per_strand"] != 1:
            raise ValueError("Unexpected Evo warmup")
        seconds = measured["both_strands_seconds"]
        repetitions = 1
    if not math.isfinite(seconds) or seconds <= 0 or not math.isclose(3600*ROWS/seconds, measured["variants_per_h100_hour"]):
        raise ValueError(f"Rate disagrees with the archived measurement for {kind}")
    return ROWS, seconds, 1, 1, repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Optional downloaded original timing files in subdirectories named by configuration")
    args = parser.parse_args()
    sources = json.loads((HERE / "sources.json").read_text())
    missing = {model[0] for model in MODELS} - sources["runs"].keys()
    if missing:
        raise SystemExit("Missing completed timing records: " + ", ".join(sorted(missing)))
    rows = []
    for kind, model, label, length, batch in MODELS:
        run = sources["runs"][kind]
        filename = "shard-times.json" if kind == "evo40b" else "benchmark.json"
        measured = json.loads((HERE / "records" / kind / filename).read_text())
        if args.archive:
            for name, info in run["timing_inputs"].items():
                data = (args.archive / kind / name).read_bytes()
                if len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                    raise ValueError(f"Corrupt timing input {kind}/{name}")
            if kind == "evo40b":
                original = {"partitions": [extract_shard((args.archive / kind / f"logs/partition-{i}.log").read_text())
                                            for i in range(4)]}
            else:
                original = json.loads((args.archive / kind / filename).read_text())
                if kind.startswith("evo"):
                    strands = [json.loads((args.archive / kind / f"strand-{s}.json").read_text()) for s in "+-"]
                    if any(sum(n for n, _ in strand["batches"]) != ROWS for strand in strands):
                        raise ValueError(f"Missing scored rows in {kind}")
                    if not math.isclose(sum(s["seconds"] for s in strands), original["both_strands_seconds"]):
                        raise ValueError(f"Strand durations disagree for {kind}")
            if original != measured:
                raise ValueError(f"Archived timings differ for {kind}")
        timed, seconds, gpus, workers, repeats = calculate(kind, measured, length, batch)
        worker_rate = 3600 * timed / seconds
        rows.append(dict(method_id=model, configuration=label, context_tokens=length, batch_size=batch,
                         variants_scored=ROWS, variants_timed=timed, gpus_per_worker=gpus, concurrent_workers=workers,
                         repetitions=repeats, summed_worker_scoring_seconds=f"{seconds:.6f}",
                         variants_per_worker_hour=f"{worker_rate:.6f}", variants_per_h100_hour=f"{worker_rate/gpus:.6f}",
                         iris_job=run["iris"]))
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
