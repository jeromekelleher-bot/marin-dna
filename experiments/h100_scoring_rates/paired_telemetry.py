"""Align raw two-second GPU and cgroup samples with each timed strand of the paired check."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median

SEGMENTS = ("01-bf16", "02-fp8", "03-fp8", "04-bf16")


def gpu_samples(path):
    with path.open() as stream:
        reader = csv.reader(stream, skipinitialspace=True)
        header = next(reader)
        if header[:4] != ["timestamp", "uuid", "name", "pstate"] or len(header) != 12:
            raise ValueError(f"Unexpected nvidia-smi header {header}")
        for row in reader:
            timestamp, uuid, _, _, sm, memory, temperature, power, _, util, _, _ = row
            instant = datetime.fromisoformat(timestamp.replace("/", "-")).replace(tzinfo=timezone.utc)
            yield instant.timestamp(), uuid, float(sm), float(memory), float(temperature), float(power), float(util)


def summarize(directory):
    gpu = list(gpu_samples(directory / "gpu-telemetry.csv"))
    host = [json.loads(line) for line in (directory / "host-telemetry.jsonl").read_text().splitlines()]
    if len({s[1] for s in gpu}) != 1:
        raise ValueError("Expected one GPU UUID throughout the paired check")
    for segment in SEGMENTS:
        bench = json.loads((directory / segment / "benchmark.json").read_text())
        intervals = [json.loads(line) for line in (directory / segment / "intervals.jsonl").read_text().splitlines()]
        expected = {(s, i): t for s, key in (("+", "forward_seconds"), ("-", "reverse_seconds"))
                    for i, t in enumerate(bench[key], 1)}
        if [(r["strand"], r["repetition"]) for r in intervals] != list(expected):
            raise ValueError(f"Unexpected strand intervals in {segment}")
        for interval in intervals:
            start, end = interval["start_epoch"], interval["end_epoch"]
            strand, rep = interval["strand"], interval["repetition"]
            seconds = expected[strand, rep]
            if seconds != interval["measured_seconds"] or abs(end - start - seconds) > 0.05:
                raise ValueError(f"Strand clocks disagree in {segment}")
            samples = [s for s in gpu if start <= s[0] <= end]
            cpu = [s for s in host if start <= s["epoch"] <= end]
            if not samples or len(cpu) < 2:
                raise ValueError(f"Insufficient telemetry in {segment}")
            def cpu_stat(sample):
                return dict(line.split() for line in sample["/sys/fs/cgroup/cpu.stat"].splitlines())
            a, b = cpu_stat(cpu[0]), cpu_stat(cpu[-1])
            yield dict(segment=segment, strand=strand, repetition=rep, measured_seconds=seconds,
                       gpu_samples=len(samples), gpu_uuid=samples[0][1],
                       sm_mhz_min=min(s[2] for s in samples), sm_mhz_median=median(s[2] for s in samples),
                       memory_mhz_median=median(s[3] for s in samples), temp_c_max=max(s[4] for s in samples),
                       power_w_median=median(s[5] for s in samples), gpu_util_pct_median=median(s[6] for s in samples),
                       host_samples=len(cpu), host_sample_span_seconds=round(cpu[-1]["epoch"] - cpu[0]["epoch"], 6),
                       throttled_periods=int(b["nr_throttled"]) - int(a["nr_throttled"]),
                       throttled_usec=int(b["throttled_usec"]) - int(a["throttled_usec"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = list(summarize(args.archive))
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
