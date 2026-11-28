# H100 variant scoring rates at deployed context lengths

This experiment times complete variant scoring for Marin DNA 1B (compiled BF16 and delayed FP8) and Evo 2 (1B base, 7B, and 40B) on CoreWeave H100 80 GB GPUs.
All five configurations score the same 9,490 held-out human Mendelian variants, with both alleles, both strands, likelihood ratios, next-token Jensen–Shannon divergence, and mean-pooled final-layer embeddings.
The H100 pair used for each Evo 2 40B model replica follows the earlier numerical validation. [Hardware comparison](https://github.com/jeromekelleher-bot/marin-dna/tree/b5916b3dc11d32c2de6237cde2a864dfd463b73c/experiments/evo2_hardware) · [Held-out evaluation](https://github.com/Open-Athena/marin-dna/issues/595)

The original BF16 and FP8 submissions exited during environment setup because the container selected `/app/.venv`.
The successful replacements explicitly selected the benchmark project environment.
Their three paired-strand intervals were 34.13, 33.50 and 33.41 seconds for BF16, and 27.32, 42.11 and 60.82 seconds for FP8.
The FP8 run converted 19 layers of each reported fused projection type and reproduced both scalar scores exactly on 32 rows when the harness toggled embedding output.
The spread in FP8 timing motivated a same-GPU check with a prespecified BF16, FP8, FP8, BF16 sequence. Each segment used a fresh Python process and compiler caches, one warmup batch per strand and three timed repetitions. The primary Marin rates pool all six paired-strand durations per precision and divide the row count by their median. The separately launched measurements remain in `records/bf16` and `records/fp8`.

Each segment saved wall-clock endpoints outside the timed region. We aligned these with two-second GPU and host cgroup samples to summarize clocks, power, utilization, temperature and CPU throttling during each timed strand. [Per-strand measurements](records/paired/strand-telemetry.csv)

The median paired-strand durations were 43.03 seconds for BF16 and 53.71 seconds for FP8. The resulting rates were 793,924 and 636,094 complete variants per H100-hour, respectively. BF16's first three durations were 34.45, 33.83 and 33.81 seconds, and its last three were 51.61, 60.14 and 61.84 seconds. The six FP8 durations were 52.57, 68.98, 54.85, 50.43, 47.89 and 61.10 seconds. Median sampled GPU utilization was zero in eight of twelve timed FP8 strand intervals and one of twelve BF16 intervals. None of the covered cgroup sampling intervals showed CPU throttling. The cause of the idle periods remains undetermined.

**Run status:** The four same-GPU segments and the four original completed configurations have verified outputs. The final Evo 2 7B output is pending.

## Timing definitions

| Configuration | Context | Batch size in variants | Scoring interval |
|---|---:|---:|---|
| Marin DNA 1B, compiled BF16 and optimized FP8 | 256 tokens | 128 | Median of six paired forward- and reverse-strand passes across two processes per precision |
| Evo 2 1B base | 8,192 bases | 16 | Sum of two full strand passes after one separate warmup batch per strand |
| Evo 2 7B | 8,192 bases | 8 | Same procedure as Evo 2 1B |
| Evo 2 40B | 8,192 bases | 1 | Four disjoint partitions; sum the two strand intervals from the end of batch one to the end of each strand on each worker |

Marin uses 255 bases plus BOS, the released checkpoint at step 59,158, and the same embedding-enabled harness and batch size in both precisions.
The FP8 configuration uses Transformer Engine delayed E4M3 scaling with a one-entry history and fused MLP and QKV projections.
Each timed Marin pass includes data loading from pretokenized tensors, host/device transfers, model evaluation, output copying and collation.
Padding the final batch repeats the last row; the numerator counts only distinct released variants.
For Evo, tokenization and transfer of the input arrays to the device precede the strand timers.
The Evo loops include the native forward and copying the score bundle back to CPU memory.
All rates exclude container setup, weights loading, initial compilation, final on-disk output validation and upload.

The throughput per worker is `3600 × timed variants / summed worker scoring seconds`.
Divide by H100s per worker for complete variants per H100-hour.
Each 40B worker used two GPUs; four such workers processed the dataset concurrently.
Its denominator sums the active scoring intervals of the four workers, and its numerator excludes one variant from each partition (9,486 timed, 9,490 scored).
The original progress bars report elapsed time to whole seconds, so this rate is approximate.
These are rates for the deployed contexts and software configurations. The models process different numbers of input tokens per variant.

## Reproduce the export

From this directory run the following CPU-only command with Python 3.11 or newer.

```sh
python export_rates.py --output h100_scoring_rates.csv
```

The exporter uses the archived benchmark summaries and the first/final batch timestamps extracted from the 40B logs.
For an independent check, download the timing inputs named in `sources.json` with a CoreWeave storage profile, then regenerate the CSV.

```sh
uv run --script fetch_timings.py --profile coreweave ./timing-cache
python export_rates.py --archive ./timing-cache --output h100_scoring_rates.csv
```

This verifies each input's SHA-256 and byte count, recomputes the 40B timings from all four raw logs, checks both complete strand passes for 1B and 7B, and verifies the four paired-control benchmark summaries, intervals and raw telemetry files.

To reproduce the per-strand telemetry summary after downloading the timing inputs, run the following command.

```sh
python paired_telemetry.py ./timing-cache/paired strand-telemetry.csv
```

For CPU throttle counters the table uses the first and last cgroup samples inside each timed interval. `host_sample_span_seconds` records the covered duration.

`uv run --script check_scores.py CONFIGURATION test.parquet scores.parquet` verifies the score Parquet's row count, release identities and labels, scalar finiteness and strand averages, and the shape, dtype and finiteness of both allele embeddings.
Its script metadata pins the CPU analysis packages.
The durable manifests identify the exact input archives, checkpoint revisions, runner archives, job environments, and checksum-verified score files.
The original preflight also independently checked the two 255-base Marin strand windows against the reference genome for every variant. [Preflight](records/preflight.json)

## Repeat GPU scoring

The checked-in worker files and the archived runner packages describe the original Iris entry points.
The manifest associates each run with its digest-pinned task image, accelerator request, input bundle and worker directory.
For a replication, give the corresponding runner a signed URL for the input object as the Iris secret `FULL_INPUT_URL`, set `FULL_INPUT_SHA256` to the recorded digest, and select a fresh `FULL_OUTPUT_KEY` under the same durable bucket.
The four original single-GPU runs also use `RATE_KIND` (`bf16`, `fp8`, `evo1b`, or `evo7b`).
Use `bash run.sh` from that runner's workspace.
On success each original task wrote `complete.json` after reading its outputs back and verifying their checksums.
The paired control uses `RATE_KIND=paired` and the same input archive as the isolated Marin runs.
The exact runner archive is stored at `paired_control.runner.uri` in the manifest. The directory `workers/paired` also links its common files to the Marin worker.
Its top-level completion receipt follows successful publication of the benchmark, intervals, logs and scores from all four segments.
