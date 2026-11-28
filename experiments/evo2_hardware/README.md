# Evo 2 40B variant scoring on GB200 and H100

On 28 November 2026 we compared the original Arc Evo 2 40B on one GB200 and on two H100 80 GB GPUs against the archived single-GH200 development scores.
The H100s held one sharded model.
Four of eight strand-averaged likelihood ratios changed sign on GB200.
On the H100 pair, all scalar scores and every pooled embedding element matched the historical archive.
All eight variants were negative examples.
The comparison addresses numerical reproducibility for these hardware and software configurations. [Paper issue](https://github.com/Open-Athena/marin-dna-paper/issues/27) · [Related held-out evaluation](https://github.com/Open-Athena/marin-dna/issues/595)

| Relative to GH200 | Acceptance threshold | GB200 | H100 pair |
|---|---:|---:|---:|
| Maximum absolute change in strand-averaged log-likelihood ratio | ≤ 0.10 | 7.89275 | 0 |
| Pearson correlation | ≥ 0.99 | 0.53817 | 1 |
| Minimum cosine across both allele embeddings | ≥ 0.995 | 0.77800 | 0.9999993 |
| Maximum absolute change in frozen probe probability | ≤ 0.05 | 0.29200 | 0 |

## Rebuild the paper CSV on CPU

From the repository root, with uv 0.11.31, Python 3.13 and the AWS CLI available, run the command below.
The AWS profile reads the historical GH200 scores and fitted classifiers from `oa-bolinas`.
The CoreWeave profile is an AWS-compatible credentials profile for `marin-us-east-02a`.
The public Hugging Face object supplies the variant identities and labels.
Use your own profile names.

```sh
AWS_PROFILE=marin-dna CW_PROFILE=coreweave \
  uv run --locked --script experiments/evo2_hardware/export_scores.py \
  --fetch --cache ~/.cache/marin/evo2-hardware --output evo2_hardware_scores.csv
sha256sum evo2_hardware_scores.csv
```

The exported file has 24 rows and SHA-256 `a961201fdde6a9a39c1310cc776580c011cbd751221255d647cbf3a603202019` in the locked Linux x86-64 analysis environment.
If the five files listed under `export_inputs` are already in the cache with those filenames, omit `--fetch`.
The exporter verifies their sizes and SHA-256 digests before reading scores or loading the fitted classifier.
It aligns variants by chromosome, 1-based position, reference and alternate allele, and checks label, subset and match group against the first eight release rows.
It applies each subset's original frozen classifier to the concatenated reference and alternate-minus-reference embeddings.
It checks its results against the original validation reports, including exact elementwise H100/GH200 agreement.
The two cosine columns reproduce the original float32 calculation; for identical vectors rounding can yield values slightly above or below one.
Spearman correlation of the eight strand-averaged scores was 0.52381 (GB200) and 1 (H100).

## Original run record

| | Blackwell | Hopper replication |
|---|---|---|
| Iris task | [One GB200](https://iris.oa.dev/#/job/%2Fubuntu%2Fevo595-gb200-dev8-20261128-115108) | [Two H100s](https://iris.oa.dev/#/job/%2Fjk%2Fevo595-h100x2-dev8-20261128-121216) |
| Host and runtime | Arm64, NVIDIA PyTorch 25.04 | x86-64, NVIDIA PyTorch 25.04 |
| Scripts and environment from the run | [GB200](runs/gb200/) | [H100](runs/h100/) |

Both runs used Evo 2 0.6.0, Vortex 1.1.0, PyTorch 2.7.0a0 (NVIDIA build `79aa17489c`), CUDA 12.9 and FlashAttention 2.7.3 with FP8 input projections enabled.
Both verified the two parts of checkpoint `arcinstitute/evo2_40b@d529aa57c30771814217ad89baaeaf6e2315c7d7`.
The original scorer ran on 8,192-base Ensembl release 115 GRCh38 windows, batch size one, both alleles and both strands, with mean-pooled final-layer embeddings stored in FP32.
The validation configuration and thresholds were set against the eight-row GH200 archive before comparing either new run.
The input audit independently checked all 16 strand windows and four scalar fields across 16,140 historical variants.
The numerical cause of the GB200 discrepancy was not isolated. [Input audit](runs/gb200/audit.json) · [H100 elementwise check](runs/h100/exact-differences.json) · [Upstream report](https://github.com/ArcInstitute/evo2/issues/157#issuecomment-3226598292)

The files in `runs/` preserve the original worker scripts, submission receipts, package inventories and validation code.
Some original launcher and audit scripts refer to paths on the original workstation.
The CPU export script, its lockfile, and the consolidated manifest were assembled afterward for the paper.
The [manifest](sources.json) identifies the scorer revision, release, reference genome and indexes, historical files, checkpoint parts, per-platform image digests, Iris jobs, and durable object URIs with sizes and hashes.

## Repeat GPU inference

Use the platform-specific image digest and GPU count in the manifest and start from the corresponding `runs/<gpu>/workspace/` directory on an Iris worker.
Each run's `input_bundle` points to a durable, byte-identical copy of the original launch tar with the pinned scorer, genome and indexes, development release, and platform wheelhouse.
The launch receipts also retain the original one-day scratch keys.

1. Supply the selected archive's SHA-256 as `SMOKE_INPUT_SHA256` and a readable URL for the archive through the `SMOKE_INPUT_URL` Iris secret.
2. Set `SMOKE_OUTPUT_KEY` to a fresh key prefix under `marin-us-east-02a` for this replication.
3. Use `bash run.sh` as the task command from the matching worker directory. The worker checks the hardware, verifies the archive, installs its wheels, fetches and verifies the model weights, scores the eight variants, and publishes its Parquet and environment inventory.
4. Compare the result by variant with the archived GH200 reference. The saved validation reports and runtime log locations are in this directory and the manifest.
