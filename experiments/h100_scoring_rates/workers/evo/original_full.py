"""Score four disjoint human test partitions, verify them, and publish the ordered release."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import boto3
from botocore.config import Config
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARTS = 4
N = 9490
KEY = ["chrom", "pos", "ref", "alt"]
META = KEY + ["subset", "label", "match_group"]
TEST_SHA = "658dd3c49b2c171d7c5c973e014dd3d2e8d425e25187eb30513df64d9ef58883"
BUCKET = "marin-us-east-02a"
PARQUET = "evo2_40b_test.parquet"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def release(path):
    if digest(path) != TEST_SHA:
        raise ValueError("Test release checksum mismatch")
    frame = pq.read_table(path)
    meta = frame.select(META).to_pandas()
    if len(meta) != N or meta.duplicated(KEY).any() or meta.isna().any().any():
        raise ValueError("Unexpected variant keys or missing release metadata")
    if set(meta.chrom.astype(str)) != {str(i) for i in range(2,23,2)} | {"Y"} or meta.label.sum() != 949:
        raise ValueError("Unexpected release chromosomes or labels")
    return frame


def partition(frame, index):
    start, end = index*N//PARTS, (index+1)*N//PARTS
    return frame.slice(start, end-start)


def validate(table, expected):
    pd.testing.assert_frame_equal(table.select(META).to_pandas(), expected.select(META).to_pandas(),
                                  check_dtype=False, check_categorical=False)
    for base in ("llr","minus_llr","abs_llr","next_token_jsd_mean"):
        for suffix in ("","_fwd","_rev"):
            if not np.isfinite(table[base+suffix].to_numpy()).all():
                raise ValueError(f"Nonfinite {base+suffix}")
    for field in ("emb_ref", "emb_alt"):
        for chunk in table[field].chunks:
            if chunk.type.value_type != pa.float32() or chunk.null_count or not np.all(chunk.value_lengths().to_numpy()==8192):
                raise ValueError(f"Invalid {field} shape or dtype")
            if not np.isfinite(chunk.flatten().to_numpy()).all():
                raise ValueError(f"Nonfinite {field}")


def store(endpoint="http://cwlota.com"):
    return boto3.client("s3",region_name="US-EAST-02A",endpoint_url=endpoint,
                        config=Config(s3={"addressing_style":"virtual"},retries={"max_attempts":5}))


def verify_object(client, key, size, checksum):
    for attempt in range(5):
        try:
            with client.get_object(Bucket=BUCKET, Key=key)["Body"] as body:
                hashed = hashlib.sha256()
                received = 0
                while block := body.read(8*1024**2):
                    hashed.update(block)
                    received += len(block)
            if (received, hashed.hexdigest()) == (size, checksum):
                return
            raise ValueError(f"Read-back mismatch for {key}: {received} bytes, sha256={hashed.hexdigest()}")
        except Exception:
            if attempt == 4:
                raise
            print(f"Retrying read-back for {key} (attempt {attempt+2})", flush=True)
            time.sleep(2**attempt)
            if attempt >= 1:
                client = store("https://cwobject.com")


def publish(path, name):
    client = store()
    key = os.environ["FULL_OUTPUT_KEY"]+"/"+name
    checksum = digest(path)
    client.upload_file(str(path),BUCKET,key,ExtraArgs={"Metadata":{"sha256":checksum}})
    verify_object(client, key, path.stat().st_size, checksum)
    return {"file":name,"sha256":checksum,"bytes":path.stat().st_size}


def score(index, source):
    folder = Path(f"partition-{index}")
    folder.mkdir()
    expected = partition(source,index)
    pq.write_table(expected,folder/"test.parquet")
    output = Path(f"scores-{index}.parquet")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES":f"{2*index},{2*index+1}", "FULL_PARTITION":str(index)}
    env.pop("LOCAL_RANK",None)
    env.pop("RANK",None)
    args = [sys.executable,"scripts/evo2_eval/eval_matched_pair.py","--model","evo2_40b",
            "--dataset-name","mendelian_traits","--dataset-hf-path",str(folder.resolve()),"--split","test",
            "--genome-path","genome.fa.gz","--window-size","8192","--batch-size","1","--return-embeddings",
            "--emb-layer","norm","--emb-dtype","float32","--skip-metrics","--output",str(output)]
    print(f"Starting partition {index}: {expected.num_rows} test variants on GPU pair {env['CUDA_VISIBLE_DEVICES']}",flush=True)
    log = Path(f"partition-{index}.log")
    with log.open("w") as stream:
        result = subprocess.run(args,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(errors="replace")[-3000:],flush=True)
        raise RuntimeError(f"Partition {index} exited with {result.returncode}")
    validate(pq.read_table(output),expected)
    receipt = publish(output,"shards/"+output.name)
    publish(log,"logs/"+log.name)
    print(f"Verified and published partition {index}",flush=True)
    return index, receipt


def main():
    frame = release(Path("test.parquet"))
    publish(Path("environment.txt"),"environment.txt")
    receipts = {}
    with ThreadPoolExecutor(PARTS) as pool:
        pending = {pool.submit(score,i,frame) for i in range(PARTS)}
        previous = {}
        while pending:
            completed,pending = wait(pending,timeout=60,return_when=FIRST_COMPLETED)
            for i in range(PARTS):
                path = Path(f"progress-{i}.txt")
                if path.exists():
                    current = path.read_text()
                    if current != previous.get(i):
                        print(f"Partition {i}: {current}",flush=True)
                        previous[i] = current
            for result in completed:
                index,receipt = result.result()
                receipts[index]=receipt
    merged = pa.concat_tables([pq.read_table(f"scores-{i}.parquet") for i in range(PARTS)])
    target = Path(PARQUET)
    pq.write_table(merged,target)
    del merged
    validate(pq.read_table(target),frame)
    manifest = {"iris_task":os.environ["IRIS_TASK_ID"],"rows":N,"dataset_sha256":TEST_SHA,
                "result":publish(target,PARQUET),"partitions":[receipts[i] for i in range(PARTS)]}
    marker = Path("complete.json")
    marker.write_text(json.dumps(manifest,indent=2)+"\n")
    publish(marker,marker.name)
    print(f"Full test release complete: {N} variants, sha256={manifest['result']['sha256']}",flush=True)


if __name__=="__main__":
    main()
