"""Stage the eight-row development smoke test, submit once, and retrieve its output."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

BASE = Path(__file__).resolve().parent
MARIN = Path("/home/ubuntu/agents-work/marin-community/marin")
IMAGE = "nvcr.io/nvidia/pytorch@sha256:36c950490fed05b87814bd831929ec7dbdd8282d9c21a2f6691836fbeb6054e2"
SCRATCH = "marin-us-east-08a"
DURABLE = "marin-us-east-02a"
RECEIPT = BASE / "run.json"

def client(region, endpoint="https://cwobject.com"):
    return boto3.client("s3", endpoint_url=endpoint, region_name=region,
        aws_access_key_id=os.environ["CW_KEY_ID"], aws_secret_access_key=os.environ["CW_KEY_SECRET"],
        config=Config(signature_version="s3v4",s3={"addressing_style":"virtual"},retries={"max_attempts":5}))

def load_keys():
    p=Path.home()/".config/marin/cw-smoke.env"
    if p.exists():
        if p.stat().st_mode & 0o077:
            raise RuntimeError("Credential file must be private")
        for line in p.read_text().splitlines():
            tag,pair=shlex.split(line)
            name,value=pair.split("=",1)
            if tag!="export" or name not in ("CW_KEY_ID","CW_KEY_SECRET") or not value:
                raise ValueError("Unexpected credential file format")
            os.environ[name]=value

def put_file(s3,bucket,key,path):
    s3.upload_file(str(path),bucket,key)
    assert s3.head_object(Bucket=bucket,Key=key)["ContentLength"]==path.stat().st_size

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["stage","submit","fetch"])
    args=parser.parse_args()
    load_keys()
    if args.action=="stage" and not RECEIPT.exists():
        now=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        name="evo595-h100x2-dev8-"+now
        rec={"job_name":name,"image":IMAGE,"scratch_key":"tmp/ttl=1d/issue-595/"+name+"/inputs.tar",
             "out_key":"marin/experiments/issue-595/"+name,"limit_seconds":1800}
        RECEIPT.write_text(json.dumps(rec,indent=2)+"\n")
    rec=json.loads(RECEIPT.read_text())
    out=client("US-EAST-02A")
    if args.action=="stage":
        put_file(out,DURABLE,rec["out_key"]+"/preflight.json",BASE/"validation/preflight.json")
        put_file(client("US-EAST-08A"),SCRATCH,rec["scratch_key"],BASE/"inputs.tar")
        rec["input_sha256"]=(BASE/"inputs.tar.sha256").read_text().split()[0]
        rec["staged"]=True
        RECEIPT.write_text(json.dumps(rec,indent=2)+"\n")
        print("Verified scratch input and durable preflight upload",flush=True)
    elif args.action=="submit":
        if rec.get("submit_invoked") or not rec.get("staged"):
            raise RuntimeError("Check existing receipt and job status before submitting")
        os.environ["SMOKE_INPUT_URL"]=client("US-EAST-08A","http://cwlota.com").generate_presigned_url(
            "get_object",Params={"Bucket":SCRATCH,"Key":rec["scratch_key"]},ExpiresIn=86400)
        cmd=[str(MARIN/".venv/bin/iris"),"--config",str(MARIN/"lib/iris/config/marin.yaml"),"job","run",
             "--target-cluster","cw-us-east-02a","--enable-extra-resources","--gpu","H100x2","--cpu","12",
             "--memory","256GB","--disk","180GB","--task-image",IMAGE,"--no-sync","--timeout","1800",
             "--max-retries","0","--user","jk","--job-name",rec["job_name"],"--no-wait",
             "--secret","SMOKE_INPUT_URL","env:SMOKE_INPUT_URL","-e","SMOKE_INPUT_SHA256",rec["input_sha256"],
             "-e","SMOKE_OUTPUT_KEY",rec["out_key"],"--","bash","run.sh"]
        rec["submit_invoked"]=True
        RECEIPT.write_text(json.dumps(rec,indent=2)+"\n")
        result=subprocess.run(cmd,cwd=BASE/"workspace",capture_output=True,text=True)
        (BASE/"submission.log").write_text(result.stderr+result.stdout)
        print(result.stderr[-2400:]+result.stdout[-450:],flush=True)
        result.check_returncode()
        rec["job_id"]=result.stdout.strip().splitlines()[-1]
        RECEIPT.write_text(json.dumps(rec,indent=2)+"\n")
    else:
        for file in ("h100-eight.parquet","environment.txt"):
            out.download_file(DURABLE,rec["out_key"]+"/"+file,str(BASE/"validation"/file))

if __name__=="__main__":
    main()
