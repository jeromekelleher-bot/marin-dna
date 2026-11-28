"""Fetch the two pinned checkpoint objects directly into one file, then verify both."""
import concurrent.futures
import hashlib
import os
import re
import time
import urllib.request
from pathlib import Path

REV = "d529aa57c30771814217ad89baaeaf6e2315c7d7"
PART = 41126745847
DIGESTS = ("3b74fa4e6158d49265e3e270ba8869390d064358f8bf3d2af0b3e1772728f485",
           "bdc4a76e0f23f8295e7061c2f0deff24f723bd916dc4cdc4d9216cac9c2d49d5")
CHUNK = 128 * 1024**2


def read_range(task):
    part, start, size, dest = task
    last = start + size - 1
    for attempt in range(3):
        try:
            url = (f"https://huggingface.co/arcinstitute/evo2_40b/resolve/{REV}/"
                   f"evo2_40b.pt.part{part}?download=true&gb200range={start}-{last}&attempt={attempt}")
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{last}"})
            with urllib.request.urlopen(request,timeout=50) as resp, dest.open("r+b",buffering=0) as out:
                if resp.status != 206 or resp.headers.get("Content-Range") != f"bytes {start}-{last}/{PART}":
                    raise ValueError("checkpoint server returned an unexpected range")
                out.seek(part*PART+start)
                left = size
                while left:
                    chunk = resp.read(min(4*1024**2,left))
                    if not chunk:
                        raise EOFError("short checkpoint response")
                    out.write(chunk)
                    left -= len(chunk)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def main():
    tmp = Path("weights.partial")
    with tmp.open("wb") as f:
        f.truncate(2*PART)
    tasks = [(part,start,min(CHUNK,PART-start),tmp) for part in range(2) for start in range(0,PART,CHUNK)]
    with concurrent.futures.ThreadPoolExecutor(12) as executor:
        for index,_ in enumerate(executor.map(read_range,tasks),1):
            if index % 64 == 0:
                print(f"Checkpoint download: {index} of {len(tasks)} ranges",flush=True)
    with tmp.open("rb") as f:
        for expected in DIGESTS:
            digest = hashlib.sha256()
            left = PART
            while left:
                block=f.read(min(left, 8*1024**2))
                if not block:
                    raise EOFError("incomplete model part")
                digest.update(block)
                left -= len(block)
            if digest.hexdigest() != expected:
                raise ValueError("checkpoint SHA-256 mismatch")
    tmp.rename("evo2_40b.pt")
    print("Both checkpoint SHA-256 digests verified",flush=True)

if __name__ == "__main__":
    main()
