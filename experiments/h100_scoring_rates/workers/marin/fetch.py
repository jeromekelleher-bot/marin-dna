import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

def checksum(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def unpack():
    secret = Path(os.getenv("RIGGING_SECRETS_DIR","/run/rigging/secrets")) / "FULL_INPUT_URL"
    bundle = Path("inputs.tar")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(secret.read_text().strip(), timeout=90) as src, bundle.open("wb") as dst:
                shutil.copyfileobj(src, dst, 4*1024**2)
            assert checksum(bundle) == os.environ["FULL_INPUT_SHA256"]
            break
        except (OSError, AssertionError):
            if attempt == 3:
                raise RuntimeError("Input bundle download or verification failed") from None
            time.sleep(2**attempt)
    with tarfile.open(bundle) as tar:
        tar.extractall(filter="data")
    bundle.unlink()

def weights(kind):
    items = json.loads(Path(__file__).with_name("models.json").read_text())
    repo, rev, name, total, expected = items[kind if kind.startswith("evo") else "marin"]
    dest = Path(name if kind.startswith("evo") else "ckpt/"+name)
    dest.parent.mkdir(exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
    chunk_size = 64 * 1024**2
    with dest.open("wb") as out:
        out.truncate(total)
    def read_chunk(start):
        end = min(total, start+chunk_size)-1
        for attempt in range(5):
            try:
                req = urllib.request.Request(f"{url}?download=true&range={start}-{end}&attempt={attempt}", headers={"Range":f"bytes={start}-{end}"})
                with urllib.request.urlopen(req,timeout=90) as data, dest.open("r+b") as out:
                    assert data.status == 206 and data.headers["Content-Range"] == f"bytes {start}-{end}/{total}"
                    out.seek(start)
                    shutil.copyfileobj(data, out, 2*1024**2)
                    assert out.tell() == end+1
                return
            except (AssertionError,OSError):
                if attempt == 4:
                    raise RuntimeError(f"Could not fetch checkpoint range {start}-{end}") from None
                time.sleep(2**attempt)
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        list(executor.map(read_chunk, range(0,total,chunk_size)))
    assert checksum(dest) == expected
    print(f"Verified {repo}@{rev}",flush=True)

if __name__ == "__main__":
    unpack()
    weights(os.environ["RATE_KIND"])
