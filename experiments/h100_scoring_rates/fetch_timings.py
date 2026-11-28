# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["boto3==1.44.1"]
# ///
"""Download only the original timing inputs (not scores or weights) from CoreWeave."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import boto3
from botocore.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--profile", help="AWS-compatible CoreWeave credentials profile")
    args = parser.parse_args()
    session = boto3.Session(profile_name=args.profile)
    store = session.client("s3", endpoint_url="https://cwobject.com", region_name="US-EAST-02A",
                           config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
    manifest = json.loads(Path(__file__).with_name("sources.json").read_text())
    for kind, run in manifest["runs"].items():
        for name, spec in run["timing_inputs"].items():
            dest = args.cache / kind / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            uri = urlsplit(spec["uri"])
            with TemporaryDirectory(dir=dest.parent) as tmp:
                path = Path(tmp) / "download"
                with store.get_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"))["Body"] as src, path.open("wb") as out:
                    while block := src.read(1024*1024):
                        out.write(block)
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if (path.stat().st_size, digest) != (spec["bytes"], spec["sha256"]):
                    raise ValueError(f"Timing input checksum mismatch at {spec['uri']}")
                path.replace(dest)


if __name__ == "__main__":
    main()
