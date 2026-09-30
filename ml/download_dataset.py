"""Download a source dataset into datasets/raw/<name>/ with resume and checksum verification.

Usage:
    python ml/download_dataset.py
    python ml/download_dataset.py --source ml/configs/sources/garbage_classification_3.yaml --force
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from utils.sources import DownloadFile, SourceConfig, load_source

DEFAULT_SOURCE = "ml/configs/sources/garbage_classification_3.yaml"
CHUNK = 1 << 20
RETRIES = 5


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, part: Path) -> None:
    """Download url into `part`, resuming from its current size if possible."""
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "wastewise-ai/1.0"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if offset and response.status != 206:  # server ignored Range: start over
            offset = 0
        total = offset + int(response.headers.get("Content-Length", 0))
        mode = "ab" if offset else "wb"
        done, next_report = offset, 0.0
        with part.open(mode) as fh:
            while chunk := response.read(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if total and done / total >= next_report:
                    print(f"    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", flush=True)
                    next_report += 0.25


def download_file(source: SourceConfig, item: DownloadFile, force: bool) -> None:
    dest = source.raw_dir / item.path
    if dest.exists() and not force:
        if item.sha256 is None or sha256_of(dest) == item.sha256:
            print(f"  ok       {item.path} (already present)")
            return
        print(f"  checksum mismatch for existing {item.path}; downloading again")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if force:
        part.unlink(missing_ok=True)

    url = f"{source.base_url}/{item.path}"
    print(f"  download {item.path}")
    for attempt in range(1, RETRIES + 1):
        try:
            _fetch(url, part)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == RETRIES:
                raise RuntimeError(f"Failed to download {url} after {RETRIES} attempts") from exc
            print(f"    attempt {attempt} failed ({exc}); resuming in {2 * attempt}s")
            time.sleep(2 * attempt)

    if item.sha256 is not None:
        actual = sha256_of(part)
        if actual != item.sha256:
            part.unlink()
            raise RuntimeError(f"{item.path}: SHA-256 mismatch (expected {item.sha256}, got {actual})")
    part.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="source config YAML")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    source = load_source(args.source)
    if not source.files:
        print(f"{source.name}: no download section; place files in {source.raw_dir} manually.")
        return 1

    print(f"Downloading '{source.name}' into {source.raw_dir}")
    for item in source.files:
        download_file(source, item, args.force)

    print(f"\nDone. License: {source.license}\nAttribution: {source.attribution}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
