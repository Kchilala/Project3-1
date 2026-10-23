"""
download_data.py - fetch the datasets from Google Drive (they are not stored in git).

Usage
-----
    python download_data.py              # download + unpack all datasets into data/
    python download_data.py har          # only the HAR dataset
    python download_data.py --pack       # (maintainers) zip data/ for uploading to Google Drive

Maintainers: after ``--pack``, upload the zip files from ``dist/`` to Google Drive,
share each file as "Anyone with the link - Viewer", and paste the share links
into ``DATASETS`` below.

Downloads use ``gdown`` (handles Google Drive's large-file confirmation page).
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

DATASETS = {
    "har": {
        "description": "Wi-Fi CSI HAR dataset of Zhuravchak et al. 2022 (Atheros, 5 GHz, 2x2, 3 rooms, 7 activities, ~1.1 GB unpacked)",
        "drive_url": "",          # <- paste Google Drive share link of wifi_csi_har_dataset.zip
        "zip": "wifi_csi_har_dataset.zip",
        "folder": "wifi_csi_har_dataset",
        "local_aliases": ["wifi_csi_har_dataset (3)"],
        "check": os.path.join("room_1", "1", "data.csv"),
        "original_source": "https://doi.org/10.6084/m9.figshare.14386892.v1",
    },
    "esp32": {
        "description": "Own ESP32 recordings: empty room / standing still / moving (2.4 GHz, 64 subcarriers, ~5 MB)",
        "drive_url": "",          # <- paste Google Drive share link of self_extracted_dataset.zip
        "zip": "self_extracted_dataset.zip",
        "folder": "self_extracted_dataset",
        "local_aliases": [],
        "check": "empty_room.csv",
        "original_source": None,
    },
}


def _present(spec) -> str | None:
    for name in [spec["folder"]] + spec["local_aliases"]:
        if os.path.exists(os.path.join(DATA, name, spec["check"])):
            return name
    return None


def _safe_extract(zip_path: str, dest: str):
    """Extract a zip, refusing members that would land outside ``dest``."""
    dest = os.path.realpath(dest)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            target = os.path.realpath(os.path.join(dest, m))
            if not target.startswith(dest + os.sep) and target != dest:
                raise RuntimeError(f"unsafe path in zip: {m}")
        z.extractall(dest)


def download(key: str):
    spec = DATASETS[key]
    have = _present(spec)
    if have:
        print(f"[{key}] already present: data/{have}")
        return
    if not spec["drive_url"]:
        print(f"[{key}] no Google Drive link configured in download_data.py.")
        if spec["original_source"]:
            print(f"       Original source: {spec['original_source']}")
        print(f"       Unpack it so that data/{spec['folder']}/{spec['check']} exists.")
        return
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required: pip install gdown")
    os.makedirs(DATA, exist_ok=True)
    zip_path = os.path.join(DATA, spec["zip"])
    print(f"[{key}] downloading {spec['description']}")
    gdown.download(spec["drive_url"], zip_path, quiet=False, fuzzy=True)
    print(f"[{key}] unpacking to data/{spec['folder']}")
    _safe_extract(zip_path, DATA)
    os.remove(zip_path)
    if not _present(spec):
        sys.exit(f"[{key}] unexpected archive layout: data/{spec['folder']}/{spec['check']} not found")
    print(f"[{key}] done")


def pack(key: str):
    """Zip data/<folder> (or a local alias) as <folder>/... into dist/<zip>."""
    spec = DATASETS[key]
    have = _present(spec)
    if not have:
        print(f"[{key}] nothing to pack (data/{spec['folder']} not found)")
        return
    os.makedirs(os.path.join(HERE, "dist"), exist_ok=True)
    out = os.path.join(HERE, "dist", spec["zip"])
    src = os.path.join(DATA, have)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src):
            for f in files:
                if f == ".DS_Store":
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, src)
                z.write(full, os.path.join(spec["folder"], rel))
    print(f"[{key}] wrote {os.path.relpath(out, HERE)} ({os.path.getsize(out) / 1e6:.0f} MB) - upload it to Google Drive")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help=f"any of {list(DATASETS)} (default: all)")
    ap.add_argument("--pack", action="store_true", help="create zip files in dist/ for uploading")
    a = ap.parse_args()
    unknown = set(a.datasets) - set(DATASETS)
    if unknown:
        ap.error(f"unknown dataset(s) {sorted(unknown)}; choose from {list(DATASETS)}")
    for k in a.datasets or list(DATASETS):
        (pack if a.pack else download)(k)
