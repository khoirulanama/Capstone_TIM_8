"""Tulis data/MANIFEST.json untuk satu folder sesi (tugas #5, metode umum bag. 7.5).

    python -m src.protokol.buat_manifest --sesi data/sesi-2026-11-06-0800 \
        --sumber "penyimpanan tim: <tautan>" --keluarkan 17 18 63

Memuat: nama sesi, sidik SHA-256 label.csv (syarat metode umum), daftar ``berkas`` yang dihitung
``src.common.manifest`` menjadi ``data_hash`` (label.csv + imu.csv), dan ``keluarkan`` = nomor
kondisi yang terganggu menurut lembar sesi. Berkas audio/citra besar TIDAK dimasukkan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.common import manifest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for potongan in iter(lambda: f.read(1 << 20), b""):
            h.update(potongan)
    return h.hexdigest()


def tulis(sesi: Path, sumber: str = "", keluarkan: list[int] | None = None, catatan: str = "") -> dict:
    sesi = sesi if sesi.is_absolute() else manifest.AKAR / sesi
    rel = sesi.relative_to(manifest.AKAR).as_posix()
    label, imu = sesi / "label.csv", sesi / "imu.csv"
    for p in (label, imu):
        if not p.exists():
            raise FileNotFoundError(p)
    m = {
        "sumber": sumber or "(isi: tautan penyimpanan tim / pengelola meja)",
        "sesi": rel,
        "sha256_label": sha256(label),
        "berkas": [f"{rel}/label.csv", f"{rel}/imu.csv"],
        "keluarkan": sorted(keluarkan or []),
        "catatan": catatan or "Isi: tanggal sesi, jumlah kondisi, kondisi yang dikeluarkan dan alasannya (lihat data/CATATAN-SESI.md).",
    }
    manifest.MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("MANIFEST.json ditulis;  sha256 label.csv =", m["sha256_label"])
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sesi", required=True)
    ap.add_argument("--sumber", default="")
    ap.add_argument("--keluarkan", type=int, nargs="*", default=[])
    ap.add_argument("--catatan", default="")
    a = ap.parse_args()
    tulis(Path(a.sesi), a.sumber, a.keluarkan, a.catatan)


if __name__ == "__main__":
    main()
