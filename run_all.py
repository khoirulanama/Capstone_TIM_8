"""Satu perintah reproduksi tim.

    python run_all.py [--seeds 1 2 3] [--final] [--hanya <nim> ...] [--tanpa-latih]

Menjalankan baseline dosen dan ``models/<nim>/run.py`` setiap anggota untuk tiap seed, lalu
merakit ``reports/hasil_komparasi.md`` dari ``results/*/metrics.json`` (hanya seed terakhir
yang tersimpan di metrics.json; sebaran antar-seed dicatat dari keluaran tiap run).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.common import manifest, split

AKAR = Path(__file__).resolve().parent


def jalankan(perintah: list[str]) -> None:
    print("$", " ".join(perintah), flush=True)
    subprocess.run(perintah, check=True, cwd=AKAR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--hanya", nargs="*", default=None, help="NIM tertentu saja")
    ap.add_argument("--tanpa-baseline", action="store_true")
    ap.add_argument("--tanpa-latih", action="store_true",
                    help="rakit tabel dari metrics.json yang sudah ada tanpa melatih ulang")
    a = ap.parse_args()

    if not a.tanpa_baseline and not a.tanpa_latih:
        jalankan([sys.executable, "-m", "src.common.baseline", "--seed", str(a.seeds[0])])
    rf_ringkas: list[dict] = []
    rf_script = AKAR / "models" / "122450039" / "non_dl.py"
    if rf_script.exists() and not a.tanpa_latih:
        for s in a.seeds:
            jalankan([sys.executable, str(rf_script), "--seed", str(s)])
            rf_ringkas.append(json.loads((AKAR / "results" / "non_dl" / "metrics.json").read_text("utf-8")))
    elif (AKAR / "results" / "non_dl" / "metrics.json").exists():
        rf_ringkas.append(json.loads((AKAR / "results" / "non_dl" / "metrics.json").read_text("utf-8")))
    anggota = sorted(p for p in (AKAR / "models").iterdir() if p.is_dir() and (p / "run.py").exists()
                     and p.name != "TEMPLATE" and (a.hanya is None or p.name in a.hanya))
    ringkas: dict[str, list[dict]] = {}
    for m in anggota:
        if not a.tanpa_latih:
            for s in a.seeds:
                perintah = [sys.executable, str(m / "run.py"), "--seed", str(s)]
                if a.final and s == a.seeds[-1]:
                    perintah.append("--final")
                jalankan(perintah)
                hasil = json.loads((AKAR / "results" / m.name / "metrics.json").read_text("utf-8"))
                ringkas.setdefault(m.name, []).append(hasil)
        else:
            hasil_path = AKAR / "results" / m.name / "metrics.json"
            if hasil_path.exists():
                ringkas[m.name] = [json.loads(hasil_path.read_text("utf-8"))]

    bl = AKAR / "results" / "baseline" / "metrics.json"
    baseline_hasil = None
    if bl.exists():
        baseline_hasil = json.loads(bl.read_text("utf-8"))

    semua_hasil = ([baseline_hasil] if baseline_hasil else []) + rf_ringkas
    semua_hasil += [h for daftar in ringkas.values() for h in daftar]
    expected_hash = manifest.data_hash(f"split:{split.SEED_SPLIT}".encode())
    hashes = {h.get("data_hash") for h in semua_hasil}
    if hashes != {expected_hash}:
        raise SystemExit(
            "data_hash tidak seragam / tidak cocok dengan data saat ini: "
            f"expected={expected_hash}, ditemukan={sorted(str(h) for h in hashes)}. "
            "Jalankan ulang semua komponen pada environment dan data yang sama."
        )
    print(f"data_hash bersama ({len(semua_hasil)} hasil): {expected_hash}")

    baris = ["| NIM | Model | seed | val_acc | val_macro_f1 | test_acc | data_hash | durasi (s) | perangkat |",
             "|---|---|---|---|---|---|---|---|---|"]
    if baseline_hasil:
        h = baseline_hasil; mt = h["metrik"]
        baris.append(f"| baseline | softmax NumPy | {h['seed']} | {mt['val_acc']:.4f} | {mt['val_macro_f1']:.4f} | — | `{h['data_hash']}` | {h.get('durasi_detik', 0):.0f} | {h.get('perangkat', '')} |")
    for h in rf_ringkas:
        mt = h["metrik"]
        baris.append(f"| non_dl | random_forest | {h['seed']} | {mt['val_acc']:.4f} | {mt['val_macro_f1']:.4f} | — | `{h['data_hash']}` | {h.get('durasi_detik', 0):.0f} | {h.get('perangkat', '')} |")
    for nim, daftar in ringkas.items():
        for h in daftar:
            mt = h["metrik"]
            test_acc = f"{mt['test_acc']:.4f}" if "test_acc" in mt else "—"
            baris.append(f"| {nim} | {h.get('model', '')} | {h['seed']} | {mt['val_acc']:.4f} | {mt['val_macro_f1']:.4f} | "
                         f"{test_acc} | `{h['data_hash']}` | {h.get('durasi_detik', 0):.0f} | {h.get('perangkat', '')} |")
    (AKAR / "reports").mkdir(exist_ok=True)
    out = AKAR / "reports" / "hasil_komparasi.md"
    out.write_text("# Hasil komparasi (dibangkitkan run_all.py)\n\n"
                   f"`data_hash` bersama: `{expected_hash}`\n\n" + "\n".join(baris) + "\n", encoding="utf-8")
    print("tabel komparasi →", out.relative_to(AKAR))


if __name__ == "__main__":
    main()
