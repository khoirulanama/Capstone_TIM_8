"""Pemeriksaan mutu data meja pada hari yang sama (tugas #5; tim-08 bag. 5 + metode umum bag. 8).

    python -m src.protokol.periksa_mutu --sesi data/sesi-2026-11-06-0800 --kondisi 96 --tulis

Pemeriksaan (LULUS / PERINGATAN / GAGAL):
  1. Jumlah kondisi = rencana final (96 bila ulangan 2; 144 bila rencana asli) dan strukturnya
     12 blok x (kondisi per blok), 3 blok per taraf mur.
  2. Laju sampel IMU kipas per kondisi (tanda masalah: < 350 sampel/detik; seharusnya ~400).
  3. Amplitudo '0 mur' pada ketiga blok serupa (per kecepatan kipas); bila beda jauh, sensor mungkin mengendur.
  4. Kesesuaian label: RMS getaran seharusnya naik bersama jumlah mur (per kecepatan); gambar disimpan.
  5. Gangguan lembar sesi: nomor kondisi di ``--keluarkan`` (atau MANIFEST) dicocokkan dengan label.csv.

Ambang 0,67 (rasio RMS '0 mur' antarblok) dan monotonik RMS adalah PERKIRAAN saya, bukan angka
dari pembimbing; sesuaikan bila data asli menunjukkan sebaran wajar yang lebih lebar.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import manifest
from . import jendela

AMBANG_FS = 350            # sampel/detik (tim-08 bag. 5 / metode umum bag. 8)
AMBANG_RASIO_0MUR = 0.67   # min/maks RMS '0 mur' antarblok (perkiraan)


def _rms_ac(seg: np.ndarray) -> float:
    """RMS komponen bolak-balik tiga sumbu akselerasi (gravitasi/offset dibuang)."""
    a = seg[:, :3] - seg[:, :3].mean(axis=0)
    return float(np.sqrt((a ** 2).sum(axis=1).mean()))


def periksa(sesi: Path, kondisi_harapan: int, keluarkan: tuple[int, ...], gambar: Path | None) -> tuple[list[tuple[str, str, str]], pd.DataFrame]:
    hasil: list[tuple[str, str, str]] = []
    label = jendela.baca_label(sesi / "label.csv")
    t, v = jendela.baca_imu(sesi / "imu.csv")

    # 1. jumlah kondisi + struktur
    per_blok = kondisi_harapan // 12 if kondisi_harapan % 12 == 0 else None
    ok_n = len(label) == kondisi_harapan
    hasil.append(("Jumlah kondisi", "LULUS" if ok_n else "GAGAL", f"{len(label)} (rencana {kondisi_harapan})"))
    masalah = jendela.periksa_struktur(label, per_blok)
    hasil.append(("Struktur blok", "LULUS" if not masalah else "GAGAL", "12 blok, 3 per taraf" if not masalah else "; ".join(masalah)))

    # 2 + 3 + 4: per kondisi
    baris = []
    for r in label.itertuples():
        i0, i1 = np.searchsorted(t, r.t_mulai), np.searchsorted(t, r.t_selesai)
        n = i1 - i0
        dur = r.t_selesai - r.t_mulai
        seg = v[i0:i1][int(jendela.FS * 1.0):]          # buang 1 detik pertama
        baris.append({"no": r.no, "blok": r.blok, "ulangan": r.ulangan, "kipas": r.kipas, "beban": r.beban,
                      "fs": n / dur if dur > 0 else 0.0, "rms": _rms_ac(seg) if len(seg) > 10 else np.nan})
    df = pd.DataFrame(baris)
    rendah = df[df["fs"] < AMBANG_FS]
    hasil.append(("Laju sampel IMU", "LULUS" if rendah.empty else "GAGAL",
                  f"median {df['fs'].median():.0f}/s" if rendah.empty else
                  f"{len(rendah)} kondisi < {AMBANG_FS}/s: nomor {rendah['no'].tolist()[:15]}"))

    nol = df[df["beban"] == 0].groupby(["kipas", "ulangan"])["rms"].median().unstack()
    rasio = (nol.min(axis=1) / nol.max(axis=1))
    buruk = rasio[rasio < AMBANG_RASIO_0MUR]
    hasil.append(("Amplitudo '0 mur' antarblok", "LULUS" if buruk.empty else "PERINGATAN",
                  f"rasio min/maks terendah {rasio.min():.2f}" + ("" if buruk.empty else f"; kecepatan {buruk.index.tolist()} berbeda jauh, sensor mungkin mengendur")))

    med = df.groupby(["kipas", "beban"])["rms"].median().unstack()
    tak_naik = [int(k) for k, baris_ in med.iterrows() if not np.all(np.diff(baris_.to_numpy()) > 0)]
    hasil.append(("Kesesuaian label (RMS naik bersama mur)", "LULUS" if not tak_naik else "PERINGATAN",
                  "RMS naik untuk semua kecepatan" if not tak_naik else f"tidak naik untuk kecepatan {tak_naik} (periksa label/sensor; pada kipas pelan beda 0 vs 1 mur memang kecil)"))
    if gambar is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for k, baris_ in med.iterrows():
            ax.plot(baris_.index, baris_.values, marker="o", label=f"kipas {k}")
        ax.set_xlabel("jumlah mur"); ax.set_ylabel("RMS AC akselerasi (g)"); ax.legend(); ax.set_title("Kesesuaian label")
        fig.tight_layout(); fig.savefig(gambar, dpi=130); plt.close(fig)

    # 5. gangguan lembar sesi
    asing = [k for k in keluarkan if k not in set(label["no"])]
    hasil.append(("Gangguan lembar sesi", "LULUS" if not asing else "GAGAL",
                  f"{len(keluarkan)} kondisi dikeluarkan" if not asing else f"nomor tidak ada di label.csv: {asing}"))
    return hasil, df


def tulis_catatan(path: Path, sesi: Path, hasil, keluarkan) -> None:
    baris = ["# Catatan sesi (data/CATATAN-SESI.md)", "",
             f"- Sesi yang dipakai: `{sesi.name}`",
             f"- Kondisi dikeluarkan: {', '.join(map(str, keluarkan)) or 'tidak ada'} — alasan: **(isi dari lembar sesi)**",
             "- Kejanggalan lain: **(isi tangan)**", "", "## Hasil pemeriksaan otomatis", "",
             "| Pemeriksaan | Status | Rincian |", "|---|---|---|"]
    baris += [f"| {a} | {b} | {c} |" for a, b, c in hasil]
    path.write_text("\n".join(baris) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sesi", required=True)
    ap.add_argument("--kondisi", type=int, default=96, help="jumlah kondisi rencana final (96 = ulangan 2; 144 = rencana asli)")
    ap.add_argument("--keluarkan", type=int, nargs="*", default=None)
    ap.add_argument("--tulis", action="store_true", help="tulis data/CATATAN-SESI.md dan data/qc_label.png")
    a = ap.parse_args()
    sesi = Path(a.sesi) if Path(a.sesi).is_absolute() else manifest.AKAR / a.sesi
    keluarkan = tuple(a.keluarkan if a.keluarkan is not None else manifest.baca().get("keluarkan", []))
    gambar = (manifest.AKAR / "data" / "qc_label.png") if a.tulis else None
    hasil, _ = periksa(sesi, a.kondisi, keluarkan, gambar)
    for nama, status, rinci in hasil:
        print(f"[{status:10s}] {nama}: {rinci}")
    if a.tulis:
        tulis_catatan(manifest.AKAR / "data" / "CATATAN-SESI.md", sesi, hasil, keluarkan)
        print("ditulis: data/CATATAN-SESI.md, data/qc_label.png")
    raise SystemExit(1 if any(s == "GAGAL" for _, s, _ in hasil) else 0)


if __name__ == "__main__":
    main()
