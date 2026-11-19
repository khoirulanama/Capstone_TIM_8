"""Generator data SIMULASI untuk menguji kode sebelum data meja asli (6 Nov) tiba.

    python -m src.protokol.simulasi --keluar data/sesi-simulasi --tulis-manifest
    python -m src.protokol.simulasi --keluar /tmp/sesi-cacat --cacat      # untuk menguji periksa_mutu

Meniru rencana Tim 8: faktor kipas {100,160,220,255}, beban_kipas {0..3 mur}, ulangan_blok=3 ->
12 blok; di tiap blok kombinasi kipas diulang ``--ulangan`` kali lalu diacak. Bawaan di sini
mengikuti penyesuaian di papan (ulangan 2, durasi 12 s -> 96 kondisi); pakai ``--ulangan 3
--durasi 18`` untuk rencana asli (144 kondisi).

PERINGATAN: format kolom label.csv/imu.csv di sini TEBAKAN berdasarkan dokumen metode (nama kolom
asli dari meja.py belum terlihat). Sinyalnya fisis-sederhana (getaran sinus pada frekuensi putar,
amplitudo ~ massa x f^2) hanya untuk menguji alur; angka F1 dari data ini TIDAK boleh masuk laporan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FS = 400
KIPAS = [100, 160, 220, 255]
BEBAN = [0, 1, 2, 3]


def _kondisi(rng, ulangan_blok: int, ulangan: int):
    blok = [b for b in BEBAN for _ in range(ulangan_blok)]
    rng.shuffle(blok)
    urut = []
    for b in blok:
        k = KIPAS * ulangan
        rng.shuffle(k)
        urut.append((b, k))
    return urut


def _sinyal(rng, pwm: int, mur: int, durasi: float, pasang: float, longgar: float = 1.0) -> np.ndarray:
    """6 kanal (ax,ay,az,gx,gy,gz) selama ``durasi`` detik. ``pasang`` = faktor khas pemasangan."""
    n = int(durasi * FS)
    t = np.arange(n) / FS
    fr = 20 + 45 * (pwm - 100) / 155                       # Hz (kira-kira 1200..3900 rpm)
    tanjak = np.clip(t / 0.8, 0, 1) ** 2                   # detik pertama: kipas masih berakselerasi
    a_dasar = 0.03 * (fr / 65) ** 1.5                      # getaran sisa motor, ada pada 0 mur
    a_mur = 0.30 * mur * (fr / 65) ** 2 / 3                # ketidakseimbangan ~ massa x f^2
    amp = (a_dasar + a_mur) * pasang * longgar
    ph = rng.uniform(0, 2 * np.pi)
    s = np.zeros((n, 6), np.float32)
    for h, rel in ((1, 1.0), (2, 0.3)):                    # putaran dan harmonik ke-2
        s[:, 0] += amp * rel * np.sin(2 * np.pi * h * fr * t + ph) * tanjak
        s[:, 1] += amp * rel * np.cos(2 * np.pi * h * fr * t + ph) * tanjak
        s[:, 2] += 0.3 * amp * rel * np.sin(2 * np.pi * h * fr * t + 2 * ph) * tanjak
    s[:, 3:] = 5 * s[:, :3]
    s += rng.normal(0, 0.02, s.shape).astype(np.float32)   # derau sensor
    s[:, 2] += 1.0                                         # gravitasi
    return s


def buat(keluar: Path, ulangan: int = 2, durasi: float = 12.0, jeda: float = 2.0, ulangan_blok: int = 3,
         seed: int = 1008, cacat: bool = False, dengan_blok: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    keluar.mkdir(parents=True, exist_ok=True)
    t = 1_793_000_000.0                                    # stempel waktu awal sembarang (epoch)
    baris_label, imu = [], []
    no = 0
    blok_cacat = next((i for i, (m, _) in enumerate(_kondisi(np.random.default_rng(seed), ulangan_blok, ulangan)) if m == 0), 0)
    rng = np.random.default_rng(seed)
    for ib, (mur, kipas_urut) in enumerate(_kondisi(rng, ulangan_blok, ulangan)):
        pasang = float(np.exp(rng.normal(0, 0.15)))        # pemasangan ulang mengubah respons ±15 %
        t += rng.uniform(20, 60)                           # waktu operator memasang mur
        for kipas in kipas_urut:
            no += 1
            longgar = 3.0 if (cacat and ib == blok_cacat) else 1.0   # sensor 'mengendur' di blok 0 mur pertama
            fs_eff = 300 if (cacat and no == 20) else FS   # contoh gangguan: sampel hilang
            s = _sinyal(rng, kipas, mur, durasi, pasang, longgar)
            n = int(durasi * fs_eff)
            idx = np.linspace(0, len(s) - 1, n).astype(int)
            ts = t + np.arange(n) / fs_eff
            imu.append(pd.DataFrame(np.column_stack([ts, np.zeros(n), s[idx]]),
                                    columns=["t_host", "imu", "ax", "ay", "az", "gx", "gy", "gz"]))
            w = pd.DataFrame(np.column_stack([ts, np.ones(n), rng.normal(0, 0.01, (n, 6))]),
                             columns=imu[-1].columns)      # sensor dinding wadah A (harus diabaikan)
            imu.append(w)
            row = {"no": no, "kipas": kipas, "beban_kipas": f"{mur} mur", "t_mulai": t, "t_selesai": t + durasi}
            if dengan_blok:
                row = {"no": no, "blok": ib + 1, **{k: v for k, v in row.items() if k != "no"}}
            baris_label.append(row)
            t += durasi + jeda
    label = pd.DataFrame(baris_label)
    imu_df = pd.concat(imu, ignore_index=True)
    imu_df["imu"] = imu_df["imu"].map({0.0: "kipas", 1.0: "dinding"})
    label.to_csv(keluar / "label.csv", index=False, lineterminator="\n")
    imu_df.to_csv(keluar / "imu.csv", index=False, float_format="%.5f", lineterminator="\n")
    return {"kondisi": no, "baris_imu": len(imu_df)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keluar", default="data/sesi-simulasi")
    ap.add_argument("--ulangan", type=int, default=2)
    ap.add_argument("--durasi", type=float, default=12.0)
    ap.add_argument("--jeda", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1008)
    ap.add_argument("--cacat", action="store_true", help="sisipkan gangguan untuk menguji periksa_mutu")
    ap.add_argument("--tanpa-kolom-blok", action="store_true", help="label.csv tanpa kolom blok")
    ap.add_argument("--tulis-manifest", action="store_true")
    a = ap.parse_args()
    info = buat(Path(a.keluar), a.ulangan, a.durasi, a.jeda, seed=a.seed, cacat=a.cacat,
                dengan_blok=not a.tanpa_kolom_blok)
    print(json.dumps(info))
    if a.tulis_manifest:
        from .buat_manifest import tulis
        tulis(Path(a.keluar), sumber="SIMULASI (bukan data meja)")


if __name__ == "__main__":
    main()
