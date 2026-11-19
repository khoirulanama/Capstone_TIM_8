"""Pemotong jendela Tim 8: label.csv + imu.csv  ->  jendela 1 detik (6 kanal, 400 Hz).

Milik tim (bagian dari protokol yang dibekukan). Aturan yang diterapkan (metode umum bagian 9):
  1. Buang 1 detik pertama setiap kondisi (alat masih berakselerasi).
  2. Hanya baris ``imu == kipas`` (sensor di badan kipas); sensor dinding wadah A diabaikan.
  3. Kondisi yang terganggu (dari lembar sesi) dikeluarkan sebelum dipotong.
  4. Setiap jendela membawa nomor blok dan nomor pemasangan (``ulangan``) agar split dapat
     dilakukan per blok, bukan per jendela.

PERHATIAN: nama kolom label.csv / imu.csv yang sebenarnya belum terlihat (paket meja.py tidak ada
di starter). Pemetaan nama kolom ada di ``KANDIDAT`` di bawah; bila keluaran meja.py memakai nama
lain, cukup tambahkan nama itu ke daftar kandidat, tidak perlu mengubah logika.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

FS = 400                                   # Hz, laju IMU
KANAL = ["ax", "ay", "az", "gx", "gy", "gz"]
JEDA_ANTAR_BLOK = 8.0                      # detik; jeda antar-kondisi sebenarnya 2 detik

KANDIDAT = {
    "no": ["no", "kondisi", "nomor", "idx", "index", "n"],
    "blok": ["blok", "block", "id_blok"],
    "t_mulai": ["t_mulai", "mulai", "t_start", "start", "t_awal", "t0"],
    "t_selesai": ["t_selesai", "selesai", "t_end", "end", "t_akhir", "t1"],
    "kipas": ["kipas", "pwm_kipas", "kipas_pwm"],
    "beban": ["beban_kipas", "beban", "mur"],
    "t_imu": ["t_host", "t", "waktu", "timestamp"],
    "sensor": ["imu", "sensor", "id_imu"],
}


def _kolom(df: pd.DataFrame, kunci: str, wajib: bool = True) -> str | None:
    for nama in KANDIDAT[kunci]:
        if nama in df.columns:
            return nama
    if wajib:
        raise KeyError(f"kolom '{kunci}' tidak ditemukan; kolom yang ada: {list(df.columns)}. "
                       f"Tambahkan nama kolom yang benar ke KANDIDAT['{kunci}'] di jendela.py.")
    return None


def _taraf_mur(nilai) -> int:
    """'2 mur' -> 2 ; 2 -> 2."""
    m = re.search(r"\d+", str(nilai))
    if not m:
        raise ValueError(f"taraf beban_kipas tidak dapat dibaca: {nilai!r}")
    return int(m.group())


def baca_label(path: Path) -> pd.DataFrame:
    """Baca label.csv menjadi tabel baku: no, t_mulai, t_selesai, kipas, beban (0..3), blok, ulangan."""
    df = pd.read_csv(path)
    df = df.sort_values(_kolom(df, "t_mulai")).reset_index(drop=True)   # urut waktu; baris sejajar dengan out
    out = pd.DataFrame({
        "t_mulai": df[_kolom(df, "t_mulai")].astype(float),
        "t_selesai": df[_kolom(df, "t_selesai")].astype(float),
        "kipas": df[_kolom(df, "kipas")].astype(int),
        "beban": df[_kolom(df, "beban")].map(_taraf_mur).astype(int),
    })
    k_no = _kolom(df, "no", wajib=False)
    out["no"] = df[k_no].astype(int) if k_no else np.arange(1, len(df) + 1)
    k_blok = _kolom(df, "blok", wajib=False)
    if k_blok:                                          # label.csv sudah memuat nomor blok
        urut = df.groupby(k_blok)[_kolom(df, "t_mulai")].min().rank(method="first").astype(int) - 1
        out["blok"] = df[k_blok].map(urut).astype(int)  # nomor blok 0.. menurut waktu mulai
    else:                                             # turunkan: blok baru bila taraf berganti / ada jeda panjang
        jeda = out["t_mulai"].to_numpy()[1:] - out["t_selesai"].to_numpy()[:-1]
        ganti = (out["beban"].to_numpy()[1:] != out["beban"].to_numpy()[:-1]) | (jeda > JEDA_ANTAR_BLOK)
        out["blok"] = np.concatenate([[0], np.cumsum(ganti)])
    # ulangan = urutan pemasangan untuk taraf yang sama (0,1,2) menurut waktu
    awal = out.groupby("blok").agg(beban=("beban", "first"), t=("t_mulai", "min")).sort_values("t")
    awal["ulangan"] = awal.groupby("beban").cumcount()
    out["ulangan"] = out["blok"].map(awal["ulangan"])
    return out


def periksa_struktur(label: pd.DataFrame, per_blok: int | None = None, n_blok: int = 12) -> list[str]:
    """Kembalikan daftar masalah struktur (kosong = baik)."""
    masalah = []
    nb = label["blok"].nunique()
    if nb != n_blok:
        masalah.append(f"jumlah blok {nb}, seharusnya {n_blok}")
    ukuran = label.groupby("blok").size()
    if per_blok is not None and (ukuran != per_blok).any():
        masalah.append(f"kondisi per blok {ukuran.to_dict()}, seharusnya {per_blok}")
    if ukuran.nunique() > 1:
        masalah.append(f"kondisi per blok tidak seragam: {ukuran.to_dict()}")
    per_taraf = label.groupby("beban")["blok"].nunique()
    if (per_taraf != 3).any():
        masalah.append(f"blok per taraf {per_taraf.to_dict()}, seharusnya 3 untuk tiap taraf")
    return masalah


def baca_imu(path: Path, sensor: str = "kipas") -> tuple[np.ndarray, np.ndarray]:
    """Waktu (detik) dan matriks (n, 6) untuk sensor di badan kipas, terurut menurut waktu."""
    df = pd.read_csv(path)
    k_s = _kolom(df, "sensor", wajib=False)
    if k_s:
        df = df[df[k_s].astype(str).str.lower() == sensor]
    k_t = _kolom(df, "t_imu")
    df = df.sort_values(k_t)
    if all(k in df.columns for k in KANAL):
        nilai = df[KANAL].to_numpy(np.float32)
    else:                                               # fallback: 6 kolom numerik selain waktu/sensor
        num = [c for c in df.select_dtypes("number").columns if c != k_t]
        if len(num) != 6:
            raise KeyError(f"imu.csv: perlu 6 kanal, ditemukan {num}. Sesuaikan KANAL di jendela.py.")
        nilai = df[num].to_numpy(np.float32)
    return df[k_t].to_numpy(np.float64), nilai


def potong(folder: str | Path, buang: float = 1.0, jendela: float = 1.0, langkah: float = 1.0,
           keluarkan: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
    """Potong satu folder sesi menjadi jendela.

    Mengembalikan dict: X (N, 6, 400) float32, y (N,) taraf mur 0..3, dan meta: kondisi, blok,
    ulangan, kipas (kecepatan). ``langkah`` < ``jendela`` berarti jendela tumpang-tindih (dipakai
    hanya untuk data latih, lihat data.py).
    """
    folder = Path(folder)
    label = baca_label(folder / "label.csv")
    label = label[~label["no"].isin(keluarkan)]
    t, v = baca_imu(folder / "imu.csv")
    w, h = int(round(jendela * FS)), int(round(langkah * FS))
    X, y, kond, blok, ulang, kipas = [], [], [], [], [], []
    for r in label.itertuples():
        # toleransi setengah sampel agar kondisi 12 s tidak kehilangan satu jendela karena pembulatan waktu
        i0 = np.searchsorted(t, r.t_mulai + buang - 0.5 / FS, side="left")
        i1 = np.searchsorted(t, r.t_selesai - 0.5 / FS, side="left")
        seg = v[i0:i1]
        for a in range(0, len(seg) - w + 1, h):         # jendela utuh saja; sisa di ujung dibuang
            X.append(seg[a:a + w].T)
            y.append(r.beban); kond.append(r.no); blok.append(r.blok); ulang.append(r.ulangan); kipas.append(r.kipas)
    if not X:
        raise RuntimeError("tidak ada jendela yang terbentuk; periksa label.csv dan imu.csv")
    return {"X": np.stack(X).astype(np.float32), "y": np.asarray(y, np.int64),
            "kondisi": np.asarray(kond), "blok": np.asarray(blok), "ulangan": np.asarray(ulang),
            "kipas": np.asarray(kipas)}
