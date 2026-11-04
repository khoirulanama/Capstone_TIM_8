"""Pemuat data kasus tim: Diagnosis Beban Tak Seimbang Kipas dari Getaran.
Protokol Sprint 0 (Tim 8).

Kontrak yang dipakai penilai dan baseline starter:
    TUGAS = "klasifikasi"
    muat() -> (X_train, y_train, X_val, y_val, X_test, y_test)

Spesifikasi Data & Pemrosesan:
- Input: Sinyal getaran kipas 400 Hz, jendela 1 detik (400 sampel per jendela).
- Target: 4 kelas (0, 1, 2, 3 mur pada bilah kipas).
- Pemotongan: 1 detik pertama tiap kondisi dibuang (transien awal).
- Resampling: 16 kHz -> 400 Hz (faktor desimasi = 40).
- Jendela: 1 s (400 sampel) tanpa tumpang tindih (stride 400).
- Pembagian data: Group-based split berdasarkan sesi/blok eksperimen.
  * Sesi 1 & 2 -> Pembagian Train & Validasi menggunakan split.indeks (seed 2026).
  * Sesi 3 -> Test set (disimpan untuk evaluasi final Sprint terakhir).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from src.common import manifest, split

TUGAS = "klasifikasi"


def muat():
    """Memuat sinyal getaran kipas, memotong jendela 1 s (400 Hz), dan membagi train/val/test."""
    m = manifest.baca()
    data_path = manifest.AKAR / m["berkas"][0]
    if not data_path.exists():
        raise FileNotFoundError(f"Berkas dataset {data_path} tidak ditemukan.")

    data = np.load(data_path)
    sinyal = data["sinyal"]       # shape: [N_kondisi, N_sampel]
    beban = data["beban"]         # shape: [N_kondisi] (0, 1, 2, 3 mur)
    sesi = data["sesi"]           # shape: [N_kondisi] (1, 2, 3)
    sr_asli = int(data["sr"])     # 16000

    sr_target = 400
    faktor_desimasi = sr_asli // sr_target  # 16000 // 400 = 40
    jendela_titik = sr_target               # 400 sampel per detik

    # 1 detik pertama dibuang (1 * sr_asli sampel)
    awal_potong = sr_asli
    sinyal_bersih = sinyal[:, awal_potong:]  # 17 detik tersisa per kondisi

    # Resampling sederhana dengan decimation rata-rata blok / subsampling
    # Bentuk: [N_kondisi, durasi_detik * 400, 40] -> mean over 40
    n_kondisi, n_sampel = sinyal_bersih.shape
    n_resample = n_sampel // faktor_desimasi
    sinyal_resampled = sinyal_bersih[:, : n_resample * faktor_desimasi].reshape(
        n_kondisi, n_resample, faktor_desimasi
    ).mean(axis=-1)  # shape: [n_kondisi, n_resample] pada 400 Hz

    # Potong menjadi jendela 1 detik (400 sampel)
    n_jendela_per_kondisi = n_resample // jendela_titik  # 17 jendela per rekaman

    windows_all = []
    labels_all = []
    sesi_all = []

    for i in range(n_kondisi):
        for j in range(n_jendela_per_kondisi):
            w = sinyal_resampled[i, j * jendela_titik : (j + 1) * jendela_titik]
            windows_all.append(w)
            labels_all.append(beban[i])
            sesi_all.append(sesi[i])

    X = np.array(windows_all, dtype=np.float32)  # [N_total, 400]
    y = np.array(labels_all, dtype=np.int64)     # [N_total]
    s = np.array(sesi_all, dtype=np.int32)

    # Pisahkan Sesi 3 sebagai Test Set (Group split)
    mask_test = (s == 3)
    mask_trainval = ~mask_test

    X_trainval, y_trainval = X[mask_trainval], y[mask_trainval]
    X_test, y_test = X[mask_test], y[mask_test]

    # Split train/val menggunakan split beku starter (seed 2026, 10% val)
    idx_train, idx_val = split.indeks(len(X_trainval), porsi_val=0.1, seed=split.SEED_SPLIT)

    X_train, y_train = X_trainval[idx_train], y_trainval[idx_train]
    X_val, y_val = X_trainval[idx_val], y_trainval[idx_val]

    return X_train, y_train, X_val, y_val, X_test, y_test
