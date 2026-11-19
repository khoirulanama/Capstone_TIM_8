"""Pemuat jendela audio Tim 8 dengan split berdasarkan ulangan eksperimen.

Jalankan ``python -m src.protokol.prepare_audio`` sekali untuk menyiapkan data
aktual. ``CAPSTONE_DATA_MODE=simulasi`` memilih generator Sprint 0 secara eksplisit.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.common import manifest

TUGAS = "klasifikasi"
MODE = os.environ.get("CAPSTONE_DATA_MODE", "aktual").strip().lower()


def _bagi_aktual(X: np.ndarray, y: np.ndarray, ulangan: np.ndarray):
    # Tiga ulangan adalah unit independen: 1=train, 2=validation, 3=test.
    # Jendela dari rekaman/ulangan yang sama tidak pernah tersebar antar-split.
    masks = [ulangan == i for i in (1, 2, 3)]
    if not all(mask.any() for mask in masks):
        raise ValueError("Data aktual harus memiliki ulangan_blok 1, 2, dan 3.")
    return tuple(v for mask in masks for v in (X[mask], y[mask]))


def muat():
    """Mengembalikan X_train, y_train, X_val, y_val, X_test, y_test."""
    if MODE == "simulasi":
        manifest.baca()
        path = manifest.AKAR / "data/rekaman_kipas_simulasi.npz"
        if not path.exists():
            raise FileNotFoundError(f"Data simulasi tidak ditemukan: {path}")
        with np.load(path) as arsip:
            sinyal, beban, sesi = arsip["sinyal"], arsip["beban"], arsip["sesi"]
            sr = int(arsip["sr"])
        ratio = sr // 400
        awal = sinyal[:, sr:]
        n = awal.shape[1] // ratio
        x400 = awal[:, :n * ratio].reshape(len(awal), n, ratio).mean(-1)
        jendela = x400.shape[1] // 400
        X = np.stack([x400[i, j*400:(j+1)*400]
                      for i in range(len(x400)) for j in range(jendela)]).astype(np.float32)
        y = np.repeat(beban, jendela).astype(np.int64)
        s = np.repeat(sesi, jendela)
        masks = [s == i for i in (1, 2, 3)]
        if not all(mask.any() for mask in masks):
            raise ValueError("Data simulasi harus memiliki sesi 1, 2, dan 3.")
        return tuple(v for mask in masks for v in (X[mask], y[mask]))
    if MODE != "aktual":
        raise ValueError("CAPSTONE_DATA_MODE harus 'aktual' atau 'simulasi'.")

    m = manifest.baca()
    path = manifest.AKAR / m["berkas"][0]
    if not path.exists():
        raise FileNotFoundError(
            f"Jendela audio aktual belum disiapkan ({path}). Jalankan "
            "python -m src.protokol.prepare_audio --session-dir <folder-sesi>."
        )
    with np.load(path) as arsip:
        X, y, ulangan = arsip["X"], arsip["y"], arsip["ulangan_blok"]
    X, y = X.astype(np.float32, copy=False), y.astype(np.int64, copy=False)
    hasil = _bagi_aktual(X, y, ulangan)
    return hasil
