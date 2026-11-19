"""Pemuat data kasus Tim 8 (diagnosis beban tak seimbang kipas) — MILIK TIM, dibekukan setelah disetujui.

    TUGAS = "klasifikasi"          # 4 kelas: 0, 1, 2, 3 mur
    muat() -> X_train, y_train, X_val, y_val, X_test, y_test    # X: (N, 6, 400) float32

Perbedaan penting dari contoh Fashion-MNIST di starter: split TIDAK diacak per jendela
(``split.indeks``), tetapi per BLOK. Jendela dari satu kondisi sangat mirip; bila dipisah acak,
F1 menjadi tinggi semu (metode umum bag. 9.2, tim-08 bag. 6). Tiap taraf mur dipasang 3 kali
(3 blok); pemasangan ke-k dari semua taraf membentuk satu LIPATAN:

    lipatan 0/1/2 = data uji ; satu lipatan lain = validasi ; satu sisanya = latih.

Urutan peran lipatan ditetapkan sekali oleh seed 2026 (seed split beku starter). Lipatan uji
dipilih lewat argumen ``lipatan`` atau variabel lingkungan ``CAPSTONE_LIPATAN`` (bawaan 0), supaya
``run.py`` model dan ``evaluate`` memakai split yang sama tanpa mengubah kontrak ``muat()``.

Normalisasi: rerata/simpangan baku per kanal dihitung dari data LATIH saja, lalu diterapkan apa
adanya pada validasi dan uji. Jendela latih tumpang-tindih 50 %; validasi/uji tidak.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from src.common import manifest, split
from . import jendela

TUGAS = "klasifikasi"
KELAS = ["0 mur", "1 mur", "2 mur", "3 mur"]
LANGKAH_LATIH = 0.5       # detik (tumpang-tindih 50 %) — hanya untuk data latih
LANGKAH_EVAL = 1.0        # detik (tanpa tumpang-tindih) — validasi dan uji
N_LIPATAN = 3


@lru_cache(maxsize=4)
def _jendela(langkah: float) -> dict:
    m = manifest.baca()
    sesi = m.get("sesi")
    if not sesi:
        raise RuntimeError("data/MANIFEST.json belum memuat kunci 'sesi' (folder sesi). "
                           "Jalankan: python -m src.protokol.buat_manifest --sesi data/<folder-sesi>")
    folder = manifest.AKAR / sesi
    return jendela.potong(folder, langkah=langkah, keluarkan=tuple(m.get("keluarkan", [])))


def peran_lipatan(lipatan: int) -> tuple[int, int, int]:
    """(ulangan_latih, ulangan_val, ulangan_uji) untuk satu lipatan, urutan dari seed split beku."""
    urutan = np.random.default_rng(split.SEED_SPLIT).permutation(N_LIPATAN)
    return (int(urutan[(lipatan + 2) % 3]), int(urutan[(lipatan + 1) % 3]), int(urutan[lipatan]))


def _ambil(d: dict, ulangan: int) -> tuple[np.ndarray, np.ndarray, dict]:
    pilih = d["ulangan"] == ulangan
    return d["X"][pilih], d["y"][pilih], {k: v[pilih] for k, v in d.items() if k not in ("X", "y")}


def muat(lipatan: int | None = None, gabung_val: bool = False, mentah: bool = False):
    """Enam larik baku starter. ``gabung_val=True`` (latih ulang final) menambahkan validasi ke latih.
    ``mentah=True`` melewati normalisasi (mis. untuk fitur statistik + FFT pembanding non-DL)."""
    if lipatan is None:
        lipatan = int(os.environ.get("CAPSTONE_LIPATAN", "0"))
    r_tr, r_va, r_te = peran_lipatan(lipatan)
    Xtr, ytr, _ = _ambil(_jendela(LANGKAH_LATIH), r_tr)
    Xva, yva, _ = _ambil(_jendela(LANGKAH_EVAL), r_va)
    Xte, yte, _ = _ambil(_jendela(LANGKAH_EVAL), r_te)
    if gabung_val:
        Xv_l, yv_l, _ = _ambil(_jendela(LANGKAH_LATIH), r_va)
        Xtr, ytr = np.concatenate([Xtr, Xv_l]), np.concatenate([ytr, yv_l])
    if not mentah:
        mu = Xtr.mean(axis=(0, 2), keepdims=True)                 # per kanal, dari data latih saja
        sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
        Xtr, Xva, Xte = (((x - mu) / sd).astype(np.float32) for x in (Xtr, Xva, Xte))
    return Xtr, ytr, Xva, yva, Xte, yte


def muat_meta(lipatan: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Meta (kondisi, blok, ulangan, kipas) tiap jendela val/uji — untuk analisis kesalahan per kecepatan."""
    if lipatan is None:
        lipatan = int(os.environ.get("CAPSTONE_LIPATAN", "0"))
    _, r_va, r_te = peran_lipatan(lipatan)
    d = _jendela(LANGKAH_EVAL)
    return {"val": _ambil(d, r_va)[2], "test": _ambil(d, r_te)[2]}


def periksa_pemisahan(lipatan: int | None = None) -> dict:
    """Pastikan tidak ada satuan pemisahan (blok, kondisi, pemasangan) yang muncul di dua bagian sekaligus.

    Memeriksa pasangan latih-validasi, latih-uji, validasi-uji pada data yang sama dengan ``muat()``.
    Melempar AssertionError bila ada irisan; mengembalikan ringkasan jumlah blok/kondisi/jendela per bagian.
    """
    if lipatan is None:
        lipatan = int(os.environ.get("CAPSTONE_LIPATAN", "0"))
    r_tr, r_va, r_te = peran_lipatan(lipatan)
    bagian = {"latih": _ambil(_jendela(LANGKAH_LATIH), r_tr)[2],
              "val": _ambil(_jendela(LANGKAH_EVAL), r_va)[2],
              "uji": _ambil(_jendela(LANGKAH_EVAL), r_te)[2]}
    ringkas = {}
    for nama, meta in bagian.items():
        ringkas[nama] = {"blok": len(set(meta["blok"])), "kondisi": len(set(meta["kondisi"])), "jendela": len(meta["blok"])}
    nama_ = list(bagian)
    for i in range(3):
        for j in range(i + 1, 3):
            for kunci in ("blok", "kondisi", "ulangan"):
                irisan = set(bagian[nama_[i]][kunci]) & set(bagian[nama_[j]][kunci])
                assert not irisan, f"lipatan {lipatan}: {kunci} {sorted(irisan)} ada di {nama_[i]} dan {nama_[j]}"
    # tiap bagian harus memuat keempat taraf mur
    for nama in ("latih", "val", "uji"):
        y = {"latih": _ambil(_jendela(LANGKAH_LATIH), r_tr), "val": _ambil(_jendela(LANGKAH_EVAL), r_va),
             "uji": _ambil(_jendela(LANGKAH_EVAL), r_te)}[nama][1]
        assert set(y.tolist()) == {0, 1, 2, 3}, f"{nama}: taraf mur tidak lengkap {sorted(set(y.tolist()))}"
    return ringkas


if __name__ == "__main__":      # python -m src.protokol.data
    for k in range(N_LIPATAN):
        print(f"lipatan {k} (latih/val/uji = ulangan {peran_lipatan(k)}): OK", periksa_pemisahan(k))
