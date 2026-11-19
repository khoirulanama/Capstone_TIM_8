# Catatan Model RNN/GRU (NIM: 122450041)

Jalur: **RNN (GRU dua arah)**. Masukan protokol terbaru: IMU 6 kanal
(ax, ay, az, gx, gy, gz), jendela 1 detik, (N, 6, 400) → diubah menjadi urutan
(N, 400, 6). Normalisasi per kanal sudah dikerjakan `data.muat()` dari data latih saja.

## Alasan optimizer

- **Optimizer**: AdamW (`torch.optim.AdamW`), `lr=1e-3`, `weight_decay=1e-4`.
- **Alasan**: AdamW mengadaptasi laju pembelajaran per parameter dan memisahkan
  *weight decay* dari gradien (decoupled), sehingga regularisasi L2 tidak ikut
  diskalakan oleh momen adaptif. Pada GRU dengan 6 kanal masukan dan ~1.400 jendela
  latih, AdamW konvergen stabil tanpa tuning laju pembelajaran per layer.
- **Penjadwal**: `ReduceLROnPlateau(mode=max, factor=0.5, patience=2)` memantau
  macro-F1 validasi; laju diturunkan bila tidak membaik.
- **Gradient clipping**: norma 5.0 untuk mencegah ledakan gradien khas RNN.

## Regularisasi

1. **Dropout 0.25** di antara lapisan GRU (karena `num_layers=2`) dan sebelum
   lapisan linier keluaran.
2. **Weight decay 1e-4** (AdamW) pada seluruh parameter.
3. **Early stopping** patience 8 epoch berdasarkan macro-F1 validasi; bobot terbaik
   dikembalikan, bukan bobot epoch terakhir.
4. **Augmentasi data latih**: pergeseran fase acak ≤ 20 sampel (50 ms, sesuai sifat
   periodik getaran) dan noise Gaussian σ=0.01.
5. **Pembobotan kelas** pada cross-entropy (bobot berbanding terbalik dengan
   frekuensi kelas latih) agar kelas minoritas tidak diabaikan.

## Arsitektur

`GRUKipas`: GRU `input_size=6`, `hidden=64`, `num_layers=2`, `bidirectional=True`,
`batch_first=True`; penggabungan *mean pooling* + *max pooling* atas sumbu waktu,
`LayerNorm`, dropout 0.25, lalu `Linear(256 → 4)`.

## Pencarian hyperparameter (tabel)

Sprint 1 memakai konfigurasi awal (belum disetel) agar jalur hidup pada data primer.
Penyetelan pada validasi lipatan 0 dijadwalkan Sprint 2.

| Konfigurasi | hidden | layers | dropout | lr | batch | Val macro-F1 | Catatan |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 (dipakai Sprint 1) | 64 | 2 | 0.25 | 1e-3 | 64 | 0.1990 | seed 1, data primer |
| A2 | — | — | — | — | — | — | Sprint 2 |
| A3 | — | — | — | — | — | — | Sprint 2 |

## Ablation (>= 2)

Dijadwalkan Sprint 2, pada lipatan dan anggaran yang sama:

1. Tanpa augmentasi (pergeseran fase + noise) vs dengan augmentasi.
2. Tanpa pembobotan kelas vs dengan pembobotan kelas.
3. (opsional) GRU satu arah vs dua arah.

## Hasil 3 seed konfigurasi akhir

Dijadwalkan Sprint 2. Sprint 1 baru menjalankan **satu seed** sesuai DoD Demo Sprint 1.

## Uji kerangka pada data simulasi (tugas #13, 19 Nov 2026)

Tujuan: memastikan `run.py` memanggil `data.muat()`, menulis hasil lewat
`src.common.evaluate`, dan bentuk masukan/keluaran sesuai jenis tugas sebelum data
meja dipakai. Data simulasi dibangkitkan dengan
`python -m src.protokol.simulasi --keluar data/sesi-simulasi --tulis-manifest`
(96 kondisi, 12 detik, bukan data penelitian).

Perintah dan hasil (`python models/122450041/run.py --seed 1`, log lengkap di
`sprint/sprint-1/bukti-simulasi-122450041.log`):

| Butir | Nilai |
|---|---|
| Bentuk masukan `data.muat()` | train `(672, 400, 6)`, val `(352, 400, 6)`, test `(352, 400, 6)` |
| Bentuk urutan untuk GRU | `(N, 400, 6)` — 400 langkah waktu, 6 kanal IMU |
| Bentuk keluaran | 4 logits (klasifikasi 4 kelas: 0–3 mur) |
| `val_acc` / `val_macro_f1` | 1.0000 / 1.0000 |
| Durasi | 513,4 detik (CPU laptop) |
| `data_hash` simulasi | `51dbc6994ddc5b601f28d5bf8320d4d8` |

> Angka simulasi **bukan** hasil penelitian dan tidak boleh dibandingkan dengan metrik
> data primer. Setelah uji ini, `data/MANIFEST.json` dikembalikan ke sesi asli
> (`e952a9c4…`) dan artefak simulasi tidak menggantikan `results/122450041/metrics.json`.

## Hasil Sprint 1 — data primer meja (seed 1)

Perintah: `python models/122450041/run.py --seed 1` (lipatan 0, perangkat CPU laptop).

| Butir | Nilai |
|---|---|
| Data | IMU 6 kanal, sesi `sesi-20261116-182908` |
| Split | lipatan 0: latih 1.442 / validasi 809 / uji 807 jendela |
| `val_acc` | **0.2460** |
| `val_macro_f1` (metrik utama) | **0.1990** |
| Durasi pelatihan | 832,7 detik (≈ 13,9 menit; anggaran ≤ 20 menit) |
| `data_hash` | `e952a9c409e0cba3252bf3ec1e64c656` (sama dengan `data/MANIFEST.json`) |
| Epoch terbaik | 13 dari 30 (early stop) |

Pembanding non-DL tim (fitur statistik + FFT + Random Forest, seed 1, data yang sama):

| Model | val_acc | val_macro_f1 |
|---|---:|---:|
| RNN/GRU (122450041) | 0.2460 | **0.1990** |
| Random Forest non-DL | 0.1953 | 0.1735 |

Analisis kesalahan (matriks konfusi validasi, baris = label, kolom = prediksi):

```
        pred 0  pred 1  pred 2  pred 3
0 mur     102      47      18      37
1 mur     106      76       2      20
2 mur      56     135       1      11
3 mur      80      90       8      20
```

Model condong memprediksi kelas 0–1 dan hampir tidak pernah memilih kelas 2
(1 dari 203 jendela). Ini konsisten dengan catatan sesi bahwa getaran tidak naik
rapi mengikuti jumlah mur dan variasi antar-pemasangan lebih besar daripada
pengaruh mur. Belum ada tuning/ablation; hasil ini bukan estimasi akhir.
Data uji **belum** dievaluasi.
