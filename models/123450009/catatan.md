# Catatan model 123450009

## Alasan optimizer
Saya memakai AdamW (lr=1e-3, weight_decay=0.01). Transformer punya banyak
parameter dengan skala gradien yang berbeda-beda (proyeksi patch, atensi,
feed-forward, kepala klasifikasi), sehingga laju belajar adaptif per parameter
pada Adam lebih stabil daripada SGD yang memakai satu laju belajar untuk semua.
AdamW dipilih dibanding Adam biasa karena weight decay dipisahkan dari
pembaruan gradien, sehingga regularisasinya bekerja sebagaimana mestinya
dan tidak ikut diskalakan oleh statistik adaptif. Ini pilihan awal; pilihan
akhir akan dibandingkan dengan SGD dan Adam pada data meja.

## Regularisasi
- Dropout 0,1 pada setiap lapis encoder.
- Weight decay 0,01 melalui AdamW.
- Normalisasi per kanal dengan statistik dari data latih saja.
Data meja relatif kecil, sehingga model rawan overfitting. Pengaruh dropout
dan weight decay akan diuji di ablation.

## Pencarian hyperparameter (tabel)
| Konfigurasi | lr | batch | dropout | epoch | d_model | patch | lapis | catatan |
|---|---|---|---|---|---|---|---|---|
| awal | 1e-3 | 64 | 0,1 | 8 | 64 | 20 | 2 | belum diuji di data meja |

## Ablation (>= 2)
Belum dikerjakan. Rencana: (1) panjang patch 10 / 20 / 40, (2) jumlah lapis
1 / 2 / 4, (3) token CLS dibanding rata-rata patch.

## Hasil RF (pembanding non-DL, #8)

**Baseline RF (seed 1, data primer meja, 16 Nov 2026):**
- val_acc = 0.241
- val_macro_f1 = 0.222

Hasil ini mendekati tebakan acak (0.25 untuk 4 kelas). Dari pemeriksaan RMS getaran
sebelumnya, amplitudo '0 mur' sampai '3 mur' pada kecepatan kipas yang sama ternyata
hampir sama — jadi fitur statistik dasar (RMS, puncak, kurtosis) kemungkinan tidak
cukup membedakan kelas beban. Sinyal pembeda mungkin ada di domain frekuensi
(energi pita FFT), yang sudah dimasukkan sebagai fitur tapi belum dominan.

## Hasil 3 seed konfigurasi akhir
Belum ada. Menunggu data meja dan pelatihan (seed 1, 2, 3).