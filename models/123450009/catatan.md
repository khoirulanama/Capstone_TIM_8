# Catatan model 123450009

## Alasan optimizer
Saya memakai AdamW (lr=1e-3, weight_decay=0.01). Transformer punya banyak
parameter dengan skala gradien yang berbeda-beda (proyeksi patch, atensi,
feed-forward, kepala klasifikasi), sehingga laju belajar adaptif per parameter
pada Adam lebih stabil daripada SGD yang memakai satu laju belajar untuk semua.
AdamW dipilih dibanding Adam biasa karena weight decay dipisahkan dari
pembaruan gradien, sehingga regularisasinya bekerja sebagaimana mestinya
dan tidak ikut diskalakan oleh statistik adaptif.

## Regularisasi
- Dropout 0,1 pada setiap lapis encoder.
- Weight decay 0,01 melalui AdamW.
Keduanya dipilih karena data meja relatif kecil (sekitar 260 jendela per kelas
setelah jumlah ulangan dikurangi), sehingga model rawan overfitting.

## Pencarian hyperparameter (tabel)
(diisi setelah data meja tersedia)

## Ablation (>= 2)
(diisi setelah data meja tersedia: panjang patch, jumlah lapis)

## Hasil 3 seed konfigurasi akhir
(diisi setelah menjalankan --seed 1, 2, 3)