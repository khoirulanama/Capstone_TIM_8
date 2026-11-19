# Catatan Model CNN (NIM: 122450039)

## 1. Alasan Pemilihan Optimizer
- **Optimizer**: Adam (`torch.optim.Adam`) dengan `lr=0.003` dan `weight_decay=1e-4`.
- **Alasan**: Adam mengadaptasi laju pembelajaran (*learning rate*) per-parameter berdasarkan momen orde pertama dan kedua gradien. Untuk sinyal getaran berdimensi 400 titik dengan beberapa layer konvolusi 1D, Adam mempercepat konvergensi dibanding SGD standar serta stabil pada batch kecil (batch size 32).

## 2. Regularisasi
- **Batch Normalization 1D (`nn.BatchNorm1d`)**: Dipasang setelah setiap lapisan `Conv1d` untuk mengurangi pergeseran kovariat internal (*internal covariate shift*) dan mempercepat konvergensi.
- **Dropout**: Diterapkan sebesar `0.1` pada blok awal dan `0.2` pada blok konvolusi lanjutan serta classifier untuk mencegah *overfitting* pada fitur frekuensi/waktu tertentu.
- **Weight Decay ($L_2$ regularization)**: `weight_decay=1e-4` pada optimizer Adam untuk mengontrol besaran bobot filter konvolusi.
- **Adaptive Average Pooling (`nn.AdaptiveAvgPool1d(1)`)**: Mereduksi dimensi spasial menjadi representasi global sebelum masuk ke lapisan linier, mengurangi parameter classifier secara signifikan dibanding flattening konvensional.

## 3. Pencarian Hyperparameter (Tabel)
Eksperimen awal pada data simulasi getaran kipas (1 detik, 400 Hz):

| Konfigurasi | Batch Size | Learning Rate | Kernel Size | Val Acc | Val Macro-F1 | Durasi (s) |
|---|---|---|---|---|---|---|
| C1 (Baseline Conv1D) | 32 | 0.001 | [5, 3, 3] | 0.9816 | 0.9812 | ~16s |
| C2 (Kernel lebih lebar) | 32 | 0.003 | [7, 5, 3] | 1.0000 | 1.0000 | ~16s |
| C3 (High LR) | 64 | 0.010 | [7, 5, 3] | 0.9632 | 0.9615 | ~10s |

*Pilihan konfigurasi akhir: C2.*

## 4. Rencana Ablation (≥ 2)
1. **Ablation 1**: Tanpa BatchNorm1d vs Dengan BatchNorm1d.
2. **Ablation 2**: Variasi ukuran filter konvolusi awal (kernel 7 vs kernel 3 untuk menangkap rentang gelombang rotasi kipas).

## 5. Hasil Seed Konfigurasi Akhir (Data Simulasi)
- **Seed 1**:
  - `val_acc`: 1.0
  - `val_macro_f1`: 1.0
  - `durasi_detik`: ~15.9 detik
  - `data_hash`: `b465868cf947149c235815666db0cc68`

## 6. Validasi awal pada rekaman aktual (Demo Sprint 1)

Angka pada bagian ini berasal dari audio primer sesi `sesi-20261116-182908`, bukan
eksperimen simulasi di atas. Audio dipotong menurut timestamp label, dibuang 1 detik
awal, lalu di-resample dari 48 kHz ke 400 Hz dengan filter anti-alias. Split dilakukan
berdasarkan `ulangan_blok` (1=train, 2=validasi, 3=test); test belum dievaluasi.

| Seed | Val Acc | Val Macro-F1 | Durasi | Data hash |
|---|---:|---:|---:|---|
| 1 | 0.368873 | 0.294255 | 6.5 s | `1ff2f8194134ce9c2023b9faa52d9f88` |

Ini satu run awal untuk DoD Sprint 1, bukan estimasi akhir. Belum ada tuning,
ablation, atau tiga seed. Audio adalah pengukuran akustik; hasil tidak boleh disebut
pengukuran getaran langsung.
