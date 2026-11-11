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
