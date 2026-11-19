# Catatan Model CNN (NIM: 122450039)

## Konfigurasi untuk data aktual

Masukan adalah audio mono 1 detik pada 400 Hz (400 sampel), bukan sinyal
getaran langsung. `run.py` mengubah tiap jendela menjadi log-magnitudo STFT
ukuran 65 frekuensi × 9 frame waktu (FFT 128, hop 32), lalu memakai CNN 2D
ringkas dengan 16/32/48 kanal. Normalisasi fitur dihitung dari train saja.
AdamW (`lr=0.001`, `weight_decay=1e-3`), dropout, dan pemilihan checkpoint
berdasarkan macro-F1 validasi mengendalikan overfitting pada data yang terbatas.
Pelatihan berhenti setelah 15 epoch tanpa peningkatan, dengan batas 100 epoch.

Empat kelas memiliki 36 interval berlabel per kelas pada data mentah. Walau
jumlah jendela tampak besar (816 per split), jendela-jendela dalam interval
yang sama berkaitan erat; karena itu split protokol tetap dikelompokkan menurut
ulangan eksperimen.

## 1. Alasan Pemilihan Optimizer
- **Optimizer**: AdamW (`torch.optim.AdamW`) dengan `lr=0.001` dan `weight_decay=1e-3`.
- **Alasan**: AdamW memberi pembaruan adaptif dan pemisahan weight decay, dengan laju belajar moderat untuk CNN kecil serta data berjumlah terbatas.

## 2. Regularisasi
- BatchNorm2d dan Dropout2d pada blok fitur; Dropout pada classifier.
- Weight decay `1e-3` melalui AdamW.
- Early stopping memakai macro-F1 validasi (patience 15, maksimum 100 epoch), lalu memulihkan checkpoint terbaik.

## 3. Pencarian Hyperparameter (catatan historis)
Eksperimen berikut dilakukan dengan arsitektur CNN 1D lama pada data simulasi;
hasilnya tidak mewakili arsitektur STFT baru maupun data aktual:

| Konfigurasi | Batch Size | Learning Rate | Kernel Size | Val Acc | Val Macro-F1 | Durasi (s) |
|---|---|---|---|---|---|---|
| C1 (Baseline Conv1D) | 32 | 0.001 | [5, 3, 3] | 0.9816 | 0.9812 | ~16s |
| C2 (Kernel lebih lebar) | 32 | 0.003 | [7, 5, 3] | 1.0000 | 1.0000 | ~16s |
| C3 (High LR) | 64 | 0.010 | [7, 5, 3] | 0.9632 | 0.9615 | ~10s |

Konfigurasi tabel ini adalah catatan historis untuk model 1D simulasi. Tuning
untuk model 2D pada audio aktual belum dilakukan.

## 4. Rencana Ablation (≥ 2)
1. **Ablation 1**: Tanpa normalisasi STFT vs normalisasi dengan statistik train.
2. **Ablation 2**: Bandingkan STFT FFT 128 vs FFT 64 untuk menguji resolusi frekuensi.

## 5. Hasil historis seed 1 (CNN 1D, data simulasi)
- **Seed 1**:
  - `val_acc`: 1.0
  - `val_macro_f1`: 1.0
  - `durasi_detik`: ~15.9 detik
  - `data_hash`: `b465868cf947149c235815666db0cc68`

## 6. Validasi awal pada rekaman aktual (arsitektur CNN 1D sebelumnya)

Angka pada bagian ini berasal dari audio primer sesi `sesi-20261116-182908`, bukan
eksperimen simulasi di atas. Audio dipotong menurut timestamp label, dibuang 1 detik
awal, lalu di-resample dari 48 kHz ke 400 Hz dengan filter anti-alias. Split dilakukan
berdasarkan `ulangan_blok` (1=train, 2=validasi, 3=test); test belum dievaluasi.

| Seed | Val Acc | Val Macro-F1 | Durasi | Data hash |
|---|---:|---:|---:|---|
| 1 | 0.368873 | 0.294255 | 6.5 s | `1ff2f8194134ce9c2023b9faa52d9f88` |

Angka ini berasal dari arsitektur CNN 1D pada waveform mentah sebelum penyesuaian
STFT di `run.py`; jangan dipakai sebagai hasil konfigurasi baru. Konfigurasi STFT
perlu dijalankan ulang untuk memperbarui hasil dan selanjutnya diuji dengan tiga
seed. Audio adalah pengukuran akustik; hasil tidak boleh disebut pengukuran
getaran langsung.
