# Protokol Evaluasi Tim 8 — Diagnosis Beban Tak Seimbang Kipas dari Getaran

Diisi Penjaga Data & Evaluasi pada Sprint 0, digabung ke `main` lewat *pull request* yang disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request* baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Diagnosis beban tak seimbang kipas dari getaran: klasifikasi 4 kelas dari getaran kipas 400 Hz, jendela 1 s, dengan pembanding non-deep-learning dan batasan ukuran/latensi untuk dipasang kembali di laptop meja. |
| Sumber dataset | Rekaman meja uji tim (mikrofon 16 kHz), variasi PWM [100, 160, 220, 255] dan beban [0, 1, 2, 3 mur], ulangan 3 × 3 blok, 18 s per kondisi. (Sprint 0: simulasi terstruktur di `data/rekaman_kipas_simulasi.npz`) |
| Jenis tugas (`TUGAS`) | klasifikasi |
| Target / label | 4 kelas (0: 0 mur, 1: 1 mur, 2: 2 mur, 3: 3 mur pada bilah kipas) |
| Ukuran data latih / validasi / uji | 1469 sampel jendela / 163 sampel jendela / 816 sampel jendela (jendela 1 detik @ 400 Hz) |
| Cara split | Group Split berbasis Sesi/Partisipan: Sesi 1 & 2 untuk Train-Val, Sesi 3 untuk Uji. Split latih/val menggunakan `split.indeks` seed 2026, porsi validasi 10%. |
| Metrik utama untuk komparasi | macro-F1 (klasifikasi) dan akurasi; komparasi akhir: rata-rata ± simpangan baku dari 3 seed konfigurasi akhir; ukuran ONNX (KB) dan latensi inferensi (ms) di laptop meja |
| Praproses yang sama untuk semua model | 1. Pembuangan 1 detik pertama tiap rekaman kondisi (transien awal dibuang).<br>2. Desimasi/resampling frekuensi dari 16 kHz ke 400 Hz.<br>3. Pemotongan jendela geser 1 s (400 titik sampel).<br>4. Normalisasi/standarisasi fitur dihitung HANYA dari data latih (*fit on train only*). |
| Anggaran komputasi per run | ≤ 20 menit pelatihan per run di CPU laptop (atau GPU lokal/Colab bila didokumentasikan); catat `--durasi` dan `--perangkat`. |
| Evaluasi data uji | Sekali, oleh Penjaga Data, pada sprint terakhir (`--final`). |
| `data_hash` saat dibekukan | `b465868cf947149c235815666db0cc68` (dari `results/baseline/metrics.json`) |

---

### Pembagian Jalur Model Tim 8 (4 Anggota & Baseline):
1. **Jalur CNN**: 1D-CNN untuk menangkap pola lokal getaran domain waktu.
2. **Jalur RNN**: LSTM / GRU untuk pemodelan sekuensial deret waktu getaran.
3. **Jalur Transformer**: Self-Attention Transformer encoder untuk sinyal 1D.
4. **Jalur Autoencoder (AE)**: Pembelajaran representasi nir-terawasi lalu klasifikasi kondisi kipas.
5. **Pembanding Non-DL**: Ekstraksi fitur statistik + FFT (RMS, puncak, kurtosis, energi pita frekuensi) + Random Forest Classifier.

### Jadwal Meja Uji:
* **Sesi Operator (Perekaman Data Aktual)**: Jumat, 6 November 2026 (08.00–09.00).
* **Uji Purwarupa Sprint 2**: Selasa, 17 November 2026 (11.30–12.00).

Disetujui pembimbing pada: *(tautan pull request ke main)*
