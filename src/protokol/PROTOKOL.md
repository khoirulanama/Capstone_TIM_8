# Protokol evaluasi tim — Tim 8 (Diagnosis beban tak seimbang kipas dari getaran)

Diisi Penjaga Data & Evaluasi (Devi Rahayu) pada Sprint 0/1, digabung ke `main` lewat *pull request*
yang disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request* baru
yang disetujui pembimbing.

> **STATUS: DRAF — menunggu review rekan dan persetujuan pembimbing (belum dibekukan).** Angka bertanda ⟨…⟩ diisi setelah data meja diterima (slot semula 10 Nov, diundur karena alat belum siap) dan lolos
> `periksa_mutu`. Nama kolom `label.csv`/`imu.csv` dan jumlah kondisi (96 vs 144) harus
> dicocokkan dengan rencana final yang benar-benar diserahkan ke pengelola meja.

| Butir | Isi |
|---|---|
| Kasus (kartu Kasus tim) | Diagnosis beban tak seimbang kipas dari getaran; pilar Mesin dan energi; pembimbing Christyan Tamaro Nadeak |
| Sumber dataset | Data primer meja uji Ruang Cerdas 5 V, slot operator semula Sel 10 Nov 2026 08.00–09.00 (diundur; jadwal baru ⟨isi⟩), rencana `t08-kipas-tak-seimbang.yaml`. Tautan penyimpanan tim: ⟨isi⟩. (`github.com/sains-data/capstone-dl-2026-starter` hanya berisi kode, bukan data.) |
| Jenis tugas (`TUGAS`) | klasifikasi, 4 kelas |
| Target / label | taraf `beban_kipas` ∈ {0, 1, 2, 3 mur} — berasal dari desain eksperimen dan diperiksa pembimbing sebelum Enter; bukan dari penebakan pasca-rekam |
| Masukan model | `imu.csv` baris `imu = kipas`, 6 kanal (ax, ay, az, gx, gy, gz), 400 Hz; jendela 1 detik = tensor (6, 400); 1 detik pertama tiap kondisi dibuang; kondisi terganggu (lembar sesi) dikeluarkan |
| Rencana yang direkam | 4 kecepatan kipas (PWM 100/160/220/255) × 4 taraf mur; 12 blok (3 pemasangan per taraf). Rencana final: ulangan 2, durasi 12 detik ⇒ 8 kondisi/blok, **96 kondisi** ⟨cocokkan dengan YAML final; rencana asli = 144 kondisi, 18 detik⟩ |
| Faktor pengganggu | Kecepatan kipas (empat taraf) bukan label; dilaporkan sebagai analisis kesalahan per kecepatan |
| Satuan pemisahan | **Blok** (satu pemasangan mur). Tiga blok per taraf ⇒ tiga lipatan: lipatan *k* memuat pemasangan ke-*k* dari keempat taraf |
| Cara split | Satu lipatan = uji, satu = validasi, satu = latih. Urutan peran ditetapkan sekali oleh `np.random.default_rng(2026).permutation(3)` (seed split beku starter). Pengembangan (Sprint 1–2) memakai lipatan uji 0; data uji tidak dilihat. `src/protokol/data.py: muat(lipatan)` |
| Jendela | latih: langkah 0,5 detik (tumpang-tindih 50 %); validasi dan uji: langkah 1 detik. Ukuran per lipatan: latih ⟨≈ 672⟩, validasi ⟨≈ 352⟩, uji ⟨≈ 352⟩ jendela (hitungan dari simulasi 96 kondisi; isi ulang dengan data asli) |
| Normalisasi | rerata dan simpangan baku per kanal dari data **latih** saja; diterapkan apa adanya pada validasi dan uji |
| Praproses yang sama untuk semua model | seluruh model memanggil `data.muat()` (sudah ternormalisasi). Pembanding non-DL (fitur statistik + FFT) memakai `data.muat(mentah=True)` agar RMS/kurtosis bermakna fisik |
| Metrik utama untuk komparasi | macro-F1 (juga akurasi); rerata ± simpangan baku dari 3 seed; ukuran ONNX (KB) dan latensi (ms) di laptop meja |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop (data kecil: ≈ 0,5–2 menit per run pada simulasi) |
| Evaluasi data uji | Sekali per lipatan, oleh Penjaga Data, pada sprint terakhir: `--final` untuk lipatan 0, 1, 2 (masing-masing 3 seed). Tidak ada pemilihan model atau penyetelan berdasarkan data uji |
| `data_hash` saat dibekukan | ⟨dari `results/baseline/metrics.json` setelah data asli masuk⟩ |

Disetujui pembimbing pada: ⟨…⟩ (tautan *pull request*: ⟨…⟩)

## Jalur model dan jadwal meja

1. **CNN**: 1D-CNN untuk pola lokal getaran domain waktu.
2. **RNN**: LSTM/GRU untuk pemodelan sekuensial.
3. **Transformer**: encoder self-attention atas potongan sinyal.
4. **Autoencoder (AE)**: representasi lalu klasifikasi kondisi kipas.
5. **Pembanding non-DL**: fitur statistik + FFT (RMS, puncak, kurtosis, energi pita) + Random Forest.

Jadwal meja uji: sesi operator semula Selasa 10 Nov 2026 08.00–09.00 (diundur; jadwal baru ⟨isi⟩);
uji purwarupa Sprint 2 Senin 23 Nov 2026 11.30–12.00 (menurut kartu Kasus tim; cocokkan bila berubah).

## Perubahan dari versi Sprint 0 (4 Nov) dan alasannya

Versi sebelumnya memakai `generate_dummy.py` dan `data/rekaman_kipas_simulasi.npz`. Saat membandingkannya
dengan kartu Kasus tim dan `Rencana/Rencana.yaml`, ditemukan hal berikut; semuanya diubah di PR ini:

| Butir | Versi Sprint 0 | Kasus tim / rencana final | Di PR ini |
|---|---|---|---|
| Sumber masukan | mikrofon 16 kHz diturunkan ke 400 Hz | `imu.csv`, baris `imu = kipas`, 6 kanal, 400 Hz; audio hanya kanal tambahan | IMU 6 kanal, bentuk (N, 6, 400) |
| Ukuran rencana | 144 rekaman, 18 s, 9 blok | `ulangan: 2`, `durasi_kondisi_detik: 12`, 12 blok ⇒ 96 kondisi | 96 kondisi, 12 blok × 8 |
| Arti blok | tiap blok memuat semua taraf mur | mur dilepas dan dipasang ulang tiap blok; satu taraf per blok, 3 blok per taraf | blok = satu pemasangan, 3 lipatan |
| Split validasi | `split.indeks` per jendela dari sesi 1–2 | tim-08: jangan memisahkan per jendela | validasi juga per blok |
| Normalisasi | tertulis di protokol, belum ada di `muat()` | dari data latih saja | diterapkan di `muat()` |
| Berkas data | `.npz` di `data/` (tidak di-commit) | label.csv dan imu.csv + MANIFEST | `MANIFEST.json` menunjuk `data/sesi-simulasi` (dibuat ulang dengan `simulasi.py`, deterministik) |

**Dampak untuk rekan setim:** `muat()` kini mengembalikan X berbentuk **(N, 6, 400)**, bukan (N, 400).
Model CNN/RNN/Transformer perlu menyesuaikan jumlah kanal masukan (6), dan RNN/Transformer menyesuaikan
bentuk urutan. `generate_dummy.py` tidak lagi dipakai oleh `data.py` (tidak dihapus).

## Mengapa berbeda dari contoh starter (Fashion-MNIST)

1. **Split per blok, bukan `split.indeks` per jendela.** Starter mengacak jendela (`split.indeks`). Pada
   kasus ini jendela dari satu kondisi sangat mirip, sehingga split acak memberi F1 tinggi semu;
   metode umum bag. 9.2 dan tim-08 bag. 6 melarangnya. Seed 2026 tetap dipakai, tetapi hanya untuk
   menetapkan urutan peran lipatan.
2. **`data/MANIFEST.json`** memakai kunci tambahan `sesi`, `sha256_label`, `keluarkan` selain
   `sumber`/`berkas`. `data_hash` tetap dihitung `src.common.manifest` dari `label.csv` dan `imu.csv`.
3. Berkas `data/*` tidak di-commit (aturan `.gitignore` starter); yang di-commit adalah
   `MANIFEST.json` dan `CATATAN-SESI.md`. Sebelum menjalankan model: `python -m src.protokol.simulasi --keluar data/sesi-simulasi`.

## Keterbatasan yang diketahui

- Hanya ada **satu pemasangan per taraf per lipatan** sebagai data latih (satu lipatan latih, satu
  validasi, satu uji). Data kecil: variasi antar-pemasangan mur terhadap ketidakseimbangan akan
  menjadi sumber galat utama; laporkan sebaran antar-lipatan, bukan hanya antar-seed.
- Hyperparameter disetel pada validasi lipatan 0. Pemasangan yang menjadi validasi di lipatan 0
  menjadi data uji di salah satu lipatan lain, sehingga hasil lipatan itu sedikit optimistis. Cantumkan
  di laporan tim.
- Kecepatan kipas diatur skrip, tetapi respons sensor dapat bergeser bila sensor mengendur; lihat
  `data/CATATAN-SESI.md`.
