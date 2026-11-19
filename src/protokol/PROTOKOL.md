# Protokol Evaluasi Tim 8 — Klasifikasi Beban Tak Seimbang Kipas dari Audio

Dokumen ini memperbarui rancangan Sprint 0 setelah rekaman aktual sesi
`sesi-20261116-182908` tersedia. Perubahan protokol tetap perlu ditinjau dan
disetujui pembimbing sebelum dijadikan protokol resmi.

| Butir | Isi |
|---|---|
| Kasus | Klasifikasi empat kondisi beban kipas (0–3 mur) dari audio meja uji. Audio adalah pengukuran akustik sebagai proksi kondisi kipas, bukan pengukuran akselerasi/getaran langsung. |
| Data primer | 144 interval berlabel; 66 berkas WAV mono, laju aktual 48 kHz. Sumber label dan waktu: `label.csv`; indeks waktu/berkas: `audio_index.csv`. Variabel pendukung ada di `imu.csv`, `aktuator.csv`, `daya.csv`, `lingkungan.csv`, dan `digital.csv`. |
| Masukan model | Audio mono, jendela 1 detik pada 400 Hz (400 sampel). Potong 1 detik awal tiap interval untuk transien, lalu resample 48 kHz ke 400 Hz dengan filter anti-alias (`scipy.signal.resample_poly`). Pemotongan mengambil audio berdasarkan waktu host dan dapat menyambung dua WAV bila interval melintasi batas berkas. |
| Target | `beban_kipas`: 0, 1, 2, atau 3 mur. |
| Split | Group split berdasarkan `ulangan_blok`: ulangan 1 untuk train, ulangan 2 untuk validasi, ulangan 3 untuk test. Seluruh jendela dari satu ulangan tetap pada split yang sama; test tidak digunakan untuk pemilihan model. Seed split tidak mengubah penetapan grup. |
| Ukuran yang diharapkan | 816 jendela per split (17 jendela × 48 interval berlabel per ulangan); total 2448. Distribusi kelas harus dilaporkan dari hasil ekstraksi sebelum pelatihan. |
| Normalisasi | Parameter normalisasi dihitung hanya dari data train. Semua jalur menggunakan split dan data yang sama. |
| Metrik | Macro-F1 (utama), akurasi (sekundernya). Sprint 1: minimal satu seed per jalur, pembanding non-DL, dan `metrics.json` valid. Sprint 2: tiga seed, penyetelan, ablation, dan komparasi lengkap. |
| Pembanding non-DL | Fitur statistik domain waktu dan energi beberapa pita FFT, diikuti Random Forest. |
| Evaluasi test | Hanya saat tahap final, dikoordinasikan Penjaga Data. Sampai itu, test dipakai untuk pemeriksaan teknis split saja; jangan hitung atau gunakan metrik test untuk memilih model. |
| Mode simulasi | `CAPSTONE_DATA_MODE=simulasi` tersedia untuk latihan alur sebelum sesi primer. Hasil simulasi bukan hasil penelitian dan tidak boleh dibandingkan dengan metrik aktual. |

## Jalur model dan DoD

Jalur tim: CNN, RNN, Transformer, dan Autoencoder; pembanding non-DL Random Forest.
Untuk Demo Sprint 1 (11 hari persiapan setelah data sesi lengkap): setiap jalur
dilatih pada data primer dengan minimal satu seed, pembanding berjalan, dan berkas
`metrics.json` valid. Tiga seed, tuning, serta perbandingan lengkap dikerjakan pada
Sprint 2 (28 November–12 Desember).

## Reproduksi

```bash
python -m src.protokol.prepare_audio --session-dir ../sesi-20261116-182908
python models/122450039/run.py --seed 1
python models/122450039/non_dl.py --seed 1
```

`rekaman_kipas_aktual.npz` adalah artefak lokal yang dihasilkan dari berkas sesi;
data mentah dan NPZ tidak di-commit. Rekaman aktual berjalan pada 48 kHz meskipun
rencana awal menyebut 16 kHz; parameter pemrosesan di atas mengikuti laju aktual.
