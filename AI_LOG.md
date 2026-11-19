# AI_LOG — Tim 8

Satu baris per pemakaian AI yang bermakna (ChatGPT, Claude, Copilot, Bravais, dan sejenisnya).
Ringkas dan jujur; transkrip penuh tidak diminta. Yang dinilai bukan "pakai AI atau tidak",
melainkan "mengerti atau tidak" — isi kolom terakhir dengan apa yang Anda ubah dari keluarannya.

| Tanggal | Siapa (NIM) | Alat | Untuk apa | Apa yang diubah dari keluarannya |
|---|---|---|---|---|
<<<<<<< HEAD
| 2026-11-09 | 123450009 | Claude | kerangka model Transformer patch di models/123450009/run.py (#15) | ganti JALUR ke trf, 4 kelas, tidak meratakan sinyal, AdamW; memeriksa alur patch-CLS-encoder; uji bentuk (8,4) dan uji data acak (test_lokal.py) lolos; belum bisa jalan di data simulasi karena data.muat() kosong |
| 2026-11-09 | 123450009 | Claude | draf catatan.md dan penyempurnaan run.py (normalisasi, cetak loss, batas waktu) | disesuaikan ke bentuk (N, 400, 6); alasan optimizer akan ditulis ulang dengan kata sendiri |
| 2026-10-31 | 122450039 | Antigravity | Sprint 0: Generator simulasi meja uji, protokol tim & baseline | Menyesuaikan pemotongan 1 detik awal transien, desimasi 400 Hz, windowing 1 s, serta group split sesi sesuai rancangan meja uji |
| 2026-11-17 | 123450009 | Claude | kerangka pembanding non-DL di models/123450009/baseline_rf.py (#8): fitur statistik + FFT + Random Forest | menguji sendiri bentuk fitur (200, 48) pada data acak (test_rf.py); belum dijalankan pada data meja karena data.muat() belum terhubung |
=======
| 2026-10-31 | 122450039 | Antigravity | Sprint 0: Generator simulasi meja uji, protokol tim & baseline | Menyesuaikan pemotongan 1 detik awal transien, desimasi 400 Hz, windowing 1 s, serta group split sesi sesuai rancangan meja uji |
| 2026-11-10 | 123450010 | Claude | Draf pemotong jendela, data.py (split per blok), periksa_mutu, dan kerangka AE pada data simulasi | Menjalankan simulasi.py dan data.py sendiri (3 lipatan lolos tanpa kebocoran), menjalankan run.py AE --seed 1 pada data simulasi, menyesuaikan berkas dengan isi repo tim, memperbaiki deskripsi PR #8, dan mencatat kekurangannya (data masih simulasi, nama kolom asli belum diverifikasi) |
| 2026-11-10 | 122450041 | OpenCode (DeepSeek) | Gladi rencana: harness `perekam/meja.py`, rencana 144 kondisi, catatan & komentar tugas | Memeriksa ulang skema YAML, memastikan keluaran 144 kondisi, memperbaiki indentasi `rencana/Rencana.yaml`, dan menandai selisih 96 vs 144 serta struktur `generate_dummy.py` |
| 2026-11-16 | 122450041 | OpenCode (DeepSeek) | Menyusun draf Log Sprint 1 Workbench (selesai/berikutnya/hambatan/bukti) dari artefak repo | Memeriksa ulang angka 144 kondisi, 1.036.800 sampel 400 Hz, dan metrics seed 1; memilih tautan commit/PR yang benar; memperbaiki nama pada catatan gladi (Khoirul Anam -> Octavia Nurwinda Puspitasari) |
| 2026-11-09 | 123450009 | Claude | Kerangka model Transformer patch di models/123450009/run.py (#15) | Mengatur jalur trf dan 4 kelas, mempertahankan dimensi sinyal, memakai AdamW, serta menguji bentuk keluaran; belum berjalan pada data primer |
| 2026-11-09 | 123450009 | Claude | Draf catatan.md dan penyempurnaan run.py | Menyesuaikan kode pada bentuk (N, 400, 6); alasan optimizer akan ditulis ulang dengan kata sendiri |
| 2026-11-19 | 122450039 | ChatGPT | Integrasi rekaman aktual: pemotongan jendela audio, split per ulangan, pembanding Random Forest, dan run awal CNN | Meninjau pemetaan timestamp dan keterbatasan akustik; memeriksa hasil 2.448 jendela, memperjelas masukan aktual audio 48 kHz, dan memakai split ulangan 1/2/3 |
>>>>>>> 5271fe0f3f7bb7a022b9060fa25efab05956deea
