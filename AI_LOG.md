# AI_LOG — Tim 8

Satu baris per pemakaian AI yang bermakna (ChatGPT, Claude, Copilot, Bravais, dan sejenisnya).
Ringkas dan jujur; transkrip penuh tidak diminta. Yang dinilai bukan "pakai AI atau tidak",
melainkan "mengerti atau tidak" — isi kolom terakhir dengan apa yang Anda ubah dari keluarannya.

| Tanggal | Siapa (NIM) | Alat | Untuk apa | Apa yang diubah dari keluarannya |
|---|---|---|---|---|
<<<<<<< HEAD
| 2026-11-09 | 123450009 | Claude | kerangka model Transformer patch di models/123450009/run.py (#15) | ganti JALUR ke trf, 4 kelas, tidak meratakan sinyal, AdamW; memeriksa alur patch-CLS-encoder; uji bentuk (8,4) dan uji data acak (test_lokal.py) lolos; belum bisa jalan di data simulasi karena data.muat() kosong |
| 2026-11-09 | 123450009 | Claude | draf catatan.md dan penyempurnaan run.py (normalisasi, cetak loss, batas waktu) | disesuaikan ke bentuk (N, 400, 6); alasan optimizer akan ditulis ulang dengan kata sendiri |
=======
| 2026-10-31 | 122450039 | Antigravity | Sprint 0: Generator simulasi meja uji, protokol tim & baseline | Menyesuaikan pemotongan 1 detik awal transien, desimasi 400 Hz, windowing 1 s, serta group split sesi sesuai rancangan meja uji |
>>>>>>> origin/main
