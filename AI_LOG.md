# AI_LOG — Tim 8

Satu baris per pemakaian AI yang bermakna (ChatGPT, Claude, Copilot, Bravais, dan sejenisnya).
Ringkas dan jujur; transkrip penuh tidak diminta. Yang dinilai bukan "pakai AI atau tidak",
melainkan "mengerti atau tidak" — isi kolom terakhir dengan apa yang Anda ubah dari keluarannya.

| Tanggal | Siapa (NIM) | Alat | Untuk apa | Apa yang diubah dari keluarannya |
|---|---|---|---|---|
| 2026-10-31 | 122450039 | Antigravity | Sprint 0: Generator simulasi meja uji, protokol tim & baseline | Menyesuaikan pemotongan 1 detik awal transien, desimasi 400 Hz, windowing 1 s, serta group split sesi sesuai rancangan meja uji |
| 2026-11-10 | 123450010 | Claude | Draf pemotong jendela, data.py (split per blok), periksa_mutu, dan kerangka AE pada data simulasi | Menjalankan simulasi.py dan data.py sendiri (3 lipatan lolos tanpa kebocoran), menjalankan run.py AE --seed 1 pada data simulasi, menyesuaikan berkas dengan isi repo tim, memperbaiki deskripsi PR #8, dan mencatat kekurangannya (data masih simulasi, nama kolom asli belum diverifikasi) |