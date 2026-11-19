# Catatan sesi (data/CATATAN-SESI.md)

- Sesi yang dipakai: `sesi-20261116-182908` (direkam 16 Nov 2026, sekitar 18.29-19.34 WIB; data asli meja uji, diterima 19 Nov 2026)
- Kondisi dikeluarkan: tidak ada (0 dari 144) — alasan: lembar sesi tidak mencatat gangguan yang membuat suatu kondisi harus dibuang. Kondisi yang datanya bolong tetap dipakai, tetapi jendela 1 detik yang bolong (kurang dari 360 sampel atau celah lebih dari 0,05 detik) dibuang otomatis oleh pemotong jendela: 88 dari 2448 jendela. Sisanya diinterpolasi linear ke 400 sampel.
- Kejanggalan lain:
  - Laju IMU rata-rata sekitar 386 Hz, bukan 400 Hz seperti rencana.
  - Wi-Fi ESP32 sempat putus beberapa kali (tercatat di `log.txt`), sehingga 10 kondisi punya laju di bawah 350/s (daftar di tabel). Kondisi 128 paling rendah (sekitar 199 Hz). Kondisi 96 dan 98 juga punya celah lebih dari 0,1 detik (dicatat May di MANIFEST-nya).
  - Audio direkam 48 kHz (rencana 16 kHz). Tidak dipakai, karena model memakai IMU.
  - Getaran tidak naik rapi mengikuti jumlah mur, dan beda antarblok pemasangan ulang lebih besar daripada pengaruh mur. Ini keterbatasan data dan dicatat apa adanya, tidak diperbaiki.
- sha256 `label.csv`: 234086bd36699773f62fe3b8989b0c9ff181b56bfd4f29fd3e343bc9b7dcea71 (daftar berkas lengkap ada di `data/MANIFEST.json`)

## Hasil pemeriksaan otomatis

| Pemeriksaan | Status | Rincian |
|---|---|---|
| Jumlah kondisi | LULUS | 144 (rencana 144) |
| Struktur blok | LULUS | 12 blok, 3 per taraf |
| Laju sampel IMU | GAGAL | 10 kondisi < 350/s: nomor [38, 95, 100, 119, 124, 125, 127, 128, 143, 144] |
| Amplitudo '0 mur' antarblok | PERINGATAN | rasio min/maks terendah 0.61; kecepatan [255] berbeda jauh, sensor mungkin mengendur |
| Kesesuaian label (RMS naik bersama mur) | PERINGATAN | tidak naik untuk kecepatan [160, 220, 255] (periksa label/sensor; pada kipas pelan beda 0 vs 1 mur memang kecil) |
| Gangguan lembar sesi | LULUS | 0 kondisi dikeluarkan |

## Pemisahan data
Per blok pemasangan ulang (3 lipatan), bukan per jendela; normalisasi dari data latih saja. Pemeriksaan kebocoran lulus untuk ketiga lipatan (`py -m src.protokol.data`).