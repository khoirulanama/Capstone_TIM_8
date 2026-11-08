**Gladi rencana — 8 Nov 2026**

✅ **Mode simulasi**
```bash
cd perekam
python meja.py --simulasi jalankan ../rencana/t08-kipas-tak-seimbang.yaml
```
→ mencetak **144 kondisi** (12 blok × 12 kondisi; 4 taraf beban × 3 `ulangan_blok`, 4 PWM kipas × 3 `ulangan`). Estimasi **1 jam 12 menit** (48 menit rekam + 24 menit ganti blok).

✅ **Folder sesi simulasi**
`data/tim08/sesi-20261108-234403/` — `label.csv` 144 baris, `manifest.json`, `imu.csv` 400 Hz (1.036.800 sampel), `aktuator.csv`, `audio_index.csv`, `lingkungan.csv`.
Bukti ringkas dibagikan lewat repo: `sprint/gladi-rencana/sesi-simulasi/` (folder lengkap 56 MB tidak di-commit karena `data/*` di-ignore).

✅ **Pemeriksaan**
- 144 kondisi; 12 blok; 12 kondisi/blok.
- Tiap taraf beban 36 kondisi; tiap PWM 36 kondisi.
- IMU 400 sampel/detik; 1 detik pertama disimulasikan sebagai peralihan.
- Label dan instruksi manual terbaca dari `label.csv`.

⚠️ **Selisih rencana yang perlu keputusan pembimbing**
- `rencana/Rencana.yaml` di `main` memakai `ulangan: 2`, `durasi: 12` → **96 kondisi / 46 menit** (usulan revisi `tim-08.pdf` §7).
- Kartu tugas meminta **144 kondisi** (ulangan 3, 18 detik).
- Simulasi di atas memakai versi 144. Indentasi YAML yang rusak sudah diperbaiki.
- `src/protokol/generate_dummy.py` masih memakai struktur 9 blok × 16 kondisi (bukan 12 × 12) dan `PROTOKOL.md` split per sesi (bukan per blok/pemasangan ulang). Perlu diselaraskan setelah keputusan.

🧰 **Bahan slot (checklist; verifikasi fisik oleh tim sebelum slot)**
- Spidol permanen untuk menandai bilah kipas.
- 3 mur M3.
- Selotip potong sama panjang: 12 potong untuk 12 blok + cadangan; taraf 0 mur tetap memakai selotip kosong.
- Lembar sesi + alat tulis.
- Flash disk tim.
- (Dicek bersama pengelola saat slot) MPU6050 menempel kuat; posisi alat sesuai tanda selotip.

🏃 **Latihan kering**
Tidak ada faktor ucapan/isyarat/gerakan partisipan, jadi latihan kering partisipan tidak relevan. Latihan kering **operator** untuk pergantian blok sudah disiapkan (`sprint/gladi-rencana/latihan-kering.md`): 3 putaran dengan stopwatch, target ≤ 2 menit/blok. Eksekusi fisik oleh operator alat sebelum slot.

📌 **Tindak lanjut**
1. Pembimbing memutuskan 144 vs 96 kondisi.
2. Operator alat menjalankan latihan kering 3 putaran dan mengisi lembar waktu.
3. Penjaga Data memeriksa `label.csv` + `manifest.json` di folder sesi.
