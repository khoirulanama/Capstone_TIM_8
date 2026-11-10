# `perekam/meja.py` — pemutar rencana meja uji (mode simulasi)

Berkas ini adalah **harness simulasi lokal** untuk rencana eksperimen tim.
Paket resmi mata kuliah (`perekam/meja.py` + `rencana/`) tidak ada di repositori
tim, sehingga harness ini dibuat mengikuti antarmuka dan aturan yang tertulis di
`00-metode-eksperimen-lapangan.pdf` bagian 3 dan 5.1.

## Pemakaian

```bash
cd perekam

# simulasi: tidak butuh perangkat, menulis folder sesi + data sintetis
python meja.py --simulasi jalankan ../rencana/t08-kipas-tak-seimbang.yaml

# periksa perangkat (di repo ini selalu melaporkan tidak terdeteksi)
python meja.py periksa

# mode perangkat belum tersedia; akan menolak dengan pesan yang jelas
python meja.py jalankan ../rencana/t08-kipas-tak-seimbang.yaml
```

Opsi tambahan:

| Opsi | Arti |
|---|---|
| `--nama-sesi NAMA` | nama folder sesi (default `sesi-YYYYMMDD-HHMMSS`) |
| `--tanpa-sensor` | lewati `imu.csv`/`lingkungan.csv`/`digital.csv` (uji cepat) |

## Cara menyusun kondisi

Mengikuti bagian 3 dokumen metode:

1. Semua kombinasi faktor **manual** dibentuk menjadi blok, diulang
   `ulangan_blok` kali, lalu urutannya diacak dengan `seed`.
2. Di dalam setiap blok, semua kombinasi faktor **otomatis** × **isyarat**
   dibentuk, diulang `ulangan` kali, lalu diacak.
3. Jumlah kondisi = `jumlah_blok × kondisi_per_blok`.

Untuk rencana `rencana/t08-kipas-tak-seimbang.yaml`:

- 4 taraf `beban_kipas` × 3 `ulangan_blok` = **12 blok**
- 4 taraf `kipas` × 3 `ulangan` = **12 kondisi/blok**
- **144 kondisi**, estimasi 12×120 s + 144×(18+2) s = 4320 s = **1 jam 12 menit**

## Skema YAML yang didukung

- Faktor berupa daftar → faktor otomatis.
  ```yaml
  faktor:
    kipas: [100, 160, 220, 255]
  ```
- Faktor berupa mapping → memakai `jenis`, `taraf`, dan `instruksi`.
  ```yaml
  faktor:
    beban_kipas:
      jenis: manual          # manual | isyarat | otomatis
      taraf: ["0 mur", "1 mur", "2 mur", "3 mur"]
      instruksi: "Pasang {taraf} M3 ..."
  ```
- `{taraf}` pada `instruksi` diganti nilai taraf saat dicetak.
- Kunci opsional `waktu_ganti_blok_detik` (default 120) dipakai untuk estimasi
  dan timeline `label.csv`.

## Keluaran

Folder `data/timNN/sesi-.../`:

| Berkas | Isi |
|---|---|
| `label.csv` | 144 baris kondisi + selang waktu, label, dan instruksi |
| `manifest.json` | metadata sesi, jumlah kondisi, estimasi, daftar berkas |
| `imu.csv` | getaran sintetis 400 Hz, 6 kanal (1.036.800 sampel) |
| `aktuator.csv` | perintah PWM kipas saat mulai/selesai kondisi |
| `audio_index.csv` | indeks audio (WAV tidak dibuat pada mode simulasi) |
| `lingkungan.csv` | suhu/kelembapan/lux sintetis 1 Hz |
| `digital.csv` | status PIR/pintu (kerangka) |
| `instruksi.txt` | instruksi per blok untuk operator alat |
| `ringkasan.txt` | salinan keluaran layar |
| `README.md` | catatan folder sesi |

## Batas

- Data **sintetis**, bukan hasil pengukuran. Jangan dipakai sebagai data
  latih/uji model.
- Mode perangkat (ESP32, mikrofon, kamera) tidak diimplementasikan.
- `audio_index.csv` hanya menunjukkan struktur indeks; berkas WAV tidak dibuat
  agar folder sesi tidak membengkak.
- Folder `data/` di-ignore oleh `.gitignore`; bukti ringkas untuk repo disalin
  ke `sprint/gladi-rencana/sesi-simulasi/`.
