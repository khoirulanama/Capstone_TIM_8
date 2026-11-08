#!/usr/bin/env python3
"""meja.py -- pemutar rencana eksperimen meja uji (paket perekam).

Antarmuka mengikuti 00-metode-eksperimen-lapangan.pdf bagian 3 dan 5.1::

    python meja.py periksa
    python meja.py jalankan ../rencana/tNN-nama.yaml
    python meja.py --simulasi jalankan ../rencana/tNN-nama.yaml

Mode ``--simulasi`` tidak memerlukan perangkat. Skrip menyusun kondisi dari
YAML, mencetak ringkasan (jumlah blok, jumlah kondisi, estimasi durasi), lalu
menulis folder sesi di ``data/timNN/sesi-.../`` berisi ``label.csv``,
``manifest.json``, dan berkas sensor sintetis secukupnya untuk menguji
pemotong jendela tim.

Catatan penting: data yang dihasilkan mode simulasi adalah **data sintetis**,
bukan hasil pengukuran. Jangan dipakai sebagai data latih/uji model.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - pesan yang ramah untuk pengguna
    sys.exit("PyYAML belum terpasang. Jalankan: pip install pyyaml")

AKAR = Path(__file__).resolve().parents[1]
LAJU_IMU = 400  # Hz, sesuai keterangan kasus tim 8


# ---------------------------------------------------------------------------
# Membaca dan memvalidasi rencana
# ---------------------------------------------------------------------------
def muat_rencana(jalur: Path) -> dict:
    if not jalur.exists():
        raise FileNotFoundError(f"berkas rencana tidak ditemukan: {jalur}")
    with jalur.open(encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    if not isinstance(plan, dict):
        raise ValueError("isi rencana harus mapping YAML (bukan daftar/skalar)")

    wajib = ["tim", "seed", "faktor", "ulangan", "ulangan_blok",
             "durasi_kondisi_detik", "jeda_detik"]
    hilang = [k for k in wajib if k not in plan]
    if hilang:
        raise ValueError(f"kunci rencana belum lengkap: {', '.join(hilang)}")
    if not isinstance(plan["faktor"], dict) or not plan["faktor"]:
        raise ValueError("'faktor' harus berisi minimal satu faktor")

    for k in ("ulangan", "ulangan_blok"):
        if int(plan[k]) < 1:
            raise ValueError(f"'{k}' harus >= 1")
    for k in ("durasi_kondisi_detik", "jeda_detik"):
        if float(plan[k]) < 0:
            raise ValueError(f"'{k}' tidak boleh negatif")
    return plan


def baca_faktor(faktor: dict):
    """Kembalikan (otomatis, manual, isyarat) sebagai daftar tuple.

    - faktor berupa daftar  -> otomatis (taraf = isi daftar)
    - faktor berupa mapping -> jenis: otomatis/manual/isyarat, taraf: daftar
    """
    otomatis: list[tuple[str, list]] = []
    manual: list[tuple[str, list, str]] = []
    isyarat: list[tuple[str, list, str]] = []

    for nama, isi in faktor.items():
        if isinstance(isi, list):
            if not isi:
                raise ValueError(f"faktor '{nama}' tidak punya taraf")
            otomatis.append((nama, isi))
            continue
        if not isinstance(isi, dict):
            raise ValueError(f"faktor '{nama}' harus daftar atau mapping")
        taraf = isi.get("taraf")
        if not isinstance(taraf, list) or not taraf:
            raise ValueError(f"faktor '{nama}' harus punya 'taraf' berupa daftar")
        jenis = str(isi.get("jenis", "otomatis")).lower()
        instruksi = str(isi.get("instruksi", "Siapkan: " + nama + " = {taraf}"))
        if jenis == "manual":
            manual.append((nama, taraf, instruksi))
        elif jenis == "isyarat":
            isyarat.append((nama, taraf, instruksi))
        else:
            otomatis.append((nama, taraf))
    return otomatis, manual, isyarat


def _kombinasi(daftar: list[tuple[str, list]]) -> list[dict]:
    if not daftar:
        return [{}]
    nama = [d[0] for d in daftar]
    return [dict(zip(nama, nilai)) for nilai in itertools.product(*[d[1] for d in daftar])]


def susun_kondisi(plan: dict):
    """Susun blok dan kondisi persis seperti aturan bagian 3 dokumen metode."""
    otomatis, manual, isyarat = baca_faktor(plan["faktor"])
    seed = int(plan["seed"])
    ulangan = int(plan["ulangan"])
    ulangan_blok = int(plan["ulangan_blok"])
    rng = random.Random(seed)

    # Blok = kombinasi faktor manual, diulang ulangan_blok kali, lalu diacak.
    kombinasi_manual = _kombinasi([(m[0], m[1]) for m in manual])
    blok = [dict(k) for k in kombinasi_manual] * ulangan_blok
    rng.shuffle(blok)

    kombinasi_otomatis = _kombinasi(otomatis)
    kombinasi_isyarat = _kombinasi([(s[0], s[1]) for s in isyarat])

    kondisi: list[dict] = []
    nomor = 0
    for blok_ke, keadaan_manual in enumerate(blok, start=1):
        isi_blok: list[dict] = []
        for ulangan_ke in range(1, ulangan + 1):
            for auto in kombinasi_otomatis:
                for isyarat_keadaan in kombinasi_isyarat:
                    nilai = {}
                    nilai.update(auto)
                    nilai.update(keadaan_manual)
                    nilai.update(isyarat_keadaan)
                    isi_blok.append({
                        "nilai": nilai,
                        "manual": dict(keadaan_manual),
                        "isyarat": dict(isyarat_keadaan),
                        "ulangan_ke": ulangan_ke,
                    })
        rng.shuffle(isi_blok)
        for kondisi_ke, k in enumerate(isi_blok, start=1):
            nomor += 1
            k["nomor"] = nomor
            k["blok"] = blok_ke
            k["kondisi"] = kondisi_ke
            k["instruksi_manual"] = "; ".join(
                instruksi.format(taraf=k["manual"][nama])
                for nama, _, instruksi in manual
            )
            k["instruksi_isyarat"] = "; ".join(
                instruksi.format(taraf=k["isyarat"][nama])
                for nama, _, instruksi in isyarat
            )
            kondisi.append(k)

    return kondisi, blok, len(isi_blok)


def beri_waktu(kondisi: list[dict], durasi: float, jeda: float, ganti: float) -> float:
    """Isi mulai/selesai_detik; waktu berjalan termasuk jeda ganti blok."""
    t = 0.0
    for k in kondisi:
        if k["kondisi"] == 1:
            k["ganti_blok_detik"] = ganti
            t += ganti
        else:
            k["ganti_blok_detik"] = 0
        k["mulai_detik"] = t
        t += durasi
        k["selesai_detik"] = t
        k["durasi_detik"] = durasi
        k["jeda_detik"] = jeda
        t += jeda
    return t


def format_durasi(detik: float) -> str:
    menit = int(round(detik / 60))
    if menit >= 60:
        return f"{menit // 60} jam {menit % 60} menit"
    return f"{menit} menit"


# ---------------------------------------------------------------------------
# Penulisan folder sesi
# ---------------------------------------------------------------------------
def _tulis_csv(jalur: Path, header: list[str], baris) -> None:
    with jalur.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for b in baris:
            w.writerow(b)


def _kolom_faktor(kondisi: list[dict]) -> list[str]:
    nama: list[str] = []
    for k in kondisi:
        for n in k["nilai"]:
            if n not in nama:
                nama.append(n)
    return nama


def _tulis_imu(jalur: Path, kondisi: list[dict], seed: int) -> None:
    """Getaran sintetis 400 Hz; amplitudo naik dengan jumlah mur dan PWM."""
    rng = random.Random(seed)
    header = "t_host,ax,ay,az,gx,gy,gz\n"
    with jalur.open("w", encoding="utf-8", newline="") as f:
        f.write(header)
        for k in kondisi:
            beban = k["manual"].get("beban_kipas", "0 mur")
            try:
                n_mur = int(str(beban).split()[0])
            except (ValueError, IndexError):
                n_mur = 0
            pwm = float(k["nilai"].get("kipas", 0))
            amp = 0.10 + 0.18 * n_mur + 0.0008 * pwm
            f0 = 15.0 + 0.20 * pwm
            w0 = 2.0 * math.pi * f0
            noise = 0.02 + 0.002 * n_mur
            t0 = k["mulai_detik"]
            n = int(round(k["durasi_detik"] * LAJU_IMU))
            buf: list[str] = []
            for i in range(n):
                t = i / LAJU_IMU
                # detik pertama adalah peralihan; amplitudo naik perlahan
                ramp = 0.25 + 0.75 * min(1.0, t)
                s1 = math.sin(w0 * t)
                s2 = math.sin(2.0 * w0 * t + 0.7)
                s3 = math.sin(3.0 * w0 * t + 1.3)
                ax = amp * ramp * (s1 + 0.35 * s2) + rng.uniform(-noise, noise)
                ay = amp * ramp * (0.80 * s1 + 0.25 * s3) + rng.uniform(-noise, noise)
                az = 9.81 + amp * ramp * 0.15 * s2 + rng.uniform(-noise, noise)
                gx = 12.0 * amp * ramp * s1 + rng.uniform(-2.0, 2.0)
                gy = 8.0 * amp * ramp * s2 + rng.uniform(-2.0, 2.0)
                gz = 5.0 * amp * ramp * s3 + rng.uniform(-2.0, 2.0)
                buf.append(
                    f"{t0 + t:.4f},{ax:.4f},{ay:.4f},{az:.4f},"
                    f"{gx:.4f},{gy:.4f},{gz:.4f}"
                )
                if len(buf) >= 2000:
                    f.write("\n".join(buf) + "\n")
                    buf.clear()
            if buf:
                f.write("\n".join(buf) + "\n")


def tulis_sesi(plan: dict, jalur_rencana: Path, kondisi: list[dict],
               blok: list, per_blok: int, total_detik: float,
               nama_sesi: str | None, tanpa_sensor: bool) -> Path:
    tim = int(plan["tim"])
    folder = AKAR / "data" / f"tim{tim:02d}" / (nama_sesi or datetime.now().strftime("sesi-%Y%m%d-%H%M%S"))
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "audio").mkdir(exist_ok=True)

    kolom = _kolom_faktor(kondisi)
    header = (["nomor", "kondisi", "blok", "ulangan_ke"] + kolom +
              ["instruksi_manual", "instruksi_isyarat", "mulai_detik",
               "selesai_detik", "durasi_detik", "jeda_detik", "ganti_blok_detik"])
    baris = []
    for k in kondisi:
        baris.append(
            [k["nomor"], k["kondisi"], k["blok"], k["ulangan_ke"]] +
            [k["nilai"].get(n, "") for n in kolom] +
            [k["instruksi_manual"], k["instruksi_isyarat"],
             f"{k['mulai_detik']:.3f}", f"{k['selesai_detik']:.3f}",
             f"{k['durasi_detik']:.3f}", f"{k['jeda_detik']:.3f}",
             f"{k['ganti_blok_detik']:.3f}"]
        )
    _tulis_csv(folder / "label.csv", header, baris)

    # Perintah aktuator: kipas disetel di awal kondisi, dimatikan saat jeda.
    aktuator = []
    for k in kondisi:
        pwm = k["nilai"].get("kipas")
        if pwm is not None:
            aktuator.append([f"{k['mulai_detik']:.3f}", "kipas", pwm, "SET"])
            aktuator.append([f"{k['selesai_detik']:.3f}", "kipas", 0, "SET"])
    _tulis_csv(folder / "aktuator.csv", ["t_host", "alat", "nilai", "aksi"], aktuator)

    # Instruksi per blok untuk operator alat.
    instruksi = []
    for i, keadaan in enumerate(blok, start=1):
        teks = []
        for nama, _, pola in baca_faktor(plan["faktor"])[1]:
            teks.append(pola.format(taraf=keadaan[nama]))
        instruksi.append(f"Blok {i:02d}: Siapkan: {'; '.join(teks)}")
    (folder / "instruksi.txt").write_text("\n".join(instruksi) + "\n", encoding="utf-8")

    berkas = ["label.csv", "aktuator.csv", "instruksi.txt", "README.md"]

    if not tanpa_sensor:
        _tulis_imu(folder / "imu.csv", kondisi, int(plan["seed"]) + 17)
        berkas.append("imu.csv")

        # Lingkungan 1 Hz (kecil, cukup untuk memeriksa struktur).
        rng = random.Random(int(plan["seed"]) + 23)
        ling = []
        for detik in range(int(math.ceil(total_detik))):
            ling.append([f"{detik:.0f}", f"{26.5 + rng.uniform(-0.4, 0.4):.2f}",
                         f"{60.0 + rng.uniform(-2.0, 2.0):.2f}",
                         f"{320 + rng.randint(-15, 15)}"])
        _tulis_csv(folder / "lingkungan.csv",
                   ["t_host", "suhu_c", "kelembapan_pct", "lux"], ling)
        berkas.append("lingkungan.csv")

        _tulis_csv(folder / "digital.csv", ["t_host", "pir", "pintu"],
                   [["0.000", 0, 0], [f"{total_detik:.3f}", 0, 0]])
        berkas.append("digital.csv")

        audio = plan.get("rekam", {}).get("audio", {})
        if audio.get("aktif"):
            sr = int(audio.get("sample_rate", 16000))
            idx = [[k["nomor"], k["kondisi"],
                    f"audio/kondisi-{k['nomor']:03d}.wav",
                    f"{k['mulai_detik']:.3f}", f"{k['selesai_detik']:.3f}", sr]
                   for k in kondisi]
            _tulis_csv(folder / "audio_index.csv",
                       ["nomor", "kondisi", "berkas", "mulai_detik",
                        "selesai_detik", "sample_rate"], idx)
            berkas.append("audio_index.csv")

    manifest = {
        "mode": "simulasi",
        "tim": tim,
        "judul": plan.get("judul", ""),
        "pembimbing": plan.get("pembimbing", ""),
        "pilar": plan.get("pilar", ""),
        "rencana": str(jalur_rencana),
        "seed": int(plan["seed"]),
        "dibuat": datetime.now().isoformat(timespec="seconds"),
        "jumlah_blok": len(blok),
        "kondisi_per_blok": per_blok,
        "jumlah_kondisi": len(kondisi),
        "ulangan": int(plan["ulangan"]),
        "ulangan_blok": int(plan["ulangan_blok"]),
        "durasi_kondisi_detik": float(plan["durasi_kondisi_detik"]),
        "jeda_detik": float(plan["jeda_detik"]),
        "waktu_ganti_blok_detik": float(plan.get("waktu_ganti_blok_detik", 120)),
        "estimasi_detik": round(total_detik, 3),
        "estimasi_menit": round(total_detik / 60, 1),
        "berkas": berkas,
        "catatan": ("Data SINTETIS dari mode simulasi; bukan hasil pengukuran. "
                    "Jangan dipakai sebagai data latih/uji model."),
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (folder / "README.md").write_text(
        f"# Sesi simulasi tim {tim}\n\n"
        f"Dibuat oleh `perekam/meja.py --simulasi` pada {manifest['dibuat']}.\n\n"
        f"- Rencana: `{jalur_rencana.name}`\n"
        f"- {manifest['jumlah_blok']} blok x {per_blok} kondisi = "
        f"**{len(kondisi)} kondisi**\n"
        f"- Estimasi: {format_durasi(total_detik)} "
        f"({total_detik:.0f} detik)\n\n"
        "`label.csv` memuat selang waktu setiap kondisi. `imu.csv` adalah "
        "getaran sintetis 400 Hz untuk menguji pemotong jendela. Berkas audio "
        "tidak dibuat pada mode simulasi; `audio_index.csv` hanya menunjukkan "
        "struktur indeks.\n\n"
        "**Data sintetis, bukan data pengukuran.**\n",
        encoding="utf-8")

    return folder


# ---------------------------------------------------------------------------
# Perintah CLI
# ---------------------------------------------------------------------------
def cetak_ringkasan(plan: dict, jalur: Path, kondisi: list[dict], blok: list,
                    per_blok: int, total_detik: float, folder: Path) -> str:
    otomatis, manual, isyarat = baca_faktor(plan["faktor"])
    faktor_teks = []
    for nama, taraf in otomatis:
        faktor_teks.append(f"{nama} (otomatis, {len(taraf)} taraf)")
    for nama, taraf, _ in manual:
        faktor_teks.append(f"{nama} (manual, {len(taraf)} taraf)")
    for nama, taraf, _ in isyarat:
        faktor_teks.append(f"{nama} (isyarat, {len(taraf)} taraf)")

    baris = [
        "=" * 64,
        "meja.py -- mode SIMULASI (tanpa perangkat)",
        "=" * 64,
        f"Rencana   : {jalur}",
        f"Tim       : {plan.get('tim')} -- {plan.get('judul', '')}",
        f"Pembimbing: {plan.get('pembimbing', '-')}",
        f"Seed      : {plan.get('seed')}",
        f"Faktor    : {', '.join(faktor_teks)}",
        f"Blok      : {len(blok)} "
        f"({len(_kombinasi([(m[0], m[1]) for m in manual]))} kombinasi manual "
        f"x {plan['ulangan_blok']} ulangan_blok)",
        f"Kondisi   : {per_blok} per blok x {len(blok)} blok = {len(kondisi)} kondisi",
        f"Durasi    : {plan['durasi_kondisi_detik']} s + "
        f"{plan['jeda_detik']} s jeda per kondisi; "
        f"ganti blok {plan.get('waktu_ganti_blok_detik', 120)} s",
        f"Estimasi  : {total_detik:.0f} s = {format_durasi(total_detik)}",
        f"Folder    : {folder}",
        "-" * 64,
        f"Daftar {len(kondisi)} kondisi:",
    ]
    for k in kondisi:
        nilai = "  ".join(f"{n}={v}" for n, v in k["nilai"].items())
        baris.append(
            f"[{k['nomor']:>3}/{len(kondisi)}] blok {k['blok']:02d}/{len(blok)} "
            f"kondisi {k['kondisi']:02d}/{per_blok} ulangan {k['ulangan_ke']}  {nilai}"
        )
    baris.append("-" * 64)
    baris.append(f"Selesai. {len(kondisi)} kondisi tercatat di {folder / 'label.csv'}")
    return "\n".join(baris)


def perintah_periksa() -> int:
    print("meja.py -- mode PERIKSA")
    print("[ ] ESP32        : tidak terdeteksi (repositori ini hanya mode simulasi)")
    print("[ ] Mikrofon     : tidak terdeteksi")
    print("[ ] Kamera       : tidak terdeteksi")
    print("Gunakan --simulasi untuk menjalankan rencana tanpa perangkat.")
    return 1


def perintah_jalankan(jalur: Path, simulasi: bool, nama_sesi: str | None,
                      tanpa_sensor: bool) -> int:
    if not simulasi:
        print("Mode perangkat belum tersedia di repositori ini.")
        print("Jalankan mode simulasi: python meja.py --simulasi jalankan <rencana.yaml>")
        return 2

    plan = muat_rencana(jalur)
    kondisi, blok, per_blok = susun_kondisi(plan)
    durasi = float(plan["durasi_kondisi_detik"])
    jeda = float(plan["jeda_detik"])
    ganti = float(plan.get("waktu_ganti_blok_detik", 120))
    total = beri_waktu(kondisi, durasi, jeda, ganti)
    folder = tulis_sesi(plan, jalur, kondisi, blok, per_blok, total,
                        nama_sesi, tanpa_sensor)
    teks = cetak_ringkasan(plan, jalur, kondisi, blok, per_blok, total, folder)
    print(teks)
    (folder / "ringkasan.txt").write_text(teks + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pemutar rencana eksperimen meja uji (mode simulasi).")
    p.add_argument("--simulasi", action="store_true",
                   help="jalankan tanpa perangkat dan tulis data sintetis")
    sub = p.add_subparsers(dest="perintah", required=True)
    sub.add_parser("periksa", help="periksa perangkat meja")
    pj = sub.add_parser("jalankan", help="jalankan rencana")
    pj.add_argument("rencana", type=Path, help="jalur berkas YAML rencana")
    pj.add_argument("--nama-sesi", default=None,
                    help="nama folder sesi (default: sesi-YYYYMMDD-HHMMSS)")
    pj.add_argument("--tanpa-sensor", action="store_true",
                    help="lewati penulisan imu.csv/lingkungan.csv (uji cepat)")
    args = p.parse_args(argv)

    if args.perintah == "periksa":
        return perintah_periksa()
    return perintah_jalankan(args.rencana, args.simulasi, args.nama_sesi,
                             args.tanpa_sensor)


if __name__ == "__main__":
    raise SystemExit(main())
