"""Skrip generator data getaran kipas simulasi (Sprint 0).
Sesuai rancangan meja uji:
- Sampling rate: 16 kHz
- Durasi: 18 detik per rekaman
- PWM: [100, 160, 220, 255]
- Beban: 0, 1, 2, 3 mur
- Ulangan: 3 sesi x 3 blok (total 9 blok)
- Disimpan sebagai NPZ terkompresi di data/rekaman_kipas_simulasi.npz
"""
import numpy as np
from pathlib import Path

def generate_dataset():
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "rekaman_kipas_simulasi.npz"

    sr = 16000
    durasi = 18
    n_samples = sr * durasi
    t = np.linspace(0, durasi, n_samples, endpoint=False, dtype=np.float32)

    pwms = [100, 160, 220, 255]
    bebans = [0, 1, 2, 3]  # jumlah mur
    sesis = [1, 2, 3]
    bloks = [1, 2, 3]

    sinyal_list = []
    pwm_list = []
    beban_list = []
    sesi_list = []
    blok_list = []

    rng = np.random.default_rng(2026)

    for sesi in sesis:
        for blok in bloks:
            for pwm in pwms:
                # Frekuensi rotasi kipas proporsional dengan PWM (misal ~15 Hz s.d 40 Hz)
                f_rot = 15.0 + (pwm - 100) / (255 - 100) * 25.0
                for beban in bebans:
                    # Amplitudo 1x frekuensi putar meningkat seiring bertambahnya mur (unbalance)
                    amp_unbalance = 0.05 + 0.15 * beban
                    # Komponen blade pass frequency (misal 7 bilah kipas)
                    f_bpf = f_rot * 7.0
                    amp_bpf = 0.08

                    sinyal = (
                        amp_unbalance * np.sin(2 * np.pi * f_rot * t + rng.uniform(0, 2 * np.pi))
                        + amp_bpf * np.sin(2 * np.pi * f_bpf * t + rng.uniform(0, 2 * np.pi))
                        + 0.5 * amp_unbalance * np.sin(2 * np.pi * 2 * f_rot * t)  # harmonik ke-2
                        + rng.normal(0, 0.04, size=n_samples).astype(np.float32)   # noise
                    ).astype(np.float32)

                    sinyal_list.append(sinyal)
                    pwm_list.append(pwm)
                    beban_list.append(beban)
                    sesi_list.append(sesi)
                    blok_list.append(blok)

    np.savez_compressed(
        out_file,
        sinyal=np.array(sinyal_list, dtype=np.float32),
        pwm=np.array(pwm_list, dtype=np.int32),
        beban=np.array(beban_list, dtype=np.int32),
        sesi=np.array(sesi_list, dtype=np.int32),
        blok=np.array(blok_list, dtype=np.int32),
        sr=sr,
    )
    print(f"Data simulasi berhasil dibuat: {out_file} ({len(sinyal_list)} kondisi)")

if __name__ == "__main__":
    generate_dataset()
