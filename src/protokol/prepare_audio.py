"""Potong rekaman WAV aktual menurut timestamp label menjadi jendela 400 Hz.

Contoh:
  python -m src.protokol.prepare_audio --session-dir ../sesi-20261116-182908
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from src.common.manifest import AKAR

SR_TARGET = 400
DROP_SECONDS = 1
WINDOW_SECONDS = 1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_window(start: float, end: float, audio: list[dict], session: Path) -> np.ndarray:
    """Ambil rentang host-time; dukung rentang yang melintasi berkas WAV."""
    pieces: list[np.ndarray] = []
    cursor = start
    for row in audio:
        file_start = float(row["t_mulai_host"])
        sr = int(row["sample_rate"])
        n = int(row["jumlah_sampel"])
        file_end = file_start + n / sr
        left, right = max(cursor, file_start), min(end, file_end)
        if right <= left:
            continue
        if left > cursor + 0.025:
            raise ValueError(f"Ada celah audio {left-cursor:.3f}s sebelum {row['berkas']}.")
        wav_sr, samples = wavfile.read(session / "audio" / row["berkas"])
        if wav_sr != sr:
            raise ValueError(f"Sample rate WAV tidak cocok dengan audio_index: {row['berkas']}.")
        if samples.ndim == 2:
            samples = samples.astype(np.float32).mean(axis=1)
        else:
            samples = samples.astype(np.float32)
        i0 = max(0, int(round((left - file_start) * sr)))
        i1 = min(len(samples), int(round((right - file_start) * sr)))
        pieces.append(samples[i0:i1])
        cursor = min(end, file_end)
        if cursor >= end - 1e-6:
            break
    if cursor < end - 0.025 or not pieces:
        raise ValueError(f"Audio hanya mencakup {cursor-start:.3f}s dari {end-start:.3f}s.")
    return np.concatenate(pieces)


def prepare(session: Path, output: Path) -> tuple[int, dict[int, int]]:
    audio = sorted(read_csv(session / "audio_index.csv"), key=lambda r: float(r["t_mulai_host"]))
    labels = read_csv(session / "label.csv")
    windows, targets, repeats, groups = [], [], [], []
    for row in labels:
        # Label dapat berupa "0 mur" ... "3 mur".
        label = int(row["beban_kipas"].split()[0])
        repeat = int(row["ulangan_blok"])
        start = float(row["t_mulai_host"]) + DROP_SECONDS
        end = float(row["t_selesai_host"])
        raw = read_window(start, end, audio, session)
        sr = int(audio[0]["sample_rate"])
        signal = resample_poly(raw, SR_TARGET, sr).astype(np.float32)
        needed = int((end - start) * SR_TARGET)
        signal = signal[:needed]
        count = len(signal) // (SR_TARGET * WINDOW_SECONDS)
        if count != 17:
            raise ValueError(f"Label {row['no']}: seharusnya 17 jendela, diperoleh {count}.")
        for i in range(count):
            lo = i * SR_TARGET * WINDOW_SECONDS
            windows.append(signal[lo:lo + SR_TARGET * WINDOW_SECONDS])
            targets.append(label)
            repeats.append(repeat)
            groups.append(f"{repeat}-{row['blok']}")
    X = np.stack(windows).astype(np.float32)
    y = np.asarray(targets, dtype=np.int64)
    u = np.asarray(repeats, dtype=np.int8)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=X, y=y, ulangan_blok=u,
                        grup=np.asarray(groups, dtype="U16"), sample_rate=SR_TARGET)
    counts = {c: int((y == c).sum()) for c in range(4)}
    return len(y), counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-dir", type=Path,
                    default=AKAR.parent / "sesi-20261116-182908")
    ap.add_argument("--output", type=Path, default=AKAR / "data/rekaman_kipas_aktual.npz")
    a = ap.parse_args()
    n, counts = prepare(a.session_dir.resolve(), a.output.resolve())
    print(f"Tersimpan {n} jendela @400 Hz ke {a.output}; distribusi kelas: {counts}")


if __name__ == "__main__":
    main()
