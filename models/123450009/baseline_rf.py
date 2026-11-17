"""Pembanding non-DL (#8): fitur statistik + FFT + Random Forest.

    python models/123450009/baseline_rf.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
FS = 400          # Hz
N_PITA = 5        # jumlah pita frekuensi


def ke_np(X):
    X = np.asarray(X, dtype=np.float32)            # harapan (N, waktu, kanal)
    if X.shape[1] == 6 and X.shape[2] != 6:
        X = X.transpose(0, 2, 1)
    return X


def fitur(X):
    """X: (N, waktu, kanal) -> (N, kanal * (3 + N_PITA))."""
    X = ke_np(X)
    rms = np.sqrt((X ** 2).mean(axis=1))                         # (N, kanal)
    puncak = np.abs(X).max(axis=1)                               # (N, kanal)
    kurt = kurtosis(X, axis=1)                                   # (N, kanal)
    spek = np.abs(np.fft.rfft(X - X.mean(axis=1, keepdims=True), axis=1)) ** 2
    pita = np.array_split(np.arange(spek.shape[1])[1:], N_PITA)  # buang DC
    energi = np.stack([spek[:, idx, :].sum(axis=1) for idx in pita], axis=1)  # (N, N_PITA, kanal)
    energi = np.log1p(energi).reshape(len(X), -1)
    f = np.concatenate([rms, puncak, np.nan_to_num(kurt), energi], axis=1)
    return f.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    clf = RandomForestClassifier(n_estimators=300, random_state=a.seed, n_jobs=-1)
    clf.fit(fitur(Xtr), np.asarray(ytr))
    pred_val = clf.predict(fitur(Xva)).astype(np.int64)
    pred_test = clf.predict(fitur(Xte)).astype(np.int64) if a.final and Xte is not None and len(Xte) > 0 else None
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred_val)
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", "rf",
                "--seed", str(a.seed), "--pred-val", str(out / "pred_val.npy"),
                "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", pred_test)
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()