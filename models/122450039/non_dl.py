"""Pembanding non-DL: fitur statistik dan FFT + Random Forest."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.protokol import data  # noqa: E402


def fitur(X: np.ndarray) -> np.ndarray:
    """Statistik waktu ditambah proporsi energi spektrum pada pita 0–200 Hz."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    z = X - mu
    sd = np.sqrt(np.mean(z * z, axis=1)) + 1e-10
    rms = np.sqrt(np.mean(X * X, axis=1))
    absx = np.abs(X)
    skew = np.mean((z / sd[:, None]) ** 3, axis=1)
    kurt = np.mean((z / sd[:, None]) ** 4, axis=1)
    stats = np.column_stack((mu[:, 0], sd, rms, absx.max(axis=1),
                             np.percentile(absx, 95, axis=1), skew, kurt))
    power = np.abs(np.fft.rfft(z, axis=1)) ** 2
    freq = np.fft.rfftfreq(X.shape[1], d=1 / 400)
    total = power.sum(axis=1) + 1e-10
    bands = [(0, 20), (20, 50), (50, 100), (100, 200)]
    energy = np.column_stack([power[:, (freq >= lo) & (freq < hi)].sum(axis=1) / total
                              for lo, hi in bands])
    return np.column_stack((stats, energy)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    start = time.time()
    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    ftr, fva = fitur(Xtr), fitur(Xva)
    model = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                   min_samples_leaf=2, class_weight="balanced",
                                   random_state=a.seed, n_jobs=-1)
    model.fit(ftr, ytr)
    pred_val = model.predict(fva).astype(np.int64)
    out = ROOT / "results" / "non_dl"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred_val)
    duration = time.time() - start
    cmd = [sys.executable, "-m", "src.common.evaluate", "--nim", "non_dl",
           "--model", "random_forest", "--seed", str(a.seed),
           "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{duration:.1f}",
           "--perangkat", a.perangkat]
    if a.final:
        pred_test = model.predict(fitur(Xte)).astype(np.int64)
        np.save(out / "pred_test.npy", pred_test)
        cmd += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(cmd, check=True, cwd=ROOT)
    print("Confusion matrix validation:\n", confusion_matrix(data.muat()[3], pred_val))


if __name__ == "__main__":
    main()
