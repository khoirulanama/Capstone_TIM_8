import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run

rng = np.random.default_rng(0)

def blok(n):
    return rng.normal(size=(n, 400, 6)).astype("float32"), rng.integers(0, 4, n)

def palsu():
    Xtr, ytr = blok(256)
    Xva, yva = blok(64)
    Xte, yte = blok(64)
    return Xtr, ytr, Xva, yva, Xte, yte

run.data.muat = palsu
pv, pt = run.latih_dan_prediksi(1, True)
print(pv.shape, pt.shape, set(pv.tolist()))