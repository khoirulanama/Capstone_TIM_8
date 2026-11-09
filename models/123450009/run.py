"""Kerangka run.py anggota. Salin folder ini menjadi models/<nim>/ lalu isi bagian MODEL.

    python models/<nim>/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "trf"                                   # cnn | tl | rnn | trf | ae

N_KELAS = 4          # klasifikasi 4 kelas (0, 1, 2, 3 mur)
PANJANG_PATCH = 20   # 400 sampel / 20 = 20 patch per jendela
D_MODEL = 64
N_HEAD = 4
N_LAYER = 2
EPOCH = 3
BATCH = 64
LR = 1e-3


def buat_model(n_kanal: int, panjang: int):
    import torch
    from torch import nn

    class TransformerPatch(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_patch = panjang // PANJANG_PATCH
            self.proj = nn.Linear(PANJANG_PATCH * n_kanal, D_MODEL)
            self.cls = nn.Parameter(torch.zeros(1, 1, D_MODEL))
            self.pos = nn.Parameter(torch.randn(1, self.n_patch + 1, D_MODEL) * 0.02)
            lapis = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=4 * D_MODEL,
                dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(
                lapis, num_layers=N_LAYER, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(D_MODEL)
            self.kepala = nn.Linear(D_MODEL, N_KELAS)

        def forward(self, x):                      # x: (B, waktu, kanal)
            B = x.shape[0]
            x = x[:, : self.n_patch * PANJANG_PATCH]
            x = x.reshape(B, self.n_patch, -1)     # (B, n_patch, PANJANG_PATCH * kanal)
            x = self.proj(x)
            x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1) + self.pos
            x = self.encoder(x)
            return self.kepala(self.norm(x[:, 0]))  # keluaran token CLS

    return TransformerPatch()


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None)."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    def ke_tensor(X):
        X = np.asarray(X, dtype=np.float32)        # harapan (N, waktu, kanal)
        if X.shape[1] == 6 and X.shape[2] != 6:    # kalau (N, kanal, waktu), balik urutannya
            X = X.transpose(0, 2, 1)
        return torch.tensor(X)

    X, y = ke_tensor(Xtr), torch.tensor(np.asarray(ytr), dtype=torch.long)
    assert int(y.max()) < N_KELAS, "label di luar 0..3, cek data.muat()"
    model = buat_model(n_kanal=X.shape[2], panjang=X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)  # alasan: tulis di catatan.md

    model.train()
    for _ in range(EPOCH):
        urut = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            idx = urut[i:i + BATCH]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(ke_tensor(Xva)).argmax(1).numpy()
        pred_test = model(ke_tensor(Xte)).argmax(1).numpy() if final and Xte is not None else None
    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()