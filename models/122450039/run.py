
"""Implementasi Jalur CNN (1D-CNN) untuk Diagnosis Beban Tak Seimbang Kipas.

Model: 1D-CNN pada jendela getaran 1 detik (400 Hz -> 400 titik).
Output: Klasifikasi 4 kelas (0, 1, 2, 3 mur).

Penggunaan:
    python models/122450039/run.py --seed 1
    python models/122450039/run.py --seed 1 --final
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM (122450039)
JALUR = "cnn"                                   # cnn | tl | rnn | trf | ae


class ConvNet1D(nn.Module):
    """Arsitektur 1D-CNN untuk klasifikasi sinyal getaran 1D (400 sampel).
    
    Terdiri dari blok konvolusi 1D (Conv1d, BatchNorm, ReLU, MaxPool, Dropout)
    untuk ekstraksi fitur lokal domain waktu dan lapisan fully-connected untuk klasifikasi.
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            # Blok 1: [B, 1, 400] -> [B, 16, 200]
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),

            # Blok 2: [B, 16, 200] -> [B, 32, 100]
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),

            # Blok 3: [B, 32, 100] -> [B, 64, 50]
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),

            # Global average pooling: [B, 64, 50] -> [B, 64, 1]
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Masukan: [B, 1, 400]
        feats = self.features(x)
        feats = torch.flatten(feats, 1)  # [B, 64]
        out = self.classifier(feats)      # [B, num_classes]
        return out


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Latih model 1D-CNN dan lakukan inferensi pada data validasi (serta data uji jika final=True)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Muat data sesuai protokol tim
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    # Normalisasi standar (z-score) dihitung HANYA dari data latih (fit on train only)
    mean_tr = np.mean(Xtr)
    std_tr = np.std(Xtr) + 1e-8

    Xtr_norm = (Xtr - mean_tr) / std_tr
    Xva_norm = (Xva - mean_tr) / std_tr
    Xte_norm = (Xte - mean_tr) / std_tr if Xte is not None else None

    # Ubah ke tensor [N, 1, 400] untuk Conv1d
    X_train_tensor = torch.tensor(Xtr_norm[:, np.newaxis, :], dtype=torch.float32)
    y_train_tensor = torch.tensor(ytr, dtype=torch.long)

    X_val_tensor = torch.tensor(Xva_norm[:, np.newaxis, :], dtype=torch.float32)

    # Dataset & Dataloader
    batch_size = 32
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cpu")
    model = ConvNet1D(num_classes=4, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    # Pelatihan 15 epoch (cepat dan efisien di CPU)
    epochs = 15
    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    # Evaluasi model
    model.eval()
    with torch.no_grad():
        logits_val = model(X_val_tensor.to(device))
        pred_val = logits_val.argmax(dim=1).cpu().numpy()

        pred_test = None
        if final and Xte_norm is not None:
            X_test_tensor = torch.tensor(Xte_norm[:, np.newaxis, :], dtype=torch.float32).to(device)
            logits_test = model(X_test_tensor)
            pred_test = logits_test.argmax(dim=1).cpu().numpy()

    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser(description="Jalur CNN untuk Capstone Tim 8")
    ap.add_argument("--seed", type=int, required=True, help="Seed acak untuk reproduktibilitas")
    ap.add_argument("--final", action="store_true", help="Evaluasi data uji sekali di sprint terakhir")
    ap.add_argument("--perangkat", default="cpu laptop", help="Nama perangkat pengujian")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
