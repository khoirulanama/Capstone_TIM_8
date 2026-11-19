
"""CNN 2D untuk klasifikasi beban kipas dari jendela audio aktual.

Audio mono 1 detik (400 Hz, 400 sampel) diubah menjadi log-spektrogram STFT
65 x 9. CNN kecil mempelajari pola frekuensi dan perubahan antar-frame.

Penggunaan:
    python models/122450039/run.py --seed 1
    python models/122450039/run.py --seed 1 --final
"""
#Model CNN
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


class ConvNetAudio(nn.Module):
    """CNN ringkas untuk log-spektrogram [batch, 1, 65 frekuensi, 9 waktu]."""
    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(24, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.classifier(torch.flatten(feats, 1))


def log_spektrogram(x: torch.Tensor) -> torch.Tensor:
    """STFT magnitudo log; n_fft=128 memberi resolusi 3.125 Hz pada 400 Hz."""
    stft = torch.stft(x, n_fft=128, hop_length=32, win_length=128,
                      window=torch.hann_window(128, device=x.device),
                      center=False, return_complex=True)
    return torch.log1p(stft.abs())


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Latih CNN audio dan prediksi validasi (serta test bila final=True)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Muat data sesuai protokol tim
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    # Fitur STFT dihitung per jendela; statistik normalisasi hanya dari train.
    Xtr_t = torch.from_numpy(np.asarray(Xtr, dtype=np.float32))
    Xva_t = torch.from_numpy(np.asarray(Xva, dtype=np.float32))
    Xte_t = torch.from_numpy(np.asarray(Xte, dtype=np.float32)) if Xte is not None else None
    Ftr = log_spektrogram(Xtr_t)
    Fva = log_spektrogram(Xva_t)
    Fte = log_spektrogram(Xte_t) if Xte_t is not None else None
    mean_tr, std_tr = Ftr.mean(), Ftr.std().clamp_min(1e-6)
    X_train_tensor = ((Ftr - mean_tr) / std_tr).unsqueeze(1)
    y_train_tensor = torch.tensor(ytr, dtype=torch.long)
    X_val_tensor = ((Fva - mean_tr) / std_tr).unsqueeze(1)

    # Dataset & Dataloader
    batch_size = 64
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cpu")
    model = ConvNetAudio(num_classes=4, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

    # Pilih checkpoint dengan macro-F1 validasi; hentikan saat tidak membaik.
    epochs, patience = 100, 15
    best_f1, best_state, stale = -1.0, None, 0
    yva_np = np.asarray(yva, dtype=np.int64)
    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_val_tensor.to(device)).argmax(dim=1).cpu().numpy()
        f1_per_class = []
        for c in range(4):
            tp = np.sum((pred == c) & (yva_np == c))
            fp = np.sum((pred == c) & (yva_np != c))
            fn = np.sum((pred != c) & (yva_np == c))
            f1_per_class.append(2 * tp / max(2 * tp + fp + fn, 1))
        score = float(np.mean(f1_per_class))
        if score > best_f1:
            best_f1, best_state, stale = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluasi model
    model.eval()
    with torch.no_grad():
        logits_val = model(X_val_tensor.to(device))
        pred_val = logits_val.argmax(dim=1).cpu().numpy()

        pred_test = None
        if final and Fte is not None:
            X_test_tensor = ((Fte - mean_tr) / std_tr).unsqueeze(1).to(device)
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
