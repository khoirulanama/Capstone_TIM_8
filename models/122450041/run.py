"""Jalur RNN (GRU) untuk diagnosis beban tak seimbang kipas dari sinyal getaran.

Kontrak tugas (lihat ``src/protokol/data.py``):
    Input  : X [N, 400]  -> diubah menjadi [N, 400, 1]
             400 Hz, jendela 1 detik (400 sampel), satu kanal getaran.
    Output : klasifikasi 4 kelas (0--3 mur) -> logits [N, 4],
             prediksi label integer [N].

Pemakaian:
    python models/122450041/run.py --seed 1
    python models/122450041/run.py --seed 1 --final   # sprint terakhir: + data uji

Skrip ini memuat data lewat ``src.protokol.data.muat()`` dan menulis
``results/122450041/metrics.json`` lewat ``src.common.evaluate``.
"""
from __future__ import annotations

import argparse
import copy
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "rnn"
PANJANG_JENDELA = 400                          # 400 Hz x 1 detik
N_KANAL = 1                                    # satu sumbu getaran


# --------------------------------------------------------------------------- #
# Reproduksibilitas
# --------------------------------------------------------------------------- #
def atur_seed(seed: int) -> None:
    """Tetapkan seed ke Python, NumPy, dan PyTorch agar hasil dapat diulang."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def siapkan_urut(X: np.ndarray, nama: str) -> np.ndarray:
    """Ubah X [N, 400] menjadi [N, 400, 1] dan hilangkan offset DC per jendela."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[..., None]
    if X.ndim != 3 or X.shape[1:] != (PANJANG_JENDELA, N_KANAL):
        raise ValueError(
            f"{nama}: bentuk {X.shape}, seharusnya [N, {PANJANG_JENDELA}, {N_KANAL}]"
        )
    # Getaran adalah sinyal AC; rata-rata jendela (DC) dibuang agar model fokus
    # pada pola osilasi, bukan bias sensor.
    return X - X.mean(axis=1, keepdims=True)


def normalisasi(
    Xtr: np.ndarray, Xva: np.ndarray, Xte: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, float]:
    """Z-score memakai statistik data latih saja (mencegah kebocoran data).

    Skala global dipertahankan: amplitudo getaran justru merupakan ciri utama
    ketidakseimbangan, sehingga normalisasi tidak dilakukan per jendela.
    """
    mu = float(Xtr.mean())
    sd = float(Xtr.std())
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    if Xte is not None:
        Xte = (Xte - mu) / sd
    return Xtr, Xva, Xte, mu, sd


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GRUKipas(nn.Module):
    """GRU dua arah atas 400 langkah waktu + pooling rata-rata/maksimum.

    Masukan : [batch, 400, 1]
    Keluaran: [batch, n_keluaran]  (4 logits untuk klasifikasi 4 kelas)
    """

    def __init__(self, n_keluaran: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.25) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=N_KANAL,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        lebar = hidden * 2                       # dua arah
        self.norm = nn.LayerNorm(lebar * 2)      # mean + max pooling
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(lebar * 2, n_keluaran)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)                       # [B, 400, 2*hidden]
        fitur = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1)
        return self.fc(self.drop(self.norm(fitur)))


# --------------------------------------------------------------------------- #
# Metrik bantu untuk pemilihan model (bukan pengganti penilai resmi)
# --------------------------------------------------------------------------- #
def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f1 = []
    for c in np.unique(np.concatenate([y, p])):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1)) if f1 else 0.0


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y.astype(np.float64) - p.astype(np.float64)) ** 2)))


# --------------------------------------------------------------------------- #
# Pelatihan dan prediksi
# --------------------------------------------------------------------------- #
def latih_dan_prediksi(
    seed: int,
    final: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    layers: int,
    dropout: float,
    patience: int,
    augmentasi: bool,
    bobot_kelas: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Latih GRU dan kembalikan (pred_val, pred_test|None)."""
    atur_seed(seed)

    # ---- Muat data lewat protokol tim (kontrak wajib) ----
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()
    tugas = getattr(data, "TUGAS", "klasifikasi")
    regresi = tugas == "regresi"

    Xtr = siapkan_urut(Xtr, "X_train")
    Xva = siapkan_urut(Xva, "X_val")
    Xte = siapkan_urut(Xte, "X_test") if Xte is not None else None
    Xtr, Xva, Xte, mu, sd = normalisasi(Xtr, Xva, Xte)

    if regresi:
        n_keluaran = 1
        ytr_t = torch.tensor(np.asarray(ytr, dtype=np.float32).reshape(-1, 1))
        yva_np = np.asarray(yva, dtype=np.float64).reshape(-1)
    else:
        n_keluaran = int(np.max(ytr)) + 1
        ytr_t = torch.tensor(np.asarray(ytr, dtype=np.int64))
        yva_np = np.asarray(yva, dtype=np.int64).reshape(-1)

    print(
        f"[data] train={tuple(Xtr.shape)} val={tuple(Xva.shape)} "
        f"test={tuple(Xte.shape) if Xte is not None else None} | "
        f"tugas={tugas} | keluaran={n_keluaran} | "
        f"normalisasi mean={mu:.5f} std={sd:.5f}"
    )

    Xtr_t = torch.from_numpy(Xtr)
    Xva_t = torch.from_numpy(Xva)
    Xte_t = torch.from_numpy(Xte) if Xte is not None else None

    model = GRUKipas(n_keluaran, hidden=hidden, layers=layers, dropout=dropout)

    # ---- Loss, optimizer, dan regularisasi ----
    if regresi:
        kriteria: nn.Module = nn.MSELoss()
    else:
        bobot = None
        if bobot_kelas:
            kelas, jumlah = np.unique(np.asarray(ytr, dtype=np.int64), return_counts=True)
            w = np.ones(n_keluaran, dtype=np.float32)
            for c, n_c in zip(kelas, jumlah):
                w[int(c)] = len(ytr) / (n_keluaran * float(n_c))
            bobot = torch.tensor(w)
        kriteria = nn.CrossEntropyLoss(weight=bobot)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=max(1, patience // 3)
    )

    rng = np.random.default_rng(seed)
    n = len(Xtr_t)
    terbaik = copy.deepcopy(model.state_dict())
    skor_terbaik = -float("inf")
    epoch_terbaik = 0
    sabar = 0

    print(f"[model] GRU bidirectional hidden={hidden} layers={layers} "
          f"dropout={dropout} | AdamW lr={lr} wd=1e-4 | batch={batch_size}")
    for epoch in range(1, epochs + 1):
        model.train()
        idx = rng.permutation(n)
        total_loss = 0.0
        for awal in range(0, n, batch_size):
            pilih = idx[awal:awal + batch_size]
            xb = Xtr_t[pilih].clone()
            yb = ytr_t[pilih]

            if augmentasi:
                # Pergeseran fase acak (<= 20 sampel = 50 ms) dan noise kecil.
                for j in range(len(xb)):
                    geser = int(rng.integers(-20, 21))
                    if geser:
                        xb[j] = torch.roll(xb[j], shifts=geser, dims=0)
                xb += torch.from_numpy(
                    rng.normal(0.0, 0.01, size=tuple(xb.shape)).astype(np.float32)
                )

            opt.zero_grad(set_to_none=True)
            keluaran = model(xb)
            loss = kriteria(keluaran, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += float(loss.item()) * len(pilih)

        # ---- Validasi untuk early stopping ----
        model.eval()
        with torch.no_grad():
            keluar_val = model(Xva_t)
        if regresi:
            pred_val_epoch = keluar_val.detach().numpy().reshape(-1)
            skor = -rmse(yva_np, pred_val_epoch)
        else:
            pred_val_epoch = keluar_val.argmax(dim=1).numpy()
            skor = macro_f1(yva_np, pred_val_epoch)
        scheduler.step(skor)

        tanda = ""
        if skor > skor_terbaik + 1e-6:
            skor_terbaik = skor
            epoch_terbaik = epoch
            terbaik = copy.deepcopy(model.state_dict())
            sabar = 0
            tanda = " *"
        else:
            sabar += 1

        metrik_epoch = (
            f"val_rmse={-skor:.4f}" if regresi else f"val_macro_f1={skor:.4f}"
        )
        print(
            f"epoch {epoch:02d}/{epochs} loss={total_loss / n:.4f} "
            f"{metrik_epoch} lr={opt.param_groups[0]['lr']:.2e}{tanda}",
            flush=True,
        )
        if sabar >= patience:
            print(f"[early-stop] tidak membaik {patience} epoch; kembali ke epoch {epoch_terbaik}")
            break

    model.load_state_dict(terbaik)
    model.eval()

    # ---- Prediksi akhir memakai bobot terbaik ----
    with torch.no_grad():
        if regresi:
            pred_val = model(Xva_t).numpy().reshape(-1).astype(np.float32)
            pred_test = (
                model(Xte_t).numpy().reshape(-1).astype(np.float32)
                if final and Xte_t is not None else None
            )
        else:
            pred_val = model(Xva_t).argmax(dim=1).numpy().astype(np.int64)
            pred_test = (
                model(Xte_t).argmax(dim=1).numpy().astype(np.int64)
                if final and Xte_t is not None else None
            )

    if not regresi:
        print(f"[selesai] epoch terbaik={epoch_terbaik} val_macro_f1={skor_terbaik:.4f}")
    else:
        print(f"[selesai] epoch terbaik={epoch_terbaik} val_rmse={-skor_terbaik:.4f}")
    return pred_val, pred_test


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Jalur RNN/GRU untuk klasifikasi beban kipas dari getaran 400 Hz."
    )
    ap.add_argument("--seed", type=int, required=True, help="seed reproduksibilitas")
    ap.add_argument("--final", action="store_true",
                    help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--tanpa-augmentasi", action="store_true",
                    help="matikan pergeseran fase dan noise saat latih")
    ap.add_argument("--tanpa-bobot-kelas", action="store_true",
                    help="matikan pembobotan kelas pada cross-entropy")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(
        seed=a.seed,
        final=a.final,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        hidden=a.hidden,
        layers=a.layers,
        dropout=a.dropout,
        patience=a.patience,
        augmentasi=not a.tanpa_augmentasi,
        bobot_kelas=not a.tanpa_bobot_kelas,
    )
    durasi = time.time() - t0

    # ---- Simpan prediksi dan panggil penilai resmi (kontrak wajib) ----
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))

    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)
    print(f"[output] pred_val={tuple(np.asarray(pred_val).shape)} "
          f"durasi={durasi:.1f} s -> {out.relative_to(AKAR)}")


if __name__ == "__main__":
    main()
