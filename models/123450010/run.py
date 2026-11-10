"""Jalur AE (Autoencoder) — Devi Rahayu (123450010), Tim 8: diagnosis beban tak seimbang kipas.

    python models/123450010/run.py --seed 1 [--final] [--lipatan 0]

Gagasan: autoencoder 1D-konvolusi dipralatih TANPA label untuk merekonstruksi jendela getaran
(6 kanal x 400 sampel), lalu vektor laten z dipakai oleh kepala linear kecil untuk
mengklasifikasikan 0/1/2/3 mur (representasi -> klasifikasi, jalur 'ae' di README).

Kontrak starter dipenuhi: data lewat ``src.protokol.data.muat()``, seed dari ``--seed``, prediksi
ke results/<nim>/pred_val.npy (dan pred_test.npy hanya dengan --final), metrik dihitung oleh
``src.common.evaluate`` (bukan di sini). Konfigurasi bawaan = konfigurasi akhir.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.common.evaluate import macro_f1  # noqa: E402  (penilai dosen; dipakai juga untuk memilih epoch)
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "ae"

# Konfigurasi akhir. Alasan tiap pilihan ditulis di catatan.md; ubah hanya lewat hasil pencarian yang dicatat.
# finetune=True: pada data simulasi, encoder BEKU + kepala linear hanya ~0,35 macro-F1, sedangkan fine-tune ~0,94
# (representasi rekonstruksi murni kurang memuat amplitudo relatif-terhadap-frekuensi). VERIFIKASI ULANG pada data meja.
BAWAAN = dict(latent=16, derau=0.10, epoch_ae=60, epoch_kepala=150, lr=1e-3, wd=1e-2, batch=64,
              dropout=0.2, pralatih=True, finetune=True)


def bangun(latent: int, dropout: float):
    """Encoder, decoder, kepala. Bentuk: (B,6,400) -> z (B,latent) -> rekonstruksi (B,6,400)."""
    import torch.nn as nn

    class AE(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(6, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.ReLU(),     # 400 -> 200
                nn.Conv1d(16, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(),    # 200 -> 100
                nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.ReLU(),    # 100 -> 50
                nn.Flatten(), nn.Linear(64 * 50, d))
            self.dec_fc = nn.Linear(d, 64 * 50)
            self.dec = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.BatchNorm1d(32), nn.ReLU(),   # 50 -> 100
                nn.ConvTranspose1d(32, 16, 4, 2, 1), nn.BatchNorm1d(16), nn.ReLU(),   # 100 -> 200
                nn.ConvTranspose1d(16, 6, 4, 2, 1))                                   # 200 -> 400

        def forward(self, x):
            z = self.enc(x)
            return self.dec(self.dec_fc(z).view(-1, 64, 50)), z

    ae = AE(latent)
    kepala = nn.Sequential(nn.BatchNorm1d(latent), nn.Dropout(dropout), nn.Linear(latent, 4))
    return ae, kepala


def _pralatih(torch, ae, Xtr, Xva, c, g):
    """Tahap 1 (tanpa label): minimalkan MSE rekonstruksi dengan derau masukan (denoising)."""
    opt = torch.optim.AdamW(ae.parameters(), lr=c["lr"], weight_decay=c["wd"])
    terbaik, simpan, sabar = 1e9, None, 0
    for ep in range(c["epoch_ae"]):
        ae.train()
        for i in torch.randperm(len(Xtr), generator=g).split(c["batch"]):
            x = Xtr[i]
            rec, _ = ae(x + c["derau"] * torch.randn(x.shape, generator=g))
            loss = torch.nn.functional.mse_loss(rec, x)        # target = sinyal bersih
            opt.zero_grad(); loss.backward(); opt.step()
        ae.eval()
        with torch.no_grad():
            lv = torch.nn.functional.mse_loss(ae(Xva)[0], Xva).item()   # validasi rekonstruksi (tanpa label)
        if lv < terbaik - 1e-5:
            terbaik, simpan, sabar = lv, copy.deepcopy(ae.state_dict()), 0
        else:
            sabar += 1
            if sabar >= 15:
                break
    if simpan is not None:
        ae.load_state_dict(simpan)
    return terbaik


def latih_dan_prediksi(seed: int, final: bool, cfg: dict | None = None):
    """Kembalikan (pred_val, pred_test|None, info, model_klasifikasi). Tidak menulis berkas."""
    import torch

    c = {**BAWAAN, **(cfg or {})}
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    g = torch.Generator().manual_seed(seed)
    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    Xtr, Xva = torch.tensor(Xtr), torch.tensor(Xva)
    ytr_t = torch.tensor(ytr)
    ae, kepala = bangun(c["latent"], c["dropout"])

    info = {"rekon_val_mse": None}
    if c["pralatih"]:
        info["rekon_val_mse"] = _pralatih(torch, ae, Xtr, Xva, c, g)
    ae.eval()

    def embed(X):
        with torch.no_grad():
            return ae.enc(X)

    # Tahap 2: kepala klasifikasi. Encoder di-fine-tune (bawaan) atau dibekukan (ablation A3).
    params = [{"params": kepala.parameters(), "lr": c["lr"] * 10}]
    if c["finetune"]:
        params.append({"params": ae.enc.parameters(), "lr": c["lr"] * 0.1})
    opt = torch.optim.AdamW(params, weight_decay=c["wd"])
    Ztr, Zva = embed(Xtr), embed(Xva)
    terbaik, simpan = -1.0, None
    for ep in range(c["epoch_kepala"]):
        kepala.train()
        if c["finetune"]:
            ae.enc.train()
        for i in torch.randperm(len(Xtr), generator=g).split(c["batch"]):
            z = ae.enc(Xtr[i]) if c["finetune"] else Ztr[i]
            loss = torch.nn.functional.cross_entropy(kepala(z), ytr_t[i])
            opt.zero_grad(); loss.backward(); opt.step()
        kepala.eval(); ae.eval()
        with torch.no_grad():
            zv = ae.enc(Xva) if c["finetune"] else Zva
            f1 = macro_f1(np.asarray(yva), kepala(zv).argmax(1).numpy())
        if f1 > terbaik:                               # pilih epoch dengan macro-F1 validasi terbaik
            terbaik, simpan = f1, (copy.deepcopy(kepala.state_dict()), copy.deepcopy(ae.state_dict()))
    kepala.load_state_dict(simpan[0]); ae.load_state_dict(simpan[1])
    kepala.eval(); ae.eval()

    class Klasifikasi(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.enc, self.kepala = ae.enc, kepala
        def forward(self, x):
            return self.kepala(self.enc(x))
    model = Klasifikasi().eval()
    with torch.no_grad():
        pred_val = model(Xva).argmax(1).numpy()
        pred_test = model(torch.tensor(Xte)).argmax(1).numpy() if final and Xte is not None else None
    info["n_parameter"] = int(sum(p.numel() for p in model.parameters()))
    return pred_val, pred_test, info, model


def ukur(model, keluar: Path) -> dict:
    """Ukuran ONNX (KB) dan latensi (ms) 1 jendela di CPU. Dijalankan lagi di laptop meja untuk angka resmi."""
    import torch
    hasil = {}
    x = torch.zeros(1, 6, 400)
    with torch.no_grad():
        for _ in range(20):
            model(x)
        t = []
        for _ in range(200):
            t0 = time.perf_counter(); model(x); t.append((time.perf_counter() - t0) * 1000)
    hasil["latensi_torch_ms_median"] = round(float(np.median(t)), 3)
    try:
        p = keluar / "model.onnx"
        torch.onnx.export(model, x, str(p), input_names=["getaran"], output_names=["logit"], opset_version=17)
        hasil["onnx_kb"] = round(p.stat().st_size / 1024, 1)
        try:
            import onnxruntime as ort
            s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            xn = x.numpy(); [s.run(None, {"getaran": xn}) for _ in range(20)]
            t = []
            for _ in range(200):
                t0 = time.perf_counter(); s.run(None, {"getaran": xn}); t.append((time.perf_counter() - t0) * 1000)
            hasil["latensi_onnx_ms_median"] = round(float(np.median(t)), 3)
        except ImportError:
            hasil["catatan"] = "onnxruntime tidak terpasang; latensi ONNX belum diukur"
    except Exception as e:  # noqa: BLE001
        hasil["catatan"] = f"ekspor ONNX gagal: {type(e).__name__}: {str(e)[:120]}"
    return hasil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument("--lipatan", type=int, default=None, help="lipatan uji 0/1/2 (bawaan: CAPSTONE_LIPATAN atau 0)")
    ap.add_argument("--ukur", action="store_true", help="ukur ukuran ONNX dan latensi")
    for k, v in BAWAAN.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "ya"), default=v)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()
    if a.lipatan is not None:
        os.environ["CAPSTONE_LIPATAN"] = str(a.lipatan)      # diwarisi subproses evaluate
    cfg = {k: getattr(a, k) for k in BAWAAN}
    t0 = time.time()
    pred_val, pred_test, info, model = latih_dan_prediksi(a.seed, a.final, cfg)
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
    # simpan hasil tiap seed (metrics.json hanya menyimpan seed terakhir) untuk tabel rerata ± sb
    hasil = json.loads((out / "metrics.json").read_text("utf-8"))
    per_seed = out / "per_seed.json"
    sdh = json.loads(per_seed.read_text("utf-8")) if per_seed.exists() else {}
    sdh[str(a.seed)] = {"metrik": hasil["metrik"], "durasi_detik": hasil.get("durasi_detik"), "konfigurasi": cfg, **info}
    per_seed.write_text(json.dumps(sdh, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.ukur:
        print(json.dumps(ukur(model, out)))


if __name__ == "__main__":
    main()
