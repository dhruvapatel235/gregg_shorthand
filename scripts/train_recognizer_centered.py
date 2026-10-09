"""
Trains the CNN-BiGRU-CTC recognizer.
  --target grascii : image -> Grascii string          (Branch B)
  --target english : image -> English letters         (Branch A)
Alphabet = characters of the TRAINING labels only. Saves checkpoint + config + per-epoch log in
outputs/<target>/ so every run is reproducible.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUT
from src.dataset_centered import WordDataset, collate
from src.manifest import read_manifest, rows_for_split
from src.model import CRNN
from src.utils import cer, greedy_decode
from src.vocab import Vocab


def evaluate(model, loader, vocab, device):
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for x, _, _, labels in loader:
            lp = model(x.to(device)).cpu().numpy()
            for j, w in enumerate(labels):
                refs.append(w)
                hyps.append(greedy_decode(lp[:, j, :], vocab))
    return cer(refs, hyps), sum(r == h for r, h in zip(refs, hyps)) / max(1, len(refs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--target", choices=["grascii", "english"], default="grascii")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=1, help="repeat the train set N times per epoch (small datasets)")
    ap.add_argument("--val-noise", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", help="default: outputs/<target>")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run = Path(a.run_dir or OUT / a.target)
    run.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df, meta = read_manifest(a.data_dir)
    train_rows = rows_for_split(df, meta, "train", a.target)
    if len(train_rows) == 0:
        sys.exit(f"No training rows with a '{a.target}' label. Run prepare_manifest.py / audit_data.py.")
    vocab = Vocab.from_labels(train_rows[a.target])
    print(f"device {device} | target {a.target} | alphabet ({len(vocab.chars)}): {''.join(vocab.chars)}")

    tr = WordDataset(a.data_dir, "train", a.target, vocab, augment=True, repeat=a.repeat, seed=a.seed, aug_strength=0.5)
    va = WordDataset(a.data_dir, "val", a.target, vocab, noise=a.val_noise)
    tl = DataLoader(tr, a.batch, shuffle=True, collate_fn=collate, num_workers=a.workers)
    vl = DataLoader(va, 64, collate_fn=collate)
    print(f"train samples/epoch {len(tr)} | val {len(va)} | split_mode {meta['split_mode']}")

    model = CRNN(vocab.size, a.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl))
    ckpt = run / "recognizer.pt"
    (run / "train_config.json").write_text(json.dumps(
        dict(vars(a), alphabet="".join(vocab.chars), n_train=len(tr), n_val=len(va), manifest_meta=meta),
        indent=2), encoding="utf-8")
    log = open(run / "train_log.csv", "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["epoch", "train_loss", "val_cer", "val_exact", "seconds"])
    best = 9.9
    for ep in range(1, a.epochs + 1):
        tr.set_epoch(ep)
        model.train()
        t0, tot = time.time(), 0.0
        for x, tg, ln, _ in tl:
            lp = model(x.to(device))
            T, B, _ = lp.shape
            loss = F.ctc_loss(lp, tg, torch.full((B,), T, dtype=torch.long), ln, blank=0, zero_infinity=True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item()
        v_cer, v_acc = evaluate(model, vl, vocab, device)
        flag = ""
        if v_cer <= best:
            best = v_cer
            torch.save({"model": model.state_dict(), "vocab": vocab.chars, "target": a.target,
                        "hidden": a.hidden}, ckpt)
            flag = "  *saved*"
        lw.writerow([ep, round(tot / len(tl), 4), round(v_cer, 4), round(v_acc, 4), round(time.time() - t0)])
        log.flush()
        print(f"epoch {ep:3d}  loss {tot/len(tl):.3f}  val {a.target} CER {v_cer:.3f}  val exact {v_acc:.3f}"
              f"  ({time.time()-t0:.0f}s){flag}")
    print("best val CER:", round(best, 4), "->", ckpt)


if __name__ == "__main__":
    main()
