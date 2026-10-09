import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import WordDataset, collate
from src.manifest import read_manifest, rows_for_split
from src.model import CRNN
from src.utils import cer, greedy_decode
from src.vocab import Vocab


WEAK_CHARS = set("&|()ZVI C J~".replace(" ", ""))


def evaluate(model, loader, vocab, device):
    model.eval()
    refs, hyps = [], []

    with torch.no_grad():
        for x, _, _, labels in loader:
            lp = model(x.to(device)).cpu().numpy()

            for j, w in enumerate(labels):
                refs.append(w)
                hyps.append(greedy_decode(lp[:, j, :], vocab))

    return (
        cer(refs, hyps),
        sum(r == h for r, h in zip(refs, hyps)) / max(1, len(refs))
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-dir",
        default="data/phase2_real_full/manifest_open"
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=16
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    ap.add_argument(
        "--hidden",
        type=int,
        default=128
    )

    ap.add_argument(
        "--weak-weight",
        type=float,
        default=3.0
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=0
    )

    ap.add_argument(
        "--run-dir",
        default="outputs/phase2_oversample"
    )

    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    df, meta = read_manifest(a.data_dir)

    train_rows = rows_for_split(
        df,
        meta,
        "train",
        "grascii"
    )

    val_rows = rows_for_split(
        df,
        meta,
        "val",
        "grascii"
    )

    if len(train_rows) == 0:
        raise RuntimeError("No training rows found.")

    vocab = Vocab.from_labels(
        train_rows["grascii"]
    )

    print(
        f"device {device} | "
        f"alphabet ({len(vocab.chars)}): "
        f"{''.join(vocab.chars)}"
    )

    print(
        "Training samples:",
        len(train_rows)
    )

    # --------------------------------------------------------
    # Determine sample weights
    # --------------------------------------------------------

    weights = []

    weak_count = 0

    for label in train_rows["grascii"]:

        label = str(label)

        has_weak = any(
            c in WEAK_CHARS
            for c in label
        )

        if has_weak:
            weights.append(a.weak_weight)
            weak_count += 1
        else:
            weights.append(1.0)

    weights = torch.tensor(
        weights,
        dtype=torch.double
    )

    print(
        "Samples containing weak characters:",
        weak_count
    )

    print(
        "Weak-character sample weight:",
        a.weak_weight
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    tr = WordDataset(
        a.data_dir,
        "train",
        "grascii",
        vocab,
        augment=True,
        seed=a.seed
    )

    va = WordDataset(
        a.data_dir,
        "val",
        "grascii",
        vocab
    )

    # --------------------------------------------------------
    # Weighted sampler
    # --------------------------------------------------------

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_rows),
        replacement=True
    )

    tl = DataLoader(
        tr,
        batch_size=a.batch,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0
    )

    vl = DataLoader(
        va,
        batch_size=64,
        shuffle=False,
        collate_fn=collate,
        num_workers=0
    )

    print(
        f"train samples/epoch {len(tr)} | "
        f"val {len(va)} | "
        f"split_mode {meta['split_mode']}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CRNN(
        vocab.size,
        a.hidden
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=a.lr,
        weight_decay=1e-4
    )

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=a.lr,
        total_steps=a.epochs * len(tl)
    )

    ckpt = run / "recognizer.pt"

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    (run / "train_config.json").write_text(
        json.dumps(
            dict(
                vars(a),
                alphabet="".join(vocab.chars),
                n_train=len(tr),
                n_val=len(va),
                weak_chars="".join(sorted(WEAK_CHARS)),
                weak_samples=weak_count,
                manifest_meta=meta
            ),
            indent=2
        ),
        encoding="utf-8"
    )

    log = open(
        run / "train_log.csv",
        "w",
        newline=""
    )

    lw = csv.writer(log)

    lw.writerow(
        [
            "epoch",
            "train_loss",
            "val_cer",
            "val_exact",
            "seconds"
        ]
    )

    best = 9.9

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for ep in range(1, a.epochs + 1):

        tr.set_epoch(ep)

        model.train()

        t0 = time.time()
        total_loss = 0.0

        for x, tg, ln, _ in tl:

            lp = model(
                x.to(device)
            )

            T, B, _ = lp.shape

            loss = F.ctc_loss(
                lp,
                tg,
                torch.full(
                    (B,),
                    T,
                    dtype=torch.long
                ),
                ln,
                blank=0,
                zero_infinity=True
            )

            opt.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            opt.step()
            sched.step()

            total_loss += loss.item()

        v_cer, v_acc = evaluate(
            model,
            vl,
            vocab,
            device
        )

        flag = ""

        if v_cer <= best:

            best = v_cer

            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab.chars,
                    "target": "grascii",
                    "hidden": a.hidden
                },
                ckpt
            )

            flag = "  *saved*"

        elapsed = time.time() - t0

        lw.writerow(
            [
                ep,
                round(
                    total_loss / len(tl),
                    4
                ),
                round(v_cer, 4),
                round(v_acc, 4),
                round(elapsed, 1)
            ]
        )

        log.flush()

        print(
            f"epoch {ep:3d}  "
            f"loss {total_loss / len(tl):.3f}  "
            f"val grascii CER {v_cer:.3f}  "
            f"val exact {v_acc:.3f}  "
            f"({elapsed:.0f}s)"
            f"{flag}"
        )

    log.close()

    print()
    print(
        "best val CER:",
        round(best, 4)
    )

    print(
        "checkpoint:",
        ckpt
    )


if __name__ == "__main__":
    main()