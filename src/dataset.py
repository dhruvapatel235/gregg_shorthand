import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import IMG_H, IMG_W
from src.manifest import read_manifest, rows_for_split


def load_gray(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return img


def to_ink(gray):
    """Return image with ink = bright, background = dark (uint8)."""
    return 255 - gray if gray.mean() > 127 else gray


def degrade(ink, rng, s=1.0):
    """Random rotation/shear/scale/thickness/noise. s = strength (0..1+)."""
    h, w = ink.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3) * s, 1.0)
    M[0, 1] += rng.uniform(-0.15, 0.15) * s
    M[0, 0] *= 1.0 + rng.uniform(-0.1, 0.1) * s
    out = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    r = rng.rand()
    if r < 0.3 * s:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8))
    elif r < 0.5 * s:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))
    if rng.rand() < 0.3 * s:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    out = out.astype(np.float32) + rng.normal(0, 8 * s, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_to_ink(ink, pad=3):
    """Tight crop around the strokes (+pad). Applied to training images AND to words cut from a
    page, so both look the same to the recognizer regardless of scan margins."""
    ys, xs = np.where(ink > 40)
    if len(xs) == 0:
        return ink
    return ink[max(0, ys.min() - pad):ys.max() + pad + 1, max(0, xs.min() - pad):xs.max() + pad + 1]


def to_canvas(ink):
    """Crop to ink, resize to height IMG_H keeping aspect ratio, pad/squash to IMG_W. float32 (H, W) in 0..1."""
    ink = crop_to_ink(ink)
    h, w = ink.shape
    nw = min(max(8, int(round(w * IMG_H / h))), IMG_W)
    interp = cv2.INTER_AREA if (IMG_H < h or nw < w) else cv2.INTER_LINEAR
    res = cv2.resize(ink, (nw, IMG_H), interpolation=interp)
    canvas = np.zeros((IMG_H, IMG_W), np.uint8)
    canvas[:, :nw] = res
    return canvas.astype(np.float32) / 255.0


def load_word_image(path, seed=None, noise=0.0):
    ink = to_ink(load_gray(path))
    if noise > 0 and seed is not None:
        ink = degrade(ink, np.random.RandomState(seed), noise)
    return ink


class WordDataset(Dataset):
    """Manifest-driven dataset. target = 'english' | 'grascii' (what the model must output)."""

    def __init__(self, manifest, split, target, vocab, augment=False, noise=0.0, repeat=1, seed=0, aug_strength=1.0):
        df, meta = read_manifest(manifest)
        df = rows_for_split(df, meta, split, target)
        df = df[df[target].map(vocab.valid)].reset_index(drop=True)
        self.paths, self.labels = list(df["abs_path"]), list(df[target])
        self.vocab, self.augment, self.noise = vocab, augment, noise
        self.repeat, self.seed, self.epoch = repeat, seed, 0
        self.aug_strength = aug_strength
        self._cache = {}

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.labels) * self.repeat

    def __getitem__(self, k):
        i = k % len(self.labels)
        if i not in self._cache:
            self._cache[i] = to_ink(load_gray(self.paths[i]))
        ink = self._cache[i]
        if self.augment:
            ink = degrade(ink, np.random.RandomState((self.seed * 1_000_003 + self.epoch * 7919 + k) % 2**32), self.aug_strength)
        elif self.noise > 0:
            ink = degrade(ink, np.random.RandomState(1000 + i), self.noise)
        x = torch.from_numpy(to_canvas(ink)).unsqueeze(0)
        return x, torch.tensor(self.vocab.encode(self.labels[i]), dtype=torch.long), self.labels[i]


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    return xs, targets, lengths, [b[2] for b in batch]
