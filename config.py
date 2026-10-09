from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GRASCII_DICT = DATA / "grascii" / "dictionary_preanniversary.tsv"

# Word-image canvas fed to the recognizer (CNN output width = 64 time steps)
IMG_H = 48
IMG_W = 256
