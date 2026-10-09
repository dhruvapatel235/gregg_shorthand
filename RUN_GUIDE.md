# Gregg Shorthand → English: recognition + Grascii + context-aware language model

```
page image / PDF
   -> line & word segmentation
   -> CNN + BiGRU + CTC recognizer          (target = GRASCII  [Branch B]   or   English letters [Branch A])
   -> lexicon candidates (CTC-scored)       Grascii "TH)" -> their | there   (same outline = real ambiguity)
   -> language model (N-gram / Transformer) picks the word that fits the sentence
   -> English text   (+ the recognized Grascii as a separate stage-1 output)
```
**Branch B (default, `--target grascii`)**: the model reads the outline into Grascii first; English is a separate, separately-evaluated step. Same-outline words (hair/hare, to/too, their/there …) get the *same* recognizer score, so only context can separate them – this is the research question.
**Branch A (`--target english`)**: the model outputs English letters directly. Same code, same evaluation, so you can compare A vs B in your report.

## Folder structure
```
gregg_shorthand_nlp/
├── RUN_GUIDE.md
├── requirements.txt
├── config.py                          image size, paths
├── data/
│   ├── grascii/
│   │   ├── dictionary_preanniversary.tsv   15,832 words with Grascii (from the Grascii project, MIT)
│   │   ├── brief_forms_extra.tsv           EMPTY template: add missing common words (see Limitations)
│   │   └── LICENSE_grascii.txt
│   └── sample_corpus.txt
├── scripts/
│   ├── make_toy_dataset.py        fake shorthand images + toy corpus (pipeline test only, NOT real Gregg)
│   ├── prepare_manifest.py        YOUR images -> manifest.csv, lexicons, Grascii spec (deterministic splits)
│   ├── audit_data.py              checks files/labels/duplicates/leakage/alphabet + sample sheet picture
│   ├── download_corpus.py         (optional) big English corpus (NLTK Brown) for N-gram / test sentences
│   ├── build_sentence_data.py     sentence-level test set + demo page
│   ├── train_recognizer.py        trains CNN-BiGRU-CTC (--target grascii | english)
│   ├── evaluate.py                Grascii/English CER, top-k, WER, ambiguity accuracy, ablation, error CSV
│   ├── transcribe.py              MAIN: shorthand document -> English (+ Grascii)
│   └── import_grascii_dictionary.py   rebuild the dictionary file from `pip install grascii` (optional)
├── src/                           vocab, manifest, grascii, model, dataset, decode, lexicon, candidates,
│                                  lm_ngram, lm_transformer, resolver, segment, pipeline
└── outputs/                       audit reports, checkpoints, logs, results, transcriptions
```

## 1. Setup (once, Windows PowerShell; use `source venv/bin/activate` on Mac/Linux)
```
cd D:\gregg_shorthand_nlp
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.9+. The Transformer (distilgpt2) downloads automatically the first time it is used. GPU optional.

## 2. Self-test with toy data (≈ 10 min on CPU) – run this first
The toy "shorthand" is invented strokes drawn from real Grascii strings, so it only proves the code works.
```
python scripts/make_toy_dataset.py
python scripts/audit_data.py --data-dir data/toy
python scripts/build_sentence_data.py --data-dir data/toy --corpus data/toy/corpus.txt --max-sentences 60
python scripts/train_recognizer.py --data-dir data/toy --target grascii --epochs 25 --repeat 6
python scripts/evaluate.py --data-dir data/toy --target grascii --noise 1.0 --ablation
python scripts/transcribe.py --input outputs/demo_page.png --data-dir data/toy --target grascii --lm ngram --ngram-corpus data/toy/corpus_train.txt
```
Branch A on the same data: repeat the `train_recognizer.py` / `evaluate.py` lines with `--target english`.

## 3. Real run
**Step 1 – data.** Put your real Gregg word images in one folder; each file named after its English word (`about.png`, `about_2.png`; `00012_about.png` also works) – or give `--csv` with columns `path,word` (optional: `grascii,source_id,writer_id,notes`). *You must verify the image source, licence and that the word matches the image* – this project cannot do that for you.

**Step 2 – manifest (Grascii is attached automatically from the bundled dictionary).**
```
python scripts/prepare_manifest.py --images D:\data\gregg_words --out data/gregg --mode closed
```
`--mode closed`: one image per word, fixed lexicon (val/test are degraded copies – optimistic, say so in the report).
`--mode open --group-by english` (or `source_id`): test words/sources never seen in training – strict test.

**Step 3 – audit. Do not skip.**
```
python scripts/audit_data.py --data-dir data/gregg
```
Read `outputs/audit_gregg.json` and LOOK at `outputs/audit_gregg_samples.png` (image + English + Grascii). Fix what it reports (unreadable files, wrong labels, words without Grascii, duplicate images with different labels, leakage).

**Step 4 – train (GPU 30–60 min; CPU: many hours, try `--epochs 15` first).**
```
python scripts/train_recognizer.py --data-dir data/gregg --target grascii --epochs 60
python scripts/train_recognizer.py --data-dir data/gregg --target english --epochs 60     (Branch A, for comparison)
```
Alphabet comes from the training labels only. Saved in `outputs/<target>/`: `recognizer.pt`, `train_config.json`, `train_log.csv`.

**Step 5 – sentence test data + evaluation.**
```
python scripts/download_corpus.py
python scripts/build_sentence_data.py --data-dir data/gregg --corpus data/corpus.txt --max-sentences 300
python scripts/evaluate.py --data-dir data/gregg --target grascii --ablation
python scripts/evaluate.py --data-dir data/gregg --target english --ablation
```
Reported separately: (1) recognizer CER/exact in its own target (Grascii or English), (2) top-1/top-k candidate accuracy, (3) English WER, word accuracy, ambiguity-resolution accuracy split into *same-outline* vs *recognition-uncertain*, sentence exact match – for Recognition-only / Dictionary / N-gram / Transformer, plus an LM-weight ablation. `outputs/<target>/error_analysis.csv` lists every error and whether context helped or hurt.

**Step 6 – transcribe your document (the product).**
```
python scripts/transcribe.py --input D:\scans\page1.png --data-dir data/gregg --target grascii --ngram-corpus data/corpus.txt
```
Writes `outputs/grascii/transcription.txt` (English), `.grascii.txt` (stage-1 Grascii), `.details.json` (candidates per word), `segmentation_page*.png` (red boxes = how words were cut; check this first). PDFs work. `--lm none|ngram|transformer`, `--beta` = how much the LM is trusted, `--word-gap` 0.6–1.3 if words are glued/split.

## What is tested, what is not
Tested here on toy data: manifest/audit, training and evaluation of both branches, N-gram resolution of same-outline words, page → text. **Not tested:** real Gregg images, the Transformer LM (needs a download from Hugging Face; the code is unchanged from your earlier version), PDF input. Toy numbers (near-perfect) only show the code runs: toy sentences are templated and easy.

## Limitations (put these in your report)
- **Missing common words.** The bundled dictionary has no entries for many very frequent words that Gregg writes as brief forms (a, and, in, is, was, we, I, see, be …). Those words can never be output until you add them to `data/grascii/brief_forms_extra.tsv` (`GRASCII<TAB>word`) with real Grascii from your manual/dataset. Sentence test data is built from runs of covered words only, so on real corpora the test set will be smaller than the corpus suggests.
- **Phrases** in the Grascii dictionary (e.g. "had been") are not used; word level only.
- **Grascii conventions** are taken from the Grascii project's dictionary (A–Z plus `' ^ ~ _ & ( ) | , . \`). Recognition treats it as plain characters; check `grascii_spec.json` against your data. Grascii recognition is not English translation – evaluated separately.
- **One image per word** (printed dictionary) ⇒ results from `closed` mode and from sentences assembled out of word images are optimistic and say little about handwritten notebooks; those need extra annotated data (fine-tune via `--csv`).
- Segmentation is classical image processing: clean dark-on-white, flat, 150–300 dpi scans work best.
- Closed vocabulary: names and unknown words are mapped to the nearest dictionary entry.

## How this follows your plan document
Manifest + deterministic splits + audit (§4, §8) → `prepare_manifest.py`, `audit_data.py` · CNN–BiGRU–CTC (§3) → `src/model.py` · vocabulary from training labels, no hard-coded class count (§B1) → `src/vocab.py` · Branch A / Branch B with one switch (§5) → `--target` · Grascii spec + normalisation (§B2) → `src/grascii.py`, `grascii_spec.json` · context module evaluated as a separate addition with baselines and ablation (§A2, §B2) → `evaluate.py` · logs/config/checkpoints (§8 step 32) → `outputs/<target>/` · toy data for code-path tests only (§4) → `make_toy_dataset.py`.
