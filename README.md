# kashi-g2p

Japanese grapheme-to-phoneme (reading) estimation: given Japanese text with
kanji, produce the full kana reading. Candidate readings for word spans are
generated from dictionaries and rules, then scored and globally decoded by a
compact model trained end-to-end. The whole runtime ships as ONNX and runs on
CPU.

This repository contains the `kashi_g2p` Python package; `pip install -e .`
exposes it as the `kashi_g2p` module.

## How it works

1. **Span candidates** — for each input, a composite provider builds a lattice
   of candidate (surface, reading) edges from a unified lexicon pack (JMdict,
   KANJIDIC2, UniDic, and a Yomogi-derived compound dictionary), alphanumeric
   and counter-compound rules, and COPY/fallback edges.
2. **Scoring** — a small transformer over character embeddings (plus
   character-type features) scores every candidate edge in context.
3. **Global decoding** — a dynamic program selects the best consistent path
   over the whole sentence (long inputs use deterministic windows that never
   bisect a candidate edge), with per-path confidence.

The compiled lexicon resource (`lexicon.txz`) and the standalone ONNX runtime
are distributed with the release assets, not in this repository.

## Installation

```bash
pip install -e .
```

Python 3.10+ and PyTorch 2.0+ are required for the training/inference API.

## Quickstart (ONNX runtime)

Download the release assets (`g2p_onnx_runtime.py`, `model.onnx` or
`model.fp16.onnx`, `resources/lexicon.txz`) into one folder, then:

```python
from g2p_onnx_runtime import G2POnnxRuntime

rt = G2POnnxRuntime(model_file="model.onnx")
result = rt.predict("今日は良い天気ですね。")
print(result["reading"])   # きょうはいいてんきですね。
```

The runtime uses CPU by default; pass `prefer_dml=True` on Windows machines
with DirectML for GPU execution.

## Quickstart (Python API, from a training checkpoint)

```python
from kashi_g2p.pipeline import V4Pipeline

pipeline = V4Pipeline.from_checkpoint("checkpoints/best_model.pt")
result = pipeline.predict("今日は良い天気ですね。")
print(result["reading"])
print(result["confidence"])
```

`predict` returns the reading, the chosen candidate edges (with per-edge
source and lock state), a path score, and a confidence value.

## Training

Training is config-driven (`python -m kashi_g2p.train --config your_config.yaml`);
YAML configs are not shipped with the package. A minimal training config looks
like:

```yaml
model:
  hidden: 512
  layers: 6
data:
  - artifacts/labeled/train.jsonl
save: checkpoints/my_run
```

Dataset channels, provider options, batch/precision settings and checkpoint
selection are all declared in the config; `kashi_g2p/config.yaml` documents
the provider defaults. Evaluation against a labeled JSONL split:

```bash
python -m kashi_g2p.evaluate --checkpoint ... --data ...
```

## Lexicon resource

`lexicon.txz` is a single-file, tab-separated, xz-compressed bundle
(`pack` / `prior` / `vocab` sections) compiled from public dictionary
sources. It is **not** part of this repository — grab it from the release
assets. The bundle contains no IDS/component data; see
`THIRD_PARTY_NOTICES.md` for the licence picture and the deliberate
exclusions.

## Tests

```bash
pip install -e .
python -m pytest tests -q
```

## Licence

- Project code and model weights: **Apache-2.0** (see `LICENSE`).
- The compiled lexicon resource embeds data derived from
  JMdict/KANJIDIC2 (CC BY-SA 4.0), UniDic (BSD), and MIT-licensed
  dictionary projects; it is therefore distributed under CC BY-SA 4.0.
  Full attributions, licence texts, and the list of deliberately excluded
  data: `THIRD_PARTY_NOTICES.md`.
