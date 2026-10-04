"""Dictionary/rule baselines (UniDic, OpenJTalk) for the labeled splits."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

from .alignment import align_ruby
from .kanjidic import Kanjidic
from .normalization import contains_kanji, hira, is_kanji, is_trainable_kanji_text
from .rules import apply_variation

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip(): yield json.loads(line)


def load_polyphone_profile(path: str | Path, *, min_count: int = 20,
                           max_majority_share: float = 0.90,
                           cache_path: str | Path | None = None) -> set[str]:
    """Return characters with genuinely diverse supervised readings.

    This deliberately uses the training split only.  KANJIDIC candidate count
    is a dictionary property and marks almost every kanji as ambiguous; the
    profile instead measures empirical reading diversity in the corpus.
    """
    if cache_path is not None:
        cache = Path(cache_path)
        if cache.is_file():
            with cache.open(encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise ValueError(f"invalid polyphone profile cache: {cache}")
            return set(value)

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    paths = (path,) if isinstance(path, (str, Path)) else tuple(path)
    for source in paths:
        source_path = Path(source)
        handle = source_path.open(encoding="utf-8")
        rows = handle
        if tqdm is not None:
            rows = tqdm(handle, desc=f"polyphone profile: {source_path.name}",
                        unit="line", unit_scale=False, dynamic_ncols=True,
                        total=None, mininterval=1.0)
        try:
            for line in rows:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not contains_kanji(row.get("text", "")):
                    continue
                for char, reading, reliable in zip(row.get("text", ""),
                                                    row.get("B", []),
                                                    row.get("loss_mask", [])):
                    if reliable and is_kanji(char) and reading not in {"COPY", "UNK", ""}:
                        counts[char][reading] += 1
        finally:
            if rows is not handle:
                rows.close()
            handle.close()
    result = {
        char for char, readings in counts.items()
        if sum(readings.values()) >= min_count
        and len(readings) >= 2
        and max(readings.values()) / sum(readings.values()) < max_majority_share
    }
    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(sorted(result), ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return result


def _cer(ref: str, hyp: str) -> float:
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(1, len(ref))


@dataclass(frozen=True)
class _Token:
    """One surface/reading pair in the order it appears in the sentence."""

    surface: str
    reading: str


Tokenizer = Callable[[str], Sequence[_Token]]


def _unidic_tokenizer(unidic: str, reading_field: str = "kana") -> Tokenizer:
    """Tokenize with UniDic and return surface readings in hiragana.

    ``kana`` is the orthographic reading (トウキョウ) and matches the gold
    labels, which come from human ruby.  ``pron`` is the phonetic realization
    (トーキョー); scoring it against ruby gold penalizes every long vowel.
    """
    import fugashi

    tagger = fugashi.Tagger(f'-r nul -d "{unidic}"')
    base_field = {"kana": "kanaBase", "pron": "pronBase"}[reading_field]

    def tokenize(text: str) -> list[_Token]:
        tokens: list[_Token] = []
        for word in tagger(text):
            feature = getattr(word, "feature", None)
            reading = hira(getattr(feature, reading_field, "") or
                           getattr(feature, base_field, "") or "")
            tokens.append(_Token(str(word.surface), reading))
        return tokens

    return tokenize


def _pyopenjtalk_tokenizer() -> Tokenizer:
    """Tokenize with OpenJTalk's frontend, the reading source of pyopenjtalk."""
    import pyopenjtalk

    def tokenize(text: str) -> list[_Token]:
        tokens: list[_Token] = []
        for entry in pyopenjtalk.run_frontend(text):
            surface = str(entry.get("string", "") or "")
            reading = hira(str(entry.get("read", "") or entry.get("pron", "") or ""))
            if surface:
                tokens.append(_Token(surface, reading))
        return tokens

    return tokenize


def build_tokenizer(engine: str, unidic: str,
                    reading_field: str = "kana") -> Tokenizer:
    if engine == "unidic":
        return _unidic_tokenizer(unidic, reading_field)
    if engine == "pyopenjtalk":
        return _pyopenjtalk_tokenizer()
    raise ValueError(f"unknown baseline engine: {engine}")


def assign_char_readings(text: str, tokens: Sequence[_Token],
                         kanjidic: Kanjidic
                         ) -> tuple[list[str | None], list[tuple[int, int]]]:
    """Project token readings onto characters in the label representation.

    Returns the per-character hypothesis (``None`` where the tokenizer gave no
    usable reading, ``""`` for characters covered by a preceding word-level
    reading) and the character spans of kanji tokens that aligned, which the
    caller uses for token-level accounting.
    """
    hyp_chars: list[str | None] = [None] * len(text)
    aligned_spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        surface = token.surface
        start = text.find(surface, cursor)
        if start < 0:
            continue
        end = start + len(surface)
        reading = token.reading
        if reading and not any(is_kanji(ch) for ch in surface) and len(reading) == len(surface):
            # This covers UniDic's contextual particle pronunciations
            # (は/へ/を) and kana-only tokens without involving DP.
            for offset, value in enumerate(reading):
                hyp_chars[start + offset] = value
        elif reading and any(is_kanji(ch) for ch in surface):
            alignment = align_ruby(surface, reading, kanjidic)
            if alignment is not None and len(alignment.b) == len(surface):
                if alignment.a and alignment.a[0] in {"熟字訓", "特殊"} and len(surface) > 1:
                    hyp_chars[start] = reading
                    for offset in range(1, len(surface)):
                        hyp_chars[start + offset] = ""
                else:
                    for offset, (base, trunc, variation) in enumerate(
                            zip(alignment.b, alignment.c, alignment.d)):
                        hyp_chars[start + offset] = apply_variation(
                            base[:-trunc] if trunc else base, variation)
                aligned_spans.append((start, end))
        cursor = end
    return hyp_chars, aligned_spans


def _gold_char(row: dict, index: int) -> str | None:
    if not row.get("loss_mask", [])[index]:
        return None
    base = row["B"][index]
    if base in {"COPY", "UNK", ""}:
        return None
    trunc = int(row.get("C", [0] * len(row["text"]))[index])
    variation = row.get("D", ["無"] * len(row["text"]))[index]
    return apply_variation(base[:-trunc] if trunc else base, variation)


def _special_gold_maps(row: dict, length: int) -> tuple[dict[int, tuple[int, str]], set[int]]:
    """Return annotated word-level ruby spans in local character coordinates."""
    starts: dict[int, tuple[int, str]] = {}
    continuations: set[int] = set()
    a_values = row.get("A", [])
    special_types = {"熟字訓", "特殊"}
    for span in row.get("special_spans", []) or []:
        try:
            start = int(span.get("start", -1)); end = int(span.get("end", start))
            reading = str(span.get("reading", ""))
        except (AttributeError, TypeError, ValueError):
            continue
        if not reading or not (0 <= start < length) or end <= start:
            continue
        if start >= len(a_values) or a_values[start] not in special_types:
            continue
        clipped_end = min(length, end)
        starts[start] = (clipped_end, reading)
        continuations.update(range(start + 1, clipped_end))
    return starts, continuations


def evaluate(path: str | Path, kanjidic: Kanjidic, unidic: str,
             limit: int | None = None,
             polyphone_chars: set[str] | None = None,
             engine: str = "unidic",
             reading_field: str = "kana") -> dict[str, object]:
    tokenize = build_tokenizer(engine, unidic, reading_field)
    rows = iter_jsonl(path)
    if tqdm is not None:
        rows = tqdm(rows, desc=f"{engine} baseline", unit="row", dynamic_ncols=True)
    total = ruby = assembled_ok = aligned = 0
    cer_sum = cer_tokens = 0.0
    poly_total = poly_ok = 0
    for row in rows:
        if limit is not None and total >= limit:
            break
        text = row.get("text", "")
        if not is_trainable_kanji_text(
                text, has_explicit_reading=bool(row.get("special_spans"))):
            continue
        gold_chars = [_gold_char(row, i) for i in range(len(text))]
        special_starts, special_continuations = _special_gold_maps(row, len(text))
        hyp_chars, aligned_spans = assign_char_readings(text, tokenize(text), kanjidic)
        for start, end in aligned_spans:
            special_token = special_starts.get(start)
            if special_token is not None and special_token[0] == end:
                # The gold special span is a single output unit even though its
                # surface contains multiple characters.
                ref = special_token[1]
                hyp = "".join(x or "" for x in hyp_chars[start:end])
                cer_sum += _cer(ref, hyp); cer_tokens += 1
            elif all(gold_chars[index] is not None for index in range(start, end)):
                ref = "".join(gold_chars[start:end])
                hyp = "".join(x or "" for x in hyp_chars[start:end])
                cer_sum += _cer(ref, hyp); cer_tokens += 1
            aligned += sum(gold_chars[index] is not None for index in range(start, end))
        for index, (gold, hyp) in enumerate(zip(gold_chars, hyp_chars)):
            if index in special_starts:
                end, reading = special_starts[index]
                gold = reading
                hyp = hyp_chars[index]
                # Continuation characters are represented by empty strings in
                # the span-aware CER below, never as raw surface characters.
            elif index in special_continuations:
                continue
            if gold is None:
                continue
            ruby += 1
            if hyp is not None:
                assembled_ok += int(hyp == gold)
            # Profile-based polyphone scores are supplied by the model eval;
            # the baseline reports the same character-level eligible count.
            if polyphone_chars is not None and text[index] in polyphone_chars:
                poly_total += 1
                poly_ok += int(hyp == gold) if hyp is not None else 0
        total += 1
    return {
        "sentences": total,
        "ruby_positions": ruby,
        "aligned_positions": aligned,
        "assembled_reading_accuracy": assembled_ok / max(1, ruby),
        "polyphone_accuracy": poly_ok / max(1, poly_total),
        "polyphone_positions": poly_total,
        "token_cer": cer_sum / max(1, cer_tokens),
        "fully_aligned_tokens": int(cer_tokens),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/labeled_full/test.jsonl")
    p.add_argument("--kanjidic", default="resources/kanjidic2.xml.gz")
    p.add_argument("--unidic", default="resources/unidic")
    p.add_argument("--limit", type=int)
    p.add_argument("--engine", choices=("unidic", "pyopenjtalk"), default="unidic")
    p.add_argument("--reading-field", choices=("kana", "pron"), default="kana",
                   help="UniDic reading field; kana matches ruby gold")
    p.add_argument("--polyphone-train")
    p.add_argument("--polyphone-min-count", type=int, default=20)
    p.add_argument("--polyphone-max-majority", type=float, default=0.90)
    p.add_argument("--polyphone-cache", help="cache the profile to avoid rescanning the training JSONL")
    p.add_argument("--output", "-o")
    a = p.parse_args()
    profile = (load_polyphone_profile(
        a.polyphone_train, min_count=a.polyphone_min_count,
        max_majority_share=a.polyphone_max_majority,
        cache_path=a.polyphone_cache,
    ) if a.polyphone_train else None)
    result = evaluate(a.data, Kanjidic.from_xml(a.kanjidic), a.unidic,
                      a.limit, profile, engine=a.engine,
                      reading_field=a.reading_field)
    result["engine"] = a.engine
    if a.engine == "unidic":
        result["reading_field"] = a.reading_field
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if a.output:
        out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved: {out}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
