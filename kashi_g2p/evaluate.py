"""Strict JSONL evaluation for the UniSpan v4 checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import time
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .normalization import is_kanji, normalize_reading, normalize_surface
from .rules import apply_variation

from .data import build_gold_edges, gold_parts as _gold_parts, iter_jsonl
from .pipeline import V4Pipeline


def _editops(reference: str, hypothesis: str) -> list[tuple[str, int, int]]:
    """Stable Levenshtein edit script without an optional dependency."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    costs = [[0] * cols for _ in range(rows)]
    steps = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0] = i; steps[i][0] = "delete"
    for j in range(1, cols):
        costs[0][j] = j; steps[0][j] = "insert"
    order = {"equal": 0, "replace": 1, "delete": 2, "insert": 3}
    for i in range(1, rows):
        for j in range(1, cols):
            diagonal = (costs[i - 1][j - 1]
                        + (reference[i - 1] != hypothesis[j - 1]))
            candidates = [(diagonal, "equal" if reference[i - 1] == hypothesis[j - 1]
                           else "replace"),
                          (costs[i - 1][j] + 1, "delete"),
                          (costs[i][j - 1] + 1, "insert")]
            costs[i][j], steps[i][j] = min(
                candidates, key=lambda item: (item[0], order[item[1]]))
    result = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        operation = steps[i][j]
        if operation in {"equal", "replace"}:
            i -= 1; j -= 1
        elif operation == "delete":
            i -= 1
        else:
            j -= 1
        if operation != "equal":
            result.append((operation, i, j))
    result.reverse()
    return result


_YOTSUGANA_FOLD = str.maketrans({
    "ぢ": "じ", "づ": "ず",
    "ヂ": "ジ", "ヅ": "ズ",
})

_VOWEL_CHARS = {
    "a": "あかさたなはまやらわがざだばぱぁゃゎアカサタナハマヤラワガザダバパァャヮ",
    "i": "いきしちにひみりぎじぢびぴぃイキシチニヒミリギジヂビピィ",
    "u": "うくすつぬふむゆるぐずづぶぷぅゅウクスツヌフムユルグズヅブプゥュ",
    "e": "えけせてねへめれげぜでべぺぇエケセテネヘメレゲゼデベペェ",
    "o": "おこそとのほもよろをごぞどぼぽぉょオコソトノホモヨロヲゴゾドボポォョ",
}
_CHAR_TO_VOWEL = {
    ch: vowel for vowel, chars in _VOWEL_CHARS.items() for ch in chars
}
_ELONGATION_MAP = {
    "a": {"あ", "ア", "ー"},
    "i": {"い", "イ", "ー"},
    "u": {"う", "ウ", "ー"},
    "e": {"い", "え", "イ", "エ", "ー"},
    "o": {"う", "お", "ウ", "オ", "ー"},
}


def fold_notation(text: str) -> str:
    """Normalize equivalent spelling conventions (yotsugana, long vowels).

    1. Yotsugana merger: ぢ->じ, づ->ず (and katakana variants).
    2. Long vowel unification: merges elongation markers (ー vs vowel prolongation)
       while preserving exact 1:1 character length.
    """
    if not text:
        return text
    text = text.translate(_YOTSUGANA_FOLD)
    result: list[str] = []
    for char in text:
        if result:
            prev_vowel = _CHAR_TO_VOWEL.get(result[-1])
            if prev_vowel and char in _ELONGATION_MAP[prev_vowel]:
                result.append("ー")
                continue
        result.append(char)
    return "".join(result)


def _bucket(char: str) -> str:
    if is_kanji(char):
        return "kanji"
    if char.isascii() and (char.isalpha() or char.isdigit()):
        return "alnum"
    if "\u3040" <= char <= "\u30ff":
        return "kana"
    return "symbol"

def _value(row: dict[str, Any], key: str, index: int, default: Any) -> Any:
    values = row.get(key) or ()
    return values[index] if index < len(values) else default


def _hypothesis_parts(text: str, prediction: dict[str, Any]) -> list[str]:
    parts = [""] * len(text)
    for edge in prediction["edges"]:
        start = int(edge["start"]); end = int(edge["end"])
        if not 0 <= start < end <= len(text):
            raise ValueError(f"decoded edge is outside input: {(start, end)}")
        parts[start] = str(edge["reading"])
    return parts


def _constraint_coverage(row: dict[str, Any], edges: Sequence[Any]) -> tuple[bool, int, int]:
    """Measure exact gold constraint reachability without injecting any edge."""
    _unchanged, compatible = build_gold_edges(
        row, edges, inject_missing=False, coarsen_segmentation=False)
    constraints: list[tuple[int, int, str]] = []
    text = normalize_surface(str(row.get("text", "")))
    special_positions = set()
    for span in row.get("special_spans", ()) or ():
        try:
            start = int(span["start"]); end = int(span["end"])
            reading = normalize_reading(str(span["reading"]))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= start < end <= len(text) and reading:
            constraints.append((start, end, reading))
            special_positions.update(range(start, end))
    for index in range(len(text)):
        if index in special_positions or not _value(row, "loss_mask", index, False):
            continue
        base = str(_value(row, "B", index, ""))
        if not base or base in {"COPY", "UNK"}:
            continue
        try:
            trim = max(0, int(_value(row, "C", index, 0)))
        except (TypeError, ValueError):
            trim = 0
        reading = apply_variation(base[:-trim] if trim else base,
                                  str(_value(row, "D", index, "無")))
        constraints.append((index, index + 1, reading))
    covered = sum(any(edge.start == start and edge.end == end
                      and edge.reading == reading for edge in edges)
                  for start, end, reading in constraints)
    return _compatible_path_exists(edges, compatible, len(text)), covered, len(constraints)

def _compatible_path_exists(edges: Sequence[Any], compatible: Sequence[bool],
                            length: int) -> bool:
    reachable = [False] * (length + 1)
    reachable[0] = True
    for start in range(length):
        if not reachable[start]:
            continue
        for edge, allowed in zip(edges, compatible):
            if allowed and edge.start == start and start < edge.end <= length:
                reachable[edge.end] = True
    return reachable[length]


def _accumulate_edits(reference: str, hypothesis: str,
                      owners: Sequence[int], text: str,
                      allowed_positions: set[int] | None,
                      errors: Counter[str]) -> int:
    count = 0
    for _operation, source, _destination in _editops(reference, hypothesis):
        if not owners:
            continue
        owner = owners[min(source, len(owners) - 1)]
        if allowed_positions is not None and owner not in allowed_positions:
            continue
        errors[_bucket(text[owner])] += 1
        count += 1
    return count


def _supervised_runs(gold_parts: Sequence[str], hypothesis_parts: Sequence[str],
                     reliable: Sequence[bool], text: str,
                     errors: Counter[str], *,
                     fold: bool = False,
                     allowed_positions: set[int] | None = None) -> tuple[int, int]:
    """Score reliable contiguous runs so masked gaps cannot create edit ops.

    When ``allowed_positions`` is given, edit operations owned by a surface
    position outside the set are ignored (used to drop gold-exempt positions
    from the strict CER numerator without altering the strict metric itself).
    """
    operations = characters = 0
    start = 0
    while start < len(text):
        if not reliable[start]:
            start += 1
            continue
        end = start + 1
        while end < len(text) and reliable[end]:
            end += 1
        reference = "".join(gold_parts[start:end])
        hypothesis = "".join(hypothesis_parts[start:end])
        if fold:
            reference = fold_notation(reference)
            hypothesis = fold_notation(hypothesis)
        owners = [index for index in range(start, end)
                  for _ in gold_parts[index]]
        operations += _accumulate_edits(
            reference, hypothesis, owners, text, allowed_positions, errors)
        characters += len(reference)
        start = end
    return operations, characters


def _load_exempt(path: str | Path | None) -> dict[int, set[int]]:
    """Load the gold-exempt manifest into {record_index: {char_index, ...}}.

    Keys mirror ``scripts/dump_gold_review_20k.py`` enumeration: ``record_index``
    is the position in the non-blank JSONL stream and ``char_index`` is the
    surface (B[]) index that owns the exempt gold reading.
    """
    if not path:
        return {}
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    exempt: dict[int, set[int]] = defaultdict(set)
    for entry in entries:
        record = int(entry["record_index"])
        char = int(entry["char_index"])
        exempt[record].add(char)
    return dict(exempt)


def evaluate_rows(pipeline: V4Pipeline, rows: Iterable[dict[str, Any]], *,
                  batch_size: int = 16, limit: int | None = None,
                  fast: bool = False,
                  exempt: dict[int, set[int]] | None = None) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    bucket_errors: Counter[str] = Counter()
    bucket_errors_folded: Counter[str] = Counter()
    bucket_chars: Counter[str] = Counter()
    pending: list[dict[str, Any]] = []
    exempt = exempt or {}

    def consume(chunk: list[dict[str, Any]]) -> None:
        texts = [normalize_surface(str(row.get("text", ""))) for row in chunk]
        predictions = pipeline.predict_batch(texts, compute_posterior=not fast)
        for row, prediction in zip(chunk, predictions):
            text, gold_parts, reliable = _gold_parts(row)
            hypothesis_parts = _hypothesis_parts(text, prediction)
            reference = "".join(gold_parts)
            hypothesis = "".join(hypothesis_parts)
            owners = [index for index, part in enumerate(gold_parts)
                      for _ in part]
            reliable_positions = {index for index, value in enumerate(reliable) if value}
            for index in reliable_positions:
                bucket_chars[_bucket(text[index])] += len(gold_parts[index])
            full_errors: Counter[str] = Counter()
            full_ops = _accumulate_edits(reference, hypothesis, owners, text,
                                         None, full_errors)
            supervised_errors: Counter[str] = Counter()
            supervised_ops, supervised_chars = _supervised_runs(
                gold_parts, hypothesis_parts, reliable, text, supervised_errors, fold=False)
            bucket_errors.update(supervised_errors)

            supervised_errors_folded: Counter[str] = Counter()
            supervised_ops_folded, _ = _supervised_runs(
                gold_parts, hypothesis_parts, reliable, text, supervised_errors_folded, fold=True)
            bucket_errors_folded.update(supervised_errors_folded)

            # Exempt-aware supervised metric (§18.9 X manifest): identical to the
            # strict path except positions in the exempt set are removed from both
            # numerator (edits) and denominator (reference chars).  The strict
            # numbers above are left untouched.
            exempt_chars = exempt.get(int(row["_record_index"]), set())
            row_exempt = exempt_chars & reliable_positions
            if row_exempt:
                allowed = reliable_positions - row_exempt
                exempt_ops, _ = _supervised_runs(
                    gold_parts, hypothesis_parts, reliable, text, Counter(),
                    fold=False, allowed_positions=allowed)
                exempt_char_count = sum(len(gold_parts[i]) for i in row_exempt)
                totals["supervised_edits_exempt"] += exempt_ops
                totals["supervised_chars_exempt"] += supervised_chars - exempt_char_count
                totals["exempt_positions_applied"] += len(row_exempt)
                # sentence-exact modulo exempt: forgive the dual_valid positions
                # (replace their hypothesis with gold) and require the full reading
                # string to match otherwise -- consistent with the strict
                # ``reference == hypothesis`` semantics, unlike the earlier
                # ``full_ops == supervised_ops`` proxy which compared two edit
                # counts produced by different alignments across the
                # reliable/unreliable boundary.
                masked_hyp = "".join(
                    gold_parts[i] if i in row_exempt
                    else (hypothesis_parts[i] if i < len(hypothesis_parts) else "")
                    for i in range(len(gold_parts)))
                totals["sentence_exact_exempt"] += int(reference == masked_hyp)
            else:
                totals["supervised_edits_exempt"] += supervised_ops
                totals["supervised_chars_exempt"] += supervised_chars
                totals["sentence_exact_exempt"] += int(reference == hypothesis)

            provider_edges = pipeline.provider.build(text)
            path_covered, constraints_covered, constraint_total = (
                _constraint_coverage(row, provider_edges))
            totals["sentences"] += 1
            totals["sentence_exact"] += int(reference == hypothesis)
            totals["sentence_exact_folded"] += int(fold_notation(reference) == fold_notation(hypothesis))
            totals["full_edits"] += full_ops
            totals["full_chars"] += len(reference)
            totals["supervised_edits"] += supervised_ops
            totals["supervised_edits_folded"] += supervised_ops_folded
            totals["supervised_chars"] += supervised_chars
            totals["gold_constraint_paths"] += int(path_covered)
            totals["constraint_covered"] += constraints_covered
            totals["constraints"] += constraint_total

    for record_index, row in enumerate(rows):
        if limit is not None and totals["sentences"] + len(pending) >= limit:
            break
        if not str(row.get("text", "")):
            continue
        row["_record_index"] = record_index
        pending.append(row)
        if len(pending) >= batch_size:
            consume(pending); pending = []
    if pending:
        consume(pending)
    buckets = {}
    buckets_folded = {}
    for name in ("kanji", "alnum", "kana", "symbol"):
        buckets[name] = {"cer": bucket_errors[name] / max(1, bucket_chars[name]),
                         "edits": bucket_errors[name],
                         "reference_chars": bucket_chars[name]}
        buckets_folded[name] = {"cer": bucket_errors_folded[name] / max(1, bucket_chars[name]),
                                "edits": bucket_errors_folded[name],
                                "reference_chars": bucket_chars[name]}
    result = {
        "sentences": totals["sentences"],
        "supervised_cer": totals["supervised_edits"] / max(1, totals["supervised_chars"]),
        "supervised_cer_folded": totals["supervised_edits_folded"] / max(1, totals["supervised_chars"]),
        "supervised_edits": totals["supervised_edits"],
        "supervised_edits_folded": totals["supervised_edits_folded"],
        "supervised_chars": totals["supervised_chars"],
        "full_cer": totals["full_edits"] / max(1, totals["full_chars"]),
        "full_edits": totals["full_edits"], "full_chars": totals["full_chars"],
        "sentence_exact": totals["sentence_exact"] / max(1, totals["sentences"]),
        "sentence_exact_folded": totals["sentence_exact_folded"] / max(1, totals["sentences"]),
        "candidate_constraint_coverage": totals["constraint_covered"] / max(1, totals["constraints"]),
        "gold_constraint_coverage": totals["gold_constraint_paths"] / max(1, totals["sentences"]),
        "candidate_constraints_covered": totals["constraint_covered"],
        "candidate_constraints": totals["constraints"],
        "gold_constraint_paths": totals["gold_constraint_paths"],
        "buckets": buckets,
        "buckets_folded": buckets_folded,
        "gold_injection": False,
    }
    if totals["exempt_positions_applied"]:
        result["supervised_cer_exempt"] = (
            totals["supervised_edits_exempt"] / max(1, totals["supervised_chars_exempt"]))
        result["supervised_edits_exempt"] = totals["supervised_edits_exempt"]
        result["supervised_chars_exempt"] = totals["supervised_chars_exempt"]
        result["sentence_exact_exempt"] = (
            totals["sentence_exact_exempt"] / max(1, totals["sentences"]))
        result["exempt_positions_applied"] = totals["exempt_positions_applied"]
    return result

def _count_rows(path: str) -> int:
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="evaluation JSONL")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    parser.add_argument("--vocab")
    parser.add_argument("--ids")
    parser.add_argument("--kanjidic")
    parser.add_argument("--jmdict")
    parser.add_argument("--unidic")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", "-o")
    parser.add_argument("--allow-resource-mismatch", action="store_true",
                        help="warn instead of failing resource_manifest validation")
    parser.add_argument("--generator-inference", action="store_true",
                        help="enable open-reading proposals in the candidate graph")
    parser.add_argument("--fast", action="store_true",
                        help="skip the per-row posterior DP; CER metrics unchanged")
    parser.add_argument("--exempt",
                        help="gold-exempt manifest (§18.9 X positions); adds a "
                             "parallel supervised_cer_exempt without changing the "
                             "strict metric. Must match --data row ordering.")
    parser.add_argument("--unified-lexicon",
                        help="path to unified lexicon pack (defaults to artifacts/lexicon_pack/unified_lexicon.pkl.gz if exists)")
    parser.add_argument("--use-legacy-lexicon-only", action="store_true",
                        help="force using legacy discrete providers instead of unified lexicon pack")
    parser.add_argument("--no-tqdm", action="store_true",
                        help="disable tqdm progress bar")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    pipeline = V4Pipeline.from_checkpoint(
        args.checkpoint, device=args.device, vocab_path=args.vocab,
        ids_path=args.ids, kanjidic_path=args.kanjidic,
        jmdict_path=args.jmdict,
        unidic_dir=args.unidic,
        unified_lexicon_path=args.unified_lexicon,
        use_legacy_lexicon_only=args.use_legacy_lexicon_only,
        strict_resources=not args.allow_resource_mismatch)
    if args.generator_inference:
        pipeline.config["generator_inference"] = True
        pipeline.config["generator_ready"] = True
    started = time.time()
    total_rows = args.limit if args.limit else _count_rows(args.data)
    progress = iter_jsonl(args.data)
    pbar = None
    if not args.no_tqdm:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_rows, desc="Evaluating", unit="sent", dynamic_ncols=True)
        except ImportError:
            pbar = None
    done = [0]

    def counting(rows):
        for row in rows:
            done[0] += 1
            if pbar is not None:
                pbar.update(1)
            elif done[0] % 5000 == 0:
                rate = done[0] / max(1e-9, time.time() - started)
                remaining = total_rows - done[0]
                print(f"evaluated {done[0]}/{total_rows} rows ({rate:.0f}/s, "
                      f"~{max(0, remaining)/max(1.0, rate)/60:.0f} min left)",
                      flush=True)
            yield row

    result = evaluate_rows(pipeline, counting(progress),
                           batch_size=args.batch_size, limit=args.limit,
                           fast=args.fast, exempt=_load_exempt(args.exempt))
    if pbar is not None:
        pbar.close()
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
