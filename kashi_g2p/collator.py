"""JSONL windowing and tensor collation for kashi-g2p."""
from __future__ import annotations

import json
import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch.utils.data import IterableDataset

from .alignment import READING_TYPES, VARIATIONS
from .kanjidic import Kanjidic, load_ids
from .lexicon import LatticeBuilder, LatticeNode
from .normalization import KANA_RE, is_kanji, is_trainable_kanji_text
from .schema import G2PSchema, schema_from_config
from .rules import apply_variation

CANDIDATE_RANKINGS = ("alpha", "frequency")
# Which lattice spans may contribute a whole-word B candidate.
#   kanji_span   span>1 and every surface character is a kanji (legacy)
#   no_kana_span span>1 and the surface contains no kana, so mixed words such
#                as ``2人`` (ふたり) and ``Ready`` become reachable
#   no_kana_all  additionally single-character dictionary readings that
#                KANJIDIC does not list
LATTICE_CANDIDATE_POLICIES = ("kanji_span", "no_kana_span", "no_kana_all")
KATAKANA_ONLY = re.compile(r"^[\u30a1-\u30fa\u30fc]+$")


def contains_kana(text: str) -> bool:
    return any(KANA_RE.match(ch) is not None for ch in text)


_MASK64 = (1 << 64) - 1


def deterministic_source_sample(byte_offset: int, source_index: int, rate: float,
                                seed: int = 20260831) -> bool:
    """Select a JSONL row reproducibly without worker-local RNG state.

    The byte offset uniquely identifies a materialized row.  SplitMix64 gives
    a stable, inexpensive uniform value and therefore produces the same
    subset regardless of DataLoader worker count.
    """
    rate = float(rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"source sample rate must be in [0, 1], got {rate}")
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    value = (int(byte_offset) ^ (int(source_index) * 0x9E3779B97F4A7C15)
             ^ int(seed)) & _MASK64
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    value ^= value >> 31
    return value < int(rate * (1 << 64))


def resume_worker_state(skip_batches: int, worker_id: int,
                        worker_count: int) -> tuple[int, int]:
    """Map a restarted DataLoader worker to its original shard and offset.

    DataLoader consumes worker outputs round-robin.  If a checkpoint stops at
    a batch count that is not divisible by the worker count, the resumed
    physical worker zero must continue from a rotated logical shard.  The
    returned pair is ``(logical_shard, batches_to_skip_in_that_shard)``.
    """
    if skip_batches < 0:
        raise ValueError("skip_batches must be >= 0")
    if worker_count < 1 or not 0 <= worker_id < worker_count:
        raise ValueError("invalid DataLoader worker id/count")
    rotation = skip_batches % worker_count
    logical_shard = (worker_id + rotation) % worker_count
    local_skip = skip_batches // worker_count
    if logical_shard < rotation:
        local_skip += 1
    return logical_shard, local_skip


def load_vocab(path: str | Path) -> dict[str, int]:
    return {line.rstrip("\n\r"): i for i, line in enumerate(Path(path).open(encoding="utf-8"))}


@dataclass
class WindowedJsonlDataset(IterableDataset):
    path: str | Sequence[str]
    max_length: int = 64
    overlap: int = 16
    # Optional buffered length bucketing.  When enabled, workers yield
    # complete homogeneous mini-batches and the DataLoader uses
    # ``batch_size=None``.  The default keeps the original one-window-per-item
    # behavior for evaluation and external callers.
    length_buckets: Sequence[int] | None = None
    bucket_buffer_size: int = 0
    bucket_batch_size: int = 0
    drop_last: bool = False
    # Training/evaluation can exclude rows with no Han characters.  Keep this
    # opt-in so inference and library callers still accept kana-only text.
    require_kanji: bool = False
    # Per-file row sampling keeps large out-of-domain corpora from dominating
    # the target lyric domain.  Sources are then consumed in fixed-size chunks
    # instead of concatenated in path order.
    source_sample_rates: Sequence[float] | None = None
    source_interleave: int = 0
    source_sample_seed: int = 20260831
    # Number of already-consumed homogeneous batches in this deterministic
    # stream.  Applied before LakeCollator so resume does not rebuild lattice
    # tensors merely to discard them.
    skip_batches: int = 0

    def _bucket_index(self, length: int, buckets: tuple[int, ...]) -> int:
        for index, limit in enumerate(buckets):
            if length <= limit:
                return index
        raise ValueError(
            f"window length {length} exceeds the largest length bucket {buckets[-1]}"
        )

    def __iter__(self) -> Iterator[dict | list[dict]]:
        # When used with a multi-worker DataLoader, shard the JSONL by byte
        # ranges. Without this every worker would iterate the full file and
        # silently duplicate training examples and disk I/O.
        try:
            from torch.utils.data import get_worker_info
            info = get_worker_info()
            worker_id = info.id if info is not None else 0
            worker_count = info.num_workers if info is not None else 1
        except Exception:
            worker_id, worker_count = 0, 1
        shard_id, local_skip_batches = resume_worker_state(
            int(self.skip_batches), worker_id, worker_count
        )
        paths = (self.path,) if isinstance(self.path, (str, Path)) else tuple(self.path)
        rates = (tuple(float(value) for value in self.source_sample_rates)
                 if self.source_sample_rates is not None else (1.0,) * len(paths))
        if len(rates) != len(paths):
            raise ValueError(
                f"source_sample_rates has {len(rates)} values for {len(paths)} data files"
            )
        if any(not 0.0 <= rate <= 1.0 for rate in rates):
            raise ValueError("source_sample_rates values must be in [0, 1]")
        if self.source_interleave < 0:
            raise ValueError("source_interleave must be >= 0")
        step = max(1, self.max_length - self.overlap)

        def source_windows(source_index: int) -> Iterator[dict]:
            path = Path(paths[source_index])
            rate = rates[source_index]
            file_size = path.stat().st_size
            start_byte = file_size * shard_id // worker_count
            end_byte = file_size * (shard_id + 1) // worker_count
            # JSONL is byte-addressable.  Range sharding avoids making every
            # worker scan the complete multi-gigabyte file just to discard rows.
            with path.open("rb") as fh:
                if start_byte:
                    fh.seek(start_byte - 1)
                    if fh.read(1) != b"\n":
                        fh.readline()  # discard the partial line crossing the range
                else:
                    fh.seek(0)
                while True:
                    line_offset = fh.tell()
                    if shard_id != worker_count - 1 and line_offset >= end_byte:
                        break
                    raw_line = fh.readline()
                    if not raw_line:
                        break
                    if not raw_line.strip():
                        continue
                    if not deterministic_source_sample(
                            line_offset, source_index, rate, self.source_sample_seed):
                        continue
                    row = json.loads(raw_line)
                    text = row["text"]
                    if not text:
                        continue
                    if self.require_kanji and not is_trainable_kanji_text(
                            text, has_explicit_reading=bool(row.get("special_spans"))):
                        continue
                    starts = list(range(0, max(1, len(text) - self.overlap), step))
                    if not starts or starts[-1] + self.max_length < len(text):
                        starts.append(max(0, len(text) - self.max_length))
                    seen = set()
                    for start in starts:
                        end = min(len(text), start + self.max_length)
                        if (start, end) in seen:
                            continue
                        seen.add((start, end))
                        sliced = {}
                        for key, value in row.items():
                            if key == "file_path":
                                continue
                            if key == "text":
                                sliced[key] = value[start:end]
                            elif key == "special_spans":
                                # Keep span coordinates local to this
                                # window.  A boundary-cut span is clipped;
                                # the explicit template still emits its
                                # reading once within the visible region.
                                spans = []
                                for span in value or []:
                                    span_start = int(span.get("start", 0))
                                    span_end = int(span.get("end", span_start))
                                    if span_start >= end or span_end <= start:
                                        continue
                                    clipped = dict(span)
                                    clipped["start"] = max(span_start, start) - start
                                    clipped["end"] = min(span_end, end) - start
                                    spans.append(clipped)
                                sliced[key] = spans
                            elif isinstance(value, list):
                                sliced[key] = value[start:end]
                            else:
                                sliced[key] = value
                        if self.require_kanji and not is_trainable_kanji_text(
                                sliced["text"],
                                has_explicit_reading=bool(sliced.get("special_spans"))):
                            continue
                        yield sliced

        def windows() -> Iterator[dict]:
            iterators = [iter(source_windows(index)) for index in range(len(paths))]
            if self.source_interleave <= 0:
                for iterator in iterators:
                    yield from iterator
                return
            active = iterators
            while active:
                remaining = []
                for iterator in active:
                    exhausted = False
                    for _ in range(self.source_interleave):
                        try:
                            yield next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                    if not exhausted:
                        remaining.append(iterator)
                active = remaining

        buckets = tuple(sorted({int(value) for value in (self.length_buckets or ())}))
        use_bucketing = bool(buckets and self.bucket_buffer_size > 0 and self.bucket_batch_size > 0)
        if not use_bucketing:
            if self.skip_batches:
                raise ValueError("skip_batches requires homogeneous length-bucket batches")
            yield from windows()
            return
        if buckets[-1] < self.max_length:
            raise ValueError(
                f"largest length bucket {buckets[-1]} is below max_length={self.max_length}"
            )
        batch_size = int(self.bucket_batch_size)
        buffer_limit = max(batch_size, int(self.bucket_buffer_size))
        pending: list[list[dict]] = [[] for _ in buckets]
        buffer: list[dict] = []

        def flush_buffer() -> Iterator[list[dict]]:
            if not buffer:
                return
            # Group by the eventual padded shape first, then by exact length so
            # each emitted mini-batch minimizes both padding and graph variants.
            buffer.sort(key=lambda item: (self._bucket_index(len(item["text"]), buckets), len(item["text"])))
            for item in buffer:
                pending[self._bucket_index(len(item["text"]), buckets)].append(item)
            buffer.clear()
            while True:
                full_index = next((i for i, items in enumerate(pending) if len(items) >= batch_size), None)
                if full_index is None:
                    break
                items = pending[full_index]
                yield items[:batch_size]
                del items[:batch_size]

        def bucketed_batches() -> Iterator[list[dict]]:
            for item in windows():
                buffer.append(item)
                if len(buffer) >= buffer_limit:
                    yield from flush_buffer()
            yield from flush_buffer()
            if not self.drop_last:
                for items in pending:
                    if items:
                        yield items

        for batch_index, items in enumerate(bucketed_batches()):
            if batch_index < local_skip_batches:
                continue
            yield items


def load_alnum_table(path: str | Path, majority: float = 0.9) -> dict[str, str]:
    """Load the mined alnum reading table as ``span text -> reading``.

    Keeps a span only when its reading is unique or the top reading holds at
    least ``majority`` of its occurrences (the content gate from the lookup
    probe: ambiguous spans like 5->ご/ふぁいぶ/いつ stay out of v1).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    table = raw.get("table", raw)
    resolved: dict[str, str] = {}
    for span, readings in table.items():
        best, count = max(readings.items(), key=lambda kv: kv[1])
        if len(readings) == 1 or count / sum(readings.values()) >= majority:
            resolved[span] = best
    return resolved


class LakeCollator:
    def __init__(self, vocab_path: str | Path, kanjidic: Kanjidic, *, ids_path: str | Path | None = None,
                 lattice: LatticeBuilder | None = None, max_candidates: int = 8,
                 max_reading_length: int = 8, max_surface_length: int = 32,
                 max_components: int = 8,
                 reading_counts: dict[str, int] | None = None,
                 pad_to_length: int | None = None,
                 length_buckets: Sequence[int] | None = None,
                 pad_candidates: bool = False,
                 node_buckets: tuple[int, ...] | None = None,
                 node_pad_multiple: int = 16,
                 node_surface_pad_multiple: int = 4,
                 inject_gold_candidates: bool = True,
                 gold_injection_prob: float = 1.0,
                 gold_injection_seed: int = 20260901,
                 ids_format: str = "chise_v1",
                 use_joint_head: bool = False,
                 max_joint_candidates: int = 128,
                 candidate_ranking: str = "alpha",
                 lattice_candidate_policy: str = "kanji_span",
                 span_candidate_budget: int = 0,
                 alnum_table: dict[str, str] | None = None,
                 alnum_singing_target: bool = False,
                 word_candidates: bool = False,
                 max_word_candidates: int = 6,
                 span_gold_rewrite: bool = False,
                 schema: str | dict | G2PSchema | None = None):
        self.vocab = load_vocab(vocab_path)
        self.unk_id = self.vocab.get("[UNK]", 1)
        self.pad_id = self.vocab.get("[PAD]", 0)
        self.kanjidic = kanjidic
        self.schema = schema_from_config(schema)
        self.ids = load_ids(ids_path, format_version=ids_format) if ids_path else {}
        self.ids_format = ids_format
        self.inject_gold_candidates = bool(inject_gold_candidates)
        # Gold injection guarantees the supervised reading is scorable, but a
        # model trained with it never sees the deployment case where no
        # candidate is correct.  A probability below one keeps part of the
        # supervision while exposing the strict candidate sets.
        self.gold_injection_prob = float(gold_injection_prob)
        if not 0.0 <= self.gold_injection_prob <= 1.0:
            raise ValueError("gold_injection_prob must be within [0, 1]")
        self.gold_injection_seed = int(gold_injection_seed)
        self.use_joint_head = bool(use_joint_head)
        self.max_joint_candidates = int(max_joint_candidates)
        # Keep IDs within the student's fixed 1024-entry component table.
        component_counts = Counter(c for values in self.ids.values() for c in values)
        ranked_components = sorted(component_counts, key=lambda c: (-component_counts[c], c))[:1022]
        self.component_vocab = {ch: i + 2 for i, ch in enumerate(ranked_components)}
        self.component_unk = 1
        self.lattice = lattice
        self.max_candidates = max_candidates
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        # ``sorted()`` orders readings by kana, so a flat cap of eight keeps
        # あ-row readings and discards せい/しょう/なま for 生.  Ranking by the
        # supervised corpus frequency makes the cap cost coverage instead of
        # decapitating the common readings.
        if candidate_ranking not in CANDIDATE_RANKINGS:
            raise ValueError(
                f"candidate_ranking must be one of {CANDIDATE_RANKINGS}, got {candidate_ranking!r}"
            )
        if lattice_candidate_policy not in LATTICE_CANDIDATE_POLICIES:
            raise ValueError(
                f"lattice_candidate_policy must be one of {LATTICE_CANDIDATE_POLICIES}, "
                f"got {lattice_candidate_policy!r}"
            )
        self.candidate_ranking = candidate_ranking
        self.lattice_candidate_policy = lattice_candidate_policy
        # Multi-character lattice spans are inserted ahead of the dictionary
        # readings, so an unbounded number of them evicts the character's own
        # KANJIDIC readings through ``max_candidates``.  With a corpus lexicon
        # attached a single anchor can carry dozens of spans, which is how exp7
        # lost 5.7 points of candidate coverage relative to exp6.  A budget
        # keeps the shortest spans, which carry the word-level readings, while
        # reserving the rest of the cap for single-character readings.  Zero
        # means unbounded, reproducing the behaviour of checkpoints created
        # before this knob existed.
        self.span_candidate_budget = max(0, int(span_candidate_budget))
        # Alnum reading table (EXPERIMENTS.md §13/§14): span text -> majority
        # reading, mined from train only.  Unlike gold injection these
        # candidates are supplied at inference too -- the training/inference
        # contract for alnum spans was broken precisely because the only
        # reading candidate they ever saw was gold-injected and therefore
        # vanished under strict decoding.
        self.alnum_table = dict(alnum_table) if alnum_table else None
        # Singing convention as training target (EXPERIMENTS.md §15.1): every
        # table-hit alnum run is treated as if annotated -- A=特殊, B=reading,
        # span recorded in special_spans, loss_mask=1 at the anchor.  Train and
        # validation labels both shift, so val/cer selects checkpoints under
        # the deployment metric.  Supervised runs keep their annotator gold.
        self.alnum_singing_target = bool(alnum_singing_target)
        # Word-level decision candidates (EXPERIMENTS.md §15.4): the lattice's
        # dictionary words (surface >= 2, encodable reading) offered to a
        # dedicated W head, scored per anchor position.  This is Yomogi's
        # decision structure grafted onto our trunk; the B head's candidate
        # table is untouched (no zero-sum).
        self.word_candidates = bool(word_candidates)
        self.max_word_candidates = int(max_word_candidates)
        # Span-equivalent gold rewrite (EXPERIMENTS.md §14.8): when a span>1
        # candidate's reading equals the concatenation of the per-character
        # golds it covers, the anchor's gold becomes that candidate and the
        # covered positions drop out of the B loss.  Without this the B loss
        # punishes row-correct span mounts (the audit's 54% phantom class).
        self.span_gold_rewrite = bool(span_gold_rewrite)
        self._base_candidate_cache: dict[str, tuple[str, ...]] = {}
        self.max_reading_length = max_reading_length
        self.max_surface_length = max_surface_length
        self.max_components = max_components
        self.reading_counts = reading_counts or {}
        self.pad_to_length = pad_to_length
        self.length_buckets = tuple(sorted({int(value) for value in (length_buckets or ())}))
        if any(value <= 0 for value in self.length_buckets):
            raise ValueError("length_buckets must contain positive values")
        self.pad_candidates = pad_candidates
        self.node_buckets = tuple(sorted(node_buckets or ()))
        if node_pad_multiple < 1:
            raise ValueError("node_pad_multiple must be positive")
        self.node_pad_multiple = int(node_pad_multiple)
        if node_surface_pad_multiple < 1:
            raise ValueError("node_surface_pad_multiple must be positive")
        self.node_surface_pad_multiple = int(node_surface_pad_multiple)
        self.a_map = {x: i for i, x in enumerate(self.schema.reading_types)}
        self.d_map = {x: i for i, x in enumerate(self.schema.variation_types)}
        self.bies_map = {x: i for i, x in enumerate(self.schema.bies_types)}
        self.pos_map = {x: i for i, x in enumerate(self.schema.pos_types)}
        self.copy_label = self.schema.copy_label

    def _id(self, ch: str, vocab: dict[str, int] | None = None) -> int:
        return (vocab or self.vocab).get(ch, self.unk_id)

    def _inject_gold(self, text: str, index: int, key: str) -> bool:
        """Return whether this position keeps its gold reading as a candidate.

        The decision is a deterministic function of the example so a position
        is treated the same way in every epoch and in every dataloader worker.
        """
        if not self.inject_gold_candidates:
            return False
        if self.gold_injection_prob >= 1.0:
            return True
        if self.gold_injection_prob <= 0.0:
            return False
        digest = zlib.crc32(
            f"{self.gold_injection_seed}\t{text}\t{index}\t{key}".encode("utf-8"))
        return digest % 1_000_000 < self.gold_injection_prob * 1_000_000

    def _base_candidates(self, ch: str) -> tuple[str, ...]:
        """Dictionary readings for one character, in truncation priority order."""
        cached = self._base_candidate_cache.get(ch)
        if cached is not None:
            return cached
        values = sorted(self.kanjidic.candidates(ch))
        if self.candidate_ranking == "frequency" and self.reading_counts:
            values.sort(key=lambda reading: (-int(self.reading_counts.get(f"{ch}\t{reading}", 0)),
                                             reading))
        result = tuple(values)
        self._base_candidate_cache[ch] = result
        return result

    def _candidate_strings(self, ch: str, gold: str,
                           extras: Sequence[tuple[str, int]] | None = None,
                           inject: bool | None = None) -> list[str]:
        if inject is None:
            inject = self.inject_gold_candidates
        # Word-level ruby templates (熟字訓/special readings, dictionary spans)
        # are anchored at the first character of their span.  They are
        # candidates for B at that position; the accompanying span length lets
        # the assembler emit the reading once and skip the remaining surface
        # characters.  A span of one is an ordinary alternative reading and is
        # appended so it cannot push dictionary readings past the cap.
        span_extras = [reading for reading, span in (extras or ()) if span > 1 and reading]
        single_extras = [reading for reading, span in (extras or ()) if span <= 1 and reading]
        if self.span_candidate_budget:
            # ``LatticeBuilder.build`` returns nodes ordered by (start, end), so
            # the head of this list is the shortest span at the anchor and the
            # tail is the long corpus-lexicon phrase match that crowds out the
            # character's own readings.
            span_extras = span_extras[: self.span_candidate_budget]
        if is_kanji(ch):
            values = list(self._base_candidates(ch))
        else:
            # ``ref_reading`` supplies the three context-sensitive particle
            # pronunciations.  Keep those alternatives in the candidate set
            # even when the current row has no gold ruby, otherwise inference
            # can never emit は->わ, へ->え, or を->お.
            particle = {"は": "わ", "へ": "え", "を": "お"}
            values = [ch] + ([particle[ch]] if ch in particle else [])
        for reading in single_extras:
            if reading not in values:
                values.append(reading)
        # Non-kanji anchors need this too: ``2人`` -> ふたり is stored on the
        # digit, so dropping span extras there made the reading unreachable at
        # inference even though training saw it as gold.
        for reading in reversed(span_extras):
            if reading not in values:
                values.insert(0, reading)
        if not values:
            values = [ch if not is_kanji(ch) else "UNK"]
        if inject and gold not in {"COPY", "UNK", ""}:
            # Keep supervised readings inside the emitted candidate budget
            # even when a dictionary already contains them beyond the cutoff.
            if gold not in values:
                values.insert(0, gold)
            elif gold not in values[:self.max_candidates]:
                values = [gold] + values[:max(0, self.max_candidates - 1)]
        if ch not in values and not is_kanji(ch):
            values.insert(0, ch)
        return values[: self.max_candidates]

    def _encode_candidates(self, chars: list[str], golds: list[str], vocab: dict[str, int],
                           extras: list[list[tuple[str, int]]] | None = None,
                           reading_length: int | None = None):
        text = "".join(chars)
        strings = [self._candidate_strings(
                       ch, gold, extras[i] if extras is not None else None,
                       self._inject_gold(text, i, gold))
                   for i, (ch, gold) in enumerate(zip(chars, golds))]
        span_maps = [dict(extras[i]) if extras is not None else {} for i in range(len(chars))]
        k = max(1, min(self.max_candidates, max(map(len, strings), default=1)))
        if reading_length is None:
            # ``max_reading_length`` is only the compact base width.  Direct
            # callers may pass long candidates too, so size this helper from
            # the actual strings instead of treating eight as a hard cap.
            r = max(self.max_reading_length,
                    max((len(value) for values in strings for value in values), default=0))
        else:
            r = int(reading_length)
        ids = torch.full((len(chars), k, r), self.pad_id, dtype=torch.long)
        mask = torch.zeros(len(chars), k, dtype=torch.bool)
        rmask = torch.zeros(len(chars), k, r, dtype=torch.bool)
        gold_idx = torch.full((len(chars),), -100, dtype=torch.long)
        prior = torch.zeros(len(chars), k, dtype=torch.float)
        span_ids = torch.ones(len(chars), k, dtype=torch.long)
        for i, (vals, gold) in enumerate(zip(strings, golds)):
            for j, value in enumerate(vals[:k]):
                value = chars[i] if value in {"COPY", "UNK"} else value
                if len(value) > r:
                    raise ValueError(
                        f"candidate reading {value!r} for {chars[i]!r} has length "
                        f"{len(value)} > requested reading_length={r}"
                    )
                encoded = [self._id(c, vocab) for c in value[:r]]
                if encoded:
                    ids[i, j, :len(encoded)] = torch.tensor(encoded)
                    rmask[i, j, :len(encoded)] = True
                mask[i, j] = True
                span_ids[i, j] = max(1, int(span_maps[i].get(value, 1)))
                pair_key = f"{chars[i]}\t{value}"
                prior[i, j] = math.log(float(self.reading_counts.get(pair_key, 1)) + 1.0)
                if value == gold or (gold == "COPY" and value == chars[i]) or (gold == "UNK" and value == "UNK"):
                    gold_idx[i] = j
        return ids, mask, rmask, gold_idx, prior, span_ids, strings

    def _joint_options(self, strings: list[list[str]], spans: torch.Tensor,
                       row: dict) -> tuple[list[list[tuple[int, int, int, int]]], list[int]]:
        options: list[list[tuple[int, int, int, int]]] = []
        targets: list[int] = []
        d_values = tuple(self.d_map)
        gold_b = row["B"]; gold_c = row.get("C", [0] * len(strings))
        gold_d = row.get("D", ["無"] * len(strings))
        for i, values in enumerate(strings):
            current: list[tuple[int, int, int, int]] = []
            for b_index, value in enumerate(values):
                span = max(1, int(spans[i, b_index]))
                if span > 1:
                    current.append((b_index, 0, self.d_map.get("無", 0), span))
                    continue
                max_trim = min(3, max(0, len(value) - 1))
                for trim in range(max_trim + 1):
                    shortened = value[:-trim] if trim else value
                    for d_index, variation in enumerate(d_values):
                        changed = apply_variation(shortened, variation)
                        if variation == "無" or changed != shortened:
                            current.append((b_index, trim, d_index, 1))
            current = current[:self.max_joint_candidates]
            target_tuple = None
            try:
                b_index = values.index(gold_b[i])
                target_tuple = (b_index, int(gold_c[i]),
                                self.d_map.get(str(gold_d[i]), -1),
                                max(1, int(spans[i, b_index])))
            except (ValueError, IndexError, TypeError):
                pass
            options.append(current)
            targets.append(current.index(target_tuple) if target_tuple in current else -100)
        return options, targets

    def _components(self, chars: list[str]) -> torch.Tensor:
        out = torch.zeros(len(chars), self.max_components, dtype=torch.long)
        for i, ch in enumerate(chars):
            leaves = self.ids.get(ch, [])
            if not leaves and is_kanji(ch):
                leaves = ["<UNK>"]
            for j, leaf in enumerate(leaves[:self.max_components]):
                out[i, j] = self.component_vocab.get(leaf, self.component_unk)
        return out

    def _bies_values(self, text: str, row: dict) -> list[str]:
        values = row.get("bies")
        if values and len(values) == len(text):
            return values
        if self.lattice is not None and getattr(self.lattice, "tagger", None) is not None:
            labels = ["S"] * len(text); cursor = 0
            try:
                for word in self.lattice.tagger(text):
                    surface = str(word.surface); start = text.find(surface, cursor)
                    if start < 0: continue
                    end = start + len(surface); n = end - start
                    labels[start:end] = ["S"] if n == 1 else ["B"] + ["I"] * (n - 2) + ["E"]
                    cursor = end
                return labels
            except Exception:
                pass
        return ["S"] * len(text)

    def _nodes(self, text: str, vocab: dict[str, int] | None = None,
               nodes: list | None = None):
        # ``nodes`` lets callers that already built the lattice (word-candidate
        # extraction) reuse it instead of paying a second dictionary walk.
        if nodes is None:
            nodes = self.lattice.build(text) if self.lattice else []
        n = max(1, len(nodes))
        sr = self.max_surface_length; rr = self.max_reading_length
        # Lattice nodes are dictionary words: 81% are a single character and a
        # batch maximum above eleven was never observed, yet every node used to
        # be padded to ``max_surface_length``.  That inflated the node encoder's
        # [batch, nodes, surface, dim] activations about fourfold, which pushed
        # backward past the 8 GiB card into WDDM host-memory spilling and cost
        # ~10x on step time.  Sizing the axis to the widest surface in this row
        # is bit-exact: padded columns are zeroed by ``smask`` before the
        # depthwise k=3 convolution, whose zero padding supplies the identical
        # neighbour at the new right edge.
        widest = min(sr, max((len(node.surface) for node in nodes), default=1))
        multiple = self.node_surface_pad_multiple
        sr = min(sr, max(multiple, ((widest + multiple - 1) // multiple) * multiple))
        vocab = vocab or self.vocab
        surface = torch.full((n, sr), self.pad_id, dtype=torch.long)
        reading = torch.full((n, rr), self.pad_id, dtype=torch.long)
        smask = torch.zeros(n, sr, dtype=torch.bool)
        rmask = torch.zeros(n, rr, dtype=torch.bool)
        pos = torch.zeros(n, dtype=torch.long)
        freq = torch.zeros(n, dtype=torch.float)
        coverage = torch.zeros(len(text), n, dtype=torch.bool)
        rel_pos = torch.full((len(text), n), 4, dtype=torch.long)
        special_extras: list[list[tuple[str, int]]] = [[] for _ in text]
        for j, node in enumerate(nodes):
            ss = [self._id(c, vocab) for c in node.surface[:sr]]; rs = [self._id(c, vocab) for c in node.reading[:rr]]
            if ss: surface[j,:len(ss)] = torch.tensor(ss); smask[j,:len(ss)] = True
            if rs: reading[j,:len(rs)] = torch.tensor(rs); rmask[j,:len(rs)] = True
            pos[j] = self.pos_map.get(node.pos, self.pos_map["その他"])
            freq[j] = node.log_frequency
            coverage[max(0,node.start):min(len(text),node.end),j] = True
            span_len = node.end - node.start
            for offset in range(max(0, node.start), min(len(text), node.end)):
                rel_pos[offset, j] = 3 if span_len == 1 else 0 if offset == node.start else 2 if offset == node.end - 1 else 1
            # Dictionary-covered readings are also available as candidates at
            # inference time.  Explicit ruby special_spans are inserted with
            # priority by _special_extras.
            # A node containing kana is an ordinary inflected/okurigana word
            # (行く, 白い).  Its reading may equal the character's full B and
            # must not overwrite that candidate's span metadata, otherwise C
            # is bypassed, so hiragana-bearing surfaces never contribute.
            # Katakana surfaces (ポイント) are exempt: their per-char B is
            # plain COPY, so a word-level span candidate cannot bypass C and
            # the mount gives the B head whole-word access (word-level
            # pipeline).
            def _hiragana_only(text: str) -> bool:
                return any("\u3041" <= ch <= "\u3096" for ch in text)

            if node.reading and 0 <= node.start < len(text) and not _hiragana_only(node.surface):
                if self.lattice_candidate_policy == "kanji_span":
                    # katakana-only surfaces pass under every policy: they
                    # cannot collide with per-char B (COPY) and word-level rows
                    # rely on them mounting
                    eligible = span_len > 1 and (
                        all(is_kanji(ch) for ch in node.surface)
                        or all(KATAKANA_ONLY.match(ch) for ch in node.surface))
                elif self.lattice_candidate_policy == "no_kana_span":
                    eligible = span_len > 1
                else:
                    eligible = True
                item = (node.reading, span_len)
                if eligible and item not in special_extras[node.start]:
                    special_extras[node.start].append(item)
        return surface, reading, smask, rmask, pos, freq, coverage, rel_pos, special_extras

    def _word_candidates_row(self, text: str, row: dict, rr: int | None = None,
                             nodes: list | None = None):
        """Word-level candidates anchored at each position.

        Returns (ids, spans, mask, gold): ids [l, k, r] reading char ids
        (pad-truncated to the batch reading width), spans [l, k] surface
        lengths, mask [l, k] validity, gold [l] index of the candidate whose
        reading equals the row-equivalent gold concatenation over its span
        (else -100 -- positions with no word-level supervision stay out of
        the W loss; exp12 taught that inventing targets overgeneralizes).
        """
        l = len(text)
        k = self.max_word_candidates
        rr = int(rr or self.max_reading_length)
        if nodes is None:
            nodes = self.lattice.build(text) if self.lattice else []
        # candidate list per anchor, frequency-ordered, deduplicated
        by_anchor: dict[int, list[tuple[str, int, float]]] = {}
        seen: set[tuple[int, int, str]] = set()
        for node in nodes:
            span_len = node.end - node.start
            if span_len < 2 or node.start >= l or not node.reading:
                continue
            # kana-only words (です、again) are valid word candidates too;
            # the is_kanji gate below only applies to span candidates.
            key = (node.start, span_len, node.reading)
            if key in seen:
                continue
            seen.add(key)
            by_anchor.setdefault(node.start, []).append(
                (node.reading, span_len, node.log_frequency))
        # row-equivalent gold concatenation helper
        B = row.get("B", [])
        lm = row.get("loss_mask", []) or [0] * l
        chars = list(text)

        def gold_concat(start: int, span_len: int) -> str | None:
            # the span's per-char golds, supervised anchor + any supervised
            # continuations; None if any covered position is unsupervised
            # Word-remounted rows (build_word_relabel, EXPERIMENTS §16.5)
            # carry the whole-word reading at the anchor and COPY at every
            # continuation.  For that shape the row-equivalent gold IS the
            # anchor's B, so accept it directly instead of concatenating
            # surface characters into a string that can never match.
            anchor = str(B[start]) if start < len(B) else "COPY"
            anchor_supervised = bool(lm[start]) if start < len(lm) else False
            if anchor not in {"COPY", "UNK", ""} and anchor_supervised:
                conts = range(start + 1, min(start + span_len, l))
                if all((str(B[j]) if j < len(B) else "COPY") in {"COPY", ""}
                       for j in conts):
                    return anchor
            parts = []
            for j in range(start, min(start + span_len, l)):
                g = str(B[j]) if j < len(B) else "COPY"
                sup = bool(lm[j]) if j < len(lm) else False
                if g in {"COPY", "UNK", ""}:
                    if sup:
                        return None
                    parts.append(chars[j])
                else:
                    if not sup:
                        return None
                    parts.append(g)
            return "".join(parts)

        ids = torch.zeros(l, k, rr, dtype=torch.long)
        spans = torch.zeros(l, k, dtype=torch.long)
        mask = torch.zeros(l, k, dtype=torch.bool)
        gold = torch.full((l,), -100, dtype=torch.long)
        for start, entries in by_anchor.items():
            entries.sort(key=lambda e: -e[2])
            for slot, (reading, span_len, _freq) in enumerate(entries[:k]):
                enc = [self._id(c, self.vocab) for c in reading[:rr]]
                if any(v == self.unk_id for v in enc):
                    continue
                ids[start, slot, : len(enc)] = torch.tensor(enc)
                spans[start, slot] = span_len
                mask[start, slot] = True
                anchor_supervised = bool(lm[start]) if start < len(lm) else False
                if gold[start] == -100 and anchor_supervised:
                    gc = gold_concat(start, span_len)
                    if gc is not None and gc == reading:
                        gold[start] = slot
        return ids, spans, mask, gold

    def _apply_singing_targets(self, row: dict) -> None:
        """Rewrite labels at unsupervised table-hit alnum runs (singing 口径)."""
        text = str(row.get("text", ""))
        n = len(text)
        B = row.get("B")
        if not text or B is None or len(B) != n or not isinstance(B, list):
            return
        A = row.get("A")
        lm = row.get("loss_mask")
        spans = row.get("special_spans")
        if not isinstance(spans, list):
            spans = []
            row["special_spans"] = spans
        ann_starts = set()
        for s in spans:
            try:
                ann_starts.add(int(s.get("start", -1)))
            except (AttributeError, TypeError, ValueError):
                continue
        j = 0
        while j < n:
            ch = text[j]
            if not (ch.isascii() and (ch.isalpha() or ch.isdigit())):
                j += 1
                continue
            alpha = ch.isalpha()
            e = j + 1
            while e < n and text[e].isascii() and (
                    (text[e].isalpha() if alpha else text[e].isdigit())):
                e += 1
            supervised = bool(lm[j]) if lm and j < len(lm) else False
            reading = self.alnum_table.get(text[j:e])
            if reading and j not in ann_starts and not supervised:
                if A is not None and j < len(A) and isinstance(A, list):
                    A[j] = "特殊"
                B[j] = reading
                for k2 in range(j + 1, min(e, n)):
                    B[k2] = "COPY"
                if lm is not None and j < len(lm):
                    lm[j] = 1
                spans.append({"start": j, "end": min(e, n), "reading": reading,
                              "type": "特殊"})
            j = e

    def _alnum_extras(self, row: dict, length: int,
                      node_extras: list[list[tuple[str, int]]]
                      ) -> tuple[list[list[tuple[str, int]]], list[bool]]:
        """Supply mined alnum span readings as span candidates (train + inference).

        The reading candidates at alnum runs must survive strict decoding, so
        unlike ``_special_extras`` this is deliberately NOT gated by
        ``_inject_gold``.  Unsupervised instances keep their gold on the COPY
        candidate (the surface character, already present via
        ``_candidate_strings``) -- and the returned mask puts those positions
        into the *B* loss (they sit outside ``loss_mask``, which would
        otherwise silence exactly the rejection signal that stops the model
        from firing readings where the metric wants letters).
        """
        extras = [list(values) for values in node_extras]
        b_mask = [False] * length
        if not self.alnum_table:
            return extras, b_mask
        text = str(row.get("text", ""))
        j = 0
        while j < length:
            ch = text[j]
            if not (ch.isascii() and (ch.isalpha() or ch.isdigit())):
                j += 1
                continue
            e = j + 1
            while e < length and text[e].isascii() and (
                    (ch.isalpha() and text[e].isalpha())
                    or (ch.isdigit() and text[e].isdigit())):
                e += 1
            reading = self.alnum_table.get(text[j:e])
            if reading and all(item[0] != reading for item in extras[j]):
                # span = run length: >1 lets the assembler emit once and skip
                # the (annotated, ref="") continuations; ==1 relies on the
                # assembler's alnum-special bypass to surface the base.
                span = min(length - j, e - j)
                extras[j].append((reading, span))
                b_mask[j] = True
            j = e
        return extras, b_mask

    def _special_extras(self, row: dict, length: int,
                        node_extras: list[list[tuple[str, int]]]) -> list[list[tuple[str, int]]]:
        extras = [list(values) for values in node_extras]
        text = str(row.get("text", ""))
        for span in row.get("special_spans", []) or []:
            try:
                start = int(span.get("start", -1))
                end = int(span.get("end", start))
                reading = str(span.get("reading", ""))
            except (AttributeError, TypeError, ValueError):
                continue
            if not reading or not (0 <= start < length) or end <= start:
                continue
            if not self._inject_gold(text, start, reading):
                continue
            item = (reading, min(length - start, end - start))
            if item not in extras[start]:
                extras[start].insert(0, item)
        return extras

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        if not rows:
            raise ValueError("empty batch")
        # Collation is intentionally length-homogeneous: WindowedJsonlDataset
        # yields <=64-char examples, but direct callers may provide shorter rows.
        actual_length = max(len(r["text"]) for r in rows)
        if self.length_buckets:
            length = next((bucket for bucket in self.length_buckets if actual_length <= bucket), None)
            if length is None:
                raise ValueError(
                    f"batch length {actual_length} exceeds the largest length bucket "
                    f"{self.length_buckets[-1]}"
                )
        else:
            if self.pad_to_length is not None and actual_length > self.pad_to_length:
                raise ValueError(f"batch length {actual_length} exceeds fixed length {self.pad_to_length}")
            length = self.pad_to_length or actual_length
        b = len(rows)
        if self.alnum_singing_target and self.alnum_table:
            for row in rows:
                self._apply_singing_targets(row)
        input_ids = torch.full((b, length), self.pad_id, dtype=torch.long)
        comp = torch.zeros(b, length, self.max_components, dtype=torch.long)
        ctype = torch.zeros(b, length, dtype=torch.long)
        attn = torch.zeros(b, length, dtype=torch.bool)
        labels = {key: torch.full((b, length), -100, dtype=torch.long) for key in ("a", "c", "d", "bies", "pos", "mora")}
        bies_input = torch.full((b, length), 3, dtype=torch.long)
        # Build the lattice once per row.  Besides node tensors, it supplies
        # word-level reading candidates for the special-span assembler and
        # (when the W head is on) whole-word candidates for exp20.
        lattice_nodes = [
            (self.lattice.build(r["text"]) if self.lattice else [])
            for r in rows
        ]
        node_values = [self._nodes(r["text"], self.vocab, nodes=ln)
                       for r, ln in zip(rows, lattice_nodes)]
        per_row_extras = [self._alnum_extras(r, len(r["text"]),
                                             self._special_extras(r, len(r["text"]),
                                                                  node_values[i][8]))
                          for i, r in enumerate(rows)]
        row_extras = [extras for extras, _mask in per_row_extras]
        alnum_b_mask = [mask for _extras, mask in per_row_extras]
        # KANJIDIC per-character readings use the fixed base length, while an
        # explicitly annotated or dictionary word template may be arbitrarily
        # longer.  Size only the candidate axis for the longest template in
        # this batch; lattice-node feature tensors stay at the compact base
        # length.  Rounding reduces torch.compile graph variants.
        longest_reading = self.max_reading_length
        for row, extras in zip(rows, row_extras):
            longest_reading = max(
                longest_reading,
                max((len(value) for value in row.get("B", [])
                     if value not in {"COPY", "UNK", ""}), default=0),
                max((len(reading) for values in extras for reading, _ in values), default=0),
            )
        candidate_reading_length = (
            (longest_reading + self.max_reading_length - 1)
            // self.max_reading_length * self.max_reading_length
        )
        b_candidates = []
        for bi, row in enumerate(rows):
            chars = list(row["text"]); l = len(chars); attn[bi,:l] = True
            input_ids[bi,:l] = torch.tensor([self._id(c, self.vocab) for c in chars])
            comp[bi,:l] = self._components(chars)
            ctype[bi,:l] = torch.tensor([0 if is_kanji(c) else 1 if "\u3040" <= c <= "\u309f" else 2 if "\u30a0" <= c <= "\u30ff" else 3 if c.isascii() and c.isalpha() else 4 if c.isdigit() else 5 for c in chars])
            for key, mapping in (("a",self.a_map),("d",self.d_map),("bies",self.bies_map),("pos",self.pos_map)):
                vals = self._bies_values(row["text"], row) if key == "bies" else row.get({"a":"A", "d":"D"}.get(key, key), [])
                if len(vals) != l:
                    vals = list(vals)[:l] + (["S"] * l if key == "bies" else ["その他"] * l)[:max(0, l-len(vals))]
                labels[key][bi,:l] = torch.tensor([
                    mapping.get(str(v), -100 if key in {"a", "d"} else 0)
                    for v in vals
                ])
            labels["c"][bi,:l] = torch.tensor(row.get("C", [0]*l))
            bies_input[bi,:l] = labels["bies"][bi,:l].clamp_min(0).clamp_max(3)
            labels["mora"][bi,:l] = torch.tensor(row.get("mora", [0]*l))
            lm = torch.tensor(row.get("loss_mask", [0]*l), dtype=torch.bool)
            labels.setdefault("loss_mask", torch.zeros(b, length, dtype=torch.bool))[bi,:l] = lm
            if "B" not in row or len(row["B"]) != l:
                raise ValueError("each row must provide a B label for every character")
            ids, cmask, rmask, gold, prior, spans, strings = self._encode_candidates(
                chars, row["B"], self.vocab, row_extras[bi], candidate_reading_length
            ); b_candidates.append((ids,cmask,rmask,gold,prior,spans,strings))
        if self.span_gold_rewrite:
            for bi, row in enumerate(rows):
                ids, cmask, rmask, gold, prior, spans_t, strings = b_candidates[bi]
                chars = list(row["text"])
                B = row["B"]
                n_pos = len(chars)
                rewritten = 0
                for a in range(n_pos):
                    if int(gold[a]) == -100:
                        continue
                    for m in range(int(cmask[a].sum())):
                        span_len = int(spans_t[a, m])
                        if span_len <= 1 or a + span_len > n_pos:
                            continue
                        val = strings[a][m] if m < len(strings[a]) else ""
                        if not val:
                            continue
                        concat_all = []
                        for k2 in range(a, a + span_len):
                            g2 = str(B[k2]) if k2 < len(B) else "COPY"
                            concat_all.append(chars[k2] if g2 in {"COPY", "UNK", ""} else g2)
                        if val != "".join(concat_all):
                            continue
                        # row-equivalent span: anchor gold -> span slot,
                        # covered continuations leave the B loss
                        gold[a] = m
                        for k2 in range(a + 1, a + span_len):
                            gold[k2] = -100
                        rewritten += 1
                        break
                    if rewritten >= 8:
                        break
        k = self.max_candidates if self.pad_candidates else max(x[0].size(1) for x in b_candidates)
        r = candidate_reading_length
        abm = torch.zeros(b, length, dtype=torch.bool)
        for i, mask_row in enumerate(alnum_b_mask):
            abm[i, :len(mask_row)] = torch.tensor(mask_row, dtype=torch.bool)
        cand = torch.full((b,length,k,r), self.pad_id, dtype=torch.long); cmask=torch.zeros(b,length,k,dtype=torch.bool); rmask=torch.zeros(b,length,k,r,dtype=torch.bool); gold=torch.full((b,length),-100,dtype=torch.long); prior=torch.zeros(b,length,k); cspan=torch.ones(b,length,k,dtype=torch.long)
        for i,(x,m,rm,g,pr,sp,_) in enumerate(b_candidates):
            cand[i,:x.size(0),:x.size(1)] = x; cmask[i,:m.size(0),:m.size(1)] = m; rmask[i,:rm.size(0),:rm.size(1)] = rm; gold[i,:g.size(0)] = g; prior[i,:pr.size(0),:pr.size(1)] = pr; cspan[i,:sp.size(0),:sp.size(1)] = sp
        n=max(x[0].size(0) for x in node_values)
        if self.node_buckets:
            n = next((bucket for bucket in self.node_buckets if n <= bucket), n)
        if self.node_pad_multiple > 1:
            n = ((n + self.node_pad_multiple - 1) // self.node_pad_multiple) * self.node_pad_multiple
        # ``_nodes`` sizes the surface axis per row, so take the batch maximum
        # instead of assuming row 0 is the widest.
        sw=max(v[0].size(1) for v in node_values); rw=max(v[1].size(1) for v in node_values)
        ns=torch.full((b,n,sw),self.pad_id,dtype=torch.long); nr=torch.full((b,n,rw),self.pad_id,dtype=torch.long); nsm=torch.zeros_like(ns,dtype=torch.bool); nrm=torch.zeros_like(nr,dtype=torch.bool); np=torch.zeros(b,n,dtype=torch.long); nf=torch.zeros(b,n); cov=torch.zeros(b,length,n,dtype=torch.bool); rel=torch.full((b,length,n),4,dtype=torch.long)
        for i,v in enumerate(node_values):
            nn=v[0].size(0); si=v[0].size(1); ri=v[1].size(1)
            ns[i,:nn,:si]=v[0]; nr[i,:nn,:ri]=v[1]; nsm[i,:nn,:si]=v[2]; nrm[i,:nn,:ri]=v[3]; np[i,:nn]=v[4]; nf[i,:nn]=v[5]; cov[i,:len(rows[i]["text"]),:nn]=v[6]; rel[i,:len(rows[i]["text"]),:nn]=v[7]
        if self.word_candidates:
            wc_ids = torch.zeros(b, length, self.max_word_candidates, r, dtype=torch.long)
            wc_spans = torch.zeros(b, length, self.max_word_candidates, dtype=torch.long)
            wc_mask = torch.zeros(b, length, self.max_word_candidates, dtype=torch.bool)
            wc_gold = torch.full((b, length), -100, dtype=torch.long)
            for i, row in enumerate(rows):
                ids_w, spans_w, mask_w, gold_w = self._word_candidates_row(
                    str(row["text"]), row, rr=r, nodes=lattice_nodes[i])
                l_i = ids_w.size(0)
                wc_ids[i, :l_i] = ids_w
                wc_spans[i, :l_i] = spans_w
                wc_mask[i, :l_i] = mask_w
                wc_gold[i, :l_i] = gold_w
        else:
            wc_ids = wc_spans = wc_mask = wc_gold = None
        output={"input_ids":input_ids,"component_ids":comp,"char_type_ids":ctype,"bies_char_ids":bies_input,"attention_mask":attn,"candidate_char_ids":cand,"candidate_mask":cmask,"candidate_reading_mask":rmask,"candidate_log_prior":prior,"candidate_span":cspan,"gold_b":gold,"gold_b_strings":[list(r["B"]) for r in rows],"alnum_b_mask":abm,"node_surface_ids":ns,"node_reading_ids":nr,"node_surface_mask":nsm,"node_reading_mask":nrm,"node_pos_ids":np,"node_frequency":nf,"coverage":cov,"rel_pos_ids":rel,"texts":[r["text"] for r in rows],"special_spans":[r.get("special_spans", []) or [] for r in rows],
            **({"word_candidate_ids": wc_ids, "word_candidate_spans": wc_spans,
                "word_candidate_mask": wc_mask, "word_candidate_gold": wc_gold}
               if self.word_candidates else {}),
            **labels}
        if self.use_joint_head:
            all_options = []
            all_targets = []
            max_joint = 1
            for row, values in zip(rows, b_candidates):
                options, targets = self._joint_options(values[6], values[5], row)
                all_options.append(options); all_targets.append(targets)
                max_joint = max(max_joint, max((len(x) for x in options), default=1))
            max_joint = (self.max_joint_candidates if self.pad_candidates
                         else min(max_joint, self.max_joint_candidates))
            jb = torch.zeros(b, length, max_joint, dtype=torch.long)
            jc = torch.zeros_like(jb); jd = torch.zeros_like(jb)
            js = torch.ones_like(jb); jm = torch.zeros(b, length, max_joint, dtype=torch.bool)
            jg = torch.full((b, length), -100, dtype=torch.long)
            for bi, (options, targets) in enumerate(zip(all_options, all_targets)):
                for li, current in enumerate(options):
                    for ji, (b_index, trim, variation, span) in enumerate(current[:max_joint]):
                        jb[bi, li, ji] = b_index; jc[bi, li, ji] = trim
                        jd[bi, li, ji] = variation; js[bi, li, ji] = span
                        jm[bi, li, ji] = True
                    jg[bi, li] = targets[li] if targets[li] < max_joint else -100
            output.update(joint_b_index=jb, joint_c=jc, joint_d=jd,
                          joint_span=js, joint_mask=jm, gold_joint=jg)
        return output
