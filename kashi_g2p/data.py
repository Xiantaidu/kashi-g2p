"""Strict sparse supervision and batching for the unified span graph."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .collator import WindowedJsonlDataset, load_vocab
from .kanjidic import load_ids
from .normalization import is_kanji, normalize_reading, normalize_surface
from .rules import apply_variation

from .candidates import CandidateProvider, pack_edges
from .decoder import allowed_edge_mask as _allowed_edge_mask
from .model_types import Edge, EdgeBatch


def _row_value(row: dict, key: str, index: int, default):
    values = row.get(key, ())
    return values[index] if index < len(values) else default


def _assembled_target(row: dict, index: int) -> str | None:
    if not _row_value(row, "loss_mask", index, 0):
        return None
    base = str(_row_value(row, "B", index, ""))
    if not base or base in {"COPY", "UNK"}:
        return None
    try:
        trim = max(0, int(_row_value(row, "C", index, 0)))
    except (TypeError, ValueError):
        trim = 0
    variation = str(_row_value(row, "D", index, "無"))
    return apply_variation(base[:-trim] if trim else base, variation)


def _special_spans(row: dict, length: int) -> list[tuple[int, int, str]]:
    result = []
    for value in row.get("special_spans", []) or []:
        try:
            start = int(value["start"])
            end = int(value["end"])
            reading = normalize_reading(str(value["reading"]))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= start < end <= length and reading:
            result.append((start, end, reading))
    result.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    selected: list[tuple[int, int, str]] = []
    for value in result:
        overlaps = any(value[0] < old[1] and old[0] < value[1]
                       for old in selected)
        if not overlaps:
            selected.append(value)
    return sorted(selected, key=lambda item: (item[0], item[1], item[2]))


_ANCHOR_SOURCES = frozenset({"jmdict", "unidic", "yomogi_dict", "pyopenjtalk"})


def trusted_gold_span_mask(rows: Sequence[dict], edges: Sequence[Sequence[Edge]],
                           *, target_offsets: Sequence[int] | None = None
                           ) -> torch.Tensor:
    """Return edges whose target-local spans have complete ruby supervision.

    ``SparseEdgeCollator`` builds candidate edges and annotation arrays in
    target-window coordinates. ``target_offsets`` only locates those spans in
    the encoder's concatenated context input; it must not shift ``B``,
    ``loss_mask``, or ``special_spans`` lookups. The optional argument remains
    accepted for callers that already pass it, but annotation coordinates stay
    target-local.
    """
    if len(rows) != len(edges):
        raise ValueError("rows and edges must have the same batch size")
    if target_offsets is not None and len(target_offsets) != len(rows):
        raise ValueError("target_offsets must match the batch size")

    masks: list[torch.Tensor] = []
    for row, row_edges in zip(rows, edges):
        trusted = [False] * len(row_edges)
        annotated_length = len(row.get("loss_mask", ()) or ())
        special: set[int] = set()
        for value in row.get("special_spans", []) or []:
            try:
                start = int(value["start"])
                end = int(value["end"])
            except (KeyError, TypeError, ValueError):
                continue
            special.update(range(start, end))

        def position_trusted(position: int) -> bool:
            supervised = bool(_row_value(row, "loss_mask", position, 0))
            base = str(_row_value(row, "B", position, ""))
            return (supervised and base not in {"COPY", "UNK", ""}
                    and position not in special)

        for index, edge in enumerate(row_edges):
            if edge.start < 0 or edge.end <= edge.start:
                continue
            if edge.end > annotated_length:
                continue
            if all(position_trusted(position)
                   for position in range(edge.start, edge.end)):
                trusted[index] = True
        masks.append(torch.tensor(trusted, dtype=torch.bool))

    width = max((mask.numel() for mask in masks), default=1)
    result = torch.zeros((len(masks), max(1, width)), dtype=torch.bool)
    for index, mask in enumerate(masks):
        result[index, :mask.numel()] = mask
    return result


def _coarsen_gold_segmentation(
        edges: Sequence[Edge], compatible: list[bool],
        special_positions: dict[int, tuple[int, int, str]]) -> int:
    """Prefer the coarsest gold segmentation in the CRF gold-partition.

    When a fully-supervised span is bridged by a gold-compatible multi-char
    dictionary word-edge, drop the strictly-shorter *interior* edges from the
    gold set so ``partial_path_nll``'s gold-partition no longer sums over the
    fragmented character-level decomposition. Without this, both the long
    word-edge and its char-fragment path carry ``gold_edge_mask=True`` and the
    loss is indifferent to segmentation granularity, which lets the scorer
    freely undervalue long edges (the SEGMENTATION error bucket) while Viterbi
    at inference then picks the miscomposing fragments.

    Anchors are chosen non-overlapping (leftmost start, then longest span) and
    only their strictly-interior edges are cleared, so every anchor still
    bridges its own span and the gold path stays reachable by construction.
    Special-span positions are skipped -- those spans are already uniquely
    constrained by the ``covered`` branch. Returns the number of edges cleared.
    """
    anchors: list[tuple[int, int]] = []
    for index, edge in enumerate(edges):
        if not compatible[index]:
            continue
        if edge.end - edge.start < 2 or edge.source not in _ANCHOR_SOURCES:
            continue
        if any(pos in special_positions
               for pos in range(edge.start, edge.end)):
            continue
        if not any(is_kanji(ch) for ch in edge.surface):
            continue
        anchors.append((edge.start, edge.end))
    if not anchors:
        return 0
    anchors.sort(key=lambda se: (se[0], -(se[1] - se[0])))
    chosen: list[tuple[int, int]] = []
    guard_end = -1
    for start, end in anchors:
        if start >= guard_end:
            chosen.append((start, end))
            guard_end = end
    cleared = 0
    for start, end in chosen:
        span = end - start
        for index, edge in enumerate(edges):
            if (compatible[index]
                    and start <= edge.start and edge.end <= end
                    and (edge.end - edge.start) < span):
                compatible[index] = False
                cleared += 1
    return cleared


def build_gold_edges(row: dict, edges: Sequence[Edge], *,
                     inject_missing: bool = False,
                     coarsen_segmentation: bool = True,
                     ) -> tuple[list[Edge], list[bool]]:
    """Return strict path-compatible edges without changing deployment input."""
    text = normalize_surface(str(row.get("text", "")))
    edges = list(edges)
    length = len(text)
    targets = [_assembled_target(row, index) for index in range(length)]
    specials = _special_spans(row, length)
    special_positions: dict[int, tuple[int, int, str]] = {}
    strict_specials: set[tuple[int, int, str]] = set()
    for start, end, reading in specials:
        available = any(edge.start == start and edge.end == end
                        and edge.reading == reading for edge in edges)
        if inject_missing or available:
            strict_specials.add((start, end, reading))
            for index in range(start, end):
                special_positions[index] = (start, end, reading)

    mounted_starts = {start for start, _end, _reading in strict_specials}
    for index in range(length):
        if (index not in mounted_starts
                and str(_row_value(row, "B", index, "")) == "COPY"):
            targets[index] = None

    def add_gold(start: int, end: int, reading: str) -> None:
        if inject_missing and not any(
            edge.start == start and edge.end == end and edge.reading == reading
            for edge in edges
        ):
            edges.append(Edge(
                start, end, text[start:end], reading, "gold",
                prior=3.0, confidence=1.0))

    for start, end, reading in strict_specials:
        add_gold(start, end, reading)
    for index, target in enumerate(targets):
        if target and index not in special_positions:
            add_gold(index, index + 1, target)

    compatible: list[bool] = []
    for edge in edges:
        positions = range(edge.start, edge.end)
        covered = {special_positions[index] for index in positions
                   if index in special_positions}
        if covered:
            compatible.append(
                len(covered) == 1
                and next(iter(covered)) == (edge.start, edge.end, edge.reading))
            continue
        supervised = [targets[index] for index in positions if targets[index]]
        fully_supervised = all(targets[index] for index in positions)
        single_anchor = (edge.end == edge.start + 1
                         and targets[edge.start] is not None)
        compatible.append(
            not supervised
            or (single_anchor and edge.reading == targets[edge.start])
            or (fully_supervised and edge.reading == "".join(supervised)))
    if coarsen_segmentation:
        _coarsen_gold_segmentation(edges, compatible, special_positions)
    return edges, compatible


def reference_reading(row: dict) -> str:
    """Assemble the complete row target while preserving unlabeled text."""
    text = normalize_surface(str(row.get("text", "")))
    specials = {start: (end, reading)
                for start, end, reading in _special_spans(row, len(text))}
    result: list[str] = []
    index = 0
    while index < len(text):
        if index in specials:
            end, reading = specials[index]
            result.append(reading)
            index = end
            continue
        target = _assembled_target(row, index)
        result.append(target if target is not None else text[index])
        index += 1
    return "".join(result)


def reliable_reading(row: dict) -> str:
    """Concatenate only explicitly supervised targets for diagnostics."""
    text = normalize_surface(str(row.get("text", "")))
    specials = {start: (end, reading)
                for start, end, reading in _special_spans(row, len(text))}
    result: list[str] = []
    index = 0
    while index < len(text):
        if index in specials:
            end, reading = specials[index]
            result.append(reading)
            index = end
            continue
        target = _assembled_target(row, index)
        if target is not None:
            result.append(target)
        index += 1
    return "".join(result)


def gold_parts(row: dict) -> tuple[str, list[str], list[bool]]:
    """Per-character gold parts and reliability flags for supervised CER.

    ``parts`` holds the expected reading (or the raw character for supervised
    COPY positions) at every index; ``reliable`` marks positions with usable
    ruby supervision, including special-span continuations. Unreliable
    positions carry placeholder text so supervised-run scoring can break runs
    there.
    """
    text = normalize_surface(str(row.get("text", "")))
    parts = [""] * len(text)
    reliable = [bool(_row_value(row, "loss_mask", index, False))
                for index in range(len(text))]
    occupied: set[int] = set()
    for start, end, reading in _special_spans(row, len(text)):
        parts[start] = reading
        reliable[start] = True
        for index in range(start + 1, end):
            parts[index] = ""
            reliable[index] = True
        occupied.update(range(start, end))
    for index, char in enumerate(text):
        if index in occupied:
            continue
        base = str(_row_value(row, "B", index, ""))
        if base and base not in {"COPY", "UNK"}:
            try:
                trim = max(0, int(_row_value(row, "C", index, 0)))
            except (TypeError, ValueError):
                trim = 0
            parts[index] = apply_variation(
                base[:-trim] if trim else base,
                str(_row_value(row, "D", index, "無")))
        else:
            parts[index] = char
    return text, parts, reliable


def gold_constraint_reachable(edges: Sequence[Edge], compatible: Sequence[bool],
                              length: int) -> bool:
    reachable = [False] * (length + 1)
    reachable[0] = True
    by_start: list[list[int]] = [[] for _ in range(length)]
    for edge, allowed in zip(edges, compatible):
        if allowed and 0 <= edge.start < length and edge.end <= length:
            by_start[edge.start].append(edge.end)
    for start, ends in enumerate(by_start):
        if reachable[start]:
            for end in ends:
                reachable[end] = True
    return reachable[length]


def _precompute_edge_metadata(
        batch: EdgeBatch,
        compatible_masks: Sequence[Sequence[bool]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           tuple[int, ...]]:
    """Build reusable CPU metadata for batched sparse dynamic programs."""
    lengths = torch.tensor([len(text) for text in batch.texts], dtype=torch.long)
    max_text_length = batch.input_ids.size(1)
    edge_width = batch.edge_mask.size(1)
    allowed = torch.zeros_like(batch.edge_mask)
    gold_reachable = torch.zeros(len(batch.edges), dtype=torch.bool)
    start_buckets: list[list[int]] = [[] for _ in range(max_text_length)]

    for row, (edges, compatible) in enumerate(
            zip(batch.edges, compatible_masks)):
        length = len(batch.texts[row])
        row_allowed = _allowed_edge_mask(edges, length)
        allowed[row, :len(edges)] = row_allowed
        effective_gold = [compatible[index] and bool(row_allowed[index])
                          for index in range(len(edges))]
        gold_reachable[row] = gold_constraint_reachable(
            edges, effective_gold, length)
        for edge_index, edge in enumerate(edges):
            start_buckets[edge.start].append(row * edge_width + edge_index)

    edge_order: list[int] = []
    start_offsets = [0]
    for bucket in start_buckets:
        edge_order.extend(bucket)
        start_offsets.append(len(edge_order))
    return (lengths, allowed, gold_reachable,
            torch.tensor(edge_order, dtype=torch.long), tuple(start_offsets))


def _char_type(char: str) -> int:
    if is_kanji(char):
        return 1
    code = ord(char)
    if 0x3040 <= code <= 0x309F:
        return 2
    if 0x30A0 <= code <= 0x30FF:
        return 3
    if char.isascii() and char.isalpha():
        return 4
    if char.isdigit():
        return 5
    return 6


def _component_vocab(ids: dict[str, list[str]], limit: int = 1022) -> dict[str, int]:
    counts: dict[str, int] = {}
    for values in ids.values():
        for value in values:
            counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts, key=lambda value: (-counts[value], value))[:limit]
    return {value: index + 2 for index, value in enumerate(ordered)}


def _safe_token_ids(text: str, vocab: dict[str, int]) -> list[int]:
    unknown = int(vocab.get("[UNK]", 1))
    return [int(vocab.get(char, unknown)) for char in text]


@dataclass
class SparseEdgeCollator:
    provider: CandidateProvider
    vocab: dict[str, int]
    max_reading_length: int = 64
    inject_gold: bool = False
    gold_injection_prob: float = 0.0
    dynamic_reading_length: bool = True
    deduplicate_readings: bool = False
    ids: dict[str, list[str]] | None = None
    component_vocab: dict[str, int] | None = None
    max_components: int = 8
    max_span_length: int = 16
    generator_supervision: bool = True
    coarsen_segmentation: bool = True
    # Contrast ablation opt-in. "all" (default) leaves the collator output
    # byte-identical to the pre-experiment behavior: no trusted mask is built.
    # "hard" additionally computes trusted_gold_span_mask per batch so the
    # model's hard-negative contrast only draws from provably supervised spans.
    contrast_mode: str = "all"

    def __post_init__(self) -> None:
        self.ids = self.ids or {}
        self.component_vocab = self.component_vocab or _component_vocab(self.ids)
        self.gold_injection_prob = float(self.gold_injection_prob)
        if self.contrast_mode not in {"all", "hard"}:
            raise ValueError(
                f"contrast_mode must be 'all' or 'hard'; got {self.contrast_mode!r}")
        if not 0.0 <= self.gold_injection_prob <= 1.0:
            raise ValueError("gold_injection_prob must be within [0.0, 1.0]")

    def _should_inject(self, text: str) -> bool:
        if self.inject_gold:
            return True
        if self.gold_injection_prob <= 0.0:
            return False
        if self.gold_injection_prob >= 1.0:
            return True
        digest = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        return (digest % 1_000_000) < int(self.gold_injection_prob * 1_000_000)

    def __call__(self, rows: list[dict]) -> EdgeBatch:
        if not rows:
            raise ValueError("empty batch")
        texts = [normalize_surface(str(row["text"])) for row in rows]
        has_context = any(bool(row.get("left_context") or row.get("right_context")) for row in rows)
        if has_context:
            full_texts = []
            target_offsets = []
            target_lengths = []
            for row, text in zip(rows, texts):
                l_ctx = normalize_surface(str(row.get("left_context", "") or ""))
                r_ctx = normalize_surface(str(row.get("right_context", "") or ""))
                full_texts.append(l_ctx + text + r_ctx)
                target_offsets.append(len(l_ctx))
                target_lengths.append(len(text))
        else:
            full_texts = texts
            target_offsets = None
            target_lengths = None

        all_edges: list[list[Edge]] = []
        all_masks: list[list[bool]] = []
        specials_by_row: list[list[tuple[int, int, str]]] = []
        for row, text in zip(rows, texts):
            should_inject = self._should_inject(text)
            edges, compatible = build_gold_edges(
                row, self.provider.build(text), inject_missing=should_inject,
                coarsen_segmentation=self.coarsen_segmentation)
            all_edges.append(edges)
            all_masks.append(compatible)
            specials_by_row.append(_special_spans(row, len(text)))
        batch = pack_edges(
            full_texts, all_edges, self.vocab,
            max_reading_length=self.max_reading_length,
            gold_edge_masks=all_masks,
            dynamic_reading_length=self.dynamic_reading_length,
            deduplicate_readings=self.deduplicate_readings,
            prior_index=getattr(self.provider, "prior_index", None),
            target_offsets=target_offsets,
            target_lengths=target_lengths)
        batch.texts = texts
        if self.contrast_mode == "hard":
            # Opt-in only: with the default "all" mode the batch carries no
            # trusted mask and every consumer stays exactly as before.
            batch.trusted_gold_span_mask = trusted_gold_span_mask(
                rows, batch.edges,
                target_offsets=(target_offsets
                                if target_offsets is not None else None))
        (batch.lengths, batch.allowed_edge_mask, batch.gold_reachable,
         batch.dp_edge_order, batch.dp_start_offsets) = _precompute_edge_metadata(
             batch, all_masks)
        batch.component_ids, batch.char_type_ids = self._input_features(full_texts)
        batch.boundary_targets = self._boundary_targets(texts, specials_by_row)
        batch.reference_readings = [reference_reading(row) for row in rows]
        batch.reliable_readings = [reliable_reading(row) for row in rows]
        supervised = [gold_parts(row) for row in rows]
        batch.supervised_gold = [parts for _text, parts, _reliable in supervised]
        batch.supervised_reliable = [reliable for _text, _parts, reliable in supervised]
        if self.generator_supervision:
            self._generator_targets(batch, specials_by_row)
        return batch

    def _input_features(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(texts)
        length = max(1, max(map(len, texts)))
        components = torch.zeros(
            (batch, length, self.max_components), dtype=torch.long)
        char_types = torch.zeros((batch, length), dtype=torch.long)
        unknown = 1
        for batch_index, text in enumerate(texts):
            for index, char in enumerate(text):
                char_types[batch_index, index] = _char_type(char)
                values = (self.ids or {}).get(char, ())[:self.max_components]
                for component_index, value in enumerate(values):
                    components[batch_index, index, component_index] = int(
                        (self.component_vocab or {}).get(value, unknown))
        return components, char_types

    def _boundary_targets(self, texts: list[str],
                          spans: list[list[tuple[int, int, str]]]) -> torch.Tensor:
        length = max(1, max(map(len, texts)))
        targets = torch.zeros(
            (len(texts), length, self.max_span_length), dtype=torch.float32)
        for batch_index, values in enumerate(spans):
            for start, end, _reading in values:
                span_length = end - start
                if span_length <= self.max_span_length:
                    targets[batch_index, start, span_length - 1] = 1.0
        return targets

    def _generator_targets(self, batch: EdgeBatch,
                           spans: list[list[tuple[int, int, str]]]) -> None:
        rows = [(batch_index, start, end, reading)
                for batch_index, values in enumerate(spans)
                for start, end, reading in values if reading]
        if not rows:
            return
        bos = int(self.vocab.get("[CLS]", 2))
        eos = int(self.vocab.get("[SEP]", 3))
        pad = int(self.vocab.get("[PAD]", 0))
        # The generator is a bounded reading proposer: skip supervision rows
        # whose target cannot fit instead of failing the whole batch.
        limit = self.max_reading_length + 1
        rows = [row for row in rows
                if len(row[3]) + 2 <= limit]
        if not rows:
            return
        encoded = [[bos, *_safe_token_ids(reading, self.vocab), eos]
                   for _batch, _start, _end, reading in rows]
        width = min(limit, max(len(value) - 1 for value in encoded))
        inputs = torch.full((len(rows), width), pad, dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        for index, value in enumerate(encoded):
            source = value[:-1][:width]
            target = value[1:][:width]
            inputs[index, :len(source)] = torch.tensor(source)
            labels[index, :len(target)] = torch.tensor(target)
        batch.generator_batch_index = torch.tensor(
            [row[0] for row in rows], dtype=torch.long)
        batch.generator_span_start = torch.tensor(
            [row[1] for row in rows], dtype=torch.long)
        batch.generator_span_end = torch.tensor(
            [row[2] for row in rows], dtype=torch.long)
        batch.decoder_input_ids = inputs
        batch.decoder_labels = labels


def generator_context(model, hidden: torch.Tensor, batch: EdgeBatch) -> torch.Tensor | None:
    if batch.generator_batch_index is None:
        return None
    if batch.generator_span_start is None or batch.generator_span_end is None:
        raise ValueError("generator span metadata is incomplete")
    selected_hidden = hidden[batch.generator_batch_index]
    starts = batch.generator_span_start.unsqueeze(1)
    ends = batch.generator_span_end.unsqueeze(1)
    return model._span_features(selected_hidden, starts, ends)[:, 0]


def auxiliary_losses(output, model, batch: EdgeBatch, *,
                     boundary_weight: float = 0.1,
                     generator_weight: float = 0.2) -> tuple[torch.Tensor, dict[str, Tensor]]:
    total = output.edge_scores.new_zeros(())
    values: dict[str, Tensor] = {}
    if batch.boundary_targets is not None and output.boundary_logits is not None:
        # boundary_targets are built over the target-window texts (target-local
        # coordinates), while boundary_logits cover the full concatenated input.
        # Slice the logits to each row's target window so the shapes align;
        # without context, the window IS the full text and this is a no-op.
        target = batch.boundary_targets.to(output.boundary_logits.device)
        valid = batch.attention_mask.to(output.boundary_logits.device).unsqueeze(-1)
        if batch.target_offset is not None:
            offsets = batch.target_offset.reshape(-1).tolist()
            lengths = [len(text) for text in batch.texts]
            sliced = []
            sliced_valid = []
            for row_index, (offset, length) in enumerate(zip(offsets, lengths)):
                sliced.append(output.boundary_logits[row_index, offset:offset + length])
                sliced_valid.append(valid[row_index, offset:offset + length])
            width = max(part.size(0) for part in sliced)
            logits_window = output.boundary_logits.new_full(
                (len(sliced), width, output.boundary_logits.size(2)), 0.0)
            valid_window = valid.new_full((len(sliced_valid), width, 1), False)
            for row_index, (part, keep) in enumerate(zip(sliced, sliced_valid)):
                logits_window[row_index, :part.size(0)] = part
                valid_window[row_index, :keep.size(0)] = keep
            boundary = F.binary_cross_entropy_with_logits(
                logits_window[valid_window.expand_as(logits_window)],
                target[valid_window.expand_as(target)])
        else:
            boundary = F.binary_cross_entropy_with_logits(
                output.boundary_logits[valid.expand_as(output.boundary_logits)],
                target[valid.expand_as(target)])
        values["boundary"] = boundary
        total = total + boundary_weight * boundary
    if batch.decoder_labels is not None and model.generator is not None:
        context = generator_context(model, output.hidden, batch)
        logits = model.generator(
            context, batch.decoder_input_ids.to(output.hidden.device))
        generation = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.decoder_labels.to(logits.device).reshape(-1), ignore_index=-100)
        values["generator"] = generation
        total = total + generator_weight * generation
    return total, values


def iter_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_loader(path: str | Path | Sequence[str | Path],
                 collator: SparseEdgeCollator, *, batch_size: int = 16,
                 max_length: int = 64, overlap: int = 16,
                 num_workers: int = 0, require_kanji: bool = True,
                 pin_memory: bool = False,
                 shuffle_buffer: int = 0,
                 length_buckets: Sequence[int] | None = None,
                 source_sample_rates: Sequence[float] | None = None,
                 source_interleave: int = 0,
                 prefetch_factor: int = 4) -> DataLoader:
    dataset = WindowedJsonlDataset(
        path, max_length=max_length, overlap=overlap,
        require_kanji=require_kanji,
        source_sample_rates=source_sample_rates,
        source_interleave=source_interleave,
        length_buckets=length_buckets,
        bucket_buffer_size=shuffle_buffer,
        bucket_batch_size=batch_size if shuffle_buffer > 0 else 0)
    use_bucketing = bool(length_buckets and shuffle_buffer > 0)
    return DataLoader(
        dataset, batch_size=None if use_bucketing else batch_size,
        collate_fn=collator, num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=(max(1, int(prefetch_factor))
                         if num_workers > 0 else None))


def default_vocab(path: str | Path) -> dict[str, int]:
    return load_vocab(path)


def load_component_ids(path: str | Path | None,
                       format_version: str = "chise_v1") -> dict[str, list[str]]:
    return load_ids(path, format_version=format_version) if path else {}
