from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor


SOURCE_NAMES = (
    "rules",
    "haqumei",
    "lyric_memory",
    "kanjidic",
    "pyopenjtalk",
    "gold",
    "copy",
    "unknown",
    "jmdict",
    "unidic",
    "generated",
    "alnum",
    "yomogi_dict",
)
SOURCE_TO_ID = {name: index for index, name in enumerate(SOURCE_NAMES)}


@dataclass(frozen=True, slots=True)
class Edge:
    """One complete surface-span to reading candidate."""

    start: int
    end: int
    surface: str
    reading: str
    source: str = "unknown"
    prior: float = 0.0
    confidence: float = 0.0
    locked: bool = False

    def validate(self, text: str) -> None:
        if not 0 <= self.start < self.end <= len(text):
            raise ValueError(f"edge span {(self.start, self.end)} is outside {text!r}")
        if text[self.start:self.end] != self.surface:
            raise ValueError(
                f"edge surface {self.surface!r} does not match "
                f"text[{self.start}:{self.end}]={text[self.start:self.end]!r}"
            )
        if not self.reading:
            raise ValueError("edge reading must be non-empty")


@dataclass
class EdgeBatch:
    """Padded tensors and sparse generation supervision for a text batch."""

    input_ids: Tensor
    attention_mask: Tensor
    edge_start: Tensor
    edge_end: Tensor
    edge_reading_ids: Tensor | None
    edge_reading_mask: Tensor | None
    edge_mask: Tensor
    edge_locked: Tensor
    edge_source_ids: Tensor
    edge_prior: Tensor
    texts: list[str]
    edges: list[list[Edge]]
    source_to_id: dict[str, int]
    gold_edge_mask: Tensor | None = None
    edge_pack_ids: Tensor | None = None
    component_ids: Tensor | None = None
    char_type_ids: Tensor | None = None
    boundary_targets: Tensor | None = None
    generator_batch_index: Tensor | None = None
    generator_span_start: Tensor | None = None
    generator_span_end: Tensor | None = None
    decoder_input_ids: Tensor | None = None
    decoder_labels: Tensor | None = None
    reference_readings: list[str] | None = None
    reliable_readings: list[str] | None = None
    supervised_gold: list[list[str]] | None = None
    supervised_reliable: list[list[bool]] | None = None
    lengths: Tensor | None = None
    allowed_edge_mask: Tensor | None = None
    gold_reachable: Tensor | None = None
    dp_edge_order: Tensor | None = None
    dp_start_offsets: tuple[int, ...] | None = None
    unique_reading_ids: Tensor | None = None
    unique_reading_mask: Tensor | None = None
    edge_reading_inverse: Tensor | None = None
    target_offset: Tensor | None = None
    target_length: Tensor | None = None
    # Per-edge span-trust mask [rows, edges]: True only for edges whose span
    # carries provably complete, non-special ruby supervision. Optional; when
    # absent no consumer may assume trusted supervision.
    trusted_gold_span_mask: Tensor | None = None

    def __post_init__(self) -> None:
        unique = (self.unique_reading_ids, self.unique_reading_mask,
                  self.edge_reading_inverse)
        if any(value is not None for value in unique) and not all(
                value is not None for value in unique):
            raise ValueError("unique reading inputs must be provided together")
        if all(value is None for value in unique) and (
                self.edge_reading_ids is None or self.edge_reading_mask is None):
            raise ValueError("dense or unique reading inputs are required")

    def _transform_tensors(self, transform) -> "EdgeBatch":
        values = {
            field.name: (
                {
                    key: transform(item) if isinstance(item, Tensor) else item
                    for key, item in value.items()
                } if isinstance(value, dict) and any(
                    isinstance(item, Tensor) for item in value.values()
                ) else [
                    transform(item) if isinstance(item, Tensor) else item
                    for item in value
                ] if isinstance(value, list) and any(
                    isinstance(item, Tensor) for item in value
                ) else transform(value) if isinstance(value, Tensor) else value
            )
            for field in fields(self)
            for value in (getattr(self, field.name),)
        }
        return EdgeBatch(**values)

    def to(self, device: torch.device | str,
           non_blocking: bool = False) -> "EdgeBatch":
        return self._transform_tensors(
            lambda value: value.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> "EdgeBatch":
        return self._transform_tensors(lambda value: value.pin_memory())

    def model_kwargs(self) -> dict[str, Tensor | None]:
        result: dict[str, Tensor | None] = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "edge_start": self.edge_start,
            "edge_end": self.edge_end,
            "edge_mask": self.edge_mask,
            "edge_locked": self.edge_locked,
            "edge_source_ids": self.edge_source_ids,
            "edge_prior": self.edge_prior,
        }
        if self.target_offset is not None:
            result["target_offset"] = self.target_offset
        if self.target_length is not None:
            result["target_length"] = self.target_length
        if self.edge_pack_ids is not None:
            result["edge_pack_ids"] = self.edge_pack_ids
        if self.trusted_gold_span_mask is not None:
            result["trusted_gold_span_mask"] = self.trusted_gold_span_mask
        if (self.unique_reading_ids is not None
                and self.unique_reading_mask is not None
                and self.edge_reading_inverse is not None):
            result["unique_reading_ids"] = self.unique_reading_ids
            result["unique_reading_mask"] = self.unique_reading_mask
            result["edge_reading_inverse"] = self.edge_reading_inverse
        else:
            result["edge_reading_ids"] = self.edge_reading_ids
            result["edge_reading_mask"] = self.edge_reading_mask
        if self.component_ids is not None:
            result["component_ids"] = self.component_ids
        if self.char_type_ids is not None:
            result["char_type_ids"] = self.char_type_ids
        return result
