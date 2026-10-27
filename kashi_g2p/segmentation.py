"""Deterministic word segmentation for fixed-boundary decoding.

The v4/v5 span models treat segmentation as a decoding decision: every edge is
a ``(start, end, reading)`` candidate and Viterbi picks a non-overlapping cover,
so the chosen path *is* the tokenization.  This module adds the opposite
contract for deployments that must not let the model move word boundaries:
an external tokenizer (or the caller) fixes the cover first, and the model only
chooses readings inside those spans.

Segments are always a contiguous, non-overlapping cover of ``[0, len(text))``.
Every segmenter here is deterministic for a given input string: no sampling, no
language-model context, and spans are derived from character offsets rather
than re-tokenized text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


def make_tagger(dicdir: str | Path):
    """Create a fugashi tagger using the dictionary's mecabrc when present."""
    import fugashi

    path = Path(dicdir).expanduser()
    rc = path / "mecabrc"
    rc_path = rc if rc.is_file() else Path(os.devnull)
    return fugashi.Tagger(
        f'-r "{rc_path.as_posix()}" -d "{path.as_posix()}"')


class SegmentationError(ValueError):
    """Raised when a segmentation does not cover the text exactly once."""


@dataclass(frozen=True, slots=True)
class Segment:
    """One fixed word span in normalized input text."""

    start: int
    end: int
    surface: str

    @property
    def length(self) -> int:
        return self.end - self.start


class Segmenter(Protocol):
    """Turn normalized text into a deterministic contiguous word cover."""

    name: str

    def segment(self, text: str) -> list[Segment]:
        ...


def _segments_from_spans(text: str, spans: Sequence[tuple[int, int]],
                         *, source: str = "fixed") -> list[Segment]:
    """Validate offsets and build ``Segment`` objects."""
    length = len(text)
    segments: list[Segment] = []
    cursor = 0
    for start, end in spans:
        start = int(start)
        end = int(end)
        if not 0 <= start < end <= length:
            raise SegmentationError(
                f"{source} segment span {(start, end)} is outside text length {length}")
        if start != cursor:
            kind = "overlaps" if start < cursor else "leaves a gap before"
            raise SegmentationError(
                f"{source} segment {(start, end)} {kind} offset {cursor}")
        segments.append(Segment(start, end, text[start:end]))
        cursor = end
    if cursor != length:
        raise SegmentationError(
            f"{source} segmentation covers [0, {cursor}) but text length is {length}")
    if not segments and length:
        raise SegmentationError(f"{source} segmentation is empty for non-empty text")
    return segments


def _segments_from_surfaces(text: str, surfaces: Sequence[str],
                            *, source: str,
                            gaps: str = "absorb") -> list[Segment]:
    """Locate tokenizer surfaces and return a contiguous cover.

    ``gaps="segment"`` makes uncovered runs their own segments; "absorb"
    extends the neighbouring segment.
    """
    surfaces = [str(surface) for surface in surfaces]
    joined = "".join(surfaces)
    if joined == text:
        spans = []
        cursor = 0
        for surface in surfaces:
            if not surface:
                continue
            spans.append((cursor, cursor + len(surface)))
            cursor += len(surface)
        return _segments_from_spans(text, spans, source=source)

    if gaps == "segment":
        spans = []
        cursor = 0
        for surface in surfaces:
            if not surface:
                continue
            found = text.find(surface, cursor)
            if found < 0:
                continue
            if found > cursor:
                spans.append((cursor, found))
            spans.append((found, found + len(surface)))
            cursor = found + len(surface)
        if cursor < len(text):
            spans.append((cursor, len(text)))
        if not spans and text:
            spans = [(0, len(text))]
        return _segments_from_spans(text, spans, source=source)

    spans: list[tuple[int, int]] = []
    cursor = 0
    for surface in surfaces:
        if not surface:
            continue
        found = text.find(surface, cursor)
        if found < 0:
            # The tokenizer rewrote or dropped the surface; leave the run for
            # the neighbour instead of guessing an offset.
            continue
        spans.append((found, found + len(surface)))
        cursor = found + len(surface)

    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] < end:
            merged[-1][1] = end
        elif merged and end <= merged[-1][1]:
            continue
        else:
            if merged and start > merged[-1][1]:
                # Absorb the uncovered run into the previous segment.
                merged[-1][1] = end
                continue
            merged.append([start, end])
    # Keep windows that start after 0 from being dropped.
    if merged and merged[0][0] > 0:
        merged[0][0] = 0
    if merged and merged[-1][1] < len(text):
        merged[-1][1] = len(text)
    if not merged and text:
        merged = [[0, len(text)]]
    return _segments_from_spans(
        text, [(start, end) for start, end in merged], source=source)


class CharSegmenter:
    """Degenerate deterministic cover: one segment per character."""

    name = "chars"

    def segment(self, text: str) -> list[Segment]:
        return [Segment(index, index + 1, char)
                for index, char in enumerate(text)]


# Counters whose reading is determined together with the preceding numeral
# (10 -> じゅっ standalone, but 10月 -> じゅうがつ).  UniDic splits the numeral
# from the counter, which would leave the numeral without an exact-span
# reading and force a digit-by-digit fallback (2024 -> つーおうつー4).
_COUNTER_HEADS = frozenset("年月日時分秒人個本枚台番曲目匹通件点度階杯冊倍段歳才回歩")


def _is_numeral_surface(surface: str) -> bool:
    return bool(surface) and all(char.isdigit() for char in surface)


def _is_counter_surface(surface: str) -> bool:
    return bool(surface) and (surface[0] in _COUNTER_HEADS or surface[0] == "つ")


def merge_numeral_counters(segments: Sequence[Segment],
                           text: str) -> list[Segment]:
    """Merge ``numeral`` + ``counter`` tokenizer segments deterministically.

    This is a fixed, reviewable normalisation of the tokenizer output, not a
    model decision: the same segments always merge the same way.  It keeps the
    cover intact and only removes boundaries inside number+counter units.
    """
    merged: list[Segment] = []
    index = 0
    while index < len(segments):
        current = segments[index]
        if (index + 1 < len(segments)
                and current.end == segments[index + 1].start
                and _is_numeral_surface(current.surface)
                and _is_counter_surface(segments[index + 1].surface)):
            following = segments[index + 1]
            merged.append(Segment(current.start, following.end,
                                  text[current.start:following.end]))
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


class _NormalizedSegmenter:
    """Apply deterministic boundary normalisation on top of a tokenizer."""

    def __init__(self, inner: Segmenter, *, merge_numerals: bool = True):
        self.inner = inner
        self.name = getattr(inner, "name", "segmenter")
        self.merge_numerals = bool(merge_numerals)

    def segment(self, text: str) -> list[Segment]:
        segments = list(self.inner.segment(text))
        if self.merge_numerals:
            segments = merge_numeral_counters(segments, text)
        return segments


class FixedSegmenter:
    """Use caller-provided word surfaces or character offsets verbatim.

    ``spans`` may be a sequence of ``(start, end)`` integer pairs or a sequence
    of surface strings.  Surfaces are located left-to-right; a mismatch raises
    instead of silently re-aligning, because the caller asked for exact
    boundaries.
    """

    name = "fixed"

    def __init__(self, spans: Sequence[Any]):
        if not spans:
            raise SegmentationError("fixed segmentation requires at least one segment")
        self.spans = list(spans)

    def segment(self, text: str) -> list[Segment]:
        if all(isinstance(span, str) for span in self.spans):
            surfaces = [str(span) for span in self.spans]
            if "".join(surfaces) != text:
                raise SegmentationError(
                    "fixed segment surfaces do not concatenate to the input text")
            spans = []
            cursor = 0
            for surface in surfaces:
                if not surface:
                    raise SegmentationError("fixed segment must not be empty")
                spans.append((cursor, cursor + len(surface)))
                cursor += len(surface)
        else:
            spans = []
            for span in self.spans:
                if (isinstance(span, (str, bytes))
                        or not hasattr(span, "__len__") or len(span) != 2):
                    raise SegmentationError(
                        "fixed segment must be a surface string or a (start, end) pair")
                spans.append((int(span[0]), int(span[1])))
        return _segments_from_spans(text, spans, source=self.name)


class UnidicSegmenter:
    """UniDic (MeCab) word boundaries, independent of the lattice provider."""

    name = "unidic"

    def __init__(self, unidic_dir: str | Path):
        path = Path(unidic_dir).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"UniDic dictionary does not exist: {path}")
        self.unidic_dir = str(path)
        self.tagger = make_tagger(self.unidic_dir)

    def segment(self, text: str) -> list[Segment]:
        surfaces = [str(word.surface) for word in self.tagger(text)]
        return _segments_from_surfaces(text, surfaces, source=self.name,
                                       gaps="segment")


class PyOpenJTalkSegmenter:
    """pyopenjtalk frontend word boundaries for TTS-side consistency."""

    name = "pyopenjtalk"

    def __init__(self):
        try:
            import pyopenjtalk  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "pyopenjtalk segmentation requires the pyopenjtalk package") from exc

    def segment(self, text: str) -> list[Segment]:
        import pyopenjtalk

        entries = pyopenjtalk.run_frontend(text)
        surfaces = [str(entry.get("string", "") or "") for entry in entries]
        return _segments_from_surfaces(text, surfaces, source=self.name)


_FREE_SPECS = frozenset({"", "none", "free", "off", "false", "viterbi"})


def resolve_segmenter(spec: Any, *, unidic_dir: str | Path | None = None,
                      default_unidic_dir: str | Path | None = None) -> Segmenter | None:
    """Build a segmenter from a config value.

    Accepts ``None``/``"free"`` (Viterbi chooses boundaries), a segmenter
    instance, one of ``"unidic"``/``"pyopenjtalk"``/``"chars"``, or a mapping
    such as ``{"mode": "unidic", "unidic_dir": "resources/unidic"}``.
    """
    if spec is None:
        return None
    if hasattr(spec, "segment") and getattr(spec, "name", None):
        return spec  # type: ignore[return-value]
    if isinstance(spec, Mapping):
        mode = spec.get("mode", spec.get("segmenter"))
        directory = spec.get("unidic_dir") or spec.get("unidic")
        merge_numerals = bool(spec.get("merge_numerals", True))
        segmenter = resolve_segmenter(
            mode, unidic_dir=directory,
            default_unidic_dir=default_unidic_dir)
        if segmenter is None or not merge_numerals:
            return segmenter
        return _NormalizedSegmenter(segmenter, merge_numerals=True)
    if isinstance(spec, str):
        mode = spec.strip().lower()
        if mode in _FREE_SPECS:
            return None
        if mode == "chars":
            return CharSegmenter()
        if mode == "unidic":
            directory = unidic_dir or default_unidic_dir
            if not directory:
                raise ValueError(
                    "unidic segmentation needs a UniDic directory; pass "
                    "unidic_dir= or set `unidic` in the checkpoint config")
            return _NormalizedSegmenter(UnidicSegmenter(directory))
        if mode == "pyopenjtalk":
            return _NormalizedSegmenter(PyOpenJTalkSegmenter())
    raise ValueError(
        f"unsupported segmentation spec {spec!r}; expected 'free', 'unidic', "
        "'pyopenjtalk', 'chars', a segmenter, or a mapping")


class DeferredSegmenter:
    """Pickle-safe proxy that builds the real segmenter inside each process.

    DataLoader workers spawn on Windows and cannot pickle fugashi taggers, so
    the configuration is stored and the tagger is created on first use.
    """

    def __init__(self, spec: Any, *, unidic_dir: str | Path | None = None):
        self.spec = spec
        self.unidic_dir = str(unidic_dir) if unidic_dir else None
        self._inner: Segmenter | None = None

    @property
    def name(self) -> str:
        if self._inner is not None:
            return self._inner.name
        return str(self.spec) if self.spec is not None else "free"

    def segment(self, text: str) -> list[Segment]:
        if self._inner is None:
            inner = resolve_segmenter(self.spec, unidic_dir=self.unidic_dir)
            if inner is None:
                raise SegmentationError(
                    "deferred segmentation spec resolved to free mode")
            self._inner = inner
        return self._inner.segment(text)

    def __getstate__(self) -> dict[str, Any]:
        return {"spec": self.spec, "unidic_dir": self.unidic_dir}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.spec = state["spec"]
        self.unidic_dir = state["unidic_dir"]
        self._inner = None


def segments_from_argument(text: str, segments: Sequence[Any],
                           *, source: str = "fixed") -> list[Segment]:
    """Normalize a per-text segmentation argument (surfaces or offsets)."""
    if not segments:
        raise SegmentationError("segmentation argument must not be empty")
    return FixedSegmenter(segments).segment(text)
