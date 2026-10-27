from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest
import torch

from kashi_g2p.candidates import CompositeProvider, CopyProvider, KanjidicProvider
from kashi_g2p.decoder import decode_fixed_segments, decode_viterbi
from kashi_g2p.kanjidic import Kanjidic
from kashi_g2p.model import SpanG2P
from kashi_g2p.model_types import Edge
from kashi_g2p.pipeline import V4Pipeline
from kashi_g2p.segmentation import (CharSegmenter, FixedSegmenter,
                                    SegmentationError, merge_numeral_counters,
                                    resolve_segmenter, segments_from_argument)


def _edges(*specs):
    return [Edge(*spec) if len(spec) > 4 else Edge(*spec, "test")
            for spec in specs]


def test_char_segmenter_is_a_complete_cover():
    segments = CharSegmenter().segment("東京へ")
    assert [(s.start, s.end, s.surface) for s in segments] == [
        (0, 1, "東"), (1, 2, "京"), (2, 3, "へ")]


def test_fixed_segmenter_accepts_surfaces_and_spans():
    by_surface = FixedSegmenter(["東京", "へ"]).segment("東京へ")
    by_span = FixedSegmenter([(0, 2), (2, 3)]).segment("東京へ")
    assert by_surface == by_span
    assert [s.surface for s in by_surface] == ["東京", "へ"]


@pytest.mark.parametrize("spans", [
    [(0, 1), (2, 3)],       # gap
    [(0, 2), (1, 3)],       # overlap
    [(0, 1)],               # does not cover the text
    [(0, 4)],               # outside the text
])
def test_fixed_segmenter_rejects_invalid_covers(spans):
    with pytest.raises(SegmentationError):
        FixedSegmenter(spans).segment("東京へ")


def test_fixed_segmenter_surface_mismatch_raises():
    with pytest.raises(SegmentationError):
        FixedSegmenter(["大阪"]).segment("東京")


def test_resolve_segmenter_modes():
    assert resolve_segmenter(None) is None
    assert resolve_segmenter("free") is None
    assert isinstance(resolve_segmenter("chars"), CharSegmenter)
    with pytest.raises(ValueError):
        resolve_segmenter("unidic")  # no directory available


def test_merge_numeral_counters_keeps_other_boundaries():
    text = "2024年10月25日に3人で行った"
    tokenizer_like = segments_from_argument(
        text, ["2024", "年", "10", "月", "25", "日", "に", "3", "人", "で", "行っ", "た"])
    merged = merge_numeral_counters(tokenizer_like, text)
    assert [segment.surface for segment in merged] == [
        "2024年", "10月", "25日", "に", "3人", "で", "行っ", "た"]
    assert "".join(segment.surface for segment in merged) == text


def test_merge_numeral_counters_requires_a_real_counter():
    text = "3月の2人"
    tokenizer_like = segments_from_argument(text, ["3", "月", "の", "2", "人"])
    assert [s.surface for s in merge_numeral_counters(tokenizer_like, text)] == [
        "3月", "の", "2人"]


def test_char_segmenter_is_not_merged_by_chars_mode():
    assert isinstance(resolve_segmenter("chars"), CharSegmenter)


def test_decode_fixed_segments_uses_the_best_path_inside_the_boundary():
    # The word edge wins when it outscores the character path.
    edges = _edges(
        (0, 1, "明", "めい", "kanjidic"),
        (1, 2, "日", "にち", "kanjidic"),
        (0, 2, "明日", "あす", "jmdict", 0.0, 0.5),
    )
    path = decode_fixed_segments(torch.tensor([1.0, 1.0, 9.0]), edges, 2,
                                 [(0, 2)])
    assert path.reading == "あす"
    assert [unit.mode for unit in path.units] == ["span"]
    assert path.edge_indices == [2]


def test_decode_fixed_segments_reports_one_unit_even_when_decomposed():
    # A higher-scoring character path is allowed; the reported unit boundary
    # still comes from the fixed segment, not from the chosen sub-edges.
    edges = _edges(
        (0, 1, "明", "めい", "kanjidic"),
        (1, 2, "日", "にち", "kanjidic"),
        (0, 2, "明日", "あす", "jmdict", 0.0, 0.5),
    )
    path = decode_fixed_segments(torch.tensor([5.0, 5.0, 0.5]), edges, 2,
                                 [(0, 2)])
    assert path.reading == "めいにち"
    assert [(u.start, u.end, u.mode) for u in path.units] == [
        (0, 2, "decomposed")]


def test_decode_fixed_segments_decomposes_only_inside_the_boundary():
    edges = _edges(
        (0, 1, "東", "とう", "kanjidic"),
        (1, 2, "京", "きょう", "kanjidic"),
        (2, 3, "へ", "へ", "copy"),
        (0, 2, "東京", "とうきょう", "jmdict", 0.0, 0.5),
        (0, 3, "東京へ", "とうきょうへ", "jmdict", 0.0, 0.5),
    )
    path = decode_fixed_segments(torch.tensor([1.0, 2.0, 3.0, 9.0, 20.0]),
                                 edges, 3, [(0, 1), (1, 3)])
    assert [unit.mode for unit in path.units] == ["span", "decomposed"]
    assert path.reading == "とうきょうへ"
    # The crossing edges (0, 2) and (0, 3) are never used.
    assert path.edge_indices == [0, 1, 2]
    assert [unit.surface for unit in path.units] == ["東", "京へ"]


def test_decode_fixed_segments_respects_locked_spans():
    edges = _edges(
        (0, 2, "AB", "えーびー", "haqumei", 0.0, 0.0),
        (0, 1, "A", "あ", "copy"),
        (1, 2, "B", "び", "copy"),
    )
    edges[0] = replace(edges[0], locked=True)
    path = decode_fixed_segments(torch.tensor([0.0, 100.0, 100.0]), edges, 2,
                                 [(0, 2)])
    assert path.reading == "えーびー"


def test_decode_fixed_segments_tie_break_is_deterministic():
    edges = _edges(
        (0, 2, "明日", "あす", "jmdict", 0.0, 0.5),
        (0, 2, "明日", "みょうにち", "jmdict", 0.0, 0.5),
    )
    first = decode_fixed_segments(torch.tensor([1.0, 1.0]), edges, 2, [(0, 2)])
    second = decode_fixed_segments(torch.tensor([1.0, 1.0]), edges, 2, [(0, 2)])
    assert first.reading == second.reading
    assert first.reading == "あす"  # first candidate wins an exact score tie


def test_decode_fixed_segments_rejects_non_cover():
    edges = _edges((0, 1, "東", "とう", "kanjidic"))
    with pytest.raises(ValueError):
        decode_fixed_segments(torch.tensor([0.0]), edges, 2, [(0, 1)])
    with pytest.raises(ValueError):
        decode_fixed_segments(torch.tensor([0.0]), edges, 2, [(0, 1), (1, 3)])


def test_decode_fixed_segments_score_is_the_sum_of_units():
    edges = _edges(
        (0, 1, "東", "とう", "kanjidic"),
        (1, 2, "京", "きょう", "kanjidic"),
    )
    path = decode_fixed_segments(torch.tensor([1.5, 2.5]), edges, 2,
                                 [(0, 1), (1, 2)])
    assert path.score == pytest.approx(4.0)


def test_decoder_free_mode_is_unchanged():
    edges = _edges(
        (0, 1, "明", "めい", "kanjidic"),
        (1, 2, "日", "にち", "kanjidic"),
        (0, 2, "明日", "あす", "jmdict"),
    )
    path = decode_viterbi(torch.tensor([0.1, 0.1, 2.0]), edges, 2)
    assert path.reading == "あす"


class _WordProvider:
    """Tiny in-memory provider for pipeline-level tests."""

    def __init__(self, words: dict[str, str]):
        self.words = words

    def build(self, text: str):
        edges = CopyProvider().build(text)
        for start in range(len(text)):
            for surface, reading in self.words.items():
                if text.startswith(surface, start):
                    edges.append(Edge(start, start + len(surface), surface,
                                      reading, "jmdict", 0.0, 0.9))
        return edges


def _tiny_pipeline() -> V4Pipeline:
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
             "東": 4, "京": 5, "へ": 6, "行": 7, "っ": 8, "た": 9,
             "と": 10, "う": 11, "き": 12, "ょ": 13, "へ": 14}
    for index, char in enumerate("あいうえおかきくけこさしすせそたちつてと"):
        vocab.setdefault(char, len(vocab))
    model = SpanG2P(len(vocab), layers=1, dim=16, heads=2, d_ff=32,
                    edge_dim=8, component_dim=8, fusion_layers=[1],
                    max_seq_len=32, max_reading_len=16,
                    use_open_generator=False)
    provider = CompositeProvider([_WordProvider({"東京": "とうきょう"}),
                                  CopyProvider()])
    return V4Pipeline(model, provider, vocab, device="cpu", max_length=32,
                      config={"deduplicate_readings": False,
                              "dynamic_reading_length": True})


def test_pipeline_fixed_segmentation_reports_word_units():
    pipeline = _tiny_pipeline()
    result = pipeline.predict("東京へ", segments=["東京", "へ"])
    assert result["mode"] == "fixed"
    units = result["segmentation"]["segments"]
    assert [unit["surface"] for unit in units] == ["東京", "へ"]
    assert [unit["start"] for unit in units] == [0, 2]
    assert [unit["end"] for unit in units] == [2, 3]
    # Unit readings concatenate to the sentence reading, and every selected
    # edge stays inside its own unit.
    assert "".join(unit["reading"] for unit in units) == result["reading"]
    for unit in units:
        for edge in result["edges"]:
            if unit["start"] <= edge["start"] < unit["end"]:
                assert edge["end"] <= unit["end"]


def test_pipeline_fixed_segmentation_can_force_character_split():
    pipeline = _tiny_pipeline()
    result = pipeline.predict("東京", segmentation="chars")
    units = result["segmentation"]["segments"]
    assert [unit["surface"] for unit in units] == ["東", "京"]
    # The whole-word candidate 東京->とうきょう exists but can no longer span
    # both segments: every chosen edge stays inside one character.
    assert len(result["edges"]) == 2
    assert all(edge["end"] - edge["start"] == 1 for edge in result["edges"])


def test_pipeline_free_mode_has_no_fixed_segmentation_keys():
    pipeline = _tiny_pipeline()
    result = pipeline.predict("東京", segmentation="free")
    assert "segmentation" not in result
    assert "mode" not in result
    assert result["reading"]


def test_pipeline_predict_segmented_requires_a_source():
    pipeline = _tiny_pipeline()
    with pytest.raises(ValueError):
        pipeline.predict_segmented("東京")


def test_pipeline_predict_segmented_round_trip():
    pipeline = _tiny_pipeline()
    result = pipeline.predict_segmented("東京へ", segments=["東京", "へ"])
    assert result["mode"] == "fixed"
    assert [unit["surface"] for unit in result["segmentation"]["segments"]] == [
        "東京", "へ"]
    assert result["reading"]


def test_segments_from_argument_supports_surfaces_and_spans():
    assert [s.surface for s in segments_from_argument("東京へ", ["東京", "へ"])] == [
        "東京", "へ"]
    assert [s.surface for s in segments_from_argument("東京へ", [(0, 2), (2, 3)])] == [
        "東京", "へ"]


_UNIDIC = Path(__file__).resolve().parents[1] / "resources" / "unidic"


@pytest.mark.skipif(not _UNIDIC.is_dir(), reason="UniDic resources not present")
def test_unidic_segmenter_covers_input():
    pytest.importorskip("fugashi")
    from kashi_g2p.segmentation import UnidicSegmenter

    segments = UnidicSegmenter(_UNIDIC).segment("東京都に行った。")
    assert segments[0].surface == "東京"
    assert "".join(segment.surface for segment in segments) == "東京都に行った。"
    assert segments[0].start == 0 and segments[-1].end == 8


def test_merge_numeral_counters_does_not_cross_a_gap():
    from kashi_g2p.segmentation import Segment
    text = "3 人"
    segments = [Segment(0, 1, "3"), Segment(2, 3, "人")]
    assert [segment.surface for segment in
            merge_numeral_counters(segments, text)] == ["3", "人"]


@pytest.mark.skipif(not _UNIDIC.is_dir(), reason="UniDic resources not present")
def test_unidic_segmenter_keeps_whitespace_as_a_segment():
    pytest.importorskip("fugashi")
    from kashi_g2p.segmentation import UnidicSegmenter

    text = "東京 へ"
    segments = UnidicSegmenter(_UNIDIC).segment(text)
    assert [segment.surface for segment in segments] == ["東京", " ", "へ"]
    assert "".join(segment.surface for segment in segments) == text
    assert [(segment.start, segment.end) for segment in segments] == [
        (0, 2), (2, 3), (3, 4)]


def test_contained_edges_drops_cross_segment_spans():
    from kashi_g2p.pipeline import _contained_edges
    from kashi_g2p.model_types import Edge
    edges = [Edge(0, 3, "大学生", "だいがくせい", "jmdict"),
             Edge(0, 2, "大学", "だいがく", "jmdict"),
             Edge(2, 3, "生", "せい", "jmdict")]
    kept = _contained_edges(edges, [(0, 2), (2, 3)], 3)
    assert [(edge.start, edge.end) for edge in kept] == [(0, 2), (2, 3)]
