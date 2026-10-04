from __future__ import annotations

import math
import random

import pytest
import torch

from kashi_g2p.kanjidic import Kanjidic

from kashi_g2p.candidates import (AlnumTableProvider, CompositeProvider, EnglishNumberProvider,
                         KanjidicProvider, LegacyLexiconProvider,
                         LyricMemoryProvider, pack_batch, pack_edges)
from kashi_g2p.decoder import (batched_edge_posterior, batched_log_partition,
                      batched_partial_path_nll, decode_viterbi,
                      log_partition, partial_path_nll, _morpho_penalty)
from kashi_g2p.data import build_gold_edges, gold_constraint_reachable
from kashi_g2p.data import SparseEdgeCollator
from kashi_g2p.train import _batch_loss
from kashi_g2p.model import SpanG2P, _attention_bias, _position_rows, parameter_count
from kashi_g2p.model_types import Edge


def test_viterbi_prefers_a_complete_word_span():
    edges = [
        Edge(0, 1, "明", "めい", "kanjidic"),
        Edge(1, 2, "日", "にち", "kanjidic"),
        Edge(0, 2, "明日", "あす", "lyric_memory"),
    ]
    result = decode_viterbi(torch.tensor([0.1, 0.1, 2.0]), edges, 2)
    assert result.reading == "あす"
    assert result.edge_indices == [2]


def test_locked_non_japanese_edge_blocks_overlapping_edges():
    edges = [
        Edge(0, 2, "AB", "えーびー", "haqumei", locked=True),
        Edge(0, 1, "A", "あ", "copy"),
        Edge(1, 2, "B", "び", "copy"),
    ]
    result = decode_viterbi(torch.tensor([0.0, 100.0, 100.0]), edges, 2)
    assert result.reading == "えーびー"


def test_partial_path_nll_is_differentiable():
    edges = [Edge(0, 1, "行", "い", "kanjidic"),
             Edge(0, 1, "行", "こう", "kanjidic"),
             Edge(1, 2, "く", "く", "copy")]
    scores = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
    starts = torch.tensor([edge.start for edge in edges])
    ends = torch.tensor([edge.end for edge in edges])
    mask = torch.ones(3, dtype=torch.bool)
    gold = torch.tensor([False, True, True])
    loss = partial_path_nll(scores, starts, ends, mask, gold, 2)
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None


def test_log_partition_unreachable_branch_has_finite_gradients():
    # The 1->3 edge starts at an unreachable state. Including it in
    # logaddexp(-inf, -inf) preserves the forward value but creates NaN grads.
    scores = torch.zeros(3, requires_grad=True)
    starts = torch.tensor([0, 1, 2])
    ends = torch.tensor([2, 3, 3])
    value = log_partition(
        scores, starts, ends, torch.ones(3, dtype=torch.bool), length=3
    )
    value.backward()
    assert torch.isfinite(value)
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.tolist() == [1.0, 0.0, 1.0]


def test_kanjidic_and_lyric_memory_providers():
    kd = Kanjidic({"明": {"めい": "音"}, "日": {"にち": "音"}})
    memory = LyricMemoryProvider({"明日": {"あす": 3}})
    provider = CompositeProvider([memory, KanjidicProvider(kd)])
    edges = provider.build("明日")
    assert any(edge.reading == "あす" and edge.end == 2 for edge in edges)
    assert any(edge.reading == "めい" and edge.end == 1 for edge in edges)


def test_source_priority_keeps_lyric_memory_over_same_span_lower_sources():
    memory = LyricMemoryProvider({"明日": {"あす": 3}})
    class LowerProvider:
        def build(self, text):
            return [Edge(0, 2, "明日", "あす", "kanjidic", prior=99.0)]
    provider = CompositeProvider(
        [LowerProvider(), memory],
        source_priority=("lyric_memory", "kanjidic"),
    )
    edge = next(edge for edge in provider.build("明日") if edge.reading == "あす")
    assert edge.source == "lyric_memory"


def test_composite_provider_can_remove_train_only_memory_without_reloading():
    memory = LyricMemoryProvider({"明日": {"あす": 3}})
    kanjidic = KanjidicProvider(Kanjidic({"明": {"めい": "音"}}))
    provider = CompositeProvider(
        [memory, kanjidic], source_priority=("lyric_memory", "kanjidic")
    )
    evaluation = provider.without_provider_types(LyricMemoryProvider)
    assert evaluation.providers == [kanjidic]
    assert evaluation.source_priority == provider.source_priority


def test_legacy_lexicon_adapter_preserves_word_span_shape():
    class FakeBuilder:
        def build(self, text):
            from kashi_g2p.lexicon import LatticeNode
            return [LatticeNode(0, 2, "明日", "あした", log_frequency=2.0,
                                source="jmdict")]
    provider = LegacyLexiconProvider(FakeBuilder())
    edges = provider.build("明日")
    assert len(edges) == 1
    assert edges[0].start == 0 and edges[0].end == 2
    assert edges[0].surface == "明日"
    assert edges[0].reading == "あした"
    assert edges[0].source == "jmdict"
    assert edges[0].prior == pytest.approx(0.7)
    assert edges[0].confidence == pytest.approx(0.82)


def test_rules_provider_handles_numbers_words_and_unknown_spelling():
    provider = EnglishNumberProvider()
    edges = provider.build("2人 Ready ZX")
    assert any((e.surface, e.reading, e.locked) == ("2人", "ふたり", False)
               for e in edges)
    assert any((e.surface, e.reading) == ("Ready", "れでぃ") for e in edges)
    assert any((e.surface, e.reading) == ("ZX", "ぜっとえっくす") for e in edges)


def test_source_ids_are_stable_across_batches():
    vocab = {"[PAD]": 0, "[UNK]": 1, "明": 2, "日": 3, "R": 4}
    provider = CompositeProvider([
        LyricMemoryProvider({"明日": {"あす": 2}}),
        KanjidicProvider(Kanjidic({"明": {"めい": "音"}, "日": {"にち": "音"}})),
    ])
    first = pack_batch(["明日"], provider, vocab)
    second = pack_batch(["明"], provider, vocab)
    assert first.source_to_id == second.source_to_id
    assert first.source_to_id["lyric_memory"] == second.source_to_id["lyric_memory"]


def test_gold_coarsening_prefers_long_dict_edge_over_char_fragments():
    """Tier-1 loss fix: when a fully-supervised kanji span is bridged by a
    gold-compatible multi-char dictionary word-edge, its strictly-interior
    char fragments are dropped from the gold set so the CRF gold-partition no
    longer sums over the fragmented decomposition (which made the loss
    segmentation-indifferent and let the scorer undervalue long edges)."""
    row = {
        "text": "時間",
        "B": ["じ", "かん"],
        "C": [0, 0],
        "D": ["無", "無"],
        "loss_mask": [1, 1],
    }
    edges = [Edge(0, 2, "時間", "じかん", "jmdict", prior=1.0, confidence=0.85),
             Edge(0, 1, "時", "じ", "kanjidic"),
             Edge(1, 2, "間", "かん", "kanjidic")]
    _result, mask = build_gold_edges(row, edges, coarsen_segmentation=True)
    assert mask[0]          # long dict word-edge stays gold-compatible
    assert not mask[1]      # interior char fragment dropped from gold
    assert not mask[2]      # interior char fragment dropped from gold

    # With coarsening OFF, the legacy behaviour keeps all three compatible.
    _r2, mask_off = build_gold_edges(row, edges, coarsen_segmentation=False)
    assert mask_off[0] and mask_off[1] and mask_off[2]


def test_gold_coarsening_never_orphans_a_non_dict_boundary():
    """A char position not covered by any anchor keeps its own gold edge, so
    the coarsened gold set still admits a complete path."""
    row = {
        "text": "時間差",
        "B": ["じ", "かん", "さ"],
        "C": [0, 0, 0],
        "D": ["無", "無", "無"],
        "loss_mask": [1, 1, 1],
    }
    edges = [Edge(0, 2, "時間", "じかん", "jmdict", prior=1.0, confidence=0.85),
             Edge(0, 1, "時", "じ", "kanjidic"),
             Edge(1, 2, "間", "かん", "kanjidic"),
             Edge(2, 3, "差", "さ", "kanjidic")]
    result, mask = build_gold_edges(row, edges, coarsen_segmentation=True)
    assert mask[0] and not mask[1] and not mask[2]
    assert mask[3]          # 差 has no covering anchor -> stays gold
    assert gold_constraint_reachable(result, mask, 3)


def test_sparse_gold_prefers_explicit_span_and_keeps_unlabelled_latent():
    row = {
        "text": "明日A",
        "B": ["めい", "日", "A"],
        "C": [0, 0, 0],
        "D": ["無", "無", "無"],
        "loss_mask": [1, 0, 0],
        "special_spans": [{"start": 0, "end": 2, "reading": "あす"}],
    }
    edges = [Edge(0, 2, "明日", "あす", "lyric_memory"),
             Edge(0, 1, "明", "めい", "kanjidic"),
             Edge(1, 2, "日", "ひ", "kanjidic"),
             Edge(2, 3, "A", "えー", "rules", locked=True)]
    result, mask = build_gold_edges(row, edges)
    gold_index = next(i for i, edge in enumerate(result)
                      if edge.start == 0 and edge.end == 2
                      and edge.reading == "あす")
    assert mask[gold_index]
    assert not mask[1]
    assert mask[3]


def test_pack_batch_and_model_forward():
    vocab = {"[PAD]": 0, "[UNK]": 1, "明": 2, "日": 3, "あ": 4, "す": 5}
    provider = CompositeProvider([
        LyricMemoryProvider({"明日": {"あす": 2}}),
        KanjidicProvider(Kanjidic({"明": {"めい": "音"}, "日": {"にち": "音"}})),
    ])
    batch = pack_batch(["明日"], provider, vocab, max_reading_length=4)
    model = SpanG2P(vocab_size=6, layers=2, dim=64, heads=4, d_ff=128,
                    max_seq_len=8, max_reading_len=4, edge_dim=32)
    output = model(**batch.model_kwargs())
    assert output.edge_scores.shape == batch.edge_mask.shape
    assert torch.isfinite(output.edge_scores).all()


def test_rule_edges_compete_with_copy_in_unified_score_space():
    vocab = {"[PAD]": 0, "[UNK]": 1, "A": 2}
    batch = pack_batch(["A"], EnglishNumberProvider(), vocab)
    model = SpanG2P(vocab_size=3, layers=1, dim=32, heads=4, d_ff=64,
                    max_seq_len=8, max_reading_len=8, edge_dim=16)
    output = model(**batch.model_kwargs())
    assert not batch.edge_locked.any()
    assert output.edge_scores.shape == batch.edge_mask.shape


def test_sparse_collator_and_training_loss_use_real_batch_contract():
    vocab = {"[PAD]": 0, "[UNK]": 1, "明": 2, "日": 3, "A": 4}
    provider = CompositeProvider([
        EnglishNumberProvider(),
        LyricMemoryProvider({"明日": {"あす": 2}}),
        KanjidicProvider(Kanjidic({"明": {"めい": "音"}, "日": {"にち": "音"}})),
    ])
    row = {"text": "明日A", "B": ["めい", "日", "A"],
           "C": [0, 0, 0], "D": ["無", "無", "無"],
           "loss_mask": [1, 0, 0],
           "special_spans": [{"start": 0, "end": 2, "reading": "あす"}]}
    batch = SparseEdgeCollator(provider, vocab)([row])
    model = SpanG2P(vocab_size=5, layers=1, dim=32, heads=4, d_ff=64,
                    max_seq_len=8, max_reading_len=64, edge_dim=16)
    loss = _batch_loss(model, batch, torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()


def test_long_reading_uses_position_extrapolation_instead_of_zero_padding():
    table = torch.nn.Embedding(4, 8)
    rows = _position_rows(table, 7, torch.device("cpu"))
    assert rows.shape == (7, 8)
    assert torch.equal(rows[:4], table.weight)
    assert torch.count_nonzero(rows[4:]) > 0

    model = SpanG2P(vocab_size=2, layers=1, dim=32, heads=4, d_ff=64,
                    max_seq_len=32, max_reading_len=4, edge_dim=16)
    ids = torch.ones((1, 1, 12), dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    encoded = model._encode_readings(ids, mask)
    assert encoded.shape == (1, 1, 32)
    assert torch.isfinite(encoded).all()


def test_attention_bias_is_float_additive_and_masks_invalid_keys():
    bias = _attention_bias(
        torch.tensor([[True, True, False]]), dtype=torch.float32)
    assert bias.shape == (1, 1, 1, 3)
    assert bias.dtype == torch.float32
    assert bias[0, 0, 0, 0].item() == 0.0
    assert torch.isneginf(bias[0, 0, 0, 2])


def test_small_model_is_finite_under_cpu_bfloat16_autocast():
    vocab = {"[PAD]": 0, "[UNK]": 1, "明": 2, "日": 3,
             "あ": 4, "す": 5}
    provider = CompositeProvider([
        LyricMemoryProvider({"明日": {"あす": 2}}),
        KanjidicProvider(Kanjidic({"明": {"めい": "音"}, "日": {"にち": "音"}})),
    ])
    batch = pack_batch(["明日"], provider, vocab, max_reading_length=4)
    model = SpanG2P(vocab_size=6, layers=1, dim=32, heads=4, d_ff=64,
                    max_seq_len=8, max_reading_len=4, edge_dim=16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(**batch.model_kwargs())
    assert torch.isfinite(output.edge_scores).all()


def test_checkpointed_model_backpropagates_in_reading_chunks():
    vocab = {"[PAD]": 0, "[UNK]": 1, "明": 2, "日": 3,
             "あ": 4, "す": 5}
    provider = CompositeProvider([
        LyricMemoryProvider({"明日": {"あす": 2}}),
        KanjidicProvider(Kanjidic({"明": {"めい": "音"}, "日": {"にち": "音"}})),
    ])
    batch = pack_batch(["明日", "明日"], provider, vocab,
                       max_reading_length=4, dynamic_reading_length=True)
    model = SpanG2P(vocab_size=6, layers=2, dim=32, heads=4, d_ff=64,
                    max_seq_len=8, max_reading_len=4, edge_dim=16,
                    activation_checkpointing=True, reading_chunk_size=1)
    output = model(**batch.model_kwargs())
    loss = output.edge_scores[batch.edge_mask].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.char_embed.weight.grad is not None


def test_dynamic_reading_padding_uses_batch_actual_maximum():
    vocab = {"[PAD]": 0, "[UNK]": 1, "明": 2, "日": 3,
             "あ": 4, "す": 5}
    provider = LyricMemoryProvider({"明日": {"あす": 2}})
    batch = pack_batch(["明日"], provider, vocab, max_reading_length=64,
                       dynamic_reading_length=True)
    assert batch.edge_reading_ids.size(-1) == 2


def test_character_sequence_overflow_is_explicitly_rejected():
    model = SpanG2P(vocab_size=2, layers=1, dim=32, heads=4, d_ff=64,
                    max_seq_len=2, max_reading_len=4, edge_dim=16)
    ids = torch.ones((1, 3), dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    try:
        model.encode(ids, mask)
    except ValueError as exc:
        assert "max_seq_len" in str(exc)
    else:
        raise AssertionError("sequence overflow must not be silently accepted")


def test_batched_dp_matches_reference_and_returns_posteriors():
    scores = torch.tensor([[0.1, 0.2, 1.0], [0.4, 0.3, -9.0]],
                          requires_grad=True)
    starts = torch.tensor([[0, 1, 0], [0, 1, 0]])
    ends = torch.tensor([[1, 2, 2], [1, 2, 1]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    lengths = torch.tensor([2, 2])
    batched = batched_log_partition(scores, starts, ends, mask, lengths)
    reference = torch.stack([
        log_partition(scores[index], starts[index], ends[index], mask[index], 2)
        for index in range(2)
    ])
    assert torch.allclose(batched, reference)
    gold = torch.tensor([[False, False, True], [True, True, False]])
    loss = batched_partial_path_nll(
        scores, starts, ends, mask, gold, lengths).mean()
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    posterior = batched_edge_posterior(
        scores.detach(), starts, ends, mask, lengths)
    assert torch.all((posterior >= 0) & (posterior <= 1))
    assert float(posterior[0, 0] + posterior[0, 2]) == pytest.approx(1.0)


def _sparse_topology(starts, mask, max_length):
    order = []
    offsets = [0]
    flat_starts = starts.reshape(-1)
    flat_mask = mask.reshape(-1)
    for start in range(max_length):
        order.extend(torch.where(flat_mask & (flat_starts == start))[0].tolist())
        offsets.append(len(order))
    return torch.tensor(order, dtype=torch.long), tuple(offsets)


def test_sparse_fused_nll_matches_dense_with_padding_and_constraints():
    batch_size, edge_count, max_length = 4, 10, 5
    lengths = torch.tensor([2, 3, 4, 5])
    starts = torch.zeros((batch_size, edge_count), dtype=torch.long)
    ends = torch.ones_like(starts)
    mask = torch.zeros_like(starts, dtype=torch.bool)
    gold = torch.zeros_like(mask)
    allowed = torch.zeros_like(mask)
    generator = torch.Generator().manual_seed(71)
    for row, length in enumerate(lengths.tolist()):
        starts[row, :length] = torch.arange(length)
        ends[row, :length] = torch.arange(1, length + 1)
        mask[row, :length] = True
        gold[row, :length] = True
        allowed[row, :length] = True
        for edge_index in range(length, min(edge_count, length + 3)):
            start = int(torch.randint(length, (), generator=generator))
            starts[row, edge_index] = start
            ends[row, edge_index] = min(length, start + 2)
            mask[row, edge_index] = True
            allowed[row, edge_index] = edge_index % 2 == 0
            gold[row, edge_index] = edge_index % 3 == 0
    order, offsets = _sparse_topology(starts, mask, max_length)
    dense_scores = torch.randn(
        (batch_size, edge_count), generator=generator, requires_grad=True)
    sparse_scores = dense_scores.detach().clone().requires_grad_(True)
    dense = batched_partial_path_nll(
        dense_scores, starts, ends, mask, gold, lengths, allowed_mask=allowed)
    sparse = batched_partial_path_nll(
        sparse_scores, starts, ends, mask, gold, lengths,
        allowed_mask=allowed, edge_order=order, start_offsets=offsets,
        max_length=max_length)
    assert torch.allclose(sparse, dense, atol=1e-5, rtol=1e-5)
    dense.sum().backward()
    sparse.sum().backward()
    assert torch.allclose(sparse_scores.grad, dense_scores.grad,
                          atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(sparse_scores.grad[~mask]) == 0
    with torch.no_grad():
        inference = batched_partial_path_nll(
            sparse_scores, starts, ends, mask, gold, lengths,
            allowed_mask=allowed, edge_order=order, start_offsets=offsets,
            max_length=max_length)
    assert torch.allclose(inference, dense.detach(), atol=1e-5, rtol=1e-5)


def test_composite_keeps_copy_path_under_aggressive_caps():
    class DenseProvider:
        def build(self, text):
            return [Edge(0, len(text), text, f"よみ{index}", "unknown",
                         confidence=float(index)) for index in range(20)]

    provider = CompositeProvider(
        [DenseProvider()], per_span_reading_cap=2, total_edge_cap=1)
    edges = provider.build("未知")
    copies = [edge for edge in edges if edge.source == "copy"]
    assert [(edge.start, edge.end) for edge in copies] == [(0, 1), (1, 2)]
    copy_scores = torch.tensor([
        10.0 if edge.source == "copy" else 0.0 for edge in edges])
    assert decode_viterbi(copy_scores, edges, 2).reading == "未知"


def test_strict_collator_never_injects_missing_special_reading():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
             "痛": 4, "ぇ": 5}
    provider = CompositeProvider([])
    row = {"text": "痛ぇ", "B": ["いて", "ぇ"], "C": [0, 0],
           "D": ["無", "無"], "loss_mask": [1, 0],
           "special_spans": [{"start": 0, "end": 1, "reading": "いて"}]}
    batch = SparseEdgeCollator(provider, vocab, inject_gold=False)([row])
    assert all(edge.source != "gold" for edge in batch.edges[0])
    assert batch.gold_edge_mask is not None
    assert any(batch.gold_edge_mask[0, :len(batch.edges[0])])


def test_sparse_fused_nll_random_graph_values_and_gradients():
    random.seed(7)
    generator = torch.Generator().manual_seed(7)
    for _trial in range(20):
        batch_size, edge_count, max_length = 5, 14, 9
        starts = torch.zeros((batch_size, edge_count), dtype=torch.long)
        ends = torch.ones_like(starts)
        mask = torch.zeros_like(starts, dtype=torch.bool)
        gold = torch.zeros_like(mask)
        allowed = torch.zeros_like(mask)
        lengths = torch.randint(3, max_length + 1, (batch_size,),
                                generator=generator)
        for row, length in enumerate(lengths.tolist()):
            starts[row, :length] = torch.arange(length)
            ends[row, :length] = torch.arange(1, length + 1)
            mask[row, :length] = True
            gold[row, :length] = True
            allowed[row, :length] = True
            for edge_index in range(length, edge_count):
                start = random.randint(0, length - 1)
                starts[row, edge_index] = start
                ends[row, edge_index] = random.randint(
                    start + 1, min(length, start + 3))
                mask[row, edge_index] = random.random() > 0.2
                allowed[row, edge_index] = random.random() > 0.25
                gold[row, edge_index] = random.random() > 0.5
        order, offsets = _sparse_topology(starts, mask, max_length)
        dense_scores = torch.randn(
            (batch_size, edge_count), generator=generator, requires_grad=True)
        sparse_scores = dense_scores.detach().clone().requires_grad_(True)
        dense = batched_partial_path_nll(
            dense_scores, starts, ends, mask, gold, lengths,
            allowed_mask=allowed)
        sparse = batched_partial_path_nll(
            sparse_scores, starts, ends, mask, gold, lengths,
            allowed_mask=allowed, edge_order=order, start_offsets=offsets,
            max_length=max_length)
        assert torch.allclose(sparse, dense, atol=1e-5, rtol=1e-5)
        dense_gradient = torch.autograd.grad(dense.sum(), dense_scores)[0]
        sparse_gradient = torch.autograd.grad(sparse.sum(), sparse_scores)[0]
        assert torch.allclose(sparse_gradient, dense_gradient,
                              atol=1e-5, rtol=1e-5)


def test_collator_precomputes_sparse_dp_metadata():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
             "明": 4, "日": 5, "あ": 6, "す": 7, "め": 8, "い": 9}
    provider = LyricMemoryProvider({"明日": {"あす": 2}, "明": {"めい": 2}})
    rows = [
        {"text": "明日", "B": ["あ", "す"], "C": [0, 0],
         "D": ["無", "無"], "loss_mask": [1, 1]},
        {"text": "明", "B": ["めい"], "C": [0], "D": ["無"],
         "loss_mask": [1]},
    ]
    batch = SparseEdgeCollator(provider, vocab)(rows)
    assert batch.lengths.tolist() == [2, 1]
    assert batch.allowed_edge_mask.shape == batch.edge_mask.shape
    assert batch.gold_reachable.tolist() == [True, True]
    assert len(batch.dp_start_offsets) == batch.input_ids.size(1) + 1
    ordered = batch.dp_edge_order.tolist()
    assert len(ordered) == int(batch.edge_mask.sum())
    assert len(set(ordered)) == len(ordered)
    flattened_starts = batch.edge_start.reshape(-1)
    for start, (left, right) in enumerate(zip(
            batch.dp_start_offsets, batch.dp_start_offsets[1:])):
        assert all(int(flattened_starts[index]) == start
                   for index in ordered[left:right])


def test_collator_reachability_respects_locked_allowed_edges():
    class LockedConflictProvider:
        def build(self, text):
            return [
                Edge(0, 2, text, "えーびー", "haqumei", locked=True),
                Edge(0, 1, text[0], "あ", "unknown"),
                Edge(1, 2, text[1], "び", "unknown"),
            ]

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
             "A": 4, "B": 5, "あ": 6, "び": 7}
    row = {"text": "AB", "B": ["あ", "び"], "C": [0, 0],
           "D": ["無", "無"], "loss_mask": [1, 1]}
    batch = SparseEdgeCollator(LockedConflictProvider(), vocab)([row])
    assert batch.gold_reachable.tolist() == [False]
    assert batch.allowed_edge_mask[0, :3].tolist() == [True, False, False]


def test_generator_context_batches_spans_in_one_model_call():
    from kashi_g2p.data import generator_context

    class Recorder:
        calls = 0

        def _span_features(self, hidden, starts, ends):
            self.calls += 1
            return hidden.gather(
                1, starts.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))

    model = Recorder()
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    batch = pack_batch(["明日", "明"], LyricMemoryProvider({}), {
        "[PAD]": 0, "[UNK]": 1, "明": 2, "日": 3})
    batch.generator_batch_index = torch.tensor([0, 1])
    batch.generator_span_start = torch.tensor([1, 0])
    batch.generator_span_end = torch.tensor([2, 1])
    context = generator_context(model, hidden, batch)
    assert model.calls == 1
    assert torch.equal(context, torch.stack((hidden[0, 1], hidden[1, 0])))


def _deduplicated_test_batch():
    vocab = {"[PAD]": 0, "[UNK]": 1, "明": 2, "日": 3,
             "あ": 4, "す": 5, "め": 6, "い": 7}
    edges = [
        [Edge(0, 1, "明", "あす", "kanjidic"),
         Edge(0, 2, "明日", "あす", "lyric_memory"),
         Edge(1, 2, "日", "めい", "kanjidic")],
        [Edge(0, 1, "明", "あす", "copy")],
    ]
    return pack_edges(
        ["明日", "明"], edges, vocab, max_reading_length=4,
        dynamic_reading_length=True, deduplicate_readings=True)


def test_pack_edges_deduplicates_valid_readings_stably_and_keeps_metadata():
    batch = _deduplicated_test_batch()
    assert batch.unique_reading_ids is not None
    assert batch.unique_reading_mask is not None
    assert batch.edge_reading_inverse is not None
    assert batch.edge_reading_inverse.tolist() == [[0, 0, 1], [0, 0, 0]]
    assert batch.unique_reading_ids.tolist() == [[4, 5], [6, 7]]
    assert batch.unique_reading_mask.tolist() == [[True, True], [True, True]]
    moved = batch.to("cpu")
    assert batch.texts == ["明日", "明"]
    assert moved.texts == batch.texts
    assert moved.edges == batch.edges
    assert moved.source_to_id == batch.source_to_id
    assert [[edge.source for edge in row] for row in batch.edges] == [
        ["kanjidic", "lyric_memory", "kanjidic"], ["copy"]]
    assert batch.source_to_id["copy"] != batch.source_to_id["kanjidic"]
    assert batch.edge_reading_ids is not None
    assert batch.edge_reading_mask is not None
    assert not batch.edge_reading_mask[1, 1:].any()


def test_unique_model_kwargs_avoid_moving_dense_readings():
    batch = _deduplicated_test_batch()
    dense_ids = batch.edge_reading_ids
    dense_mask = batch.edge_reading_mask
    moved = batch.to("cpu")
    assert moved.edge_reading_ids is dense_ids
    assert moved.edge_reading_mask is dense_mask
    kwargs = moved.model_kwargs()
    assert "unique_reading_ids" in kwargs
    assert "edge_reading_inverse" in kwargs
    assert "edge_reading_ids" not in kwargs
    assert "edge_reading_mask" not in kwargs


def _small_dedup_model(*, reading_chunk_size=0, checkpointing=False):
    return SpanG2P(
        vocab_size=8, layers=2, dim=32, heads=4, d_ff=64,
        max_seq_len=8, max_reading_len=4, edge_dim=16, dropout=0.0,
        activation_checkpointing=checkpointing,
        reading_chunk_size=reading_chunk_size, use_open_generator=False)


def _dense_and_unique_kwargs(batch):
    common = {
        "input_ids": batch.input_ids, "attention_mask": batch.attention_mask,
        "edge_start": batch.edge_start, "edge_end": batch.edge_end,
        "edge_mask": batch.edge_mask, "edge_locked": batch.edge_locked,
        "edge_source_ids": batch.edge_source_ids, "edge_prior": batch.edge_prior,
    }
    dense = {**common, "edge_reading_ids": batch.edge_reading_ids,
             "edge_reading_mask": batch.edge_reading_mask}
    unique = {**common, "unique_reading_ids": batch.unique_reading_ids,
              "unique_reading_mask": batch.unique_reading_mask,
              "edge_reading_inverse": batch.edge_reading_inverse}
    return dense, unique


def test_dense_and_unique_outputs_losses_and_reading_gradients_match():
    torch.manual_seed(23)
    batch = _deduplicated_test_batch()
    dense_kwargs, unique_kwargs = _dense_and_unique_kwargs(batch)
    dense_model = _small_dedup_model()
    unique_model = _small_dedup_model()
    unique_model.load_state_dict(dense_model.state_dict(), strict=True)
    dense_output = dense_model(**dense_kwargs)
    unique_output = unique_model(**unique_kwargs)
    assert torch.allclose(
        dense_output.edge_scores, unique_output.edge_scores, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        dense_output.edge_features, unique_output.edge_features,
        atol=1e-6, rtol=1e-6)
    dense_loss = dense_output.edge_scores[batch.edge_mask].square().mean()
    unique_loss = unique_output.edge_scores[batch.edge_mask].square().mean()
    assert torch.allclose(dense_loss, unique_loss, atol=1e-7, rtol=1e-7)
    dense_loss.backward()
    unique_loss.backward()
    for name in ("char_embed.weight", "reading_conv.weight",
                 "reading_norm.weight", "reading_proj.weight"):
        dense_grad = dict(dense_model.named_parameters())[name].grad
        unique_grad = dict(unique_model.named_parameters())[name].grad
        assert torch.allclose(dense_grad, unique_grad, atol=1e-6, rtol=1e-5), name


def test_unique_reading_chunk_sizes_match_and_checkpoint_backward():
    torch.manual_seed(29)
    batch = _deduplicated_test_batch()
    _dense, unique_kwargs = _dense_and_unique_kwargs(batch)
    baseline = _small_dedup_model(reading_chunk_size=0)
    baseline_output = baseline(**unique_kwargs).edge_scores
    state = baseline.state_dict()
    assert set(state) == set(_small_dedup_model().state_dict())
    for chunk_size in (1, 16):
        model = _small_dedup_model(reading_chunk_size=chunk_size)
        model.load_state_dict(state, strict=True)
        output = model(**unique_kwargs).edge_scores
        assert torch.allclose(output, baseline_output, atol=1e-6, rtol=1e-6)
    checkpointed = _small_dedup_model(
        reading_chunk_size=1, checkpointing=True)
    checkpointed.load_state_dict(state, strict=True)
    loss = checkpointed(**unique_kwargs).edge_scores[batch.edge_mask].sum()
    loss.backward()
    assert checkpointed.reading_proj.weight.grad is not None
    assert torch.isfinite(checkpointed.reading_proj.weight.grad).all()


def test_unique_path_handles_no_valid_edges_without_padding_pollution():
    batch = pack_edges(
        [""], [[]], {"[PAD]": 0, "[UNK]": 1},
        max_reading_length=2, deduplicate_readings=True)
    assert batch.unique_reading_ids.shape == (0, 2)
    assert batch.edge_reading_inverse.tolist() == [[0]]
    model = SpanG2P(
        vocab_size=2, layers=1, dim=32, heads=4, d_ff=64,
        max_seq_len=2, max_reading_len=2, edge_dim=16, dropout=0.0,
        use_open_generator=False)
    output = model(**batch.model_kwargs())
    assert torch.count_nonzero(output.edge_features[0, 0, 16:32]) == 0
    assert torch.isfinite(output.edge_features).all()
    assert torch.all(output.edge_scores == torch.finfo(output.edge_scores.dtype).min)


def test_default_model_stays_below_the_hard_parameter_budget():
    model = SpanG2P()
    count = parameter_count(model)
    assert 30_000_000 <= count < 40_000_000, count


def test_fold_notation_equivalence():
    from kashi_g2p.evaluate import fold_notation

    assert fold_notation("") == ""
    # Yotsugana
    assert fold_notation("いちにちぢゅう") == fold_notation("いちにちじゅう")
    assert fold_notation("つづく") == fold_notation("つずく")
    # Long vowels (ー vs elongation)
    assert fold_notation("そう") == fold_notation("そー") == "そー"
    assert fold_notation("れい") == fold_notation("れー") == "れー"
    assert fold_notation("ちょうりゅう") == fold_notation("ちょーりゅー") == "ちょーりゅー"
    assert fold_notation("おおきい") == fold_notation("おーきい") == "おーきー"
    assert fold_notation("いいえ") == fold_notation("いーえ") == "いーえ"
    # Preserves string length
    s = "いちにちじゅう"
    folded = fold_notation(s)
    assert len(folded) == len(s)


def test_gold_injection_prob_controls_missing_edge_injection():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "痛": 4, "ぇ": 5}
    provider = CompositeProvider([])
    row = {"text": "痛ぇ", "B": ["いて", "ぇ"], "C": [0, 0],
           "D": ["無", "無"], "loss_mask": [1, 0],
           "special_spans": [{"start": 0, "end": 1, "reading": "いて"}]}

    # 0.0 -> missing special reading never injected
    batch_zero = SparseEdgeCollator(provider, vocab, gold_injection_prob=0.0)([row])
    assert all(edge.source != "gold" for edge in batch_zero.edges[0])

    # 1.0 -> missing special reading always injected
    batch_one = SparseEdgeCollator(provider, vocab, gold_injection_prob=1.0)([row])
    assert any(edge.source == "gold" for edge in batch_one.edges[0])

    # Invalid range raises ValueError
    with pytest.raises(ValueError):
        SparseEdgeCollator(provider, vocab, gold_injection_prob=-0.1)
    with pytest.raises(ValueError):
        SparseEdgeCollator(provider, vocab, gold_injection_prob=1.5)


def test_pyopenjtalk_provider():
    from kashi_g2p.candidates import PyOpenJTalkProvider

    provider = PyOpenJTalkProvider()
    text = "夜が更ける頃に君を思い出す"
    edges = provider.build(text)
    assert len(edges) > 0
    for edge in edges:
        assert edge.source == "pyopenjtalk"
        assert 0.0 < edge.confidence <= 1.0
        edge.validate(text)
    readings = {(edge.start, edge.end, edge.surface): edge.reading for edge in edges}
    assert (0, 1, "夜") in readings
    assert readings[(0, 1, "夜")] == "よる"


def test_composite_provider_with_pyopenjtalk():
    provider = CompositeProvider.from_resources(
        use_pyopenjtalk=True,
        use_rules=True,
        use_haqumei=False,
        use_legacy_lexicon=False,
    )
    edges = provider.build("夜が更ける頃に君を思い出す")
    sources = {e.source for e in edges}
    assert "pyopenjtalk" in sources
    assert "copy" in sources
    # Validate entire path
    assert any(e.source == "pyopenjtalk" and e.surface == "夜" and e.reading == "よる" for e in edges)


def test_okurigana_truncation_filter():
    from kashi_g2p.candidates import CompositeProvider
    from kashi_g2p.model_types import Edge
    text = "この気持ちだって"
    e_short = Edge(2, 4, "気持", "きもち", "jmdict")
    e_long = Edge(2, 5, "気持ち", "きもち", "pyopenjtalk")
    e_copy = Edge(4, 5, "ち", "ち", "copy")
    filtered = CompositeProvider._filter_truncated_okurigana([e_short, e_long, e_copy], text)
    surfaces = [e.surface for e in filtered]
    assert "気持" not in surfaces
    assert "気持ち" in surfaces
    assert "ち" in surfaces


def test_derive_okurigana_stems():
    from kashi_g2p.candidates import CompositeProvider
    from kashi_g2p.model_types import Edge

    def stems(edges, text):
        out = CompositeProvider._derive_okurigana_stems(edges, text)
        return {(e.start, e.end, e.surface, e.reading) for e in out} - {
            (e.start, e.end, e.surface, e.reading) for e in edges}

    # kanji + trailing okurigana: derive the truncated single-kanji stem
    text = "どれも失くして"
    word = Edge(3, 6, "失くし", "なくし", "jmdict")
    assert (3, 4, "失", "な") in stems([word], text)

    # multi-kanji stem (気付く -> きづく) derives 気付 -> きづ over the kanji block
    text2 = "気付いて"
    word2 = Edge(0, 3, "気付い", "きづい", "unidic")
    assert (0, 2, "気付", "きづ") in stems([word2], text2)

    # compound where the target kanji is not the prefix (真っ青) has no trailing
    # hiragana okurigana -> nothing derived (pure-kanji-stem guard)
    text3 = "真っ青だ"
    word3 = Edge(0, 3, "真っ青", "まっさお", "jmdict")
    assert stems([word3], text3) == set()

    # reading must literally end with the okurigana kana; otherwise skip
    text4 = "灯り"
    bad = Edge(0, 2, "灯り", "あかるい", "jmdict")   # does not end with り
    assert stems([bad], text4) == set()

    # no duplicate when the stem already exists
    text5 = "失くし"
    word5 = Edge(0, 3, "失くし", "なくし", "jmdict")
    have = Edge(0, 1, "失", "な", "kanjidic")
    assert stems([word5, have], text5) == set()


def test_rendaku_trap_filter():
    from kashi_g2p.candidates import CompositeProvider
    from kashi_g2p.model_types import Edge
    # Former hardcoded cases still handled by the general voicing rule.
    e_archaic = Edge(0, 3, "大丈夫", "だいじょうふ", "jmdict")
    e_modern = Edge(0, 3, "大丈夫", "だいじょうぶ", "pyopenjtalk")
    readings = [e.reading for e in
                CompositeProvider._filter_rendaku_traps([e_archaic, e_modern])]
    assert "だいじょうふ" not in readings
    assert "だいじょうぶ" in readings

    # Generalizes to any word not in any table (medial こ→ご voicing).
    e_unvoiced = Edge(0, 2, "山川", "やまかわ", "jmdict")
    e_voiced = Edge(0, 2, "山川", "やまがわ", "unidic")
    readings = [e.reading for e in
                CompositeProvider._filter_rendaku_traps([e_unvoiced, e_voiced])]
    assert "やまかわ" not in readings and "やまがわ" in readings

    # Does NOT fire on initial-mora voicing (that is _expand_word_variants'
    # job; suppressing it here would drop legitimate non-rendaku readings).
    e_a = Edge(0, 2, "母", "はは", "jmdict")
    e_b = Edge(0, 2, "母", "ばば", "jmdict")
    readings = [e.reading for e in
                CompositeProvider._filter_rendaku_traps([e_a, e_b])]
    assert "はは" in readings and "ばば" in readings

    # A lone unvoiced reading (no voiced sibling) is untouched.
    solo = Edge(0, 2, "水川", "みずかわ", "jmdict")
    readings = [e.reading for e in CompositeProvider._filter_rendaku_traps([solo])]
    assert readings == ["みずかわ"]


def test_first_order_viterbi_matches_zero_order_when_transition_zero():
    edges = [
        Edge(0, 1, "明", "めい", "kanjidic"),
        Edge(1, 2, "日", "にち", "kanjidic"),
        Edge(0, 2, "明日", "あす", "lyric_memory"),
    ]
    scores = torch.tensor([0.1, 0.1, 2.0])
    trans = torch.zeros((len(edges), len(edges)))
    res_zero = decode_viterbi(scores, edges, 2)
    res_first = decode_viterbi(scores, edges, 2, transition_scores=trans)
    assert res_zero.reading == res_first.reading
    assert res_zero.edge_indices == res_first.edge_indices
    assert math.isclose(res_zero.score, res_first.score, rel_tol=1e-5)


def test_first_order_viterbi_influences_disambiguation():
    # 何(なに: score=2.0 vs なん: score=1.8) + だろう(score=1.0)
    # 单边分 "なに" 稍高，但转移偏置给 "なん" + "だろう" +1.0
    edges = [
        Edge(0, 1, "何", "なに", "kanjidic"),   # index 0
        Edge(0, 1, "何", "なん", "kanjidic"),   # index 1
        Edge(1, 4, "だろう", "だろう", "rules"), # index 2
    ]
    scores = torch.tensor([2.0, 1.8, 1.0])
    # 无转移时，首选 "なに" + "だろう" -> "なにだろう"
    res_0 = decode_viterbi(scores, edges, 4)
    assert res_0.reading == "なにだろう"

    # 加入边对转移：1 -> 2 ("なん" -> "だろう") 奖励 +1.0
    trans = torch.zeros((len(edges), len(edges)))
    trans[1, 2] = 1.0
    res_1 = decode_viterbi(scores, edges, 4, transition_scores=trans)
    assert res_1.reading == "なんだろう"
    assert res_1.edge_indices == [1, 2]


def _enumerate_covering_paths(starts, ends, length, allowed):
    """All non-overlapping edge-index paths covering ``[0, length)``."""
    out = []

    def walk(pos, acc):
        if pos == length:
            out.append(list(acc))
            return
        for i in allowed:
            if starts[i] == pos and ends[i] <= length:
                acc.append(i)
                walk(ends[i], acc)
                acc.pop()

    walk(0, [])
    return out


def _ref_first_order_nll(scores, trans, starts, ends, length, gold_mask):
    """Differentiable brute-force first-order full/gold NLL over a tiny graph."""
    def log_z(allowed):
        terms = []
        for path in _enumerate_covering_paths(starts, ends, length, allowed):
            score = sum(scores[i] for i in path)
            for a, b in zip(path, path[1:]):
                score = score + trans[a, b]
            terms.append(score)
        return torch.logsumexp(torch.stack(terms), 0)

    full = list(range(len(starts)))
    gold = [i for i in range(len(starts)) if gold_mask[i]]
    return log_z(full) - log_z(gold)


# 明/日/明日 lattice over "明日": path {明,日} vs the single word edge {明日}.
_FO_STARTS = [0, 1, 0]
_FO_ENDS = [1, 2, 2]
_FO_GOLD = [True, True, False]   # gold admits only the char path 明+日


def test_first_order_partition_reduces_to_zeroth_when_transition_zero():
    scores = torch.tensor([[0.3, -0.4, 1.1]])
    starts = torch.tensor([_FO_STARTS])
    ends = torch.tensor([_FO_ENDS])
    mask = torch.ones(1, 3, dtype=torch.bool)
    gold = torch.tensor([_FO_GOLD])
    lengths = torch.tensor([2])
    zero_trans = torch.zeros(1, 3, 3)

    zeroth = batched_partial_path_nll(scores, starts, ends, mask, gold, lengths)
    first = batched_partial_path_nll(
        scores, starts, ends, mask, gold, lengths, transition=zero_trans)
    assert torch.allclose(zeroth, first, atol=1e-5)


def test_first_order_partition_matches_enumeration():
    from kashi_g2p.decoder import _first_order_forward_backward

    scores = torch.tensor([[0.3, -0.4, 1.1]], dtype=torch.float64)
    trans = torch.tensor([[[0.0, 0.7, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0]]], dtype=torch.float64)
    starts = torch.tensor([_FO_STARTS])
    ends = torch.tensor([_FO_ENDS])
    valid = torch.ones(1, 3, dtype=torch.bool)
    gold_valid = torch.tensor([_FO_GOLD])
    lengths = torch.tensor([2])

    z_full, _, _ = _first_order_forward_backward(
        scores, trans, starts, ends, valid, lengths, 2)
    z_gold, _, _ = _first_order_forward_backward(
        scores, trans, starts, ends, gold_valid, lengths, 2)

    # full = {明,日}(0.3-0.4+0.7) and {明日}(1.1); gold = {明,日} only.
    ref_full = torch.logsumexp(torch.tensor(
        [0.3 - 0.4 + 0.7, 1.1], dtype=torch.float64), 0)
    ref_gold = torch.tensor(0.3 - 0.4 + 0.7, dtype=torch.float64)
    assert torch.allclose(z_full[0], ref_full, atol=1e-9)
    assert torch.allclose(z_gold[0], ref_gold, atol=1e-9)


def test_first_order_partition_handles_mixed_lengths_in_a_batch():
    # Two rows of different length share one padded batch. A shorter row's
    # terminal edge sits below the batch max length; the backward pass must not
    # wipe it out. With zero transition the first-order NLL must match the
    # exact 0th-order NLL row for row.
    starts = torch.tensor([[0, 1, 0, 0],
                           [0, 1, 2, 0]])
    ends = torch.tensor([[1, 2, 2, 0],
                         [1, 2, 3, 3]])
    mask = torch.tensor([[True, True, True, False],
                         [True, True, True, True]])
    gold = torch.tensor([[True, True, False, False],
                         [True, True, True, False]])
    scores = torch.tensor([[0.3, -0.4, 1.1, 0.0],
                           [0.2, 0.5, -0.1, 0.9]])
    lengths = torch.tensor([2, 3])
    zero_trans = torch.zeros(2, 4, 4)

    zeroth = batched_partial_path_nll(scores, starts, ends, mask, gold, lengths)
    first = batched_partial_path_nll(
        scores, starts, ends, mask, gold, lengths, transition=zero_trans)
    assert torch.allclose(zeroth, first, atol=1e-5)


def test_first_order_partition_gradients_match_enumeration():
    starts = torch.tensor([_FO_STARTS])
    ends = torch.tensor([_FO_ENDS])
    mask = torch.ones(1, 3, dtype=torch.bool)
    gold = torch.tensor([_FO_GOLD])
    lengths = torch.tensor([2])

    scores = torch.tensor([[0.3, -0.4, 1.1]], requires_grad=True)
    trans = torch.zeros(1, 3, 3)
    with torch.no_grad():
        trans[0, 0, 1] = 0.7
    trans.requires_grad_(True)
    nll = batched_partial_path_nll(
        scores, starts, ends, mask, gold, lengths, transition=trans)
    nll.sum().backward()

    scores_ref = torch.tensor([0.3, -0.4, 1.1], requires_grad=True)
    trans_ref = torch.zeros(3, 3)
    trans_ref[0, 1] = 0.7
    trans_ref.requires_grad_(True)
    ref = _ref_first_order_nll(
        scores_ref, trans_ref, _FO_STARTS, _FO_ENDS, 2, _FO_GOLD)
    ref.backward()

    assert torch.allclose(scores.grad[0], scores_ref.grad, atol=1e-5)
    assert torch.allclose(trans.grad[0], trans_ref.grad, atol=1e-5)


def _tiny_transition_batch():
    vocab = {"[PAD]": 0, "[UNK]": 1, "明": 2, "日": 3, "A": 4}
    provider = CompositeProvider([
        EnglishNumberProvider(),
        LyricMemoryProvider({"明日": {"あす": 2}}),
        KanjidicProvider(Kanjidic({"明": {"めい": "音"}, "日": {"にち": "音"}})),
    ])
    row = {"text": "明日A", "B": ["めい", "日", "A"],
           "C": [0, 0, 0], "D": ["無", "無", "無"],
           "loss_mask": [1, 0, 0],
           "special_spans": [{"start": 0, "end": 2, "reading": "あす"}]}
    return SparseEdgeCollator(provider, vocab)([row])


def test_first_order_loss_trains_the_transition_head():
    # With first_order_loss the transition head must receive gradient through
    # the path loss -- the whole point of the fix.
    batch = _tiny_transition_batch()
    model = SpanG2P(vocab_size=5, layers=1, dim=32, heads=4, d_ff=64,
                    max_seq_len=8, max_reading_len=64, edge_dim=16,
                    use_transition_head=True, transition_dim=8,
                    first_order_loss=True)
    loss = _batch_loss(model, batch, torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    grad = model.transition_head.left_proj.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_zeroth_order_leaves_transition_head_untrained():
    # Control: without first_order_loss the head stays a detached decode-only
    # feature and receives no gradient (the pre-fix behavior, kept for exp35).
    batch = _tiny_transition_batch()
    model = SpanG2P(vocab_size=5, layers=1, dim=32, heads=4, d_ff=64,
                    max_seq_len=8, max_reading_len=64, edge_dim=16,
                    use_transition_head=True, transition_dim=8,
                    first_order_loss=False)
    loss = _batch_loss(model, batch, torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.transition_head.left_proj.weight.grad is None


def test_model_with_per_layer_seeds_and_transition_head():
    m = SpanG2P(
        vocab_size=100, layers=4, dim=64, heads=2, d_ff=128,
        fusion_layers=[2, 4], use_per_layer_edge_seed=True,
        use_transition_head=True, transition_dim=32,
        use_open_generator=False, use_prior_table=False
    )
    assert m.use_per_layer_edge_seed
    assert "2" in m.edge_seeds and "4" in m.edge_seeds
    assert m.transition_head is not None


def test_windowed_context_collator_and_forward():
    class DummyProvider:
        def build(self, text):
            return [Edge(0, len(text), text, text, "copy")]

    vocab = {"[PAD]": 0, "[UNK]": 1, "私": 2, "は": 3, "猫": 4, "で": 5, "す": 6}
    collator = SparseEdgeCollator(
        DummyProvider(), vocab, max_reading_length=16,
        generator_supervision=False)

    rows = [{
        "text": "猫です",
        "left_context": "私は",
        "right_context": "。",
        "loss_mask": [1, 1, 1],
        "B": ["ねこ", "で", "す"],
    }]

    batch = collator(rows)
    assert batch.target_offset is not None
    assert int(batch.target_offset[0]) == 2  # len("私は") == 2
    assert int(batch.target_length[0]) == 3  # len("猫です") == 3

    model = SpanG2P(
        vocab_size=100, layers=2, dim=32, heads=2, d_ff=64,
        fusion_layers=[2], use_open_generator=False, use_prior_table=False)
    output = model(**batch.model_kwargs())
    assert output.edge_scores.shape == batch.edge_mask.shape


def test_english_number_provider_counters():
    provider = EnglishNumberProvider()
    
    # 3人 -> さんにん
    edges_3nin = {(e.surface, e.reading) for e in provider.build("3人のブギーマン")}
    assert ("3人", "さんにん") in edges_3nin
    assert ("3", "さん") in {(e.surface, e.reading) for e in provider.build("3")}

    # 10曲 -> じゅっきょく
    edges_10kyoku = {(e.surface, e.reading) for e in provider.build("10曲目")}
    assert ("10曲", "じゅっきょく") in edges_10kyoku

    # 24時間 -> にじゅうよじかん
    edges_24h = {(e.surface, e.reading) for e in provider.build("24時間営業")}
    assert ("24時間", "にじゅうよじかん") in edges_24h

    # 1つ -> ひとつ, 2つ -> ふたつ
    edges_tsu = {(e.surface, e.reading) for e in provider.build("1つ2つ")}
    assert ("1つ", "ひとつ") in edges_tsu
    assert ("2つ", "ふたつ") in edges_tsu


def test_english_word_no_letter_spelling_explosion():
    provider = EnglishNumberProvider()
    
    # Uppercase acronyms <= 4 chars are spelled out
    edges_dj = {e.surface: e.reading for e in provider.build("DJ")}
    assert "DJ" in edges_dj and edges_dj["DJ"] == "でぃーじぇー"

    # Known lyric word
    edges_joyful = {e.surface: e.reading for e in provider.build("Joyful")}
    assert "Joyful" in edges_joyful and edges_joyful["Joyful"] == "じょいふる"

    # Unseen normal lowercase/mixed word should NOT explode into letter spelling
    edges_unknown = {e.surface: e.reading for e in provider.build("covered")}
    assert "covered" not in edges_unknown


def test_alnum_table_provider_case_insensitive():
    table = {"JOYFUL": {"じょいふる": 100}}
    provider = AlnumTableProvider(table)
    edges = {e.surface: e.reading for e in provider.build("Joyful day")}
    assert "Joyful" in edges
    assert edges["Joyful"] == "じょいふる"


def test_filter_truncated_okurigana_drops_duplicate():
    # When text is '気持ち' and dictionary gives both '気持' -> 'きもち' and '気持ち' -> 'きもち',
    # the shorter '気持' -> 'きもち' is dropped to prevent 'きもちち' stutter.
    text = "気持ち"
    edges = [
        Edge(0, 2, "気持", "きもち", "yomogi_dict"),
        Edge(0, 3, "気持ち", "きもち", "yomogi_dict"),
        Edge(2, 3, "ち", "ち", "copy"),
    ]
    filtered = CompositeProvider._filter_truncated_okurigana(edges, text)
    surfaces = {(e.start, e.end, e.surface, e.reading) for e in filtered}
    assert (0, 3, "気持ち", "きもち") in surfaces
    assert (0, 2, "気持", "きもち") not in surfaces
    assert (2, 3, "ち", "ち") in surfaces


def test_morpho_penalty_digit_copy_and_verb_inflection():
    # 1. Digit COPY + Kanji counter transition gets penalized
    e_digit_copy = Edge(0, 1, "8", "8", "copy")
    e_gatsu = Edge(1, 2, "月", "がつ", "jmdict")
    assert _morpho_penalty(e_digit_copy, e_gatsu) == -10.0

    # Phonetic digit + Kanji counter has NO penalty
    e_digit_hachi = Edge(0, 1, "8", "はち", "rules")
    assert _morpho_penalty(e_digit_hachi, e_gatsu) == 0.0

    # 2. On-yomi noun stem + verb te-form inflection gets penalized
    e_bishou = Edge(0, 2, "微笑", "びしょう", "jmdict")
    e_nde = Edge(2, 4, "んで", "んで", "copy")
    assert _morpho_penalty(e_bishou, e_nde) == -10.0

    # Kun-yomi verb stem + verb te-form inflection has NO penalty
    e_hohoe = Edge(0, 2, "微笑", "ほほえ", "pyopenjtalk")
    assert _morpho_penalty(e_hohoe, e_nde) == 0.0


def test_unified_lexicon_provider():
    from pathlib import Path
    pack_path = Path("artifacts/lexicon_pack/unified_lexicon.pkl.gz")
    if not pack_path.exists():
        return
    from kashi_g2p.unified_provider import UnifiedLexiconProvider
    provider = UnifiedLexiconProvider.from_pack(pack_path, use_rules=True, use_copy=True)

    # 1. Alphanumeric
    edges = list(provider.build("DJ"))
    assert any(e.surface == "DJ" and e.reading == "でぃーじぇい" for e in edges)

    # 2. Counter compound
    edges_counter = list(provider.build("3人"))
    assert any(e.surface == "3人" and e.reading == "さんにん" for e in edges_counter)

    # 3. Kanji word with furigana
    edges_word = list(provider.build("言葉"))
    assert any(e.surface == "言葉" and e.reading == "ことば" for e in edges_word)



