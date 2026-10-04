"""Train the compact v4 span reranker.

Example:
    python -m kashi_g2p.train --config kashi_g2p/config.yaml
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import time
import math
from pathlib import Path
import random
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None

from .candidates import CompositeProvider
from .data import (SparseEdgeCollator, auxiliary_losses, build_loader,
                   default_vocab, load_component_ids)
from .decoder import (allowed_edge_mask, batched_partial_path_nll,
                      decode_viterbi)
from .model import SpanG2P, parameter_count
from .v5_model import SpanG2Pv5, kana_token_ids


DEFAULT_CONFIG: dict[str, Any] = {
    "data": ["artifacts/labeled_lyrics_v3/train.jsonl"],
    "eval_data": ["artifacts/labeled_lyrics_v3/dev_sample50k_kanji.jsonl"],
    "vocab": "models/bert-base-japanese-char-v3/vocab.txt",
    "kanjidic": "resources/kanjidic2.xml.gz",
    "jmdict": "resources/JMdict_e.gz",
    "unidic": "resources/unidic",
    "ids": "resources/ids.txt",
    "ids_format": "chise_v1",
    "source_sample_rates": None,
    "source_interleave": 256,
    "max_length": 64,
    "overlap": 16,
    "max_reading_length": 64,
    # batch_size is the physical micro-batch. grad_accum restores the desired
    # optimizer batch without keeping all examples' activations alive.
    "batch_size": 8,
    "grad_accum": 2,
    "eval_batch_size": 8,
    "num_workers": 0,
    "pin_memory": None,
    "bucket_buffer_size": 128,
    "length_buckets": [16, 32, 48, 64],
    "dynamic_reading_length": True,
    "deduplicate_readings": True,
    "lexicon_pack": None,
    "use_prior_table": False,
    "prior_dim": 4,
    "yomogi_warm_start": False,
    "yomogi_model": None,
    "amp_dtype": "auto",
    "activation_checkpointing": True,
    "reading_chunk_size": 16,
    "steps": 10000,
    # Optional absolute optimizer-step cap for the while loop (short-process
    # chunked runs). When set, training stops (and saves a full checkpoint)
    # at this step, but config["steps"] stays the cosine LR schedule horizon.
    "stop_after_step": None,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "float32_matmul_precision": "high",
    "use_tqdm": True,
    "tensorboard": False,
    "tensorboard_logdir": None,
    "eval_every": 500,
    "eval_row_limit": None,
    "save_every": 500,
    "save": "artifacts/experiments/v4/checkpoints/model.pt",
    "best_save": "artifacts/experiments/v4/checkpoints/best_model.pt",
    "resume": None,
    "profile_steps": False,
    "empty_cache_every": 200,
    "init_from": None,
    "device": None,
    "use_haqumei": True,
    "use_rules": True,
    "use_legacy_lexicon": True,
    "legacy_node_cap": 512,
    "use_lyric_memory": False,
    "lyric_min_count": 2,
    "alnum_table": None,
    "yomogi_dict": None,
    "max_edges": 256,
    "max_readings_per_span": 8,
    "layers": 8,
    "dim": 512,
    "heads": 8,
    "d_ff": 1280,
    "edge_dim": 256,
    "component_dim": 256,
    "fusion_layers": [2, 4, 6, 8],
    "fusion_heads": 8,
    "max_span_length": 16,
    "use_open_generator": True,
    "generator_inference": False,
    "generator_dim": 256,
    "generator_layers": 3,
    "generator_heads": 8,
    "generator_d_ff": 640,
    "generator_max_length": 64,
    "boundary_loss_weight": 0.1,
    "generator_loss_weight": 0.2,
    "warmup_steps": 500,
    "min_lr_ratio": 0.1,
    "best_metric": "cer",
    "resource_manifest": True,
    "parameter_limit": 40000000,
    "seed": 42,
    "dropout": 0.1,
    "reading_kernel_size": 5,
    "max_seq_len": 256,
    "source_count": 16,
    "gold_injection_prob": 0.0,
    "use_pyopenjtalk": False,
    "use_per_layer_edge_seed": False,
    "use_transition_head": False,
    "transition_dim": 64,
    "transition_features": "context",
    "first_order_loss": False,
    "coarsen_segmentation": True,
    "transition_lr": None,
    "transition_weight_decay": 0.0,
    "transition_init_std": 0.02,
    # V5 architecture (docs/V5_ARCHITECTURE.md). model_arch selects the class;
    # everything below only applies when model_arch == "v5".
    "model_arch": "v4",
    # model_arch == "v4bert": directory written by
    # kashi_g2p_bert.bootstrap (config.json + sliced.pt).
    "bert_slice_dir": "models/bert-char-v3-slice6",
    "use_edge_fusion": False,
    "reading_layers": 4,
    "reading_heads": 8,
    "reading_d_ff": 1024,
    "use_relation_attention": False,
    "relation_layers": 2,
    "relation_heads": 8,
    "relation_d_ff": 1024,
    "read_loss_weight": 0.5,
    "contrast_loss_weight": 0.1,
    "contrast_margin": 1.0,
    "max_contrast_pairs": 2048,
    "contrast_mode": "all",
    "edge_dim_v5": 384,
}


def _load_config(path: str | Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("--config requires PyYAML; install pyyaml") from exc
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("v4 config must be a mapping")
    if isinstance(value.get("training"), dict):
        value = value["training"]
    unknown = sorted(set(value) - set(config))
    if unknown:
        raise ValueError("unknown v4 config key(s): " + ", ".join(unknown))
    config.update(value)
    return config


def _as_paths(value: str | Path | list[str | Path]) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else [str(value)]


def _device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_pin_memory(value: Any, device: torch.device) -> bool:
    """Enable pinned host batches by default only for CUDA training."""
    if value is None or (isinstance(value, str)
                         and value.strip().lower() in {"", "auto", "none"}):
        return device.type == "cuda"
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        name = value.strip().lower()
        if name in {"true", "yes", "on", "1"}:
            return True
        if name in {"false", "no", "off", "0"}:
            return False
    raise ValueError("pin_memory must be null/auto or a boolean")


def _resolve_amp_dtype(value: Any, device: torch.device) -> torch.dtype | None:
    """Resolve the portable AMP setting once, before the training loop."""
    name = "auto" if value is None else str(value).strip().lower()
    if name in {"none", "off", "false", "fp32", "float32"}:
        return None
    if name in {"auto", ""}:
        if device.type == "cuda":
            return (torch.bfloat16 if torch.cuda.is_bf16_supported()
                    else torch.float16)
        if device.type == "cpu":
            return torch.bfloat16
        return None
    if name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 AMP was requested but this CUDA device does not support it")
        if device.type not in {"cuda", "cpu"}:
            raise ValueError(f"BF16 AMP is not supported on device {device}")
        return dtype
    if name in {"fp16", "float16", "half"}:
        if device.type != "cuda":
            raise ValueError("FP16 AMP is supported only on CUDA; use bf16 or none on CPU")
        return torch.float16
    raise ValueError(
        "amp_dtype must be one of auto, bf16, fp16, or none; "
        f"got {value!r}"
    )


def _autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _make_scaler(device: torch.device, dtype: torch.dtype | None):
    """BF16 needs no loss scaler; FP16 gets one for CUDA underflow control."""
    if device.type != "cuda" or dtype != torch.float16:
        return None
    return torch.amp.GradScaler("cuda")


def _provider(config: dict[str, Any], *, memory: bool) -> CompositeProvider:
    return CompositeProvider.from_resources(
        kanjidic_path=config.get("kanjidic"),
        jmdict_path=config.get("jmdict"),
        unidic_dir=config.get("unidic"),
        lyric_train=(config.get("data") if memory and config.get("use_lyric_memory") else None),
        include_okurigana=True,
        use_haqumei=bool(config.get("use_haqumei", True)),
        use_pyopenjtalk=bool(config.get("use_pyopenjtalk", False)),
        use_rules=bool(config.get("use_rules", True)),
        use_legacy_lexicon=bool(config.get("use_legacy_lexicon", True)),
        legacy_node_cap=int(config.get("legacy_node_cap", 512)),
        lyric_min_count=int(config.get("lyric_min_count", 2)),
        alnum_table=config.get("alnum_table"),
        yomogi_dict=config.get("yomogi_dict"),
        lexicon_pack=(config.get("lexicon_pack")
                      if bool(config.get("use_prior_table", False)) else None),
        per_span_reading_cap=int(config.get("max_readings_per_span", 8)),
        total_edge_cap=int(config.get("max_edges", 256)),
    )


_drop_counter: list[int] = [0]


def _row_reachable(edges, compatible: Sequence[bool], length: int) -> bool:
    reachable = [False] * (length + 1)
    reachable[0] = True
    for start in range(length):
        if not reachable[start]:
            continue
        for edge, ok in zip(edges, compatible):
            if ok and edge.start == start and start < edge.end <= length:
                reachable[edge.end] = True
    return reachable[length]


def _zero_path_loss(output, device_batch) -> torch.Tensor:
    return output.edge_scores[device_batch.edge_mask].sum() * 0.0


def _warm_start_prior_table(model: SpanG2P, *, prior_index, yomogi_tsv: str,
                            yomogi_ckpt: str) -> int:
    """Seed the per-entry prior embeddings with Yomogi's entry scores.

    Yomogi's output_layer assigns every dictionary entry a learned 32-dim
    quality vector trained on 72M pairs; entries shared with our pack are
    projected to prior_dim via PCA and rescaled, giving the table a warm
    start that encodes external reading-quality knowledge (MIT licensed).
    """
    import unicodedata

    state = torch.load(yomogi_ckpt, map_location="cpu", weights_only=False)
    weights = (state.get("model", state)["output_layer.weight"]
               .float() if isinstance(state, dict) else None)
    if weights is None:
        return 0

    def to_hira(text: str) -> str:
        return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c
                       for c in text)

    ids: list[int] = []
    vectors: list[Tensor] = []
    with Path(yomogi_tsv).open(encoding="utf-8") as handle:
        for line in handle:
            row = line.rstrip(chr(10)).split(chr(9))
            if len(row) != 4:
                continue
            try:
                dict_id = int(row[0])
            except ValueError:
                continue
            surface = unicodedata.normalize("NFKC", row[1])
            reading = to_hira(row[2])
            pack_id = prior_index.get((surface, reading))
            if pack_id is not None:
                if not 0 <= dict_id < weights.shape[0]:
                    raise ValueError(
                        f"yomogi TSV id {dict_id} outside checkpoint weights "
                        f"{tuple(weights.shape)}; warm start expects a single "
                        "TSV whose ids index yomogi output_layer rows")
                ids.append(pack_id)
                vectors.append(weights[dict_id])
    if not vectors:
        return 0
    matrix = torch.stack(vectors)
    matrix = matrix - matrix.mean(0, keepdim=True)
    _, _, components = torch.pca_lowrank(matrix, q=4)
    projected = matrix @ components
    scale = 0.2 / projected.std().clamp_min(1e-6)
    projected = projected * scale
    with torch.no_grad():
        for row_id, vector in zip(ids, projected):
            model.prior_table.weight[row_id] = vector.to(
                model.prior_table.weight.device)
    return len(ids)


def _log_dropped_rows(kept: int, total: int) -> None:
    _drop_counter[0] += 1
    if _drop_counter[0] % 500 == 1:
        print(json.dumps({
            "event": "dropped_unreachable_gold_rows",
            "occurrences": _drop_counter[0],
            "kept": kept, "total": total,
        }, ensure_ascii=False), flush=True)


def _path_loss_from_output(output, batch, device_batch) -> torch.Tensor:
    if batch.gold_edge_mask is None:
        return _zero_path_loss(output, device_batch)

    # The declared objective also applies during no-grad validation. Retain
    # grad detection for callers constructing older/custom output objects.
    trans = getattr(output, "transition_scores", None)
    use_fo = trans is not None and (
        getattr(output, "first_order_loss", False) or trans.requires_grad)

    metadata = (
        getattr(batch, "lengths", None),
        getattr(batch, "allowed_edge_mask", None),
        getattr(batch, "gold_reachable", None),
        getattr(batch, "dp_edge_order", None),
        getattr(batch, "dp_start_offsets", None),
    )
    if all(value is not None for value in metadata):
        reachable = torch.as_tensor(
            batch.gold_reachable, dtype=torch.bool, device="cpu")
        if bool(reachable.all()):
            losses = batched_partial_path_nll(
                output.edge_scores, device_batch.edge_start,
                device_batch.edge_end, device_batch.edge_mask,
                device_batch.gold_edge_mask, device_batch.lengths,
                transition=trans if use_fo else None,
                allowed_mask=device_batch.allowed_edge_mask,
                edge_order=device_batch.dp_edge_order,
                start_offsets=device_batch.dp_start_offsets,
                max_length=len(batch.dp_start_offsets) - 1,
                check_finite=False)
            return losses.mean()

        keep_rows = reachable.nonzero(as_tuple=False).flatten().tolist()
        if not keep_rows:
            return _zero_path_loss(output, device_batch)
        index_tensor = torch.tensor(
            keep_rows, device=output.edge_scores.device, dtype=torch.long)
        losses = batched_partial_path_nll(
            output.edge_scores[index_tensor],
            device_batch.edge_start[index_tensor],
            device_batch.edge_end[index_tensor],
            device_batch.edge_mask[index_tensor],
            device_batch.gold_edge_mask[index_tensor],
            device_batch.lengths[index_tensor],
            transition=trans[index_tensor] if use_fo else None,
            allowed_mask=device_batch.allowed_edge_mask[index_tensor])
        path_loss = losses.mean()
        _log_dropped_rows(len(keep_rows), len(batch.texts))
        return path_loss

    # Compatibility path for hand-built/legacy batches without collator metadata.
    lengths = torch.tensor(
        [len(text) for text in batch.texts], dtype=torch.long,
        device=output.edge_scores.device)
    allowed_cpu = torch.zeros_like(batch.edge_mask)
    keep_rows: list[int] = []
    for index, edges in enumerate(batch.edges):
        row_allowed = allowed_edge_mask(
            edges, len(batch.texts[index]), device=torch.device("cpu"))
        allowed_cpu[index, :len(edges)] = row_allowed
        compatible = batch.gold_edge_mask[index, :len(edges)].tolist()
        if _row_reachable(edges, compatible, len(batch.texts[index])):
            keep_rows.append(index)
    if not keep_rows:
        return _zero_path_loss(output, device_batch)
    allowed = allowed_cpu.to(output.edge_scores.device)
    index_tensor = torch.tensor(
        keep_rows, device=output.edge_scores.device, dtype=torch.long)
    losses = batched_partial_path_nll(
        output.edge_scores[index_tensor], device_batch.edge_start[index_tensor],
        device_batch.edge_end[index_tensor], device_batch.edge_mask[index_tensor],
        device_batch.gold_edge_mask[index_tensor], lengths[index_tensor],
        transition=trans[index_tensor] if use_fo else None,
        allowed_mask=allowed[index_tensor])
    if len(keep_rows) < len(batch.texts):
        _log_dropped_rows(len(keep_rows), len(batch.texts))
    return losses.mean()


def _v5_auxiliary_losses(output, batch, device_batch) -> torch.Tensor:
    """V5 auxiliary terms: L_read CE and the L_contrast margin loss.

    The model computes them only when gradients are enabled (eval skips them,
    mirroring exp39 semantics). Weights are resolved per run in ``run()``.
    """
    total = output.edge_scores.new_zeros(())
    read = getattr(output, "read_loss", None)
    if read is not None:
        total = total + float(_read_weight_current[0]) * read
    contrast = getattr(output, "contrast_loss", None)
    if contrast is not None:
        total = total + float(_contrast_weight_current[0]) * contrast
    return total


# Mutable cells so a resumed/changed run can set weights without globals gymnastics.
_read_weight_current = [0.0]
_contrast_weight_current = [0.0]

# Contrast pair logging cadence: every Nth grad-enabled micro-batch, mirroring
# the periodic style of _log_dropped_rows. len(negatives) is a host-side int;
# logging never syncs the GPU.
_CONTRAST_LOG_EVERY = 200
_counters: dict[str, int] = {}


def _batch_objective(model, batch, device: torch.device):
    non_blocking = device.type == "cuda" and batch.input_ids.is_pinned()
    device_batch = batch.to(device, non_blocking=non_blocking)
    kwargs = device_batch.model_kwargs()
    if isinstance(model, SpanG2Pv5):
        kwargs["gold_edge_mask"] = device_batch.gold_edge_mask
        # Fail-closed contrast wiring: in hard mode the collator must have
        # produced the trusted span mask, otherwise the model would silently
        # contribute zero contrast pairs for the whole run.
        output = model(**kwargs)
        # Guard and pair logging only under grad-enabled training: evaluation
        # runs model.eval() and the model skips L_contrast entirely there, so
        # a mask-free legacy batch is legitimate for eval and must not raise.
        if torch.is_grad_enabled():
            if getattr(model, "contrast_mode", "all") == "hard" and (
                    device_batch.trusted_gold_span_mask is None
                    or not isinstance(device_batch.trusted_gold_span_mask, Tensor)):
                raise ValueError(
                    "contrast_mode='hard' requires trusted_gold_span_mask on the "
                    "batch; the collator was not built with contrast_mode='hard' "
                    "or the mask was dropped before the model call")
            if (getattr(model, "contrast_mode", "all") == "hard"
                    and getattr(output, "contrast_loss", None) is not None):
                negatives = getattr(output, "contrast_negatives", None)
                # A supervised step with zero eligible pairs is observable (not
                # an error: an unsupervised batch may legitimately have zero
                # pairs). Logged periodically (not per micro-batch) to keep the
                # stream readable; pairs stay observable without GPU sync.
                key = "_contrast_zero_count"
                if negatives is None or len(negatives) == 0:
                    _counters[key] = _counters.get(key, 0) + 1
                    if _counters[key] % _CONTRAST_LOG_EVERY == 1:
                        print(json.dumps({
                            "event": "contrast_pairs_zero",
                            "occurrences": _counters[key],
                        }, ensure_ascii=False), flush=True)
                else:
                    _counters["_contrast_pair_total"] = (
                        _counters.get("_contrast_pair_total", 0) + len(negatives))
                    _counters["_contrast_events"] = (
                        _counters.get("_contrast_events", 0) + 1)
                    if _counters["_contrast_events"] % _CONTRAST_LOG_EVERY == 1:
                        print(json.dumps({
                            "event": "contrast_pairs",
                            "pairs": int(len(negatives)),
                            "occurrences": _counters["_contrast_events"],
                            "total_pairs": _counters["_contrast_pair_total"],
                        }, ensure_ascii=False), flush=True)
        loss = _path_loss_from_output(output, batch, device_batch)
        loss = loss + _v5_auxiliary_losses(output, batch, device_batch)
        # V5 has no boundary/generator heads; only its own auxiliary terms.
        return loss, output
    output = model(**kwargs)
    path_loss = _path_loss_from_output(output, batch, device_batch)
    auxiliary, _values = auxiliary_losses(
        output, model, device_batch,
        boundary_weight=float(getattr(model, "boundary_loss_weight", 0.1)),
        generator_weight=float(getattr(model, "generator_loss_weight", 0.2)))
    return path_loss + auxiliary, output


def _batch_loss(model: SpanG2P, batch, device: torch.device) -> torch.Tensor:
    loss, _output = _batch_objective(model, batch, device)
    return loss


@torch.inference_mode()
def _evaluate_metrics(model: SpanG2P, loader: DataLoader,
                      device: torch.device,
                      amp_dtype: torch.dtype | None = None,
                       row_limit: int | None = None
                       ) -> tuple[float, float, float, float]:
    """Compute objective, strict accuracy, supervised CER, and folded CER in one pass."""
    from .evaluate import _editops, fold_notation

    was_training = model.training
    model.eval()
    loss_total = 0.0
    batch_count = 0
    correct = total = 0
    edits = edits_folded = reference_length = 0
    seen = 0
    for batch in loader:
        # Preserve the existing row-limit behavior: finish the current batch and
        # stop only at the next batch boundary.
        if row_limit is not None and seen >= row_limit:
            break
        with _autocast(device, amp_dtype):
            loss, output = _batch_objective(model, batch, device)
        loss_total += float(loss.detach().cpu())
        batch_count += 1
        # One transfer for all rows, then decode each row exactly once on CPU.
        edge_scores = output.edge_scores.detach().float().cpu()
        trans_scores = (
            output.transition_scores.detach().float().cpu()
            if getattr(output, "transition_scores", None) is not None else None)
        for index, text in enumerate(batch.texts):
            n_edges = len(batch.edges[index])
            path = decode_viterbi(
                edge_scores[index, :n_edges],
                batch.edges[index], len(text),
                transition_scores=(trans_scores[index, :n_edges, :n_edges]
                                   if trans_scores is not None else None))
            gold_mask = (batch.gold_edge_mask[index]
                         if batch.gold_edge_mask is not None else None)
            if gold_mask is not None and all(
                    bool(gold_mask[edge_index])
                    for edge_index in path.edge_indices):
                correct += 1
            total += 1

            gold = batch.supervised_gold
            reliable = batch.supervised_reliable
            if gold is None or reliable is None:
                reference = (batch.reference_readings or batch.texts)[index]
                edits += len(_editops(reference, path.reading))
                edits_folded += len(_editops(fold_notation(reference), fold_notation(path.reading)))
                reference_length += len(reference)
                seen += 1
                continue
            hyp_parts = [""] * len(text)
            for edge in path.edges:
                hyp_parts[edge.start] = edge.reading
            row_gold = gold[index]
            row_reliable = reliable[index]
            position = 0
            while position < len(text):
                if not row_reliable[position]:
                    position += 1
                    continue
                end = position + 1
                while end < len(text) and row_reliable[end]:
                    end += 1
                reference = "".join(row_gold[position:end])
                hypothesis = "".join(hyp_parts[position:end])
                edits += len(_editops(reference, hypothesis))
                edits_folded += len(_editops(fold_notation(reference), fold_notation(hypothesis)))
                reference_length += len(reference)
                position = end
            seen += 1
    model.train(was_training)
    return (loss_total / max(1, batch_count),
            correct / max(1, total),
            edits / max(1, reference_length),
            edits_folded / max(1, reference_length))


@torch.inference_mode()
def _evaluate(model: SpanG2P, loader: DataLoader, device: torch.device,
              amp_dtype: torch.dtype | None = None,
              row_limit: int | None = None) -> float:
    return _evaluate_metrics(
        model, loader, device, amp_dtype, row_limit=row_limit)[0]


@torch.inference_mode()
def _strict_constraint_accuracy(model: SpanG2P, loader: DataLoader,
                                device: torch.device,
                                amp_dtype: torch.dtype | None = None,
                                row_limit: int | None = None) -> float:
    return _evaluate_metrics(
        model, loader, device, amp_dtype, row_limit=row_limit)[1]


def _strict_cer(model: SpanG2P, loader: DataLoader,
                device: torch.device,
                amp_dtype: torch.dtype | None = None,
                row_limit: int | None = None) -> float:
    return _evaluate_metrics(
        model, loader, device, amp_dtype, row_limit=row_limit)[2]


def _save(path: str | Path, model: SpanG2P, optimizer: torch.optim.Optimizer,
          step: int, best_loss: float, config: dict[str, Any],
          scaler: Any = None, scheduler: Any = None,
          resource_manifest: dict[str, Any] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    payload = {
        "model": model.state_dict(),
        "config": dict(config),
        "step": int(step),
        "best_loss": float(best_loss),
        "optimizer": optimizer.state_dict(),
        "parameters": parameter_count(model),
        "resource_manifest": resource_manifest or {},
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
        },
    }
    if torch.cuda.is_available():
        payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, temporary)
    temporary.replace(output)


def _resolve_stop_after_step(config: dict[str, Any],
                             restored_step: int = 0) -> int | None:
    """Resolve the optional absolute optimizer-step cap for the while loop.

    When unset (default) this returns None and run() behaves exactly as
    before. The cap only bounds the training loop; config["steps"] remains
    the cosine LR schedule horizon, so a chunked short process keeps the
    same schedule the full 30k run would have followed. Validation is
    deterministic: the cap must be positive and <= steps, and on resume it
    must be strictly greater than the restored step (otherwise the run would
    either exit immediately with a bogus save of unadvanced state or, if the
    cap were silently clamped, never trigger a stop).
    """
    value = config.get("stop_after_step")
    if value is None:
        return None
    limit = int(value)
    steps = int(config["steps"])
    if limit < 1:
        raise ValueError(
            f"stop_after_step must be a positive optimizer step; got {limit}")
    if limit > steps:
        raise ValueError(
            f"stop_after_step ({limit}) must be <= steps ({steps}); the cap "
            "bounds the loop, it does not extend the schedule horizon")
    if limit <= restored_step:
        raise ValueError(
            f"stop_after_step ({limit}) must be greater than the restored "
            f"step ({restored_step}); a resume at or past the cap would be "
            "a no-op that re-saves unchanged state")
    return limit


def run(config: dict[str, Any]) -> dict[str, float | int]:
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(config.get("device"))
    pin_memory = _resolve_pin_memory(config.get("pin_memory"), device)
    amp_dtype = _resolve_amp_dtype(config.get("amp_dtype", "auto"), device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(str(
            config.get("float32_matmul_precision", "high")))
    scaler = None
    vocab = default_vocab(config["vocab"])
    ids = load_component_ids(config.get("ids"), config.get("ids_format", "chise_v1"))
    print(json.dumps({"status": "loading_candidate_resources"},
                     ensure_ascii=False), flush=True)
    provider = _provider(config, memory=True)
    manifest = (provider.resource_manifest
                if bool(config.get("resource_manifest", True)) else {})
    eval_provider = provider
    print(json.dumps({"status": "candidate_resources_ready"},
                     ensure_ascii=False), flush=True)
    collator = SparseEdgeCollator(
        provider, vocab, int(config["max_reading_length"]), inject_gold=False,
        gold_injection_prob=float(config.get("gold_injection_prob", 0.0)),
        dynamic_reading_length=bool(config.get("dynamic_reading_length", True)),
        deduplicate_readings=bool(config.get("deduplicate_readings", True)),
        ids=ids, max_span_length=int(config.get("max_span_length", 16)),
        coarsen_segmentation=bool(config.get("coarsen_segmentation", True)),
        contrast_mode=str(config.get("contrast_mode", "all")))
    # Eval never injects gold, but in hard mode the trusted mask must be built
    # so model_kwargs wiring matches training; the model itself skips
    # L_contrast under model.eval()/no-grad, so eval needs no contrast loss.
    eval_collator = SparseEdgeCollator(
        eval_provider, vocab, int(config["max_reading_length"]), inject_gold=False,
        gold_injection_prob=0.0,
        dynamic_reading_length=bool(config.get("dynamic_reading_length", True)),
        deduplicate_readings=bool(config.get("deduplicate_readings", True)),
        ids=ids, max_span_length=int(config.get("max_span_length", 16)),
        coarsen_segmentation=False,
        contrast_mode=str(config.get("contrast_mode", "all")))
    bucket_buffer_size = int(config.get("bucket_buffer_size", 0))
    length_buckets = config.get("length_buckets")
    train_loader = build_loader(
        _as_paths(config["data"]), collator,
        batch_size=int(config["batch_size"]), max_length=int(config["max_length"]),
        overlap=int(config["overlap"]), num_workers=int(config["num_workers"]),
        pin_memory=pin_memory,
        shuffle_buffer=bucket_buffer_size, length_buckets=length_buckets,
        source_sample_rates=config.get("source_sample_rates"),
        source_interleave=int(config.get("source_interleave", 256)),
    )
    eval_loader = None
    if config.get("eval_data"):
        eval_loader = build_loader(
            _as_paths(config["eval_data"]), eval_collator,
            batch_size=int(config["eval_batch_size"]), max_length=int(config["max_length"]),
            overlap=int(config["overlap"]), num_workers=int(config["num_workers"]),
            pin_memory=pin_memory,
            shuffle_buffer=bucket_buffer_size, length_buckets=length_buckets,
        )
    model_arch = str(config.get("model_arch", "v4")).lower()
    if model_arch == "v5":
        # V5 conditions the reading decoder on each edge's span, so identical
        # readings on different spans must not be deduplicated into one row
        # (docs/V5_ARCHITECTURE.md section 3). Fail loudly instead of letting
        # the collator return unique-reading tensors the model cannot consume.
        if bool(config.get("deduplicate_readings", False)):
            raise ValueError(
                "model_arch=v5 requires deduplicate_readings: false; the "
                "reading decoder conditions on span context, so readings "
                "cannot be deduplicated by surface form")
        model = SpanG2Pv5(
            vocab_size=max(vocab.values()) + 1,
            layers=int(config["layers"]), dim=int(config["dim"]),
            heads=int(config["heads"]), d_ff=int(config["d_ff"]),
            max_seq_len=int(config["max_seq_len"]),
            max_reading_len=int(config["max_reading_length"]),
            component_dim=int(config.get("component_dim", 256)),
            source_count=int(config["source_count"]),
            edge_dim=int(config.get("edge_dim_v5", 384)),
            dropout=float(config["dropout"]),
            fusion_layers=tuple(config.get("fusion_layers", ())),
            fusion_heads=int(config.get("fusion_heads", 8)),
            max_span_length=int(config.get("max_span_length", 16)),
            use_edge_fusion=bool(config.get("use_edge_fusion", False)),
            reading_layers=int(config.get("reading_layers", 4)),
            reading_heads=int(config.get("reading_heads", 8)),
            reading_d_ff=int(config.get("reading_d_ff", 1024)),
            use_relation_attention=bool(
                config.get("use_relation_attention", False)),
            relation_layers=int(config.get("relation_layers", 2)),
            relation_heads=int(config.get("relation_heads", 8)),
            relation_d_ff=int(config.get("relation_d_ff", 1024)),
            use_prior_table=bool(config.get("use_prior_table", False)),
            prior_table_size=((len(provider.prior_index) + 1)
                              if (bool(config.get("use_prior_table", False))
                                  and getattr(provider, "prior_index", None))
                              else 1),
            prior_dim=int(config.get("prior_dim", 4)),
            activation_checkpointing=bool(
                config.get("activation_checkpointing", True)),
            reading_chunk_size=int(config.get("reading_chunk_size", 16)),
            use_transition_head=bool(config.get("use_transition_head", True)),
            transition_dim=int(config.get("transition_dim", 64)),
            first_order_loss=bool(config.get("first_order_loss", True)),
            transition_features=str(config.get("transition_features", "context")),
            reading_token_ids=kana_token_ids(vocab),
        )
        contrast_mode = str(config.get("contrast_mode", "all")).lower()
        if contrast_mode not in {"all", "hard"}:
            raise ValueError(
                f"contrast_mode must be 'all' or 'hard'; got {contrast_mode!r}")
        if (contrast_mode == "hard"
                and float(config.get("contrast_loss_weight", 0.1)) <= 0):
            raise ValueError(
                "contrast_mode='hard' expects contrast supervision but "
                "contrast_loss_weight <= 0 disables L_contrast entirely")
        if (contrast_mode == "hard"
                and not bool(config.get("gold_injection_prob", 0.0)
                             or config.get("inject_gold", False))):
            # Gold injection is what creates complete same-span rival groups;
            # without it hard mode would contribute zero pairs on most batches.
            print(json.dumps({
                "event": "hard_contrast_without_gold_injection",
                "detail": "contrast_mode='hard' with gold_injection_prob=0: "
                          "eligible contrast pairs may be near zero",
            }, ensure_ascii=False), flush=True)
        model.contrast_mode = contrast_mode
        # The config exposes contrast_margin/max_contrast_pairs; without this
        # wiring the model kept its hardcoded 1.0/2048 defaults even when a
        # config overrode them (e.g. a hard-negative sweep varying the margin
        # would silently train with margin=1.0 in every arm).
        model.contrast_margin = float(config.get("contrast_margin", 1.0))
        model.max_contrast_pairs = int(config.get("max_contrast_pairs", 2048))
    elif model_arch == "v4bert":
        # BERT-initialized v4 (kashi_g2p_bert). The candidate graph, loss,
        # decoder, and data pipeline are byte-identical to the v4 branch; only
        # the encoder differs (pretrained layer-sliced char-BERT).
        from kashi_g2p_bert.model import SpanG2PBert
        model = SpanG2PBert(
            bert_dir=str(config["bert_slice_dir"]),
            edge_dim=int(config["edge_dim"]),
            component_dim=int(config.get("component_dim", 256)),
            source_count=int(config["source_count"]),
            dropout=float(config["dropout"]),
            reading_kernel_size=int(config["reading_kernel_size"]),
            fusion_layers=tuple(config.get("fusion_layers", ())),
            fusion_heads=int(config.get("fusion_heads", 8)),
            max_span_length=int(config.get("max_span_length", 16)),
            max_reading_len=int(config["max_reading_length"]),
            use_prior_table=bool(config.get("use_prior_table", False)),
            prior_table_size=((len(provider.prior_index) + 1)
                              if (bool(config.get("use_prior_table", False))
                                  and getattr(provider, "prior_index", None))
                              else 1),
            prior_dim=int(config.get("prior_dim", 4)),
            use_per_layer_edge_seed=bool(config.get("use_per_layer_edge_seed", False)),
            use_transition_head=bool(config.get("use_transition_head", False)),
            transition_dim=int(config.get("transition_dim", 64)),
            first_order_loss=bool(config.get("first_order_loss", False)),
            transition_features=str(config.get("transition_features", "context")),
            activation_checkpointing=bool(
                config.get("activation_checkpointing", True)),
            reading_chunk_size=int(config.get("reading_chunk_size", 16)),
        )
    else:
        model = SpanG2P(
            vocab_size=max(vocab.values()) + 1,
            layers=int(config["layers"]), dim=int(config["dim"]),
            heads=int(config["heads"]), d_ff=int(config["d_ff"]),
            max_seq_len=int(config["max_seq_len"]),
            max_reading_len=int(config["max_reading_length"]),
            component_dim=int(config.get("component_dim", 256)),
            source_count=int(config["source_count"]), edge_dim=int(config["edge_dim"]),
            dropout=float(config["dropout"]),
            reading_kernel_size=int(config["reading_kernel_size"]),
            fusion_layers=tuple(config.get("fusion_layers", [2, 4, 6, 8])),
            fusion_heads=int(config.get("fusion_heads", 8)),
            max_span_length=int(config.get("max_span_length", 16)),
            use_open_generator=bool(config.get("use_open_generator", True)),
            generator_dim=int(config.get("generator_dim", 256)),
            generator_layers=int(config.get("generator_layers", 3)),
            generator_heads=int(config.get("generator_heads", 8)),
            generator_d_ff=int(config.get("generator_d_ff", 640)),
            generator_max_length=int(config.get("generator_max_length", 64)),
            activation_checkpointing=bool(config.get("activation_checkpointing", True)),
            reading_chunk_size=int(config.get("reading_chunk_size", 16)),
            use_prior_table=bool(config.get("use_prior_table", False)),
            prior_table_size=((len(provider.prior_index) + 1)
                              if (bool(config.get("use_prior_table", False))
                                  and getattr(provider, "prior_index", None))
                              else 1),
            prior_dim=int(config.get("prior_dim", 4)),
            use_per_layer_edge_seed=bool(config.get("use_per_layer_edge_seed", False)),
            use_transition_head=bool(config.get("use_transition_head", False)),
            transition_dim=int(config.get("transition_dim", 64)),
            first_order_loss=bool(config.get("first_order_loss", False)),
            transition_features=str(config.get("transition_features", "context")),
        )
    _read_weight_current[0] = float(config.get("read_loss_weight", 0.5))
    _contrast_weight_current[0] = float(config.get("contrast_loss_weight", 0.1))
    if (model.prior_table is not None and bool(config.get("yomogi_warm_start"))
            and config.get("yomogi_dict") and config.get("yomogi_model")):
        matched = _warm_start_prior_table(
            model, prior_index=provider.prior_index,
            yomogi_tsv=config["yomogi_dict"][0]
                        if isinstance(config["yomogi_dict"], list)
                        else config["yomogi_dict"],
            yomogi_ckpt=config["yomogi_model"])
        print(json.dumps({"status": "prior_table_warm_started",
                          "matched_entries": matched}, ensure_ascii=False), flush=True)
    model = model.to(device)
    model.boundary_loss_weight = float(config.get("boundary_loss_weight", 0.1))
    model.generator_loss_weight = float(config.get("generator_loss_weight", 0.2))
    count = parameter_count(model)
    if count >= int(config.get("parameter_limit", 40_000_000)):
        raise ValueError(f"model has {count:,} parameters; budget is <40,000,000")
    base_lr = float(config["lr"])
    base_wd = float(config["weight_decay"])
    transition_lr = config.get("transition_lr")
    if (transition_lr is not None
            and getattr(model, "transition_head", None) is not None):
        transition_lr = float(transition_lr)
        transition_wd = float(config.get("transition_weight_decay", 0.0))
        transition_params = set(model.transition_head.parameters())
        head_params = [p for p in model.parameters() if p in transition_params]
        rest_params = [p for p in model.parameters() if p not in transition_params]
        optimizer = torch.optim.AdamW([
            {"params": rest_params, "lr": base_lr, "weight_decay": base_wd},
            {"params": head_params, "lr": transition_lr,
             "weight_decay": transition_wd},
        ])
        print(json.dumps({
            "status": "transition_param_group",
            "transition_lr": transition_lr,
            "transition_weight_decay": transition_wd,
            "base_lr": base_lr,
            "transition_params": sum(p.numel() for p in head_params),
        }, ensure_ascii=False), flush=True)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=base_wd)
    warmup = max(0, int(config.get("warmup_steps", 0)))
    total_steps = max(1, int(config["steps"]))
    minimum = float(config.get("min_lr_ratio", 0.1))

    def lr_scale(current: int) -> float:
        if warmup and current < warmup:
            return max(1e-8, current / warmup)
        progress = (current - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return minimum + (1.0 - minimum) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = _make_scaler(device, amp_dtype)
    step = 0
    best = math.inf
    # Fine-tuning init: inherit ONLY the weights. Unlike resume, the step
    # counter, optimizer, scheduler, and best-metric tracking all start fresh,
    # so a 30k-step fine-tune horizon works on top of a 70k-step checkpoint.
    init_from = config.get("init_from")
    if init_from:
        state = torch.load(init_from, map_location="cpu", weights_only=False)
        loaded_state = dict(state["model"])
        if getattr(model, "use_per_layer_edge_seed", False) and "edge_seed.weight" in loaded_state:
            seed_w = loaded_state.pop("edge_seed.weight")
            for layer in getattr(model, "fusion_layers", ()):
                k = f"edge_seeds.{layer}.weight"
                if k not in loaded_state:
                    loaded_state[k] = seed_w.clone()
        # Drop weights whose shape no longer matches (e.g. a v4 checkpoint's
        # 256-dim transition head against a V5 model at 384). strict=False
        # does NOT forgive size mismatches, so they would crash init; these
        # modules are re-initialized from scratch anyway (the transition head
        # is re-seeded below for first-order training).
        dropped: list[str] = []
        model_state = model.state_dict()
        for key in list(loaded_state):
            if (key in model_state
                    and model_state[key].shape != loaded_state[key].shape):
                loaded_state.pop(key)
                dropped.append(key)
        missing, unexpected = model.load_state_dict(loaded_state, strict=False)
        print(json.dumps({
            "status": "initialized_weights_only",
            "path": str(init_from),
            "source_step": int(state.get("step", 0)),
            "missing_keys_count": len(missing),
            "unexpected_keys_count": len(unexpected),
            "dropped_shape_mismatch": dropped,
        }, ensure_ascii=False), flush=True)
        # A 0th-order checkpoint carries the transition head's dead zero init.
        # Re-seed it so first-order training has a live gradient.
        if (bool(config.get("first_order_loss", False))
                and getattr(model, "transition_head", None) is not None):
            _init_std = float(config.get("transition_init_std", 0.02))
            model.reinitialize_transition_for_training(std=_init_std)
            print(json.dumps({"status": "transition_head_reinitialized_for_first_order",
                              "std": _init_std},
                             ensure_ascii=False), flush=True)
    resume = config.get("resume")
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        saved_manifest = state.get("resource_manifest", {})
        if saved_manifest and saved_manifest != manifest:
            raise ValueError("checkpoint resource manifest does not match current resources")
        model.load_state_dict(state["model"])
        if state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if state.get("rng"):
            torch.set_rng_state(state["rng"]["torch"])
            random.setstate(state["rng"]["python"])
            if torch.cuda.is_available() and state["rng"].get("cuda"):
                torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        step = int(state.get("step", 0))
        best = float(state.get("best_loss", math.inf))
    # Resolve the optional loop cap AFTER resume so validation can compare it
    # against the restored step. Unset -> None -> default behavior unchanged.
    stop_after_step = _resolve_stop_after_step(config, restored_step=step)

    model.train()
    iterator = iter(train_loader)
    last_loss = math.inf
    grad_accum = max(1, int(config.get("grad_accum", 1)))
    if int(config["batch_size"]) < 1:
        raise ValueError("batch_size must be positive")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    tensorboard_logdir = config.get("tensorboard_logdir")
    writer = None
    if bool(config.get("tensorboard")) and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=tensorboard_logdir or None)
    progress = None
    if bool(config.get("use_tqdm")) and tqdm is not None:
        experiment = Path(str(config.get("save", "experiment"))).parent.parent.name
        progress = tqdm(
            total=int(config["steps"]), initial=step, desc=experiment,
            unit="step", dynamic_ncols=True)
    skip_batches = 0
    corrupt_batches = 0
    grad_skip_limit = int(config.get("grad_skip_limit", 50))
    skipped_updates = 0
    profile_steps = bool(config.get("profile_steps", False))
    _stage = {"next_ms": 0.0, "loss_ms": 0.0, "bwd_ms": 0.0, "opt_ms": 0.0,
              "max_loss_ms": 0.0}
    # The while-loop horizon: config["steps"] normally. With stop_after_step
    # set (short-process chunking), the loop stops earlier at that absolute
    # optimizer step, while the LR schedule keeps using config["steps"].
    loop_horizon = (stop_after_step
                    if stop_after_step is not None else int(config["steps"]))
    while step < loop_horizon:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = None
        valid_micro_batches = 0
        last_batch = None
        for _ in range(grad_accum):
            _t0 = time.time() if profile_steps else 0.0
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            if profile_steps:
                torch.cuda.synchronize()
                _stage["next_ms"] += time.time() - _t0
                _t0 = time.time()
            last_batch = batch
            if profile_steps:
                _tA = time.time()
            with _autocast(device, amp_dtype):
                loss = _batch_loss(model, batch, device)
                scaled_loss = loss / grad_accum
            if profile_steps:
                torch.cuda.synchronize()
                _elapsed = time.time() - _t0
                _stage["loss_ms"] += _elapsed
                _stage["max_loss_ms"] = max(_stage["max_loss_ms"], _elapsed)
                _t0 = time.time()
            finite_loss = torch.isfinite(loss.detach())
            if not bool(finite_loss):
                if corrupt_batches < grad_skip_limit:
                    corrupt_batches += 1
                    print(json.dumps({
                        "event": "skipped_non_finite_loss_batch",
                        "step": step + 1, "loss": float(loss.detach().cpu()),
                    }, ensure_ascii=False), flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_loss = None
                    valid_micro_batches = 0
                    continue
                raise FloatingPointError(
                    f"non-finite v4 loss at step {step + 1}: {loss.item()}"
                )
            detached_loss = loss.detach()
            accumulated_loss = (detached_loss if accumulated_loss is None
                                else accumulated_loss + detached_loss)
            valid_micro_batches += 1
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if profile_steps:
                torch.cuda.synchronize()
                _stage["bwd_ms"] += time.time() - _t0
                _t0 = time.time()
        if profile_steps:
            torch.cuda.synchronize()
        if valid_micro_batches == 0:
            continue
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["grad_clip"])
        )
        finite_grad_norm = bool(torch.isfinite(grad_norm.detach()))
        if not finite_grad_norm:
            # A single poisoned batch would otherwise destroy the run. Skip
            # the update, keep training from the last good weights, and
            # surface the event in the log stream.
            skipped_updates += 1
            skip_batches += grad_accum
            print(json.dumps({
                "step": step + 1, "event": "skipped_non_finite_gradients",
                "grad_norm": None,
                "skipped_updates": skipped_updates,
            }, ensure_ascii=False), flush=True)
            if skipped_updates > grad_skip_limit:
                raise FloatingPointError(
                    f"too many non-finite gradient events: {skipped_updates}"
                )
            if scaler is not None:
                scaler.update()
            continue
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        if profile_steps:
            _stage["opt_ms"] += time.time() - _t0
        step += 1
        # Release fragmented cached memory every N steps.  Without this the
        # caching allocator's reserved pool creeps up (observed 7710 -> 7828
        # of 8188 MiB within minutes) until allocations spill into WDDM
        # host memory: step time jumps ~3x and GPU power drops from ~80 W to
        # ~30 W (micro-kernels waiting on host transfers).  The legacy
        # bench_step.py did the same after each measurement batch.
        if (device.type == "cuda" and step % int(
                config.get("empty_cache_every", 200)) == 0):
            torch.cuda.empty_cache()
        if profile_steps and step % 5 == 0:
            total = sum(_stage.values())
            print(json.dumps({
                "profile": {k: round(v * 1000 / 5, 1) for k, v in _stage.items()},
                "step_ms": round(total * 1000 / 5, 1),
            }, ensure_ascii=False), flush=True)
            _stage = {"next_ms": 0.0, "loss_ms": 0.0, "bwd_ms": 0.0, "opt_ms": 0.0,
                      "max_loss_ms": 0.0}
        if accumulated_loss is None or valid_micro_batches == 0:
            last_loss = 0.0
        else:
            last_loss = float(
                (accumulated_loss / valid_micro_batches).cpu())
        if progress is not None:
            progress.update(1)
            # tqdm owns the console; step logs go to TensorBoard instead.
            progress.set_postfix(loss=f"{last_loss:.3f}")
        else:
            if step % 10 == 0 or step == 1:
                print(json.dumps({"step": step, "loss": last_loss}, ensure_ascii=False), flush=True)
        if writer is not None:
            writer.add_scalar("train/loss", last_loss, step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
        if step == 1 and device.type == "cuda" and last_batch is not None:
            memory_bits = {
                "cuda_memory": {
                    "micro_batch": int(last_batch.input_ids.size(0)),
                    "effective_batch": int(last_batch.input_ids.size(0) * grad_accum),
                    "max_edges": int(last_batch.edge_mask.size(1)),
                    "reading_length": int(
                        last_batch.unique_reading_ids.size(1)
                        if last_batch.unique_reading_ids is not None
                        else last_batch.edge_reading_ids.size(2)),
                    "unique_readings": (
                        int(last_batch.unique_reading_ids.size(0))
                        if last_batch.unique_reading_ids is not None else None),
                    "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
                    "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
                    "amp_dtype": str(amp_dtype).replace("torch.", ""),
                }
            }
            print(json.dumps(memory_bits, ensure_ascii=False), flush=True)
            if writer is not None:
                for name in ("peak_allocated_mb", "peak_reserved_mb"):
                    writer.add_scalar(f"cuda/{name}", memory_bits["cuda_memory"][name], step)
        if step % int(config["eval_every"]) == 0 and eval_loader is not None:
            limit_value = config.get("eval_row_limit")
            row_limit = int(limit_value) if limit_value else None
            metric, strict, cer, cer_folded = _evaluate_metrics(
                model, eval_loader, device, amp_dtype, row_limit=row_limit)
            summary = {"step": step, "eval_nll": metric,
                       "supervised_cer": cer,
                       "supervised_cer_folded": cer_folded,
                       "strict_constraint_accuracy": strict}
            if progress is not None:
                progress.set_postfix(
                    loss=f"{last_loss:.3f}", cer=f"{cer*100:.2f}%",
                    cer_f=f"{cer_folded*100:.2f}%",
                    path_ok=f"{strict*100:.1f}%")
                print(json.dumps(summary, ensure_ascii=False), flush=True)
            else:
                print(json.dumps(summary, ensure_ascii=False), flush=True)
            if writer is not None:
                writer.add_scalar("eval/nll", metric, step)
                writer.add_scalar("eval/strict_cer", cer, step)
                writer.add_scalar("eval/strict_cer_folded", cer_folded, step)
                writer.add_scalar("eval/strict_constraint_accuracy", strict, step)
            if progress is not None:
                progress.set_postfix(
                    loss=f"{last_loss:.3f}",
                    cer=f"{cer*100:.2f}%", strict_bars=f"{strict*100:.1f}%")
            selection = cer if config.get("best_metric", "cer") == "cer" else metric
            if selection < best:
                best = selection
                _save(config["best_save"], model, optimizer, step, best, config,
                      scaler, scheduler, manifest)
        if step % int(config["save_every"]) == 0:
            _save(config["save"], model, optimizer, step, best, config,
                  scaler, scheduler, manifest)
    _save(config["save"], model, optimizer, step, best, config,
          scaler, scheduler, manifest)
    if not Path(config["best_save"]).is_file():
        _save(config["best_save"], model, optimizer, step, best, config,
              scaler, scheduler, manifest)
    if progress is not None:
        progress.close()
    if writer is not None:
        writer.close()
    result: dict[str, float | int] = {
        "step": step,
        "loss": last_loss,
        "best_metric_value": best,
        "parameters": parameter_count(model),
    }
    if device.type == "cuda":
        result["peak_allocated_mb"] = round(
            torch.cuda.max_memory_allocated(device) / 2**20, 1
        )
        result["peak_reserved_mb"] = round(
            torch.cuda.max_memory_reserved(device) / 2**20, 1
        )
    return result


def _configure_cuda_allocator() -> None:
    """Reduce cudaMalloc stalls from allocator fragmentation on Windows/WDDM.

    Expandable segments grow the pool in place instead of issuing new cudaMalloc
    calls per distinct tensor shape, which is what caused the periodic
    multi-hundred-ms stalls at varying batch shapes (E 233-314, S 52-62).
    """
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> None:
    _configure_cuda_allocator()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-step", type=int,
                        dest="stop_after_step")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--amp-dtype")
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.steps is not None:
        config["steps"] = args.steps
    if args.resume is not None:
        config["resume"] = args.resume
    if args.stop_after_step is not None:
        config["stop_after_step"] = args.stop_after_step
    if args.device is not None:
        config["device"] = args.device
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        config["grad_accum"] = args.grad_accum
    if args.amp_dtype is not None:
        config["amp_dtype"] = args.amp_dtype
    print(json.dumps(run(config), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
