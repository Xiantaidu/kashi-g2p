"""Checkpoint-backed strict inference for the v4 span model."""

from __future__ import annotations

from dataclasses import replace
import importlib
import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import torch
from torch import Tensor

from .collator import load_vocab
from .normalization import (is_kana_reading, normalize_reading,
                                    normalize_surface)
from .resources import dir_fingerprint, file_fingerprint

from .model import SpanG2P


def _model_class(config: Mapping[str, Any]):
    """Resolve SpanG2P vs SpanG2Pv5 vs SpanG2PBert from the config's model_arch."""
    arch = str(config.get("model_arch", "v4")).lower()
    if arch == "v5":
        from .v5_model import SpanG2Pv5
        return SpanG2Pv5
    if arch == "v4bert":
        from kashi_g2p_bert.model import SpanG2PBert
        return SpanG2PBert
    return SpanG2P


def _candidate_api():
    """Load the evolving candidate API at use time and fail actionably."""
    try:
        module = importlib.import_module("kashi_g2p.candidates")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "v4 candidate API is unavailable; expected CompositeProvider and "
            "pack_edges in kashi_g2p.candidates"
        ) from exc
    missing = [name for name in ("CompositeProvider", "pack_edges")
               if not hasattr(module, name)]
    if missing:
        raise RuntimeError("v4 candidate API is missing: " + ", ".join(missing))
    return module


def _decoder_api():
    return importlib.import_module("kashi_g2p.decoder")


def _resolve_path(value: str | Path | None, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    rooted = root / path
    return rooted if rooted.exists() else path.resolve()


def _resolve_paths(value: Any, root: Path) -> Path | list[Path] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path)):
        return [path for item in value if (path := _resolve_path(item, root)) is not None]
    return _resolve_path(value, root)


def _manifest_resource_matches(expected: Any, path: Any) -> bool:
    """Compare candidates._path_manifest entries without trusting saved paths."""
    if isinstance(path, Sequence) and not isinstance(path, (str, bytes, Path)):
        if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
            return False
        expected_rows = sorted(expected, key=lambda item: str(item.get("path", "")))
        actual_paths = sorted((Path(item) for item in path), key=lambda item: item.as_posix())
        return (len(expected_rows) == len(actual_paths)
                and all(_manifest_resource_matches(row, item)
                        for row, item in zip(expected_rows, actual_paths)))
    path = Path(path)
    if not isinstance(expected, Mapping):
        actual = dir_fingerprint(path) if path.is_dir() else file_fingerprint(path)
        value = str(expected)
        return (actual == value if value.startswith(("sha256:", "manifest-sha256:"))
                else actual is not None and actual.rsplit(":", 1)[-1] == value)
    kind = str(expected.get("kind", ""))
    if kind == "file" or "sha256" in expected:
        if not path.is_file():
            return False
        digest = file_fingerprint(path)
        return (digest is not None
                and digest.rsplit(":", 1)[-1] == str(expected.get("sha256", ""))
                and ("size" not in expected
                     or path.stat().st_size == int(expected["size"])))
    if kind == "directory" or "files" in expected:
        if not path.is_dir():
            return False
        expected_files = expected.get("files")
        if not isinstance(expected_files, Sequence):
            return False
        actual_files = [item for item in path.rglob("*") if item.is_file()]
        if len(actual_files) != len(expected_files):
            return False
        for entry in expected_files:
            if not isinstance(entry, Mapping) or "path" not in entry:
                return False
            item = path / str(entry["path"])
            if not _manifest_resource_matches({**entry, "kind": "file"}, item):
                return False
        return True
    fingerprint = (expected.get("fingerprint") or expected.get("digest"))
    return True if not fingerprint else _manifest_resource_matches(fingerprint, path)


def _validate_resource_manifest(manifest: Mapping[str, Any] | None,
                                resources: Mapping[str, Any], *,
                                strict: bool = True) -> None:
    if not manifest:
        return
    entries = manifest.get("resources", manifest)
    if not isinstance(entries, Mapping):
        raise ValueError("checkpoint resource_manifest.resources must be a mapping")
    failures = []
    for name, expected in sorted(entries.items()):
        path = resources.get(str(name))
        required = bool(expected.get("required", True)) if isinstance(expected, Mapping) else True
        paths = (path if isinstance(path, Sequence)
                 and not isinstance(path, (str, bytes, Path)) else [path])
        if path is None or any(item is None or not Path(item).exists() for item in paths):
            if required:
                failures.append(f"{name}: missing")
        elif not _manifest_resource_matches(expected, path):
            failures.append(f"{name}: fingerprint mismatch")
    if failures:
        message = "checkpoint resource_manifest validation failed: " + "; ".join(failures)
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def _provider_options(config: Mapping[str, Any],
                      manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    settings = {}
    if isinstance(manifest, Mapping):
        settings = manifest.get("settings", manifest.get("options", {}))
    if not isinstance(settings, Mapping):
        settings = {}

    def pick(setting: str, config_key: str, default: Any) -> Any:
        return settings.get(setting, config.get(config_key, default))

    def optional_int(value: Any) -> int | None:
        return None if value is None else int(value)

    return {
        "include_okurigana": bool(pick("include_okurigana", "include_okurigana", True)),
        "use_haqumei": bool(pick("use_haqumei", "use_haqumei", True)),
        "use_pyopenjtalk": bool(pick("use_pyopenjtalk", "use_pyopenjtalk", False)),
        "use_rules": bool(pick("use_rules", "use_rules", True)),
        "use_legacy_lexicon": bool(pick(
            "use_legacy_lexicon", "use_legacy_lexicon", True)),
        "legacy_node_cap": int(pick("legacy_node_cap", "legacy_node_cap", 512)),
        "lyric_min_count": int(pick("lyric_min_count", "lyric_min_count", 2)),
        "alnum_table": pick("alnum_table", "alnum_table", None),
        "yomogi_dict": pick("yomogi_dict", "yomogi_dict", None),
        "haqumei_options": pick("haqumei_options", "haqumei_options", None),
        "per_span_reading_cap": optional_int(pick(
            "per_span_reading_cap", "max_readings_per_span", 8)),
        "total_edge_cap": optional_int(pick("total_edge_cap", "max_edges", 256)),
    }


def _model_config(config: Mapping[str, Any], state: Mapping[str, Tensor]) -> dict[str, Any]:
    """Central checkpoint compatibility layer for SpanG2P constructor args."""
    model_cls = _model_class(config)
    signature = inspect.signature(model_cls.__init__)
    allowed = set(signature.parameters) - {"self"}
    aliases = {"max_reading_len": "max_reading_length",
               "bert_dir": "bert_slice_dir"}
    result: dict[str, Any] = {}
    for parameter in allowed:
        key = aliases.get(parameter, parameter)
        if key in config:
            result[parameter] = config[key]
    char_weight = state.get("char_embed.weight")
    if char_weight is not None and "vocab_size" in allowed:
        result["vocab_size"] = int(char_weight.shape[0])
        result["dim"] = int(char_weight.shape[1])
    source_weight = state.get("source_embed.weight")
    if source_weight is not None:
        result["source_count"] = int(source_weight.shape[0])
        result["source_dim"] = int(source_weight.shape[1])
    reading_pos = state.get("reading_pos.weight")
    if reading_pos is not None:
        result["max_reading_len"] = int(reading_pos.shape[0])
    component = state.get("component_embed.weight")
    if component is not None and "component_vocab_size" in allowed:
        result["component_vocab_size"] = int(component.shape[0])
        result["component_dim"] = int(component.shape[1])
    char_type = state.get("char_type_embed.weight")
    if char_type is not None:
        result["char_type_count"] = int(char_type.shape[0])
    prior_weight = state.get("prior_table.weight")
    if prior_weight is not None:
        result["use_prior_table"] = True
        result["prior_table_size"] = int(prior_weight.shape[0])
        result["prior_dim"] = int(prior_weight.shape[1])
    is_v5 = model_cls.__name__ == "SpanG2Pv5"
    if is_v5:
        block_ffn = state.get("blocks.0.ffn.gate.weight")
        if block_ffn is not None:
            result["d_ff"] = int(block_ffn.shape[0])
        block_qkv = state.get("blocks.0.qkv.weight")
        if block_qkv is not None and "heads" not in result:
            # qkv is [3*dim, dim]; heads are not recoverable from weights, so
            # require them from config and refuse a silent default mismatch.
            raise ValueError(
                "v5 checkpoint config must specify 'heads' to rebuild the "
                "text backbone attention")
        if any(key.startswith("blocks.") for key in state):
            result["layers"] = max(
                (int(key.split(".")[1]) + 1 for key in state
                 if key.startswith("blocks.") and key.endswith(".ffn.gate.weight")),
                default=int(result.get("layers", 8)))
        if any(key.startswith("reading_decoder.") for key in state):
            result["reading_layers"] = max(
                (int(key.split(".")[1]) + 1 for key in state
                 if key.startswith("reading_decoder.")
                 and key.endswith(".ffn.gate.weight")),
                default=int(result.get("reading_layers", 4)))
        reading_ffn = state.get("reading_decoder.0.ffn.gate.weight")
        if reading_ffn is not None:
            result["reading_d_ff"] = int(reading_ffn.shape[0])
        if any(key.startswith("relation_blocks.") for key in state):
            result["use_relation_attention"] = True
            result["relation_layers"] = max(
                (int(key.split(".")[1]) + 1 for key in state
                 if key.startswith("relation_blocks.")
                 and key.endswith(".ffn.gate.weight")),
                default=int(result.get("relation_layers", 2)))
            relation_weight = state.get("relation_blocks.0.ffn.gate.weight")
            if relation_weight is not None:
                result["relation_d_ff"] = int(relation_weight.shape[0])
        memory_weight = state.get("memory_proj.weight")
        if memory_weight is not None:
            # memory_proj is nn.Linear(dim, edge_dim, bias=False); a Linear
            # weight is [out_features, in_features], so edge_dim is shape[0]
            # and shape[1] is the text backbone width (dim).
            result["edge_dim"] = int(memory_weight.shape[0])
        reading_weight = state.get("reading_decoder.0.qkv.weight")
        if reading_weight is not None:
            # qkv is [3*dim, dim]; infer reading_heads only from config, keep
            # the decoder width aligned with edge_dim by construction.
            pass
        result.setdefault("use_transition_head", any(
            key.startswith("transition_head.") for key in state))
        result.setdefault("first_order_loss", True)
        result.pop("use_open_generator", None)
    elif model_cls.__name__ == "SpanG2P":
        result.setdefault("use_open_generator", any(
            key.startswith("generator.") for key in state))
    if any(key.startswith("edge_seeds.") for key in state):
        result["use_per_layer_edge_seed"] = True
    if any(key.startswith("transition_head.") for key in state):
        result["use_transition_head"] = True
    if "transition_head.reading_left_proj.weight" in state:
        result["transition_features"] = "context_reading"
    return result

def _load_model(state: Mapping[str, Tensor], config: Mapping[str, Any],
                vocab: Mapping[str, int] | None = None):
    model_cls = _model_class(config)
    model_cfg = _model_config(config, state)
    # V5 restricts its reading output head to kana tokens; the checkpoint's
    # reading_output width tells us whether it was trained that way.
    if model_cls.__name__ == "SpanG2Pv5":
        from .v5_model import kana_token_ids
        head_w = state.get("reading_output.weight")
        if (vocab is not None and head_w is not None
                and head_w.shape[0] < int(model_cfg.get("vocab_size", 0))):
            model_cfg["reading_token_ids"] = kana_token_ids(dict(vocab))
    model = model_cls(**model_cfg)
    compatible = dict(state)
    # Migrate single edge_seed.weight to edge_seeds if model uses per-layer seeds
    if model.use_per_layer_edge_seed and "edge_seed.weight" in compatible:
        seed_weight = compatible.pop("edge_seed.weight")
        for layer in model.fusion_layers:
            key = f"edge_seeds.{layer}.weight"
            if key not in compatible:
                compatible[key] = seed_weight.clone()
    try:
        model.load_state_dict(compatible, strict=True)
    except RuntimeError as strict_error:
        obsolete = [key for key in compatible if key == "position_embed.weight"]
        for key in obsolete:
            compatible.pop(key)
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        optional_prefixes = ("component_", "char_type_", "edge_seed.", "edge_seeds.",
                             "transition_head.", "fusions.", "boundary_", "generator.")
        hard_missing = [key for key in missing
                        if not key.startswith(optional_prefixes)]
        if unexpected or hard_missing:
            raise RuntimeError(
                "checkpoint is incompatible with the current SpanG2P API; "
                f"missing={hard_missing}, unexpected={unexpected}"
            ) from strict_error
        warnings.warn(
            "loaded an early v4 checkpoint without newly added optional model "
            "modules; use a current checkpoint for reported evaluation",
            RuntimeWarning, stacklevel=2)
    return model


def _edge_posteriors(scores: Tensor, edges: list[Any], length: int,
                     force_locked: bool,
                     transition_scores: Tensor | None = None) -> Tensor:
    decoder = _decoder_api()
    starts = torch.tensor([edge.start for edge in edges], device=scores.device)
    ends = torch.tensor([edge.end for edge in edges], device=scores.device)
    mask = torch.ones(len(edges), dtype=torch.bool, device=scores.device)
    allowed = (decoder.allowed_edge_mask(edges, length, device=scores.device)
               if force_locked else mask)
    if transition_scores is not None:
        # Confidence must describe the same first-order path distribution as
        # Viterbi, including its fixed morphosyntactic penalties.
        count = len(edges)
        with torch.no_grad():
            transition = transition_scores[:count, :count].detach().float().clone()
            penalties = torch.tensor([
                [decoder._morpho_penalty(left, right)
                 if left.end == right.start else 0.0 for right in edges]
                for left in edges], device=scores.device, dtype=torch.float32)
            z, marginals, _ = decoder._first_order_forward_backward(
                scores.detach().float()[None], (transition + penalties)[None],
                starts[None], ends[None], allowed[None],
                torch.tensor([length], device=scores.device), length)
            if not bool(torch.isfinite(z).all()):
                raise ValueError("candidate graph does not admit a complete path")
        return marginals[0].clamp(0.0, 1.0)
    function = getattr(decoder, "edge_posterior", None)
    if function is not None:
        return function(scores, starts, ends, mask, length,
                        allowed_mask=allowed).float()
    for name in ("edge_posteriors", "posterior_edges"):
        function = getattr(decoder, name, None)
        if function is None:
            continue
        kwargs = {}
        if "force_locked" in inspect.signature(function).parameters:
            kwargs["force_locked"] = force_locked
        value = function(scores, edges, length, **kwargs)
        if isinstance(value, tuple):
            value = value[0]
        return torch.as_tensor(value, device=scores.device, dtype=torch.float32)
    # Exact semi-Markov marginals are the derivative of log Z. This fallback is
    # deliberately centralized and can disappear when decoder exposes its API.
    with torch.enable_grad():
        differentiable = scores.detach().float().clone().requires_grad_(True)
        starts = torch.tensor([edge.start for edge in edges], device=scores.device)
        ends = torch.tensor([edge.end for edge in edges], device=scores.device)
        partition = decoder.log_partition(differentiable, starts, ends, allowed, length)
        if not torch.isfinite(partition):
            raise ValueError("candidate graph does not admit a complete path")
        posterior = torch.autograd.grad(partition, differentiable)[0]
    return posterior.detach().clamp(0.0, 1.0)

class V4Pipeline:
    """Candidate graph, checkpoint scorer, and strict global decoder."""

    def __init__(self, model: SpanG2P, provider: Any,
                 vocab: dict[str, int], *, device: torch.device | str = "cpu",
                 max_reading_length: int = 64, max_length: int | None = None,
                 config: Mapping[str, Any] | None = None,
                 checkpoint_meta: Mapping[str, Any] | None = None,
                 input_feature_builder: Any | None = None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.provider = provider
        self.vocab = dict(vocab)
        self.config = dict(config or {})
        # Checkpoints predating reading deduplication retain the dense path.
        self.deduplicate_readings = bool(
            self.config.get("deduplicate_readings", False))
        self.checkpoint_meta = dict(checkpoint_meta or {})
        self.input_feature_builder = input_feature_builder
        self.id_to_token = {index: token for token, index in self.vocab.items()}
        self.max_reading_length = int(max_reading_length)
        configured = int(max_length or self.config.get("max_length", 0) or 0)
        rope_limit = self._model_sequence_limit()
        self.max_length = min(value for value in (configured, rope_limit) if value > 0)

    def _model_sequence_limit(self) -> int:
        blocks = getattr(self.model, "blocks", ())
        if blocks and hasattr(blocks[0], "rope"):
            return int(blocks[0].rope.cos.size(-2))
        position = getattr(self.model, "position_embed", None)
        if position is not None:
            return int(position.num_embeddings)
        return int(self.config.get("max_seq_len", 256))

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *,
                        device: torch.device | str | None = None,
                        vocab_path: str | Path | None = None,
                        ids_path: str | Path | None = None,
                        kanjidic_path: str | Path | None = None,
                        jmdict_path: str | Path | None = None,
                        unidic_dir: str | Path | None = None,
                        alnum_table: str | Path | None = None,
                        yomogi_dict: str | Path | None = None,
                        unified_lexicon_path: str | Path | None = None,
                        use_legacy_lexicon_only: bool = False,
                        lyric_train: str | Path | Sequence[str | Path] | None = None,
                        strict_resources: bool = True,
                        provider: Any | None = None, **provider_overrides) -> "V4Pipeline":
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "model" not in payload:
            raise ValueError("v4 checkpoint must contain 'model' and 'config'")
        config = dict(payload.get("config") or {})
        if not config:
            raise ValueError("v4 checkpoint is missing its training config")
        parents = list(checkpoint_path.parents)
        root = next((parent for parent in parents
                     if (parent / "kashi_g2p").is_dir()), Path.cwd())
        resolved_unified = None
        if unified_lexicon_path not in (False, "", "none", "None"):
            resolved_unified = _resolve_path(
                unified_lexicon_path or config.get("unified_lexicon_path"), root)
            if resolved_unified is None and not use_legacy_lexicon_only:
                default_pack = root / "artifacts" / "lexicon_pack" / "unified_lexicon.pkl.gz"
                if default_pack.is_file():
                    resolved_unified = default_pack
        selected = {
            "vocab": _resolve_path(vocab_path or config.get("vocab"), root),
            "ids": _resolve_path(ids_path or config.get("ids"), root),
            "kanjidic": _resolve_path(kanjidic_path or config.get("kanjidic"), root),
            "jmdict": _resolve_path(jmdict_path or config.get("jmdict"), root),
            "unidic": _resolve_path(unidic_dir or config.get("unidic"), root),
            "alnum_table": _resolve_path(alnum_table or config.get("alnum_table"), root),
            "yomogi_dict": (_resolve_path(yomogi_dict, root) if yomogi_dict
                            else _resolve_paths(config.get("yomogi_dict"), root)),
            "lexicon_pack": _resolve_path(config.get("lexicon_pack"), root),
            "lyric_train": _resolve_paths(lyric_train, root),
        }
        for name, value in selected.items():
            if name == "ids" and value is None and not any(
                    key.startswith(("component_", "char_type_"))
                    for key in payload["model"]):
                continue
            if resolved_unified is not None and name in (
                    "kanjidic", "jmdict", "unidic", "alnum_table", "yomogi_dict"):
                # Unified lexicon pack supersedes discrete lexical resources
                continue
            paths = (value if isinstance(value, list) else [value])
            for path in paths:
                if path is not None and not path.exists():
                    raise FileNotFoundError(
                        f"checkpoint resource {name!r} does not exist: {path}")
        if resolved_unified is None:
            _validate_resource_manifest(payload.get("resource_manifest"), selected,
                                        strict=strict_resources)
        vocab_resource = selected["vocab"]
        if vocab_resource is None:
            raise ValueError("checkpoint config does not specify a vocabulary")
        vocab = load_vocab(vocab_resource)
        if provider is None:
            candidates = _candidate_api()
            options = {
                "kanjidic_path": selected["kanjidic"],
                "jmdict_path": selected["jmdict"], "unidic_dir": selected["unidic"],
                "lyric_train": selected["lyric_train"],
                "lexicon_pack": selected["lexicon_pack"],
                "unified_lexicon_path": resolved_unified,
                **_provider_options(config, payload.get("resource_manifest")),
            }
            options.update(provider_overrides)
            provider = candidates.CompositeProvider.from_resources(**options)
        from .data import SparseEdgeCollator, load_component_ids
        model_state = payload["model"]
        uses_input_features = any(key.startswith(("component_", "char_type_"))
                                  for key in model_state)
        feature_builder = None
        if uses_input_features:
            component_ids = load_component_ids(
                selected["ids"], str(config.get("ids_format", "chise_v1")))
            feature_builder = SparseEdgeCollator(
                provider, vocab, int(config.get("max_reading_length", 64)),
                ids=component_ids,
                max_span_length=int(config.get("max_span_length", 16)),
                deduplicate_readings=bool(
                    config.get("deduplicate_readings", False)),
                generator_supervision=False)
        model = _load_model(model_state, config, vocab=vocab)
        target_device = device or config.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, provider, vocab, device=target_device, config=config,
                   max_reading_length=int(config.get("max_reading_length", 64)),
                   max_length=int(config.get("max_length", 0) or 0) or None,
                   checkpoint_meta=payload,
                   input_feature_builder=feature_builder)

    @classmethod
    def from_resources(cls, *, vocab_path: str | Path,
                       kanjidic_path: str | Path | None = None,
                       jmdict_path: str | Path | None = None,
                       unidic_dir: str | Path | None = None,
                       lyric_train: str | Path | None = None,
                       device: torch.device | str = "cpu",
                       model: SpanG2P | None = None, **options) -> "V4Pipeline":
        """Explicit untrained-model constructor retained for tests and research."""
        candidates = _candidate_api()
        vocab = load_vocab(vocab_path)
        provider_keys = set(inspect.signature(
            candidates.CompositeProvider.from_resources).parameters)
        provider_options = {key: value for key, value in options.items()
                            if key in provider_keys}
        provider = candidates.CompositeProvider.from_resources(
            kanjidic_path=kanjidic_path, lyric_train=lyric_train,
            jmdict_path=jmdict_path, unidic_dir=unidic_dir, **provider_options)
        max_reading_length = int(options.get("max_reading_length", 64))
        config = {
            "deduplicate_readings": bool(
                options.get("deduplicate_readings", False)),
            "dynamic_reading_length": bool(
                options.get("dynamic_reading_length", False)),
        }
        return cls(model or SpanG2P(vocab_size=max(vocab.values()) + 1), provider,
                   vocab, device=device, config=config,
                   max_reading_length=max_reading_length)

    def _build_edges(self, text: str) -> list[Any]:
        edges = list(self.provider.build(text))
        if text and not edges:
            raise ValueError("candidate provider returned no edges")
        for edge in edges:
            edge.validate(text)
        return edges

    def _generator_inference_ready(self) -> bool:
        if not bool(self.config.get("generator_inference", False)):
            return False
        ready = (self.checkpoint_meta.get("generator_ready")
                 or self.config.get("generator_ready"))
        warmup = int(self.config.get("warmup_steps", 0) or 0)
        step = int(self.checkpoint_meta.get("step", 0) or 0)
        return bool(ready) and step >= warmup

    def _decode_generated_reading(self, token_ids: Tensor,
                                  length: int) -> tuple[str, int] | None:
        generator = getattr(self.model, "generator", None)
        if generator is None or length < 2:
            return None
        ids = [int(value) for value in token_ids[:length].tolist()]
        eos_id = int(getattr(generator, "eos_id", self.vocab.get("[SEP]", 3)))
        bos_id = int(getattr(generator, "bos_id", self.vocab.get("[CLS]", 2)))
        pad_id = int(getattr(generator, "pad_id", self.vocab.get("[PAD]", 0)))
        if eos_id not in ids:
            return None
        eos_index = ids.index(eos_id)
        content_ids = [value for value in ids[:eos_index]
                       if value not in {bos_id, eos_id, pad_id}]
        tokens = [self.id_to_token.get(value, "") for value in content_ids]
        if (not tokens or any(len(token) != 1 or not
                ("\u3041" <= token <= "\u3096" or token == "ー") for token in tokens)):
            return None
        reading = normalize_reading("".join(tokens))
        return (reading, len(content_ids)) if reading else None

    def _generated_edges(self, batch: Any, output: Any) -> list[list[Any]]:
        if (getattr(self.model, "generator", None) is None
                or not self._generator_inference_ready()):
            return [[] for _ in batch.texts]
        edge_type = importlib.import_module("kashi_g2p.model_types").Edge
        allowed_ids = [index for token, index in self.vocab.items()
                       if len(token) == 1 and is_kana_reading(token)]
        allowed = torch.tensor(allowed_ids, dtype=torch.long, device=self.device)
        results: list[list[Any]] = []
        top_spans = int(self.config.get("generator_top_spans", 8))
        beam_size = int(self.config.get("generator_beam_size", 4))
        for row, row_text in enumerate(batch.texts):
            generated = self.model.propose_readings(
                output.hidden[row:row + 1, :len(row_text)],
                top_spans=top_spans, beam_size=beam_size,
                max_reading_length=self.max_reading_length,
                allowed_token_ids=allowed)
            row_edges = []
            for start, end, tokens, length, score in zip(
                    generated.span_start.tolist(), generated.span_end.tolist(),
                    generated.token_ids, generated.lengths.tolist(),
                    generated.log_probs.tolist()):
                decoded = self._decode_generated_reading(tokens, int(length))
                if decoded is not None:
                    reading, token_count = decoded
                    normalized_score = float(score) / max(1, token_count + 1)
                    row_edges.append(edge_type(
                        int(start), int(end), row_text[int(start):int(end)], reading,
                        "generated", prior=normalized_score, confidence=0.0))
            results.append(row_edges)
        return results

    def _forward(self, texts: list[str], all_edges: list[list[Any]]):
        candidates = _candidate_api()

        def pack(edges: list[list[Any]]):
            packed = candidates.pack_edges(
                texts, edges, self.vocab,
                max_reading_length=self.max_reading_length,
                dynamic_reading_length=bool(
                    self.config.get("dynamic_reading_length", True)),
                deduplicate_readings=self.deduplicate_readings,
                prior_index=getattr(self.provider, "prior_index", None))
            if self.input_feature_builder is not None:
                packed.component_ids, packed.char_type_ids = (
                    self.input_feature_builder._input_features(packed.texts))
            return packed

        batch = pack(all_edges)
        device_batch = batch.to(self.device)
        output = self.model(**device_batch.model_kwargs())
        generated = self._generated_edges(batch, output)
        if any(generated):
            all_edges = [edges + additions
                         for edges, additions in zip(all_edges, generated)]
            batch = pack(all_edges)
            output = self.model(**batch.to(self.device).model_kwargs())
        scores = getattr(output, "edge_scores", None)
        if scores is None:
            raise RuntimeError("SpanG2P.forward must return edge_scores")
        return batch, output

    def _decode(self, text: str, edges: list[Any], scores: Tensor, *,
                force_locked: bool,
                compute_posterior: bool = True,
                transition_scores: Tensor | None = None) -> dict[str, Any]:
        decoder = _decoder_api()
        path = decoder.decode_viterbi(scores, edges, len(text),
                                      force_locked=force_locked,
                                      transition_scores=transition_scores)
        # The forward-backward posterior only feeds the reported per-edge
        # confidence; bulk evaluation can skip its per-row DP entirely.
        posterior = (_edge_posteriors(scores, edges, len(text), force_locked,
                                     transition_scores=transition_scores)
                     if compute_posterior else None)
        selected = []
        for index, edge in zip(path.edge_indices, path.edges):
            confidence = (float(posterior[index].detach().cpu())
                          if posterior is not None else 1.0)
            selected.append({
                "start": edge.start, "end": edge.end, "surface": edge.surface,
                "reading": edge.reading, "source": edge.source,
                "locked": edge.locked, "score": float(scores[index].detach().cpu()),
                "confidence": confidence,
            })
        confidence = min((edge["confidence"] for edge in selected), default=1.0)
        return {"text": text, "reading": path.reading, "edges": selected,
                "score": path.score, "confidence": confidence}

    def _safe_windows(self, text: str, edges: list[Any]) -> list[tuple[int, int]]:
        """Choose deterministic boundaries that never bisect a provider edge."""
        if len(text) <= self.max_length:
            return [(0, len(text))]
        windows = []
        start = 0
        while start < len(text):
            limit = min(len(text), start + self.max_length)
            end = limit
            if limit < len(text):
                while True:
                    crossing = [edge for edge in edges
                                if edge.start < end < edge.end and edge.end > start]
                    if not crossing:
                        break
                    # Moving left preserves every cross-boundary edge in the next
                    # window. Repeat because overlapping edges may form a chain.
                    moved = min(edge.start for edge in crossing)
                    if moved >= end:
                        raise RuntimeError("safe window boundary did not advance")
                    end = moved
            if end <= start:
                blockers = [edge for edge in edges
                            if edge.start <= start and edge.end > limit]
                span = max((edge.end - edge.start for edge in blockers), default=0)
                raise ValueError(
                    "provider edge exceeds the model window limit; no safe split "
                    f"at offset {start} (edge length {span}, limit {self.max_length})"
                )
            windows.append((start, end))
            start = end
        return windows

    def _local_edges(self, text: str, edges: list[Any],
                     start: int, end: int) -> list[Any]:
        result = []
        for edge in edges:
            if start <= edge.start and edge.end <= end:
                result.append(replace(edge, start=edge.start - start,
                                      end=edge.end - start))
            elif edge.start < end and start < edge.end:
                raise RuntimeError("internal error: a window bisected a provider edge")
        local_text = text[start:end]
        if local_text and not result:
            raise ValueError("candidate provider returned no edges for a safe window")
        for edge in result:
            edge.validate(local_text)
        return result

    @torch.inference_mode(False)
    def predict_batch(self, texts: Sequence[str], *, force_locked: bool = True,
                      allow_windows: bool = True,
                      compute_posterior: bool = True) -> list[dict[str, Any]]:
        normalized = [normalize_surface(str(text)) for text in texts]
        if any(not text for text in normalized):
            raise ValueError("empty input text is not supported")
        results: list[dict[str, Any] | None] = [None] * len(normalized)
        direct_indices = [index for index, text in enumerate(normalized)
                          if len(text) <= self.max_length]
        if direct_indices:
            direct_texts = [normalized[index] for index in direct_indices]
            direct_edges = [self._build_edges(text) for text in direct_texts]
            with torch.inference_mode():
                batch, output = self._forward(direct_texts, direct_edges)
            trans_scores = getattr(output, "transition_scores", None)
            for row, original_index in enumerate(direct_indices):
                row_trans = trans_scores[row] if trans_scores is not None else None
                results[original_index] = self._decode(
                    batch.texts[row], batch.edges[row],
                    output.edge_scores[row, :len(batch.edges[row])],
                    force_locked=force_locked,
                    compute_posterior=compute_posterior,
                    transition_scores=row_trans)
        for index, text in enumerate(normalized):
            if results[index] is not None:
                continue
            if not allow_windows:
                raise ValueError(
                    f"input length {len(text)} exceeds model limit {self.max_length}")
            results[index] = self._predict_long(text, force_locked=force_locked,
                                                compute_posterior=compute_posterior)
        return [result for result in results if result is not None]

    def predict(self, text: str, *, force_locked: bool = True,
                allow_windows: bool = True,
                compute_posterior: bool = True) -> dict[str, Any]:
        return self.predict_batch([text], force_locked=force_locked,
                                  allow_windows=allow_windows,
                                  compute_posterior=compute_posterior)[0]

    def _predict_long(self, text: str, *, force_locked: bool,
                      compute_posterior: bool = True) -> dict[str, Any]:
        global_edges = self._build_edges(text)
        windows = self._safe_windows(text, global_edges)
        local_texts = [text[start:end] for start, end in windows]
        local_edges = [self._local_edges(text, global_edges, start, end)
                       for start, end in windows]
        with torch.inference_mode():
            batch, output = self._forward(local_texts, local_edges)
        stitched_edges = []
        readings = []
        score = 0.0
        confidence = 1.0
        for row, (start, _end) in enumerate(windows):
            transitions = getattr(output, "transition_scores", None)
            decoded = self._decode(
                batch.texts[row], batch.edges[row],
                output.edge_scores[row, :len(batch.edges[row])],
                force_locked=force_locked,
                compute_posterior=compute_posterior,
                transition_scores=(transitions[row] if transitions is not None else None))
            readings.append(decoded["reading"])
            score += float(decoded["score"])
            confidence = min(confidence, float(decoded["confidence"]))
            for edge in decoded["edges"]:
                shifted = dict(edge)
                shifted["start"] += start
                shifted["end"] += start
                stitched_edges.append(shifted)
        return {"text": text, "reading": "".join(readings),
                "edges": stitched_edges, "score": score,
                "confidence": confidence,
                "windows": [{"start": start, "end": end}
                            for start, end in windows]}
