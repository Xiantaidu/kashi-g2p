"""kashi-g2p v4: span candidates with global decoding.

Also hosts the shared reading infrastructure: normalization, KANJIDIC,
lattice lexicon, collation, and the dictionary/rule baselines.
"""

from .candidates import (CandidateProvider, CompositeProvider,
                         EnglishNumberProvider, HaqumeiProvider,
                         KanjidicProvider, LegacyLexiconProvider,
                         LyricMemoryProvider, build_resource_manifest,
                         resource_fingerprint)
from .decoder import (BatchedDecodedPath, DecodedPath, allowed_edge_mask,
                      batched_decode_viterbi, batched_edge_posterior,
                      batched_log_partition, batched_partial_path_nll,
                      decode_viterbi, partial_path_nll)
from .data import (SparseEdgeCollator, build_gold_edges,
                   gold_constraint_reachable)
from .model import (GeneratedReadings, OpenReadingGenerator, SpanG2P,
                    SpanModelOutput, parameter_count)
from .model_types import Edge, EdgeBatch

__all__ = [
    "CandidateProvider", "CompositeProvider", "EnglishNumberProvider",
    "HaqumeiProvider",
    "KanjidicProvider", "LegacyLexiconProvider", "LyricMemoryProvider",
    "build_resource_manifest", "resource_fingerprint",
    "BatchedDecodedPath", "DecodedPath", "decode_viterbi",
    "batched_decode_viterbi", "batched_edge_posterior",
    "batched_log_partition", "batched_partial_path_nll",
    "allowed_edge_mask", "partial_path_nll",
    "SparseEdgeCollator", "build_gold_edges", "gold_constraint_reachable",
    "GeneratedReadings", "OpenReadingGenerator", "SpanG2P",
    "SpanModelOutput", "parameter_count", "Edge", "EdgeBatch",
]
