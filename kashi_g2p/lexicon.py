"""Dictionary resources and full-match lattice construction.

The lattice is deliberately a light-weight feature source: it never chooses a
single segmentation and it never receives ruby gold labels.  JMdict entries
are indexed in a trie; UniDic contributes the tokenizer's observed spans when
available.  Missing resources simply result in an empty lattice (the model's
character path remains usable).
"""
from __future__ import annotations

import gzip
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .normalization import hira


@dataclass(frozen=True)
class LatticeNode:
    start: int
    end: int
    surface: str
    reading: str
    pos: str = "その他"
    log_frequency: float = 0.0
    bies: str = "S"
    source: str = "jmdict"


def _coarse_pos(value: str | None) -> str:
    value = str(value or "")
    if "助動詞" in value: return "助動"
    if "名詞" in value or "固有" in value: return "名"
    if "動詞" in value: return "動"
    if "形容詞" in value: return "形"
    if "副詞" in value: return "副"
    if "助詞" in value: return "助"
    if "接続" in value: return "接"
    if "連体" in value: return "連体"
    if "感動" in value: return "感"
    if "接頭" in value: return "接頭"
    if "接尾" in value: return "接尾"
    return "その他"


class PrefixTrie:
    def __init__(self):
        self.root: dict[str, object] = {}

    def add(self, surface: str, value: tuple[str, str, float, str]):
        node = self.root
        for ch in surface:
            node = node.setdefault(ch, {})  # type: ignore[assignment]
        node.setdefault("\0", []).append(value)  # type: ignore[index]

    def matches(self, text: str, start: int, max_len: int = 32):
        node = self.root
        for i in range(start, min(len(text), start + max_len)):
            child = node.get(text[i])
            if child is None:
                break
            node = child  # type: ignore[assignment]
            for value in node.get("\0", []):  # type: ignore[union-attr]
                yield i + 1, value


class JMdictLexicon:
    def __init__(self):
        self.trie = PrefixTrie()
        self.size = 0

    @classmethod
    def from_xml(cls, path: str | Path) -> "JMdictLexicon":
        result = cls()
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as fh:
            for _, entry in ET.iterparse(fh, events=("end",)):
                if entry.tag != "entry":
                    continue
                kebs = [x.text or "" for x in entry.findall("k_ele/keb")]
                readings = []
                for rel in entry.findall("r_ele"):
                    reb = hira(rel.findtext("reb") or "")
                    if reb:
                        restrictions = {x.text for x in rel.findall("re_restr") if x.text}
                        readings.append((reb, restrictions))
                pos = _coarse_pos(entry.findtext("sense/pos"))
                # JMdict priority tags provide an ordinal frequency proxy.
                pri = [x.text or "" for x in entry.findall("r_ele/re_pri") + entry.findall("k_ele/ke_pri")]
                freq = float(sum(1 for p in pri if p.startswith(("news", "ichi", "spec", "nf"))))
                for surface in kebs:
                    for reading, restrictions in readings:
                        if restrictions and surface not in restrictions:
                            continue
                        surface = surface.replace("・", "")
                        if surface and reading:
                            result.trie.add(surface, (reading, pos, freq, "jmdict"))
                            result.size += 1
                entry.clear()
        return result

    def matches(self, text: str, start: int):
        for end, (reading, pos, freq, source) in self.trie.matches(text, start):
            yield LatticeNode(start, end, text[start:end], reading, pos, math.log1p(freq),
                              "S" if end - start == 1 else "B", source)


class CorpusLexicon:
    """Train-only word readings exported from explicit corpus ruby spans."""

    def __init__(self):
        self.trie = PrefixTrie()
        self.size = 0
        self.metadata: dict[str, object] = {}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CorpusLexicon":
        result = cls()
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if line_number == 1 and value.get("type") == "lake_corpus_lexicon":
                    result.metadata = value
                    continue
                surface = str(value.get("surface", ""))
                reading = hira(str(value.get("reading", "")))
                count = int(value.get("count", 0))
                pos = str(value.get("pos", "その他"))
                if surface and reading and count > 0:
                    result.trie.add(surface, (reading, pos, float(count), "corpus"))
                    result.size += 1
        return result

    def matches(self, text: str, start: int):
        for end, (reading, pos, freq, source) in self.trie.matches(text, start):
            yield LatticeNode(start, end, text[start:end], reading, pos,
                              math.log1p(freq),
                              "S" if end - start == 1 else "B", source)


class LatticeBuilder:
    def __init__(self, jmdict: JMdictLexicon | None = None, *,
                 corpus: CorpusLexicon | None = None, tagger=None,
                 node_cap: int = 2048):
        self.jmdict = jmdict
        self.corpus = corpus
        self.tagger = tagger
        self.node_cap = node_cap
        # fugashi.Tagger owns a native pointer and cannot be pickled on
        # Windows.  Keep the dictionary location so DataLoader workers can
        # reconstruct their own tagger after spawn.
        self.unidic_dir: str | None = None
        self.last_overflow = False

    @classmethod
    def from_resources(cls, jmdict_path: str | Path | None = None,
                       unidic_dir: str | Path | None = None, node_cap: int = 2048,
                       corpus_lexicon_path: str | Path | None = None):
        lex = JMdictLexicon.from_xml(jmdict_path) if jmdict_path and Path(jmdict_path).exists() else None
        corpus = (CorpusLexicon.from_jsonl(corpus_lexicon_path)
                  if corpus_lexicon_path and Path(corpus_lexicon_path).exists() else None)
        tagger = None
        if unidic_dir:
            try:
                import fugashi
                tagger = fugashi.Tagger(f'-r nul -d "{unidic_dir}"')
            except Exception:
                tagger = None
        result = cls(lex, corpus=corpus, tagger=tagger, node_cap=node_cap)
        result.unidic_dir = str(unidic_dir) if unidic_dir else None
        return result

    def __getstate__(self):
        # Do not attempt to pickle the native fugashi object.  JMdict's trie
        # is pure Python and remains shared through the serialized state.
        return {
            "jmdict": self.jmdict,
            "corpus": self.corpus,
            "node_cap": self.node_cap,
            "unidic_dir": self.unidic_dir,
        }

    def __setstate__(self, state):
        self.jmdict = state.get("jmdict")
        self.corpus = state.get("corpus")
        self.node_cap = state.get("node_cap", 2048)
        self.unidic_dir = state.get("unidic_dir")
        self.tagger = None
        if self.unidic_dir:
            try:
                import fugashi
                self.tagger = fugashi.Tagger(f'-r nul -d "{self.unidic_dir}"')
            except Exception:
                self.tagger = None
        self.last_overflow = False

    def build(self, text: str) -> list[LatticeNode]:
        nodes: list[LatticeNode] = []
        if self.jmdict is not None:
            for start in range(len(text)):
                nodes.extend(self.jmdict.matches(text, start))
        if self.corpus is not None:
            for start in range(len(text)):
                nodes.extend(self.corpus.matches(text, start))
        # Add the Viterbi spans from UniDic as an independent feature source.
        # Full-match JMdict entries remain intact; this only improves coverage
        # for inflected forms and particles not present in JMdict.
        if self.tagger is not None:
            cursor = 0
            try:
                for word in self.tagger(text):
                    surface = str(word.surface)
                    start = text.find(surface, cursor)
                    if start < 0:
                        continue
                    feature = getattr(word, "feature", None)
                    reading = hira(getattr(feature, "pron", "") or getattr(feature, "pronBase", "") or "")
                    if reading:
                        pos = _coarse_pos(getattr(feature, "pos1", None))
                        nodes.append(LatticeNode(start, start + len(surface), surface, reading,
                                                 pos, 0.0, "S" if len(surface) == 1 else "B", "unidic"))
                    cursor = start + len(surface)
            except Exception:
                pass
        # Deduplicate exact candidates while preserving source diversity.
        unique = {}
        source_rank = {"corpus": 3, "jmdict": 2, "unidic": 1}
        for node in nodes:
            key = (node.start, node.end, node.surface, node.reading, node.pos)
            previous = unique.get(key)
            if (previous is None or (node.log_frequency, source_rank.get(node.source, 0))
                    > (previous.log_frequency, source_rank.get(previous.source, 0))):
                unique[key] = node
        nodes = list(unique.values())
        self.last_overflow = len(nodes) > self.node_cap
        if self.last_overflow:
            nodes.sort(key=lambda n: (n.end - n.start, n.log_frequency, n.source), reverse=True)
            nodes = nodes[:self.node_cap]
        return sorted(nodes, key=lambda n: (n.start, n.end, n.surface))


def nodes_to_arrays(nodes: list[LatticeNode], length: int, cap: int | None = None):
    """Return node list and a [length, node_count] coverage matrix."""
    if cap is not None:
        nodes = nodes[:cap]
    coverage = [[False] * len(nodes) for _ in range(length)]
    for j, node in enumerate(nodes):
        for i in range(max(0, node.start), min(length, node.end)):
            coverage[i][j] = True
    return nodes, coverage
