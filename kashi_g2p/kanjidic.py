from __future__ import annotations

import gzip
import json
import re
import warnings
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from .normalization import hira, is_kanji

KUN_SUFFIX_RE = re.compile(r"[.].*$")
BASE_READING_RE = re.compile(r"^[\u3041-\u3096ー]+$")
# Kept as a compatibility symbol for callers that want to impose the old
# validation policy explicitly.  Candidate loading itself has no reading
# length cap; the collator grows the candidate axis for longer strings.
MAX_BASE_READING_LEN: int | None = None
BAD_READING_START = frozenset("ぁぃぅぇぉゃゅょゎっ")


@dataclass(frozen=True)
class KanjidicReading:
    text: str
    kind: str
    okurigana: str = ""


class Kanjidic:
    def __init__(self, readings: dict[str, set[str] | dict[str, str] | list[KanjidicReading]] | None = None, *, truncated: bool = False,
                 filtered_readings: int = 0, include_okurigana: bool = False,
                 okurigana: dict[str, dict[str, str]] | None = None):
        # Internally keep one type for every reading.  The constructor accepts
        # the old set-based cache format so existing artifacts remain usable.
        normalized: dict[str, dict[str, str]] = {}
        for char, values in (readings or {}).items():
            out: dict[str, str] = {}
            if isinstance(values, dict):
                out.update({str(k): str(v) for k, v in values.items()})
            else:
                for value in values:
                    if isinstance(value, KanjidicReading):
                        out[value.text] = value.kind
                    else:
                        out[str(value)] = ""
            normalized[char] = out
        self.readings = normalized
        self.truncated = truncated
        self.filtered_readings = filtered_readings
        self.include_okurigana = bool(include_okurigana)
        self.okurigana = {
            str(char): {str(reading): str(suffix) for reading, suffix in values.items()}
            for char, values in (okurigana or {}).items()
        }

    @classmethod
    def from_xml(cls, path: str | Path, *, strict: bool = False,
                 include_okurigana: bool = False) -> "Kanjidic":
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        result: dict[str, dict[str, str]] = defaultdict(dict)
        okurigana: dict[str, dict[str, str]] = defaultdict(dict)
        truncated = False
        filtered_readings = 0
        try:
            with opener(path, "rb") as fh:
                for event, elem in ET.iterparse(fh, events=("end",)):
                    if elem.tag != "character":
                        continue
                    literal = elem.findtext("literal")
                    if not literal:
                        elem.clear()
                        continue
                    group = elem.find("reading_meaning/rmgroup")
                    if group is not None:
                        for node in group.findall("reading"):
                            r_type = node.attrib.get("r_type")
                            if r_type not in {"ja_on", "ja_kun"}:
                                continue
                            # KANJIDIC2 uses ``-`` as an okurigana marker in
                            # some ja_kun entries (for example ``-ゆ.き``).
                            # It is metadata, not part of the reading.
                            raw_reading = hira(node.text or "").replace("-", "")
                            suffix = raw_reading.split(".", 1)[1] if "." in raw_reading else ""
                            reading = raw_reading
                            # In KANJIDIC ``.`` marks the start of okurigana.
                            # Legacy LAKE labels removed the suffix entirely.
                            # Exp3 keeps it in B and lets C learn the truncation.
                            reading = (reading.replace(".", "") if include_okurigana
                                       else KUN_SUFFIX_RE.sub("", reading)).strip()
                            # KANJIDIC also contains explanatory phrase-like
                            # kun readings for rare characters.  They are not
                            # usable as a per-character base reading.
                            if (reading and BASE_READING_RE.fullmatch(reading)
                                    and reading[0] not in BAD_READING_START):
                                result[literal][reading] = "音" if r_type == "ja_on" else "訓"
                                if include_okurigana and suffix:
                                    okurigana[literal][reading] = suffix
                            else:
                                filtered_readings += 1
                    elem.clear()
        except EOFError:
            if strict:
                raise
            truncated = True
            warnings.warn(f"truncated gzip resource: {path}; using partial KANJIDIC", RuntimeWarning)
        return cls(dict(result), truncated=truncated,
                   filtered_readings=filtered_readings,
                   include_okurigana=include_okurigana,
                   okurigana=dict(okurigana))

    def candidates(self, char: str) -> set[str]:
        return set(self.readings.get(char, {}))

    def typed_candidates(self, char: str) -> list[KanjidicReading]:
        return [
            KanjidicReading(text, kind or "訓", self.okurigana.get(char, {}).get(text, ""))
            for text, kind in self.readings.get(char, {}).items()
        ]

    def reading_type(self, char: str, reading: str) -> str:
        return self.readings.get(char, {}).get(reading, "訓")

    def validate(self, max_reading_len: int | None = MAX_BASE_READING_LEN) -> dict[str, list]:
        """Return structural issues in the parsed base-reading table."""
        issues: dict[str, list] = {
            "empty": [], "dup": [], "too_long": [], "not_hiragana": [], "bad_start": []
        }
        for char, values in self.readings.items():
            readings = list(values)
            if not readings:
                issues["empty"].append(char)
            if len(readings) != len(set(readings)):
                issues["dup"].append(char)
            for reading in readings:
                if (max_reading_len is not None
                        and len(reading) > max_reading_len):
                    issues["too_long"].append((char, reading))
                elif not BASE_READING_RE.fullmatch(reading):
                    issues["not_hiragana"].append((char, reading))
                elif reading[0] in "ぁぃぅぇぉゃゅょゎっ":
                    issues["bad_start"].append((char, reading))
        return issues

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.readings, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "Kanjidic":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data)


def load_reading_lexicon(path: str | Path, *,
                         include_okurigana: bool = False) -> Kanjidic:
    """Load a candidate-reading table for any character language.

    Japanese runs continue to use KANJIDIC2 XML.  Other languages can supply
    the same simple JSON shape, for example ``{"你": ["ni3", "ni"],
    "行": ["xing2", "hang2"]}``; the rest of the model only requires the
    ``candidates``/``typed_candidates`` interface.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        return Kanjidic.load_json(path)
    return Kanjidic.from_xml(path, include_okurigana=include_okurigana)


def load_ids(path: str | Path, *, format_version: str = "chise_v1") -> dict[str, list[str]]:
    """Load character decomposition data, retaining leaf components per char.

    Formats:
    - ``chise_v1``: CHISE IDS ``U+XXXX<TAB>literal<TAB>IDS`` (ids.txt /
      cjkvi-ids, GPLv2).
    - ``cjkdecomp``: amake/cjk-decomp ``literal:type(comp,comp)`` lines
      (Apache-2.0 option).  Lines whose head is a 5-digit number define
      intermediate decompositions, not glyphs; numeric component references
      are skipped for the same reason.
    - ``legacy``: old two-field parse kept for old checkpoints whose
      component branch was never trained.
    """
    if format_version not in {"legacy", "chise_v1", "cjkdecomp"}:
        raise ValueError("IDS format_version must be 'legacy', 'chise_v1' or 'cjkdecomp'")
    result: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if format_version == "cjkdecomp":
            head, sep, tail = line.partition(":")
            if not sep:
                continue
            char = head.strip()
            if len(char) != 1:
                continue
            open_idx = tail.find("(")
            close_idx = tail.rfind(")")
            if open_idx == -1 or close_idx < open_idx:
                continue
            leaves = []
            for comp in tail[open_idx + 1:close_idx].split(","):
                comp = comp.strip()
                if len(comp) == 1 and not comp.isspace():
                    leaves.append(comp)
            result[char] = leaves[:8]
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        # CHISE IDS.txt is normally ``U+XXXX<TAB>literal<TAB>IDS``.  The old
        # parser treated the codepoint as the literal and therefore returned
        # an empty table.  Keep that behavior only for old checkpoints whose
        # component branch was never trained.
        if format_version == "chise_v1" and len(fields) >= 3 and fields[0].startswith("U+"):
            char, expression = fields[1], fields[2]
        else:
            char, expression = fields[0], fields[1]
        if len(char) != 1:
            continue
        # The trailing region marker (e.g. ``[GJK]``) describes which glyph
        # forms the decomposition applies to; it is not part of the IDS tree.
        expression = re.sub(r"\[[^]]*\]\s*$", "", expression)
        leaves = [
            c for c in expression
            if not (0x2ff0 <= ord(c) <= 0x2fff)
            and not c.isspace()
            and not (0xE0100 <= ord(c) <= 0xE01EF)  # variation selectors
        ]
        result[char] = leaves[:8]
    return result
