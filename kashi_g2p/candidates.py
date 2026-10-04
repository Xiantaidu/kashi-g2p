"""Candidate providers for the v4 span model.

Providers only retrieve possible readings.  They never choose a segmentation;
that decision belongs to ``kashi_g2p.decoder``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch

from .kanjidic import Kanjidic, load_reading_lexicon
from .lexicon import LatticeBuilder
from .normalization import hira, is_kanji, normalize_surface

from .model_types import Edge, EdgeBatch, SOURCE_TO_ID


class CandidateProvider(Protocol):
    def build(self, text: str) -> list[Edge]:
        """Return valid, non-empty candidates for a normalized input text."""


_HASH_CHUNK_SIZE = 1 << 20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        return {
            "kind": "file",
            "path": resolved.as_posix(),
            "size": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
        }
    if resolved.is_dir():
        files = []
        for item in sorted(
                (entry for entry in resolved.rglob("*") if entry.is_file()),
                key=lambda entry: entry.relative_to(resolved).as_posix()):
            files.append({
                "path": item.relative_to(resolved).as_posix(),
                "size": item.stat().st_size,
                "sha256": _sha256_file(item),
            })
        return {"kind": "directory", "path": resolved.as_posix(), "files": files}
    raise FileNotFoundError(f"candidate resource does not exist: {resolved}")


def build_resource_manifest(*, kanjidic_path: str | Path | None = None,
                            jmdict_path: str | Path | None = None,
                            unidic_dir: str | Path | None = None,
                            alnum_table: str | Path | None = None,
                            yomogi_dict: str | Path | None = None,
                            lyric_train: (str | Path | Sequence[str | Path]
                                          | None) = None,
                            **settings: Any) -> dict[str, Any]:
    """Describe candidate resources and settings with full content hashes."""
    resources: dict[str, Any] = {}
    for name, path in (("kanjidic", kanjidic_path), ("jmdict", jmdict_path),
                       ("unidic", unidic_dir), ("alnum_table", alnum_table)):
        if path is not None:
            resources[name] = _path_manifest(path)
    if yomogi_dict is not None:
        yomogi_paths = ((yomogi_dict,) if isinstance(yomogi_dict, (str, Path))
                        else tuple(yomogi_dict))
        resources["yomogi_dict"] = sorted(
            (_path_manifest(item) for item in yomogi_paths),
            key=lambda item: str(item.get("path", "")))
    if lyric_train is not None:
        paths = ((lyric_train,) if isinstance(lyric_train, (str, Path))
                 else tuple(lyric_train))
        lyric_manifests = [_path_manifest(path) for path in paths]
        resources["lyric_train"] = sorted(
            lyric_manifests, key=lambda item: str(item["path"]))
    normalized_settings = json.loads(json.dumps(
        settings, ensure_ascii=False, sort_keys=True, default=str))
    return {"version": 1, "resources": resources, "settings": normalized_settings}


def resource_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 for a resource manifest."""
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class HaqumeiProvider:
    """Expose Haqumei's word-level reading mapping as span candidates.

    Haqumei's detailed mapping is preferred over its flat phoneme output because
    v4 initially targets the ruby ``read`` field.  ``pron`` remains available
    to a later singing-phonology target.
    """

    def __init__(self, *, include_non_japanese: bool = True, **options):
        try:
            import haqumei
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("HaqumeiProvider requires the haqumei package") from exc
        self.include_non_japanese = include_non_japanese
        self.engine = haqumei.Haqumei(**options)

    @staticmethod
    def _canonical_surface(text: str) -> str:
        digits = str.maketrans("0123456789", "〇一二三四五六七八九")
        return normalize_surface(text).translate(digits)

    @classmethod
    def _locate(cls, text: str, surface: str, cursor: int) -> tuple[int, int] | None:
        """Align Haqumei's normalized number surface back to original text."""
        source = cls._canonical_surface(text)
        target = cls._canonical_surface(surface)
        start = source.find(target, cursor)
        if start < 0:
            return None
        return start, start + len(surface)

    @staticmethod
    def _is_non_japanese(surface: str) -> bool:
        return any(ch.isascii() and (ch.isalpha() or ch.isdigit()) for ch in surface)

    def build(self, text: str) -> list[Edge]:
        result: list[Edge] = []
        cursor = 0
        try:
            details = self.engine.g2p_mapping_detailed(text)
        except Exception:
            return result
        for detail in details:
            surface = normalize_surface(str(getattr(detail, "word", "") or ""))
            reading = hira(str(getattr(detail, "read", "") or ""))
            if not surface or not reading:
                continue
            if not self.include_non_japanese and self._is_non_japanese(surface):
                continue
            located = self._locate(text, surface, cursor)
            if located is None:
                continue
            start, end = located
            cursor = end
            unknown = bool(getattr(detail, "is_unknown", False))
            result.append(Edge(
                start, end, text[start:end], reading, "haqumei",
                prior=-0.75 if unknown else 0.75,
                confidence=0.45 if unknown else 0.92,
                # English and number spans are deterministic preprocessing,
                # so they do not enter the neural scorer by default.
                locked=self._is_non_japanese(text[start:end]),
            ))
        return result


class PyOpenJTalkProvider:
    """Expose pyopenjtalk's G2P predictions as high-confidence candidate edges.

    Provides both word-level span edges for multi-kanji words and decomposed
    character-level edges (when Kanjidic is provided), directly populating
    the candidate graph with external dictionary/G2P knowledge.
    """

    def __init__(self, *, kanjidic: Kanjidic | None = None,
                 prior: float = 0.75, confidence: float = 0.90):
        try:
            import pyopenjtalk
        except ImportError as exc:
            raise RuntimeError("PyOpenJTalkProvider requires pyopenjtalk package") from exc
        self.kanjidic = kanjidic
        self.prior = prior
        self.confidence = confidence

    def build(self, text: str) -> list[Edge]:
        import pyopenjtalk
        try:
            entries = pyopenjtalk.run_frontend(text)
        except Exception:
            return []

        tokens: list[tuple[str, str]] = []
        for entry in entries:
            surface = str(entry.get("string", "") or "")
            reading = hira(str(entry.get("read", "") or entry.get("pron", "") or ""))
            if surface:
                tokens.append((surface, reading))
        if not tokens:
            return []

        result: list[Edge] = []
        seen: set[tuple[int, int, str]] = set()
        cursor = 0

        # 1. Word-level span edges for tokens that contain kanji
        for surface, reading in tokens:
            start = text.find(surface, cursor)
            if start < 0:
                continue
            end = start + len(surface)
            cursor = end
            if not reading or not any(is_kanji(ch) for ch in surface):
                continue
            edge_key = (start, end, reading)
            if edge_key not in seen:
                seen.add(edge_key)
                result.append(Edge(
                    start, end, text[start:end], reading, "pyopenjtalk",
                    prior=self.prior, confidence=self.confidence,
                ))

        # 2. Decompose into character-level edges if kanjidic is provided
        if self.kanjidic is not None:
            from .baseline import assign_char_readings, _Token
            token_objs = [_Token(s, r) for s, r in tokens]
            try:
                chars, _spans = assign_char_readings(text, token_objs, self.kanjidic)
                for index, char_reading in enumerate(chars):
                    if char_reading and is_kanji(text[index]):
                        edge_key = (index, index + 1, char_reading)
                        if edge_key not in seen:
                            seen.add(edge_key)
                            result.append(Edge(
                                index, index + 1, text[index], char_reading, "pyopenjtalk",
                                prior=self.prior, confidence=self.confidence,
                            ))
            except Exception:
                pass

        return result


class EnglishNumberProvider:
    """Deterministic readings for ASCII words and numbers.

    Known English words use a small, reviewable lexicon; unseen words fall
    back to spelling letter names ONLY for uppercase acronyms (<=4 chars, e.g. DJ, TV).
    Numeric runs use Japanese cardinal rules, with common counter compounds
    handled as one edge (``2人 -> ふたり``, ``3人 -> さんにん``, ``10曲 -> じゅっきょく``).
    """

    _WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
    _NUMBER_RE = re.compile(r"\d+")
    _COUNTER_RE = re.compile(r"(?P<number>\d+)(?P<counter>時間|人|日|秒|分|回|歩|歳|才|年|月|個|本|枚|台|番|曲|時|目|匹|通|件|点|度|階|杯|冊|倍|段|つ)")
    # Kanji-numeral date/counter words (五日, 七月, 十時...). pyopenjtalk only
    # offers the Yamato date reading (五日->いつか) and jmdict the wrong Sino
    # day reading (五日->ごにち), so the coexisting correct Sino reading
    # (五日->ごか) is missing from every provider and the path is forced onto
    # the wrong candidate (found in the s600 attribution: 七月五日高一生の夏).
    _KANJI_NUM_RE = re.compile(
        r"(?P<number>[一二三四五六七八九十百千]+)(?P<counter>時間|日|秒|分|回|歳|年|月|個|本|枚|台|番|曲|時|匹|件|点|度|階|杯|冊|倍|段|つ)")
    _KANJI_DIGIT_VALUE = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
        "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000,
    }

    _DIGITS = ("ぜろ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう")
    _DIGIT_READINGS = {
        0: ("ぜろ", "れい"),
        1: ("いち", "わん"),
        2: ("に", "つー"),
        3: ("さん", "すりー"),
        4: ("よん", "し", "ふぉー"),
        5: ("ご", "ふぁいぶ"),
        6: ("ろく", "しっくす"),
        7: ("なな", "しち", "せぶん"),
        8: ("はち", "えいと"),
        9: ("きゅう", "く", "ないん"),
        10: ("じゅう", "てん"),
    }
    _LETTER_NAMES = {
        "a": "えー", "b": "びー", "c": "しー", "d": "でぃー",
        "e": "いー", "f": "えふ", "g": "じー", "h": "えいち",
        "i": "あい", "j": "じぇー", "k": "けー", "l": "える",
        "m": "えむ", "n": "えぬ", "o": "おー", "p": "ぴー",
        "q": "きゅー", "r": "あーる", "s": "えす", "t": "てぃー",
        "u": "ゆー", "v": "ぶい", "w": "だぶりゅー", "x": "えっくす",
        "y": "わい", "z": "ぜっと",
    }
    _WORDS = {
        "a": "えー", "an": "あん", "are": "あー", "baby": "べいびー",
        "bye": "ばい", "come": "かむ", "day": "でい", "good": "ぐっど",
        "hello": "はろー", "how": "はう", "i": "あい", "just": "じゃすと",
        "like": "らいく", "love": "らぶ", "merry": "めりー", "me": "みー",
        "my": "まい", "only": "おんりー", "ready": "れでぃ", "sf": "えすえふ",
        "spot": "すぽっと", "the": "ざ", "to": "とぅ", "type": "たいぷ",
        "want": "うぉんと", "what": "ほわっと", "you": "ゆー", "your": "ゆあー",
        "you're": "ゆあー", "we": "うぃー", "with": "うぃず",
        "up": "あっぷ", "la": "ら", "yah": "やー", "yeah": "いぇー",
        "oh": "おー", "ah": "あー", "no": "のー", "go": "ごー",
        "all": "おーる", "one": "わん", "two": "つー", "three": "すりー",
        "four": "ふぉー", "five": "ふぁいぶ", "six": "しっくす", "seven": "せぶん",
        "eight": "えいと", "nine": "ないん", "ten": "てん", "be": "びー",
        "do": "どぅー", "so": "そー", "in": "いん", "on": "おん",
        "it": "いっと", "is": "いず", "of": "おぶ", "for": "ふぉー",
        "and": "あんど", "night": "ないと", "time": "たいむ", "music": "みゅーじっく",
        "song": "そんぐ", "heart": "はーと", "dream": "どりーむ", "star": "すたー",
        "world": "わーるど", "dance": "だんす", "party": "ぱーてぃー", "girl": "がーる",
        "boy": "ぼーい", "never": "ねばー", "always": "おーるうぇいず",
        "forever": "ふぉーえばー", "together": "とぅげざー", "tonight": "とぅないと",
        "stop": "すとっぷ", "joyful": "じょいふる", "again": "あげいん",
    }
    _SPECIAL_COUNTERS = {
        "人": {1: "ひとり", 2: "ふたり", 4: "よにん"},
        "日": {1: "ついたち", 2: "ふつか", 3: "みっか", 4: "よっか",
               5: "いつか", 6: "むいか", 7: "なのか", 8: "ようか",
               9: "ここのか", 10: "とおか", 14: "じゅうよっか",
               20: "はつか", 24: "にじゅうよっか"},
        "つ": {1: "ひとつ", 2: "ふたつ", 3: "みっつ", 4: "よっつ",
               5: "いつつ", 6: "むっつ", 7: "ななつ", 8: "やっつ",
               9: "ここのつ", 10: "とお"},
        "月": {4: "しがつ", 7: "しちがつ", 9: "くがつ"},
        "時": {4: "よじ", 9: "くじ", 14: "じゅうよじ", 24: "にじゅうよじ"},
        "時間": {4: "よじかん", 9: "くじかん", 14: "じゅうよじかん", 24: "にじゅうよじかん"},
        "年": {4: "よねん"},
    }
    _COUNTER_SUFFIX = {
        "秒": ("びょう", {}),
        "分": ("ふん", {1: "いっぷん", 3: "さんぷん", 4: "よんぷん", 6: "ろっぷん", 8: "はっぷん", 10: "じゅっぷん"}),
        "回": ("かい", {1: "いっかい", 6: "ろっかい", 8: "はっかい", 10: "じゅっかい"}),
        "歩": ("ほ", {1: "いっぽ", 3: "さんぽ", 6: "ろっぽ", 8: "はっぽ", 10: "じゅっぽ"}),
        "日": ("か", {1: "いちにち", 2: "ふつか", 3: "さんか", 4: "よんか",
                      5: "ごか", 6: "ろっか", 7: "しちか", 8: "はちか",
                      9: "きゅうか", 10: "じゅっか"}),
        "歳": ("さい", {1: "いっさい", 8: "はっさい", 10: "じゅっさい", 20: "はたち"}),
        "才": ("さい", {1: "いっさい", 8: "はっさい", 10: "じゅっさい", 20: "はたち"}),
        "人": ("にん", {1: "ひとり", 2: "ふたり", 4: "よにん"}),
        "年": ("ねん", {4: "よねん"}),
        "月": ("がつ", {4: "しがつ", 7: "しちがつ", 9: "くがつ"}),
        "個": ("こ", {1: "いっこ", 6: "ろっこ", 8: "はっこ", 10: "じゅっこ"}),
        "本": ("ほん", {1: "いっぽん", 2: "にほん", 3: "さんぼん", 6: "ろっぽん", 8: "はっぽん", 10: "じゅっぽん"}),
        "枚": ("まい", {}),
        "台": ("だい", {}),
        "番": ("ばん", {}),
        "曲": ("きょく", {1: "いっきょく", 6: "ろっきょく", 8: "はっきょく", 10: "じゅっきょく"}),
        "時": ("じ", {4: "よじ", 9: "くじ", 14: "じゅうよじ", 24: "にじゅうよじ"}),
        "時間": ("じかん", {4: "よじかん", 9: "くじかん", 14: "じゅうよじかん", 24: "にじゅうよじかん"}),
        "目": ("め", {}),
        "匹": ("ひき", {1: "いっぴき", 3: "さんびき", 6: "ろっぴき", 8: "はっぴき", 10: "じゅっぴき"}),
        "通": ("つう", {1: "いっつう", 8: "はっつう", 10: "じゅっつう"}),
        "件": ("けん", {1: "いっけん", 3: "さんげん", 6: "ろっけん", 8: "はっけん", 10: "じゅっけん"}),
        "点": ("てん", {1: "いってん", 10: "じゅってん"}),
        "度": ("ど", {1: "いちど", 2: "にど", 3: "さんど"}),
        "階": ("かい", {1: "いっかい", 3: "さんがい", 6: "ろっかい", 8: "はっかい", 10: "じゅっかい"}),
        "杯": ("はい", {1: "いっぱい", 2: "にはい", 3: "さんばい", 6: "ろっぱい", 8: "はっぱい", 10: "じゅっぱい"}),
        "冊": ("さつ", {1: "いっさつ", 8: "はっさつ", 10: "じゅっさつ"}),
        "倍": ("ばい", {}),
        "段": ("だん", {}),
        "つ": ("", {}),
    }

    def __init__(self, words: Mapping[str, str] | None = None):
        self.words = {**self._WORDS, **(words or {})}

    @classmethod
    def _under_1000(cls, value: int) -> str:
        result = ""
        hundreds, value = divmod(value, 100)
        if hundreds:
            result += {1: "ひゃく", 2: "にひゃく", 3: "さんびゃく", 4: "よんひゃく", 5: "ごひゃく", 6: "ろっぴゃく", 7: "ななひゃく", 8: "はっぴゃく", 9: "きゅうひゃく"}[hundreds]
        tens, ones = divmod(value, 10)
        if tens:
            result += "じゅう" if tens == 1 else cls._DIGITS[tens] + "じゅう"
        if ones:
            result += cls._DIGITS[ones]
        return result or cls._DIGITS[0]

    @classmethod
    def _number(cls, value: int) -> str:
        if value < 0:
            raise ValueError("number must be non-negative")
        if value == 0:
            return cls._DIGITS[0]
        if value < 1000:
            return cls._under_1000(value)
        if value < 10000:
            thousands, rest = divmod(value, 1000)
            prefix = {
                1: "せん", 2: "にせん", 3: "さんぜん", 4: "よんせん",
                5: "ごせん", 6: "ろくせん", 7: "ななせん",
                8: "はっせん", 9: "きゅうせん",
            }[thousands]
            return prefix + (cls._under_1000(rest) if rest else "")
        for divisor, name in ((10**12, "ちょう"), (10**8, "おく"), (10**4, "まん")):
            if value >= divisor:
                high, low = divmod(value, divisor)
                prefix = cls._number(high)
                return prefix + name + (cls._number(low) if low else "")
        raise ValueError("number is outside Japanese cardinal range")

    @classmethod
    def _number_readings(cls, value: int) -> list[str]:
        standard = cls._number(value)
        readings = [standard]
        for alt in cls._DIGIT_READINGS.get(value, ()):
            if alt not in readings:
                readings.append(alt)
        return readings

    @classmethod
    def _number_counter(cls, value: int, counter: str) -> str:
        special = cls._SPECIAL_COUNTERS.get(counter, {}).get(value)
        if special:
            return special
        suffix, values = cls._COUNTER_SUFFIX.get(counter, (counter, {}))
        if value in values:
            return values[value]
        return cls._number(value) + suffix

    @classmethod
    def _word_reading(cls, word: str, words: Mapping[str, str]) -> tuple[str, float]:
        lowered = word.lower()
        if lowered in words:
            return hira(words[lowered]), 0.99
        if word.isupper() and len(word) <= 4 and all(ch in cls._LETTER_NAMES for ch in lowered):
            return "".join(cls._LETTER_NAMES[ch] for ch in lowered), 0.93
        return "", 0.0

    def build(self, text: str) -> list[Edge]:
        result: list[Edge] = []
        consumed: set[int] = set()
        for match in self._COUNTER_RE.finditer(text):
            value = int(match.group("number"))
            reading = self._number_counter(value, match.group("counter"))
            start, end = match.span()
            if reading and not any(is_kanji(ch) for ch in reading):
                result.append(Edge(start, end, text[start:end], reading, "rules",
                                   prior=2.5, confidence=1.0))
            consumed.update(range(start, end))
        # Kanji-numeral counter words: emit BOTH the rule reading and the Sino
        # digit+suffix reading. Only for 1..10 numerals where the Sino form is
        # a legal counter reading; higher kanji numerals keep the rule form.
        for match in self._KANJI_NUM_RE.finditer(text):
            value = self._KANJI_DIGIT_VALUE.get(match.group("number"))
            if value is None or value > 10:
                continue
            counter = match.group("counter")
            start, end = match.span()
            rule_reading = self._number_counter(value, counter)
            readings = []
            for reading in (rule_reading, self._number(value)
                            + self._COUNTER_SUFFIX[counter][0]):
                if reading and reading not in readings and not any(
                        is_kanji(ch) for ch in reading):
                    readings.append(reading)
            for reading in readings:
                result.append(Edge(start, end, text[start:end], reading, "rules",
                                   prior=1.5, confidence=0.9))
            consumed.update(range(start, end))
        for match in self._NUMBER_RE.finditer(text):
            if any(index in consumed for index in range(*match.span())):
                continue
            start, end = match.span()
            val = int(match.group())
            for reading in self._number_readings(val):
                result.append(Edge(start, end, text[start:end], reading,
                                   "rules", prior=2.0, confidence=1.0))
        for match in self._WORD_RE.finditer(text):
            start, end = match.span()
            reading, confidence = self._word_reading(match.group(), self.words)
            if reading:
                result.append(Edge(start, end, text[start:end], reading, "rules",
                                   prior=1.8 if confidence > 0.95 else 0.5,
                                   confidence=confidence))
        return result


class KanjidicProvider:
    """Single-character fallback candidates from KANJIDIC2.

    The v3 label convention keeps okurigana inside B and lets C trim it, so a
    ruby like 伝わ→つたわる assembles the anchor 伝 to ``つた`` plus the kana
    that follow.  Alongside each full reading the provider therefore also emits
    the okurigana-trimmed base; otherwise a supervised anchor with C>0 has no
    reachable gold edge.  Rendaku/handaku/sokuon variants of short readings are
    emitted too because the per-character D label can derive a reading that no
    dictionary lists (葉→ば, 一→いっ).
    """

    _VOICED = dict(zip("かきくけこさしすせそたちつてとはひふへほ",
                       "がぎぐげござじずぜぞだぢづでどばびぶべぼ"))
    _SEMI = dict(zip("はひふへほ", "ぱぴぷぺぽ"))

    def __init__(self, dictionary: Kanjidic):
        self.dictionary = dictionary

    @classmethod
    def from_path(cls, path: str | Path, *, include_okurigana: bool = True):
        return cls(load_reading_lexicon(path, include_okurigana=include_okurigana))

    @classmethod
    def _variants(cls, reading: str) -> set[str]:
        result = {reading}
        if reading:
            result.add(cls._VOICED.get(reading[0], reading[0]) + reading[1:])
            result.add(cls._SEMI.get(reading[0], reading[0]) + reading[1:])
            if reading[-1] in "かきくけこたちつてと":
                result.add(reading[:-1] + "っ")
        return result

    def build(self, text: str) -> list[Edge]:
        result = []
        for index, char in enumerate(text):
            if not is_kanji(char):
                continue
            readings = set(self.dictionary.candidates(char))
            expanded = set(readings)
            for reading in readings:
                suffix = self.dictionary.okurigana.get(char, {}).get(reading, "")
                if suffix and reading.endswith(suffix):
                    base = reading[:-len(suffix)]
                    if base:
                        expanded.add(base)
            for reading in sorted(expanded):
                if len(reading) <= 4:
                    for variant in self._variants(reading):
                        result.append(Edge(
                            index, index + 1, char, variant, "kanjidic"))
                else:
                    result.append(Edge(index, index + 1, char, reading, "kanjidic"))
        return result


class LegacyLexiconProvider:
    """Adapt the old model's JMdict + UniDic lattice into v4 edges.

    This intentionally excludes ``CorpusLexicon``.  The old experiments
    showed that train-corpus phrase injection raises coverage while making CER
    much worse through long-span competition. JMdict and UniDic are external,
    word-level Japanese knowledge and are safe to expose as bounded candidates.
    """

    def __init__(self, builder: LatticeBuilder, *, node_cap: int = 512):
        if node_cap < 1:
            raise ValueError("node_cap must be positive")
        self.builder = builder
        self.node_cap = int(node_cap)

    @classmethod
    def from_resources(cls, *, jmdict_path: str | Path | None = None,
                       unidic_dir: str | Path | None = None,
                       node_cap: int = 512) -> "LegacyLexiconProvider":
        return cls(LatticeBuilder.from_resources(
            jmdict_path=jmdict_path,
            unidic_dir=unidic_dir,
            node_cap=node_cap,
            corpus_lexicon_path=None,
        ), node_cap=node_cap)

    def build(self, text: str) -> list[Edge]:
        try:
            nodes = self.builder.build(text)
        except Exception:
            return []
        result = []
        for node in nodes[:self.node_cap]:
            reading = hira(str(node.reading or ""))
            surface = str(node.surface or "")
            source = str(node.source or "jmdict")
            if not surface or not reading or source not in {"jmdict", "unidic"}:
                continue
            # KANJIDIC and COPY already handle kana-only material. Restricting
            # this adapter to Han-containing words avoids changing ordinary
            # kana pronunciation merely because UniDic segmented it.
            if not any(is_kanji(char) for char in surface):
                continue
            confidence = 0.82 if source == "jmdict" else 0.76
            prior = min(1.5, 0.35 * float(node.log_frequency))
            result.append(Edge(
                node.start, node.end, surface, reading, source,
                prior=prior, confidence=confidence,
            ))
        return result


class LyricMemoryProvider:
    """Train-only memory for explicit lyric special spans."""

    def __init__(self, entries: dict[str, dict[str, int]] | None = None):
        self.entries = {
            str(surface): {str(reading): int(count) for reading, count in values.items()}
            for surface, values in (entries or {}).items()
        }

    @classmethod
    def from_jsonl(cls, path: str | Path | Sequence[str | Path], *, min_count: int = 1) -> "LyricMemoryProvider":
        entries: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        paths = (path,) if isinstance(path, (str, Path)) else tuple(path)
        for item in paths:
            with Path(item).open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = str(row.get("text", ""))
                    for span in row.get("special_spans", []) or []:
                        try:
                            start = int(span["start"]); end = int(span["end"])
                            reading = hira(str(span["reading"]))
                        except (KeyError, TypeError, ValueError):
                            continue
                        if 0 <= start < end <= len(text) and reading:
                            entries[text[start:end]][reading] += 1
        filtered = {
            surface: dict(values)
            for surface, values in entries.items()
            if any(count >= min_count for count in values.values())
        }
        return cls(filtered)

    def build(self, text: str) -> list[Edge]:
        result = []
        for surface, readings in self.entries.items():
            cursor = 0
            while True:
                start = text.find(surface, cursor)
                if start < 0:
                    break
                end = start + len(surface)
                for reading, count in readings.items():
                    result.append(Edge(
                        start, end, surface, reading, "lyric_memory",
                        prior=1.0 + math.log1p(count), confidence=min(0.99, 0.6 + 0.04 * math.log1p(count)),
                    ))
                cursor = end
        return result


class AlnumTableProvider:
    """Mined latin/digit reading table from supervised training ruby.

    The legacy exp6→exp11 step showed this is the largest single supervised-CER
    lever (English words that are actually sung carry their ruby reading on the
    span start).  Entries keep only unambiguous readings: a single reading or a
    dominant reading with >= 90% of the mined count.  Table edges compete with
    COPY and rules in the unified score space; nothing is locked, so the model
    still learns to reject readings for English that is not sung.
    """

    _MIN_SHARE = 0.9

    def __init__(self, table: Mapping[str, Mapping[str, int]]):
        self.table: dict[str, tuple[str, ...]] = {}
        self._lower_table: dict[str, tuple[str, ...]] = {}
        for span, readings in (table or {}).items():
            span = str(span)
            if not span or not readings:
                continue
            total = sum(int(count) for count in readings.values())
            if total <= 0:
                continue
            ranked = sorted(readings.items(), key=lambda kv: (-int(kv[1]), kv[0]))
            if len(ranked) == 1 or ranked[0][1] / total >= self._MIN_SHARE:
                self.table[span] = (ranked[0][0],)
            else:
                # Ambiguous spans stay selectable with their top readings.
                self.table[span] = tuple(reading for reading, _ in ranked[:3])
            low = span.lower()
            if low not in self._lower_table:
                self._lower_table[low] = self.table[span]
        # First-character index: scanning all spans per row costs
        # len(table) str.find calls; bucketing by leading character reduces
        # the per-row work to spans whose initial actually occurs.
        self._by_first: dict[str, list[str]] = {}
        for span in self.table:
            self._by_first.setdefault(span[0].lower(), []).append(span)

    @classmethod
    def from_json(cls, path: str | Path) -> "AlnumTableProvider":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        table = payload.get("table", payload) if isinstance(payload, Mapping) else {}
        return cls(table)

    def build(self, text: str) -> list[Edge]:
        result: list[Edge] = []
        seen_first: set[str] = {char.lower() for char in set(text)}
        matched_spans: set[tuple[int, int]] = set()
        for initial, spans in self._by_first.items():
            if initial not in seen_first:
                continue
            for span in spans:
                cursor = 0
                while True:
                    start = text.find(span, cursor)
                    if start < 0:
                        break
                    end = start + len(span)
                    matched_spans.add((start, end))
                    for reading in self.table[span]:
                        result.append(Edge(
                            start, end, span, reading, "alnum",
                            prior=1.2, confidence=0.9))
                    cursor = end
        # Case-insensitive fallback for unmatched alphanumeric tokens
        for match in re.finditer(r"[A-Za-z0-9]+", text):
            span_range = match.span()
            if span_range not in matched_spans:
                word = match.group()
                low = word.lower()
                if low in self._lower_table:
                    for reading in self._lower_table[low]:
                        result.append(Edge(
                            span_range[0], span_range[1], word, reading, "alnum",
                            prior=1.2, confidence=0.9))
        return result


class YomogiDictProvider:
    """Wide-coverage surface->reading dictionary (Yomogi v1, MIT licensed).

    335,883 entries compiled from pyopenjtalk/NAIST-jdic, unidic-csj,
    AzooKey dictionary storage (NEologd, SudachiDict) and Mozc-UT
    (Wikipedia, personal names).  Only kanji-containing surfaces are kept --
    kana and katakana surfaces are already covered by COPY -- and katakana
    readings are converted to hiragana for the project convention.  Entries
    add candidate coverage for proper nouns and mixed alphanumeric spans that
    JMdict and UniDic lack.
    """

    _MAX_READINGS = 4
    _MAX_SURFACE = 24

    def __init__(self, table: Mapping[str, Sequence[str]]):
        self.table: dict[str, tuple[str, ...]] = {}
        for surface, readings in (table or {}).items():
            self.table[surface] = tuple(sorted(readings)[:self._MAX_READINGS])
        self._by_first: dict[str, list[str]] = {}
        for surface in self.table:
            self._by_first.setdefault(surface[0], []).append(surface)

    @classmethod
    def from_tsv(cls, path: str | Path | Sequence[str | Path]
                 ) -> "YomogiDictProvider":
        """Load one or more 4-column TSVs (id, surface, katakana reading, pron)."""
        paths = ((path,) if isinstance(path, (str, Path)) else tuple(path))
        surfaces: dict[str, set[str]] = defaultdict(set)
        for item in paths:
          with Path(item).open(encoding="utf-8") as handle:
            for line in handle:
                row = line.rstrip(chr(10)).split(chr(9))
                if len(row) != 4:
                    continue
                surface = normalize_surface(row[1])
                reading = row[2]
                if not surface or not reading or len(surface) > cls._MAX_SURFACE:
                    continue
                if not any(is_kanji(char) for char in surface) or len(surface) < 2:
                    continue
                hiragana = "".join(
                    chr(ord(char) - 0x60) if chr(0x30A1) <= char <= chr(0x30F6) else char
                    for char in reading)
                if (hiragana and all(
                        chr(0x3041) <= char <= chr(0x3096) or char == chr(0x30FC)
                        for char in hiragana)):
                    surfaces[surface].add(hiragana)
        table = {surface: sorted(readings)
                 for surface, readings in surfaces.items()}
        return cls(table)

    def build(self, text: str) -> list[Edge]:
        result: list[Edge] = []
        present = set(text)
        for initial, surfaces in self._by_first.items():
            if initial not in present:
                continue
            for surface in surfaces:
                cursor = 0
                while True:
                    start = text.find(surface, cursor)
                    if start < 0:
                        break
                    end = start + len(surface)
                    for reading in self.table[surface]:
                        result.append(Edge(
                            start, end, surface, reading, "yomogi_dict",
                            # Below UniDic (0.76): this source is the widest
                            # and noisiest, and at 0.8 its edges evicted
                            # UniDic gold under the per-span cap (exp24).
                            prior=0.6, confidence=0.75))
                    cursor = end
        return result


class CopyProvider:
    """Emit the complete one-character identity path for any input."""

    def build(self, text: str) -> list[Edge]:
        return [Edge(i, i + 1, char, char, "copy", prior=-1.0)
                for i, char in enumerate(text)]


class CompositeProvider:
    """Merge providers into a bounded graph with a complete COPY path."""

    def __init__(self, providers: Sequence[CandidateProvider], *,
                 source_priority: Sequence[str] = (),
                 per_span_reading_cap: int | None = None,
                 total_edge_cap: int | None = None):
        """Configure caps for non-COPY readings; COPY edges are exempt."""
        if per_span_reading_cap is not None and per_span_reading_cap < 1:
            raise ValueError("per_span_reading_cap must be positive or None")
        if total_edge_cap is not None and total_edge_cap < 1:
            raise ValueError("total_edge_cap must be positive or None")
        self.providers = list(providers)
        self.per_span_reading_cap = per_span_reading_cap
        self.total_edge_cap = total_edge_cap
        # Earlier names are higher priority. _rank uses normal tuple ordering,
        # so assign descending values rather than the raw list index.
        self.source_priority = {
            name: len(source_priority) - index
            for index, name in enumerate(source_priority)
        }
        self.prior_index: dict[tuple[str, str], int] | None = None
        self.resource_manifest: dict[str, Any] = {}
        self.resource_fingerprint = resource_fingerprint(self.resource_manifest)

    @classmethod
    def from_resources(cls, *, kanjidic_path: str | Path | None = None,
                       lyric_train: str | Path | Sequence[str | Path] | None = None,
                       include_okurigana: bool = True,
                       use_haqumei: bool = True,
                       use_pyopenjtalk: bool = False,
                       use_rules: bool = True,
                       jmdict_path: str | Path | None = None,
                       unidic_dir: str | Path | None = None,
                       use_legacy_lexicon: bool = True,
                       legacy_node_cap: int = 512,
                       lyric_min_count: int = 1,
                       haqumei_options: dict | None = None,
                       alnum_table: str | Path | None = None,
                       yomogi_dict: (str | Path | Sequence[str | Path] | None) = None,
                       lexicon_pack: str | Path | None = None,
                       unified_lexicon_path: str | Path | None = None,
                       per_span_reading_cap: int | None = None,
                       total_edge_cap: int | None = None) -> "CompositeProvider":
        providers: list[CandidateProvider] = []
        kanjidic_obj = None

        if unified_lexicon_path and Path(unified_lexicon_path).exists():
            from .unified_provider import UnifiedLexiconProvider
            providers.append(UnifiedLexiconProvider.from_pack(
                unified_lexicon_path, use_rules=use_rules, use_copy=False))
        else:
            if use_rules:
                providers.append(EnglishNumberProvider())
            if alnum_table:
                providers.append(AlnumTableProvider.from_json(alnum_table))
            if yomogi_dict:
                providers.append(YomogiDictProvider.from_tsv(yomogi_dict))
            if use_legacy_lexicon and (jmdict_path or unidic_dir):
                providers.append(LegacyLexiconProvider.from_resources(
                    jmdict_path=jmdict_path, unidic_dir=unidic_dir,
                    node_cap=legacy_node_cap,
                ))
            if kanjidic_path:
                kanjidic_obj = load_reading_lexicon(
                    kanjidic_path, include_okurigana=include_okurigana)
                providers.append(KanjidicProvider(kanjidic_obj))

        if use_haqumei:
            try:
                providers.append(HaqumeiProvider(
                    include_non_japanese=False, **(haqumei_options or {})))
            except RuntimeError:
                pass
        if lyric_train:
            providers.append(LyricMemoryProvider.from_jsonl(lyric_train, min_count=lyric_min_count))

        if kanjidic_obj is None and kanjidic_path:
            kanjidic_obj = load_reading_lexicon(
                kanjidic_path, include_okurigana=False)
        if use_pyopenjtalk:
            try:
                providers.append(PyOpenJTalkProvider(kanjidic=kanjidic_obj))
            except RuntimeError:
                pass
        providers.append(CopyProvider())
        result = cls(providers, source_priority=(
            "gold", "lyric_memory", "alnum", "rules", "haqumei", "pyopenjtalk", "jmdict",
            "unidic", "yomogi_dict", "kanjidic", "copy",
        ), per_span_reading_cap=per_span_reading_cap,
            total_edge_cap=total_edge_cap)
        if lexicon_pack:
            from .lexicon_pack import build_prior_index
            result.prior_index = build_prior_index(lexicon_pack)
        result.resource_manifest = build_resource_manifest(
            kanjidic_path=kanjidic_path,
            jmdict_path=jmdict_path,
            unidic_dir=unidic_dir,
            alnum_table=alnum_table,
            yomogi_dict=yomogi_dict,
            lexicon_pack=lexicon_pack,
            lyric_train=lyric_train,
            include_okurigana=include_okurigana,
            use_haqumei=use_haqumei,
            use_pyopenjtalk=use_pyopenjtalk,
            use_rules=use_rules,
            use_legacy_lexicon=use_legacy_lexicon,
            legacy_node_cap=legacy_node_cap,
            lyric_min_count=lyric_min_count,
            haqumei_options=haqumei_options,
            per_span_reading_cap=per_span_reading_cap,
            total_edge_cap=total_edge_cap,
        )
        result.resource_fingerprint = resource_fingerprint(result.resource_manifest)
        return result

    def build(self, text: str) -> list[Edge]:
        raw: list[Edge] = []
        for provider in self.providers:
            raw.extend(provider.build(text))
        for edge in raw:
            edge.validate(text)

        # An explicit lyric memory entry is allowed to override a deterministic
        # Haqumei edge for the same span.  Ambiguous memory readings stay
        # selectable; they are not silently converted into rules.
        memory_spans = {
            (edge.start, edge.end)
            for edge in raw if edge.source == "lyric_memory"
        }
        adjusted = [
            Edge(edge.start, edge.end, edge.surface, edge.reading, edge.source,
                 edge.prior, edge.confidence,
                 edge.locked and (edge.start, edge.end) not in memory_spans)
            for edge in raw
        ]

        adjusted = self._expand_word_variants(adjusted, text)

        unique: dict[tuple[int, int, str, str], Edge] = {}
        for edge in adjusted:
            copy_discriminator = "copy" if edge.source == "copy" else ""
            key = (edge.start, edge.end, edge.reading, copy_discriminator)
            old = unique.get(key)
            if old is None or self._rank(edge) > self._rank(old):
                unique[key] = edge
        result = self._prune(list(unique.values()), text)
        result = self._filter_truncated_okurigana(result, text)
        result = self._filter_rendaku_traps(result)
        result = self._repair_complete_path(result, text)
        result = self._derive_okurigana_stems(result, text)
        return sorted(result, key=self._output_key)

    @staticmethod
    def _filter_truncated_okurigana(edges: list[Edge], text: str) -> list[Edge]:
        """Drop okurigana-truncated word edges when a full kana-extended edge exists.

        For instance, when '気持ち' appears in text, dictionary entries provide both
        '気持' -> 'きもち' (span [2, 4]) and '気持ち' -> 'きもち' (span [2, 5]).
        Selecting '気持' causes the trailing 'ち' to be read twice ('きもちち').
        When an edge covering [start, end) shares its exact reading with a longer
        edge covering [start, max_end) where all intermediate characters are kana,
        the shorter edge is strictly an okurigana truncation and must be suppressed.
        """
        longer_spans: dict[tuple[int, str], int] = {}
        for e in edges:
            key = (e.start, e.reading)
            if key not in longer_spans or e.end > longer_spans[key]:
                longer_spans[key] = e.end

        filtered: list[Edge] = []
        for e in edges:
            max_end = longer_spans.get((e.start, e.reading), e.end)
            if max_end > e.end:
                ext = text[e.end:max_end]
                if ext and all("\u3040" <= ch <= "\u30ff" for ch in ext):
                    continue
            filtered.append(e)
        return filtered

    @staticmethod
    def _derive_okurigana_stems(edges: list[Edge], text: str) -> list[Edge]:
        """Additive: expose the kun-stem char reading of a kanji+okurigana word.

        A word edge like 失くし->なくし (span [s, s+3)) carries the correct reading,
        but the single-kanji stem 失->な is NOT a standalone char candidate (kanjidic
        gives しつ/うしな), so char-level decoding is forced onto a wrong reading
        (失->うしな).  §18.12 measured that the model, trained with gold injection,
        already scores 失->な highly once it is offered as a candidate.  For each
        kanji-initial word whose surface ends in a hiragana okurigana run and whose
        reading *literally* ends with that okurigana, add the truncated stem edge
        (失->な, 灯->あか, 気付->きづ).  Purely additive -- the parent word edge is
        kept, so the model still chooses between the word edge and the stem.  The
        literal okurigana-suffix guard means no reading is fabricated (kana read as
        themselves), and a pure-kanji stem requirement excludes compounds like 真っ青
        where the target kanji is not the prefix.  The stem inherits the parent
        edge's source so the batch source vocabulary is never widened.
        """
        def _hira(ch: str) -> bool:
            return "ぁ" <= ch <= "ゖ"

        existing = {(e.start, e.end, e.reading) for e in edges}
        derived: list[Edge] = []
        for e in edges:
            surf = text[e.start:e.end]
            if len(surf) < 2:
                continue
            k = len(surf)
            while k > 0 and _hira(surf[k - 1]):
                k -= 1
            okuri = surf[k:]
            stem = surf[:k]
            if not okuri or not stem or not is_kanji(stem[0]):
                continue
            if any(_hira(ch) for ch in stem):        # pure-kanji stem (excludes 真っ青)
                continue
            if not e.reading.endswith(okuri):        # okurigana kana read as themselves
                continue
            stem_reading = e.reading[:-len(okuri)]
            if not stem_reading:
                continue
            key = (e.start, e.start + k, stem_reading)
            if key in existing:
                continue
            existing.add(key)
            derived.append(Edge(e.start, e.start + k, stem, stem_reading,
                                e.source, e.prior, e.confidence, False))
        return edges + derived

    @classmethod
    def _filter_rendaku_traps(cls, edges: list[Edge]) -> list[Edge]:
        """Suppress an unvoiced reading when a rendaku-voiced sibling exists.

        General 連濁 rule (no per-word table): within one span, if a dictionary
        word reading differs from another dictionary word reading of the same
        span by voicing exactly one *non-initial* mora (per the ``_DAKUTEN`` /
        ``_HANDAKU`` phonology maps), the unvoiced form is a compound-medial
        rendaku trap and is dropped in favour of the voiced form.  This
        generalizes the former archaic-trap word list (だいじょうふ→ぶ, てんこく→ご
        are exactly this pattern) to every word without enumerating any.

        Only non-initial positions (k >= 1) are considered, so this never
        collides with the first-mora variants synthesized by
        ``_expand_word_variants``.  Locked (deterministic/copy) edges are never
        dropped.
        """
        by_span: dict[tuple[int, int], list[Edge]] = defaultdict(list)
        for edge in edges:
            by_span[(edge.start, edge.end)].append(edge)

        drop: set[int] = set()
        for span_edges in by_span.values():
            readings = {edge.reading for edge in span_edges
                        if edge.source in cls._WORD_SOURCES}
            if len(readings) < 2:
                continue
            for edge in span_edges:
                if edge.source not in cls._WORD_SOURCES or edge.locked:
                    continue
                reading = edge.reading
                for k in range(1, len(reading)):
                    voiced = cls._DAKUTEN.get(reading[k]) or cls._HANDAKU.get(reading[k])
                    if voiced and reading[:k] + voiced + reading[k + 1:] in readings:
                        drop.add(id(edge))
                        break
        if not drop:
            return edges
        return [edge for edge in edges if id(edge) not in drop]

    @staticmethod
    def _copy_edges(text: str) -> list[Edge]:
        return CopyProvider().build(text)

    @staticmethod
    def _has_complete_path(edges: Sequence[Edge], length: int) -> bool:
        """Return whether directed boundaries connect 0 to ``length``."""
        reachable = [False] * (length + 1)
        reachable[0] = True
        by_start: dict[int, list[int]] = defaultdict(list)
        for edge in edges:
            if 0 <= edge.start < edge.end <= length:
                by_start[edge.start].append(edge.end)
        for start in range(length):
            if reachable[start]:
                for end in by_start.get(start, ()):
                    reachable[end] = True
        return reachable[length]

    _DAKUTEN = dict(zip("かきくけこさしすせそたちつてとはひふへほう",
                        "がぎぐげござじずぜぞだぢづでどばびぶべぼゔ"))
    _HANDAKU = dict(zip("はひふへほ", "ぱぴぷぺぽ"))
    _WORD_SOURCES = frozenset({"jmdict", "unidic", "yomogi_dict", "pyopenjtalk"})

    _PUNCT_OR_SPACE = frozenset(" \t\r\n、。！？!?…・~～-—")

    @classmethod
    def _expand_word_variants(cls, edges: list[Edge], text: str = "") -> list[Edge]:
        """Add first-mora rendaku/handaku variants to dictionary word edges.

        Gold readings reconstructed from kana frequently differ from the
        dictionary form by exactly these transformations; without the variant
        edge the row's gold path is unreachable and the row trains nothing.
        """
        out = list(edges)
        seen = {(edge.start, edge.end, edge.reading) for edge in edges}
        for edge in edges:
            if (edge.source not in cls._WORD_SOURCES or edge.locked
                    or edge.end - edge.start < 2 or len(edge.reading) < 2):
                continue
            # Phonological guard: Rendaku (sequential voicing) is compound-medial only.
            # It cannot occur phrase-initially, after whitespace/punctuation, or on numbers.
            if edge.start == 0:
                continue
            if text and edge.start > 0 and text[edge.start - 1] in cls._PUNCT_OR_SPACE:
                continue
            if edge.surface and (edge.surface[0].isdigit() or edge.surface[0] in "0123456789０１２３４５６７８９"):
                continue
            first = edge.reading[0]
            for mapped in (cls._DAKUTEN.get(first), cls._HANDAKU.get(first)):
                if not mapped:
                    continue
                reading = mapped + edge.reading[1:]
                marker = (edge.start, edge.end, reading)
                if marker not in seen:
                    seen.add(marker)
                    out.append(Edge(
                        edge.start, edge.end, edge.surface, reading,
                        edge.source, max(0.0, edge.prior - 0.4),
                        edge.confidence * 0.85))
        return out

    def _prune(self, edges: list[Edge], text: str) -> list[Edge]:
        mandatory = {
            (edge.start, edge.end, edge.reading, edge.source)
            for edge in self._copy_edges(text)
        }
        keep = list(edges)
        if self.per_span_reading_cap is not None:
            spans: dict[tuple[int, int], list[Edge]] = defaultdict(list)
            for edge in keep:
                spans[(edge.start, edge.end)].append(edge)
            keep = []
            for span in sorted(spans):
                ranked = sorted(spans[span], key=self._prune_key)
                copies = [edge for edge in ranked
                          if (edge.start, edge.end, edge.reading,
                              edge.source) in mandatory]
                others = [edge for edge in ranked if edge not in copies]
                # Take the top-k by rank, then swap in the best fallback edges
                # until the fallback source has representation. jmdict/unidic
                # outrank kanjidic on confidence alone and would otherwise
                # crowd out every okurigana-trimmed or D-varied anchor form.
                selected = others[:self.per_span_reading_cap]
                reserve = max(1, self.per_span_reading_cap // 2)
                fallback = [edge for edge in others
                            if edge.source == "kanjidic"
                            and edge not in selected]
                for edge in fallback:
                    if sum(1 for e in selected
                           if e.source == "kanjidic") >= reserve:
                        break
                    # Replace the weakest non-fallback slot, not the tail
                    # blindly, so the swap cannot evict a better fallback.
                    replaceable = [e for e in selected
                                   if e.source != "kanjidic"]
                    if not replaceable:
                        break
                    selected[selected.index(replaceable[-1])] = edge
                keep.extend(copies)
                keep.extend(selected)
        if self.total_edge_cap is not None:
            copies = [edge for edge in keep
                      if (edge.start, edge.end, edge.reading,
                          edge.source) in mandatory]
            others = [edge for edge in keep if edge not in copies]
            keep = copies + sorted(others, key=self._prune_key)[:self.total_edge_cap]
        copy_keys = {
            (edge.start, edge.end, edge.reading, edge.source)
            for edge in keep if edge.source == "copy"
        }
        for edge in self._copy_edges(text):
            key = (edge.start, edge.end, edge.reading, edge.source)
            if key not in copy_keys:
                keep.append(edge)
                copy_keys.add(key)
        return keep

    def _repair_complete_path(self, edges: list[Edge], text: str) -> list[Edge]:
        if self._has_complete_path(edges, len(text)):
            return edges
        present = {(edge.start, edge.end, edge.reading, edge.source)
                   for edge in edges}
        for edge in self._copy_edges(text):
            key = (edge.start, edge.end, edge.reading, edge.source)
            if key not in present:
                edges.append(edge)
                present.add(key)
        if not self._has_complete_path(edges, len(text)):
            raise ValueError("candidate graph has no complete 0..L path")
        return edges

    def _prune_key(
            self, edge: Edge,
    ) -> tuple[int, float, float, int, int, str, str]:
        priority, confidence, prior = self._rank(edge)
        return (-priority, -confidence, -prior, edge.start, edge.end,
                edge.reading, edge.source)

    @staticmethod
    def _output_key(edge: Edge) -> tuple[int, int, float, str, str]:
        return (edge.start, edge.end, -edge.prior, edge.reading, edge.source)

    def _rank(self, edge: Edge) -> tuple[int, float, float]:
        return (self.source_priority.get(edge.source, -1), edge.confidence, edge.prior)

    def without_provider_types(self, *provider_types: type) -> "CompositeProvider":
        """Reuse loaded resources while removing train-only providers."""
        result = CompositeProvider([
            provider for provider in self.providers
            if not isinstance(provider, provider_types)
        ], per_span_reading_cap=self.per_span_reading_cap,
            total_edge_cap=self.total_edge_cap)
        result.source_priority = dict(self.source_priority)
        result.resource_manifest = dict(self.resource_manifest)
        result.resource_fingerprint = self.resource_fingerprint
        result.prior_index = self.prior_index
        return result


def pack_edges(texts: Sequence[str], all_edges: Sequence[Sequence[Edge]],
               vocab: dict[str, int], *, max_reading_length: int = 64,
               source_to_id: Mapping[str, int] | None = None,
               gold_edge_masks: Sequence[Sequence[bool]] | None = None,
               dynamic_reading_length: bool = False,
               deduplicate_readings: bool = False,
               prior_index: Mapping[tuple[str, str], int] | None = None,
               target_offsets: Sequence[int] | None = None,
               target_lengths: Sequence[int] | None = None,
               ) -> EdgeBatch:
    """Pack an already-built graph, optionally retaining sparse gold masks."""
    normalized = [normalize_surface(text) for text in texts]
    if len(normalized) != len(all_edges):
        raise ValueError("texts and all_edges must have the same batch size")
    all_edges = [list(edges) for edges in all_edges]
    for row_idx, (text, edges) in enumerate(zip(normalized, all_edges)):
        offset = target_offsets[row_idx] if target_offsets is not None else 0
        target_len = target_lengths[row_idx] if target_lengths is not None else len(text)
        for edge in edges:
            if not 0 <= edge.start < edge.end <= target_len:
                raise ValueError(f"edge span {(edge.start, edge.end)} is outside target length {target_len}")
            surface_in_text = text[offset + edge.start : offset + edge.end]
            if surface_in_text != edge.surface:
                raise ValueError(
                    f"edge surface {edge.surface!r} does not match "
                    f"text[{offset + edge.start}:{offset + edge.end}]={surface_in_text!r}"
                )
    if gold_edge_masks is not None:
        if len(gold_edge_masks) != len(all_edges):
            raise ValueError("gold_edge_masks must match the batch size")
        for edges, mask in zip(all_edges, gold_edge_masks):
            if len(edges) != len(mask):
                raise ValueError("each gold edge mask must match its edge list")
    length = max(1, max(map(len, normalized), default=1))
    edge_count = max(1, max((len(edges) for edges in all_edges), default=1))
    actual_reading_length = max(
        (len(edge.reading) for edges in all_edges for edge in edges), default=1
    )
    reading_length = (actual_reading_length if dynamic_reading_length else
                      max(max_reading_length, actual_reading_length))
    source_to_id = dict(SOURCE_TO_ID if source_to_id is None else source_to_id)
    missing_sources = sorted({edge.source for edges in all_edges for edge in edges}
                             - set(source_to_id))
    if missing_sources:
        raise ValueError(
            "unknown candidate source(s); pass a stable source_to_id mapping: "
            + ", ".join(missing_sources)
        )
    pad_id = int(vocab.get("[PAD]", 0)); unk_id = int(vocab.get("[UNK]", 1))
    input_ids = torch.full((len(normalized), length), pad_id, dtype=torch.long)
    attention = torch.zeros((len(normalized), length), dtype=torch.bool)
    starts = torch.zeros((len(normalized), edge_count), dtype=torch.long)
    ends = torch.ones((len(normalized), edge_count), dtype=torch.long)
    reading_ids = torch.full((len(normalized), edge_count, reading_length), pad_id, dtype=torch.long)
    reading_mask = torch.zeros_like(reading_ids, dtype=torch.bool)
    edge_mask = torch.zeros((len(normalized), edge_count), dtype=torch.bool)
    edge_locked = torch.zeros((len(normalized), edge_count), dtype=torch.bool)
    source_ids = torch.zeros((len(normalized), edge_count), dtype=torch.long)
    prior = torch.zeros((len(normalized), edge_count), dtype=torch.float32)
    pack_ids = (torch.zeros((len(normalized), edge_count), dtype=torch.long)
                if prior_index is not None else None)
    gold = (torch.zeros((len(normalized), edge_count), dtype=torch.bool)
            if gold_edge_masks is not None else None)
    unique_keys: dict[tuple[int, ...], int] = {}
    unique_rows: list[tuple[int, ...]] = []
    reading_inverse = (torch.zeros(
        (len(normalized), edge_count), dtype=torch.long)
        if deduplicate_readings else None)

    def token_id(char: str) -> int:
        return int(vocab.get(char, unk_id))

    for batch_index, (text, edges) in enumerate(zip(normalized, all_edges)):
        ids = [token_id(char) for char in text]
        if ids:
            input_ids[batch_index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[batch_index, :len(ids)] = True
        for edge_index, edge in enumerate(edges):
            starts[batch_index, edge_index] = edge.start
            ends[batch_index, edge_index] = edge.end
            edge_mask[batch_index, edge_index] = True
            edge_locked[batch_index, edge_index] = edge.locked
            source_ids[batch_index, edge_index] = source_to_id[edge.source]
            prior[batch_index, edge_index] = edge.prior
            ids = [token_id(char) for char in edge.reading[:reading_length]]
            if ids:
                reading_ids[batch_index, edge_index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                reading_mask[batch_index, edge_index, :len(ids)] = True
            if reading_inverse is not None:
                key = tuple(ids)
                unique_index = unique_keys.get(key)
                if unique_index is None:
                    unique_index = len(unique_rows)
                    unique_keys[key] = unique_index
                    unique_rows.append(key)
                reading_inverse[batch_index, edge_index] = unique_index
            if pack_ids is not None:
                pack_ids[batch_index, edge_index] = prior_index.get(
                    (edge.surface, edge.reading), 0)
            if gold is not None:
                gold[batch_index, edge_index] = bool(gold_edge_masks[batch_index][edge_index])
    unique_ids = None
    unique_mask = None
    if reading_inverse is not None:
        unique_ids = torch.full(
            (len(unique_rows), reading_length), pad_id, dtype=torch.long)
        unique_mask = torch.zeros_like(unique_ids, dtype=torch.bool)
        for unique_index, ids in enumerate(unique_rows):
            if ids:
                unique_ids[unique_index, :len(ids)] = torch.tensor(
                    ids, dtype=torch.long)
                unique_mask[unique_index, :len(ids)] = True
    offset_tensor = (torch.tensor(target_offsets, dtype=torch.long)
                     if target_offsets is not None else None)
    length_tensor = (torch.tensor(target_lengths, dtype=torch.long)
                     if target_lengths is not None else None)
    return EdgeBatch(
        input_ids=input_ids, attention_mask=attention,
        edge_start=starts, edge_end=ends,
        edge_reading_ids=reading_ids, edge_reading_mask=reading_mask,
        edge_mask=edge_mask, edge_locked=edge_locked,
        edge_source_ids=source_ids, edge_prior=prior,
        edge_pack_ids=pack_ids,
        texts=normalized, edges=all_edges, source_to_id=source_to_id,
        gold_edge_mask=gold, unique_reading_ids=unique_ids,
        unique_reading_mask=unique_mask,
        edge_reading_inverse=reading_inverse,
        target_offset=offset_tensor,
        target_length=length_tensor)


def pack_batch(texts: Sequence[str], provider: CandidateProvider,
               vocab: dict[str, int], *, max_reading_length: int = 64,
               source_to_id: Mapping[str, int] | None = None,
               dynamic_reading_length: bool = False,
               deduplicate_readings: bool = False) -> EdgeBatch:
    """Build padded tensors while retaining the original edge graph."""
    normalized = [normalize_surface(text) for text in texts]
    all_edges = [provider.build(text) for text in normalized]
    return pack_edges(normalized, all_edges, vocab,
                      max_reading_length=max_reading_length,
                      source_to_id=source_to_id,
                      dynamic_reading_length=dynamic_reading_length,
                      deduplicate_readings=deduplicate_readings)
