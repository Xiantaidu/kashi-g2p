from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .kanjidic import Kanjidic
from .normalization import is_kanji, normalize_reading

READING_TYPES = ("音", "訓", "熟字訓", "特殊", "COPY")
# Core Japanese transformations with reliable support in the training data.
# The old eight-label space is kept as a named compatibility constant so old
# checkpoints/data can still be inspected, but new collators/models use this
# compact space by default.
VARIATIONS = ("無", "連濁", "半濁", "促音化", "長音化")
LEGACY_VARIATIONS = ("無", "連濁", "半濁", "促音化", "長音化", "撥音化", "拗音化", "その他")


@dataclass
class Alignment:
    a: list[str]
    b: list[str]
    c: list[int]
    d: list[str]
    mask: list[int]
    confidence: list[float]


_VOICED = dict(zip("かきくけこさしすせそたちつてとはひふへほ", "がぎぐげござじずぜぞだぢづでどばびぶべぼ"))
_SEMI = dict(zip("はひふへほ", "ぱぴぷぺぽ"))


def _initial_change(base: str, table: dict[str, str]) -> str:
    if not base:
        return base
    # Dakuten/handakuten applies to the first mora, including a digraph.
    first = base[0]
    return table.get(first, first) + base[1:]


def _variation(base: str, target: str) -> str | None:
    if base == target:
        return "無"
    if _initial_change(base, _VOICED) == target:
        return "連濁"
    if _initial_change(base, _SEMI) == target:
        return "半濁"
    if base and target == base[:-1] + "っ" and base[-1] in "かきくけこたちつてと":
        return "促音化"
    if target in {base + "う", base + "お"} or base in {target + "う", target + "お"}:
        return "長音化"
    if target == base.replace("ん", "っ", 1):
        return "撥音化"
    return None


def align_ruby(surface: str, reading: str, dictionary: Kanjidic,
               *, okurigana_hint: str = "") -> Alignment | None:
    """DP-align a ruby span to characters and decompose reading transformations."""
    reading = normalize_reading(reading)
    chars = list(surface)
    if not chars:
        return None
    candidates: list[list[tuple[str, str, int, str, str, int]]] = []
    hint = normalize_reading(okurigana_hint)
    for char_index, ch in enumerate(chars):
        if not is_kanji(ch):
            candidates.append([(ch, "COPY", 0, "無", ch, 0)])
            continue
        values = []
        for entry in dictionary.typed_candidates(ch):
            base = entry.text
            typ = entry.kind or dictionary.reading_type(ch, base)
            hint_score = 0
            if char_index == len(chars) - 1 and entry.okurigana and hint:
                common = 0
                for expected, observed in zip(entry.okurigana, hint):
                    if expected != observed:
                        break
                    common += 1
                if common:
                    # One matching kana is enough to distinguish 伝う from
                    # 伝える; longer exact prefixes receive a small bonus.
                    hint_score = 100 + common
            for trunc in range(4):
                shortened = base[:-trunc] if trunc else base
                if not shortened:
                    continue
                values.append((shortened, typ, trunc, "無", base, hint_score))
                # Variants are generated only for alignment.  The stored B is
                # always the unmodified KANJIDIC reading (and C/D explain the
                # generated surface form).
                for variant, variation in (
                    (_initial_change(shortened, _VOICED), "連濁"),
                    (_initial_change(shortened, _SEMI), "半濁"),
                    (shortened[:-1] + "っ" if shortened[-1:] in "かきくけこたちつてと" else shortened, "促音化"),
                    (shortened + "う", "長音化"),
                ):
                    if variant != shortened:
                        values.append((variant, typ, trunc, variation, base, hint_score))
        # Keep exact surface copy as a fallback for kana/punctuation only.
        candidates.append(values)

    @lru_cache(maxsize=None)
    def solve(i: int, j: int):
        if i == len(chars):
            return (0, []) if j == len(reading) else None
        best = None
        for text, typ, trunc, variation, base, hint_score in candidates[i]:
            if reading.startswith(text, j):
                rest = solve(i + 1, j + len(text))
                if rest is None:
                    continue
                score, tail = rest
                # Prefer shorter truncation and dictionary entries that consume
                # more reading; score is only a deterministic tie-breaker.
                candidate = (score + len(text) * 10 - trunc + hint_score,
                             [(base, typ, trunc, variation)] + tail)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        return best

    solved = solve(0, 0)
    if solved is None:
        # A ruby that cannot be decomposed into KANJIDIC per-character bases
        # is normally a jukujikun/special word reading (e.g. 彼方→かなた,
        # 荒磯→ありそ, 個々→ここ) or a rare single-character reading.  Keep
        # the complete reading on the first character as a span-template-like
        # target and mark the remaining surface characters COPY/unreliable.
        # The collator adds the explicit annotated span reading as a candidate
        # at the span start, keeping the label in the dynamic candidate set.
        # Explicit ruby is authoritative, including readings longer than the
        # per-character KANJIDIC limit (花魁道中《おいらんどうちゅう》).
        # KANJIDIC candidates are length-unbounded; this fallback preserves an
        # annotation that still cannot be decomposed character by character.
        if not reading:
            return None
        special_type = "特殊" if len(chars) == 1 or any(not is_kanji(ch) for ch in chars) else "熟字訓"
        return Alignment(
            [special_type] + ["COPY"] * (len(chars) - 1),
            [reading] + ["COPY"] * (len(chars) - 1),
            [0] * len(chars),
            ["無"] * len(chars),
            [1] + [0] * (len(chars) - 1),
            [0.9] + [1.0] * (len(chars) - 1),
        )
    _, parts = solved
    return Alignment(
        [p[1] for p in parts],
        [p[0] for p in parts],
        [p[2] for p in parts],
        [p[3] for p in parts],
        [1] * len(parts),
        [1.0] * len(parts),
    )
