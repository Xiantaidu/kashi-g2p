from __future__ import annotations

import re
import unicodedata

KANA_RE = re.compile(r"^[\u3041-\u3096\u30a1-\u30fa\u30fc\u309d\u309e\u30fb]+$")
KANJI_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\U00020000-\U0002fa1f々〆ヶ]"
)
RUBY_RE = re.compile(r"<ruby(?:\s[^>]*)?>(.*?)</ruby>", re.S | re.I)
RT_RE = re.compile(r"<rt(?:\s[^>]*)?>(.*?)</rt>", re.S | re.I)
RB_RE = re.compile(r"<rb(?:\s[^>]*)?>(.*?)</rb>", re.S | re.I)
RP_RE = re.compile(r"<rp(?:\s[^>]*)?>(.*?)</rp>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")


def hira(text: str) -> str:
    """NFKC-normalize text and convert katakana to hiragana."""
    text = unicodedata.normalize("NFKC", text)
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            ch = chr(code - 0x60)
        out.append(ch)
    return "".join(out)


def normalize_reading(text: str, *, remove_spaces: bool = True) -> str:
    text = hira(text).replace("|", "")
    if remove_spaces:
        text = "".join(text.split())
    return text


def normalize_surface(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("|", "")


def compact_surface(text: str) -> str:
    return "".join(normalize_surface(text).split())


def is_kanji(ch: str) -> bool:
    return bool(ch and KANJI_RE.fullmatch(ch))


def contains_kanji(text: str) -> bool:
    """Return whether *text* contains a Japanese Han character/mark."""
    return any(is_kanji(ch) for ch in text)


def contains_latin_letter(text: str) -> bool:
    """Return whether *text* contains a Latin-script letter."""
    for ch in text:
        name = unicodedata.name(ch, "")
        if "LATIN" in name and "LETTER" in name:
            return True
    return False


def is_trainable_kanji_text(text: str, *, has_explicit_reading: bool = False) -> bool:
    """Select useful Japanese G2P training/evaluation text.

    Ordinary examples must contain at least one Han character.  Explicitly
    annotated Latin spellings are retained because their kana ruby is genuine
    supervision; unannotated Latin, pure kana, numbers and punctuation are
    excluded.
    """
    return contains_kanji(text) or (
        has_explicit_reading and contains_latin_letter(text)
    )


def is_kana_reading(text: str) -> bool:
    return bool(text and KANA_RE.fullmatch(text))


def parse_ruby_output(output: str) -> list[tuple[str, str | None]]:
    """Return (surface, reading) pieces, dropping rt/rp markup.

    The corpus uses ruby fragments, occasionally with rb/rp wrappers. Regex is
    sufficient after the dataset's validated HTML fragment format and keeps
    character offsets deterministic.
    """
    output = unicodedata.normalize("NFKC", output)
    pieces: list[tuple[str, str | None]] = []
    cursor = 0
    for match in RUBY_RE.finditer(output):
        if match.start() > cursor:
            plain = TAG_RE.sub("", output[cursor:match.start()])
            if plain:
                pieces.append((plain, None))
        body = match.group(1)
        rt = RT_RE.search(body)
        reading = normalize_reading(rt.group(1)) if rt else None
        if rt:
            surface = body[: rt.start()] + body[rt.end() :]
        else:
            surface = body
        surface = RB_RE.sub(r"\1", surface)
        surface = RP_RE.sub("", surface)
        surface = TAG_RE.sub("", surface)
        pieces.append((surface, reading))
        cursor = match.end()
    if cursor < len(output):
        plain = TAG_RE.sub("", output[cursor:])
        if plain:
            pieces.append((plain, None))
    return [(normalize_surface(s), r) for s, r in pieces if s or r]


def ruby_source(pieces: list[tuple[str, str | None]]) -> str:
    return "".join(surface for surface, _ in pieces)


def ruby_reading(pieces: list[tuple[str, str | None]]) -> str:
    return "".join(reading if reading is not None else surface for surface, reading in pieces)
