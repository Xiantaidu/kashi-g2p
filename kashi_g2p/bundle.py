"""Reader for the ``lexicon.txz`` resource bundle.

Lets the training-side provider read the same single-file bundle the
converter uses.
"""

from __future__ import annotations

import lzma
from pathlib import Path

MAGIC = "##kashi-g2p v1"
SECTION = "##section "
END = "##end"


def is_bundle(path: str | Path) -> bool:
    """True when ``path`` starts with the xz container magic."""
    try:
        with open(path, "rb") as handle:
            return handle.read(6) == b"\xfd7zXZ\x00"
    except OSError:
        return False


def read_bundle(path: str | Path):
    """Return ``(pack, prior, components, vocab)`` from a lexicon bundle.

    ``pack`` maps surface -> tuple of ``(reading, source, prior, confidence)``.
    """
    path = Path(path)
    with lzma.open(path, "rb") as handle:
        raw = handle.read()
    lines = raw.decode("utf-8").splitlines()
    if not lines or lines[0] != MAGIC:
        raise ValueError(f"not a kashi-g2p lexicon bundle: {path}")

    pack: dict[str, tuple] = {}
    prior: dict[tuple[str, str], int] = {}
    components: dict[str, list[int]] = {}
    vocab: list[str] = []
    section = None
    last_surface = None
    for line in lines[1:]:
        if line.startswith("##"):
            if line == END:
                continue
            if line.startswith(SECTION):
                section = line[len(SECTION):]
                last_surface = None
                continue
            raise ValueError(f"bad bundle header: {line!r}")
        if section == "pack":
            surface, reading, source, bonus, weight = line.split("\t")
            entry = (reading, source, float(bonus), float(weight))
            if surface == last_surface:
                pack[surface] += (entry,)
            else:
                pack[surface] = (entry,)
                last_surface = surface
        elif section == "prior":
            surface, reading, row = line.split("\t")
            prior[(surface, reading)] = int(row)
        elif section == "components":
            char, _, ids = line.partition("\t")
            components[char] = [int(value) for value in ids.split(",")] if ids else []
        elif section == "vocab":
            vocab.append(line)
    return pack, prior, components, vocab
