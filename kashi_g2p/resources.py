"""Resource fingerprints.

The lattice is part of the model (see EXPERIMENTS.md): the node encoder and
the candidate space are both built from UniDic/JMdict/KANJIDIC at runtime, so
a silent resource swap degrades CER multiplicatively, not linearly.
Checkpoints therefore pin the exact resources they were trained with, and
load-time code warns when the runtime resources do not match.

Hashing strategy: small files get a full SHA-256.  The UniDic directory is
~775 MB of binaries, so its fingerprint hashes the small config files by
content and folds every file's name and size -- a version swap always changes
either a config file or the binary set.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

_SMALL_FILE_LIMIT = 8 * 1024 * 1024


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    return "sha256:" + _sha256_stream(path)


def dir_fingerprint(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for entry in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = entry.relative_to(path).as_posix()
        size = entry.stat().st_size
        head = _sha256_stream(entry) if size <= _SMALL_FILE_LIMIT else f"size:{size}"
        digest.update(f"{rel}\x00{size}\x00{head}\n".encode("utf-8"))
    return "manifest-sha256:" + digest.hexdigest()


def compute_resource_fingerprints(*, jmdict: str | None = None,
                                  unidic: str | None = None,
                                  kanjidic: str | None = None,
                                  alnum_table: str | None = None) -> dict[str, str]:
    """Fingerprint the lattice resources; missing paths are simply omitted."""
    fingerprints: dict[str, str] = {}
    for name, raw, is_dir in (("jmdict", jmdict, False), ("unidic", unidic, True),
                              ("kanjidic", kanjidic, False),
                              ("alnum_table", alnum_table, False)):
        if not raw:
            continue
        value = dir_fingerprint(raw) if is_dir else file_fingerprint(raw)
        if value is not None:
            fingerprints[name] = value
    return fingerprints


def check_resource_fingerprints(expected: dict[str, str] | None,
                                actual: dict[str, str]) -> list[str]:
    """Return one warning per pinned resource that is missing or has changed."""
    if not expected:
        return []
    warnings = []
    for name, pinned in sorted(expected.items()):
        current = actual.get(name)
        if current is None:
            warnings.append(f"resource '{name}' is missing but the checkpoint pinned it")
        elif current != pinned:
            warnings.append(
                f"resource '{name}' differs from the one pinned in the checkpoint "
                "(dictionary swap: lattice behaviour may change)")
    return warnings
