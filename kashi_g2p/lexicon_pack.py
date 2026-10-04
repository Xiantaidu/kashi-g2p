"""Reader for the LEX1 lexicon pack produced by scripts/build_lexicon_pack.py.

Also provides :func:`build_prior_index`, which maps (surface, reading) pairs
to stable integer ids used by the model's per-entry learned prior table.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from collections import OrderedDict

MAGIC = b"LEX1"
MAX_READINGS = 6


def _kata_to_hira(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def _clean_reading(reading: str) -> str | None:
    value = hira(reading)
    if not value:
        return None
    if all("\u3041" <= ch <= "\u3096" or ch == "ー" for ch in value):
        return value
    return None


def _has_kanji(surface: str) -> bool:
    return any(is_kanji(ch) for ch in surface)




def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


class _VarintReader:
    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def read(self) -> int:
        result = 0
        shift = 0
        while True:
            byte = self.data[self.pos]
            self.pos += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7


def _front_coded(keys: list[str]) -> bytes:
    """Encode a sorted key list: varint(shared prefix) + varint(len) + suffix."""
    out = bytearray()
    previous = ""
    for key in keys:
        shared = 0
        limit = min(len(previous), len(key))
        while shared < limit and previous[shared] == key[shared]:
            shared += 1
        suffix = key[shared:].encode("utf-8")
        out += _varint(shared)
        out += _varint(len(suffix))
        out += suffix
        previous = key
    return bytes(out)


def _decode_front_coded(data: bytes):
    """Yield (start_offset, key) for every entry in a front-coded blob."""
    reader = _VarintReader(data)
    key = ""
    offset = 0
    while reader.pos < len(data):
        start_offset = offset
        shared = reader.read()
        length = reader.read()
        suffix = data[reader.pos:reader.pos + length].decode("utf-8")
        reader.pos += length
        key = key[:shared] + suffix
        yield start_offset, key
        offset = reader.pos



def build_prior_index(pack_path: str | Path) -> "OrderedDict[tuple[str, str], int]":
    """Map every (surface, reading) pair in the pack to 1-based ids.

    Id 0 is reserved for edges that do not exist in the pack (COPY, rules,
    neural proposals), which the model embeds as its own learned row.
    """
    pack = LexiconPack(pack_path)
    index: "OrderedDict[tuple[str, str], int]" = OrderedDict()
    next_id = 1
    for index_i in range(pack.surface_count):
        surface, readings = pack._entry(index_i)
        for reading in readings:
            index[(surface, reading)] = next_id
            next_id += 1
    return index


class LexiconPack:
    """Reader for the LEX1 format: exact lookup + prefix range enumeration."""

    def __init__(self, path: str | Path):
        data = Path(path).read_bytes()
        if data[:4] != MAGIC:
            raise ValueError("not a LEX1 lexicon pack")
        if hashlib.sha256(data[4:-32]).hexdigest() != data[-32:].hex():
            raise ValueError("lexicon pack checksum mismatch")
        body = data[4:-32]
        reader = _VarintReader(body)
        self.reading_count = reader.read()
        reading_blob_len = reader.read()
        reading_blob = body[reader.pos:reader.pos + reading_blob_len]
        reader.pos += reading_blob_len
        self._readings = [key for _offset, key in _decode_front_coded(reading_blob)]
        self.surface_count = reader.read()
        surface_blob_len = reader.read()
        self._surface_blob = body[reader.pos:reader.pos + surface_blob_len]
        reader.pos += surface_blob_len
        offset_count = reader.read()
        self._surface_offsets = [reader.read() for _ in range(offset_count)]
        list_len = reader.read()
        self._lists = body[reader.pos:reader.pos + list_len]
        reader.pos += list_len
        self._sparse = []
        for _ in range(reader.read()):
            offset = reader.read()
            length = reader.read()
            key = body[reader.pos:reader.pos + length].decode("utf-8")
            reader.pos += length
            self._sparse.append((offset, key))

    @staticmethod
    def _skip_record(data: bytes, pos: int) -> int:
        reader = _VarintReader(data, pos)
        reader.read()  # shared prefix length
        length = reader.read()
        return reader.pos + length

    def _seek_key(self, index: int) -> str:
        """Decode the surface at ``index`` by walking from its sparse anchor."""
        block = index // 64
        anchor_offset, anchor_key = self._sparse[block]
        anchor_index = block * 64
        if anchor_index == index:
            return anchor_key
        pos = self._skip_record(self._surface_blob, anchor_offset)
        key = anchor_key
        for _ in range(anchor_index + 1, index + 1):
            reader = _VarintReader(self._surface_blob, pos)
            shared = reader.read()
            length = reader.read()
            suffix = self._surface_blob[reader.pos:reader.pos + length].decode("utf-8")
            pos = reader.pos + length
            key = key[:shared] + suffix
        return key

    def _entry(self, index: int) -> tuple[str, list[str]]:
        key = self._seek_key(index)
        start = self._surface_offsets[index]
        end = (self._surface_offsets[index + 1]
               if index + 1 < self.surface_count else len(self._lists))
        reader = _VarintReader(self._lists, start)
        readings = []
        previous = 0
        # Offsets are BYTE positions in the varint stream; walk until the
        # next surface's byte offset instead of counting entries.
        while reader.pos < end:
            previous += reader.read()
            readings.append(self._readings[previous])
        return key, readings

    def lookup(self, surface: str) -> list[str]:
        """Exact-match readings for one surface (binary search + local walk)."""
        if not self._sparse:
            return []
        low, high, block = 0, len(self._sparse) - 1, 0
        while low < high:
            mid = (low + high + 1) // 2
            if self._sparse[mid][1] <= surface:
                block = mid
                low = mid
            else:
                high = mid - 1
        anchor_index = block * 64
        anchor_offset, anchor_key = self._sparse[block]
        pos = self._skip_record(self._surface_blob, anchor_offset)
        key = anchor_key
        for current in range(anchor_index, min(self.surface_count, anchor_index + 64)):
            if current > anchor_index:
                reader = _VarintReader(self._surface_blob, pos)
                shared = reader.read()
                length = reader.read()
                suffix = self._surface_blob[reader.pos:reader.pos + length].decode("utf-8")
                pos = reader.pos + length
                key = key[:shared] + suffix
            if key == surface:
                return self._entry(current)[1]
            if key > surface:
                return []
        return []

    def iter_range(self, prefix: str):
        """Yield (surface, readings) whose surface starts with ``prefix``."""
        if not self._sparse:
            return
        bound = prefix + "￿"
        low, high, block = 0, len(self._sparse) - 1, 0
        while low < high:
            mid = (low + high) // 2
            if self._sparse[mid][1] < prefix:
                low = mid + 1
            else:
                high = mid
        block = low
        anchor_index = block * 64
        anchor_offset, anchor_key = self._sparse[block]
        pos = self._skip_record(self._surface_blob, anchor_offset)
        key = anchor_key
        for current in range(anchor_index, self.surface_count):
            if current > anchor_index:
                reader = _VarintReader(self._surface_blob, pos)
                shared = reader.read()
                length = reader.read()
                suffix = self._surface_blob[reader.pos:reader.pos + length].decode("utf-8")
                pos = reader.pos + length
                key = key[:shared] + suffix
            if key >= bound:
                return
            if key.startswith(prefix):
                yield key, self._entry(current)[1]


