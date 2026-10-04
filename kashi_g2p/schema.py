"""Language/task schemas used by the language-independent G2P core.

The encoder and dynamic candidate scorer do not require Japanese-specific
labels.  A schema names the optional task heads and their label spaces so the
same model can later be used for Japanese kana, Mandarin pinyin, or another
grapheme-to-phoneme task.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class G2PSchema:
    """Serializable label contract for one G2P language/task."""

    name: str
    reading_types: tuple[str, ...]
    variation_types: tuple[str, ...]
    bies_types: tuple[str, ...] = ("B", "I", "E", "S")
    pos_types: tuple[str, ...] = ("その他",)
    c_classes: int = 0
    mora_classes: int = 0
    unit_name: str = "reading"
    copy_label: str = "COPY"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("reading_types", "variation_types", "bies_types", "pos_types"):
            value[key] = list(value[key])
        return value

    @classmethod
    def from_value(cls, value: "G2PSchema | str | Mapping[str, Any] | None") -> "G2PSchema":
        if value is None:
            return cls.japanese()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.lower().replace("-", "_")
            if key in {"ja", "japanese", "kana", "default"}:
                return cls.japanese()
            if key in {"zh", "zh_cn", "zh_pinyin", "chinese", "pinyin", "mandarin"}:
                return cls.chinese_pinyin()
            raise ValueError(f"unknown G2P schema: {value}")
        if isinstance(value, Mapping):
            data = dict(value)
            if "c_classes" not in data:
                data["c_classes"] = 1 if str(data.get("name", "")).lower() in {"zh", "zh_pinyin", "pinyin"} else 4
            if "mora_classes" not in data and str(data.get("name", "")).lower() in {"ja", "japanese", "kana"}:
                data["mora_classes"] = 9
            # JSON/YAML naturally decode tuples as lists.
            for key in ("reading_types", "variation_types", "bies_types", "pos_types"):
                if key in data:
                    data[key] = tuple(str(x) for x in data[key])
            return cls(**data)
        raise TypeError(f"schema must be a name, mapping, or G2PSchema; got {type(value)!r}")

    @classmethod
    def japanese(cls) -> "G2PSchema":
        # The final three historical D labels had no support in the lyric
        # validation set.  Keep only transformations that are trained and
        # assembled by the current Japanese rules.
        return cls(
            name="ja",
            reading_types=("音", "訓", "熟字訓", "特殊", "COPY"),
            variation_types=("無", "連濁", "半濁", "促音化", "長音化"),
            pos_types=("名", "動", "形", "副", "助", "助動", "接", "連体",
                       "感", "接頭", "接尾", "その他", "記号", "数", "代", "連"),
            c_classes=4,
            mora_classes=9,
            unit_name="kana",
        )

    @classmethod
    def chinese_pinyin(cls) -> "G2PSchema":
        """A conservative starting contract for Mandarin pinyin.

        A Chinese lexicon supplies candidate pinyin strings.  There is no
        Japanese okurigana/phonological-change target, so those heads are
        disabled and only the candidate scorer plus optional BIES/POS tasks
        remain.  Tone can be represented in the candidate strings (e.g.
        ``shi4``) or added as a language-specific auxiliary head later.
        """
        return cls(
            name="zh_pinyin",
            reading_types=("拼音", "COPY"),
            variation_types=("无",),
            pos_types=("名", "动", "形", "副", "介", "连", "助", "数", "量", "代", "其他"),
            c_classes=1,
            mora_classes=0,
            unit_name="pinyin",
        )


def schema_from_config(value: Any = None) -> G2PSchema:
    """Short alias used by training/collation entry points."""
    return G2PSchema.from_value(value)
