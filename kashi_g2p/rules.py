from __future__ import annotations

import re

SMALL = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")


def kana_to_phonemes(text: str) -> list[str]:
    table = {
        "し": ["sh", "i"], "ち": ["ch", "i"], "つ": ["ts", "u"],
        "じ": ["j", "i"], "ふ": ["f", "u"], "ん": ["N"], "っ": ["cl"],
        "きゃ": ["ky", "a"], "きゅ": ["ky", "u"], "きょ": ["ky", "o"],
        "しゃ": ["sh", "a"], "しゅ": ["sh", "u"], "しょ": ["sh", "o"],
        "ちゃ": ["ch", "a"], "ちゅ": ["ch", "u"], "ちょ": ["ch", "o"],
        "にゃ": ["ny", "a"], "にゅ": ["ny", "u"], "にょ": ["ny", "o"],
        "ひゃ": ["hy", "a"], "ひゅ": ["hy", "u"], "ひょ": ["hy", "o"],
        "みゃ": ["my", "a"], "みゅ": ["my", "u"], "みょ": ["my", "o"],
        "りゃ": ["ry", "a"], "りゅ": ["ry", "u"], "りょ": ["ry", "o"],
        "ぎゃ": ["gy", "a"], "ぎゅ": ["gy", "u"], "ぎょ": ["gy", "o"],
        "じゃ": ["j", "a"], "じゅ": ["j", "u"], "じょ": ["j", "o"],
        "びゃ": ["by", "a"], "びゅ": ["by", "u"], "びょ": ["by", "o"],
        "ぴゃ": ["py", "a"], "ぴゅ": ["py", "u"], "ぴょ": ["py", "o"],
    }
    result: list[str] = []
    i = 0
    while i < len(text):
        if i + 1 < len(text) and text[i:i + 2] in table:
            result.extend(table[text[i:i + 2]]); i += 2; continue
        ch = text[i]
        if ch == "ー":
            result.append("-")
        elif ch in table:
            result.extend(table[ch])
        elif "ぁ" <= ch <= "ゖ":
            # Conservative mora nucleus mapping; unknown kana remain visible.
            row = {"あ":"a","い":"i","う":"u","え":"e","お":"o"}
            result.append(row.get(ch, ch))
        else:
            result.append(ch)
        i += 1
    return result


def mora_groups(text: str) -> list[str]:
    out=[]
    for ch in text:
        if out and ch in SMALL:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def assembled_mora_count(text: str) -> int:
    return len(mora_groups(text))


def apply_variation(base: str, variation: str) -> str:
    if variation == "無": return base
    voiced = dict(zip("かきくけこさしすせそたちつてとはひふへほ", "がぎぐげござじずぜぞだぢづでどばびぶべぼ"))
    semi = dict(zip("はひふへほ", "ぱぴぷぺぽ"))
    if variation == "連濁" and base: return voiced.get(base[0], base[0]) + base[1:]
    if variation == "半濁" and base: return semi.get(base[0], base[0]) + base[1:]
    if variation == "促音化" and base and base[-1] in "かきくけこたちつてと": return base[:-1] + "っ"
    if variation == "長音化": return base + "う"
    if variation == "撥音化": return base.replace("ん", "っ", 1)
    return base
