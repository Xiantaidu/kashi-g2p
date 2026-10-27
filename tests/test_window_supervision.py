import json

from kashi_g2p.collator import WindowedJsonlDataset


def test_window_drops_supervision_for_cut_special_span(tmp_path):
    path = tmp_path / "cut_span.jsonl"
    text = "あ" * 63 + "日々" + "い" * 10
    row = {
        "text": text,
        "A": ["COPY"] * len(text),
        "B": ["COPY"] * 63 + ["ひび", "COPY"] + ["COPY"] * 10,
        "C": [0] * len(text),
        "D": ["無"] * len(text),
        "loss_mask": [0] * 63 + [1, 0] + [0] * 10,
        "special_spans": [{"start": 63, "end": 65, "reading": "ひび",
                           "type": "熟字訓"}],
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    cut_window = next(window for window in
                      WindowedJsonlDataset(str(path), 64, 16)
                      if "日" in window["text"])
    local = cut_window["text"].index("日")
    assert cut_window["special_spans"] == []
    assert cut_window["B"][local] == "COPY"
    assert cut_window["loss_mask"][local] == 0
