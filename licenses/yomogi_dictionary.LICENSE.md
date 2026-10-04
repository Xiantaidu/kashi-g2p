---
title: Yomogi v1
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
short_description: 漢字かな交じり文から読みを推定するツール
---

# Yomogi v1

## ライセンス

本リポジトリに含まれるファイルは MIT License で提供されます。

model/dictionary.tsv および model/surface_vocab.tsv は次のデータソースを利用して作成されています。

* [pyopenjtalk(-plus)](https://github.com/tsukumijima/pyopenjtalk-plus) ([LICENSE](https://github.com/tsukumijima/pyopenjtalk-plus/blob/master/LICENSE.md))
  * [NAIST-jdic](https://github.com/kazuma-t/naist-jdic) ([LICENSE](https://github.com/kazuma-t/naist-jdic/blob/main/COPYING))
  * [unidic-csj](https://clrd.ninjal.ac.jp/unidic/download.html#unidic_csj) ([LICENSE](https://clrd.ninjal.ac.jp/unidic/copying/BSD))
* [AzooKeyKanaKanjiConverter](https://github.com/azooKey/AzooKeyKanaKanjiConverter) ([LICENSE](https://github.com/azooKey/AzooKeyKanaKanjiConverter/blob/main/LICENSE))
  * [azooKey_dictionary_storage](https://github.com/azooKey/azooKey_dictionary_storage) ([LICENSE](https://github.com/azooKey/azooKey_dictionary_storage/blob/main/LICENSE))
    * [NEologd](https://github.com/neologd/mecab-unidic-neologd) ([LICENSE](https://github.com/neologd/mecab-ipadic-neologd/blob/master/COPYING))
    * [SudachiDict](https://github.com/WorksApplications/SudachiDict) ([LICENSE](https://github.com/WorksApplications/SudachiDict/blob/develop/LICENSE-2.0.txt))
      * [UniDic](https://clrd.ninjal.ac.jp/unidic/) ([LICENSE](https://clrd.ninjal.ac.jp/unidic/copying/BSD))
      * NEologd
* [Mozc UT Dictionaries](https://github.com/utuhiro78?tab=repositories&q=mozcdic-ut)
  * [ウィキペディア日本語版](https://ja.wikipedia.org/) ([LICENSE](https://ja.wikipedia.org/wiki/Wikipedia:%E3%82%A6%E3%82%A3%E3%82%AD%E3%83%9A%E3%83%87%E3%82%A3%E3%82%A2%E3%82%92%E4%BA%8C%E6%AC%A1%E5%88%A9%E7%94%A8%E3%81%99%E3%82%8B))
  * NEologd
  * [personal_names](https://github.com/utuhiro78/mozcdic-ut-personal-names) ([LICENSE](https://github.com/utuhiro78/mozcdic-ut-personal-names/blob/main/LICENSE))
  * SudachiDict
