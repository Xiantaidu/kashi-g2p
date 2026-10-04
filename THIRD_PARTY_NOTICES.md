# Third-party notices and licence obligations

This file lists the third-party data embedded in the compiled lexicon
artifact (`lexicon.txz`) and the licence conditions that apply to its
distribution. The project code itself is licensed separately (see LICENSE).

## EDRDG — JMdict / KANJIDIC2 (CC BY-SA 4.0)

The compiled lexicon contains data derived from the JMdict and KANJIDIC2
dictionary files.

These files are the property of the Electronic Dictionary Research and
Development Group, and are used in conformance with the Group's licence
(https://www.edrdg.org/edrdg/licence.html).

- Files: JMdict — https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project
- Files: KANJIDIC2 — https://www.edrdg.org/wiki/index.php/KANJIDIC_Project
- Licence: Creative Commons Attribution-ShareAlike 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/) together with the
  EDRDG General Dictionary Licence Statement reproduced at the URL above.
- Copyright (c) James William Breen and The Electronic Dictionary Research
  and Development Group.

Because the lexicon is a derivative work of these files, the compiled
`lexicon.txz` must be distributed under CC BY-SA 4.0 (or a compatible
licence) together with this acknowledgement. Any product/website that
displays readings from the lexicon must repeat this acknowledgement in its
documentation or an "About / Sources" screen.

## UniDic 3.1.0 (notcore lexicon)

Portions of the compiled lexicon are derived from the UniDic 3.1.0
"notcore" word list. UniDic is licensed under a BSD/GPL/LGPL triple
licence; this distribution uses the BSD option.

```
Copyright (c) 2011-2021, The UniDic Consortium
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

 * Neither the name of the UniDic Consortium nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Yomogi dictionary v1 (MIT)

Compound-word readings are derived from the Yomogi v1 dictionary (MIT).
Its licence file, including the full disclosure of *its* upstream data
sources, is shipped at `licenses/yomogi_dictionary.LICENSE.md`.

## JmdictFurigana (MIT)

Ruby alignments were derived from JmdictFurigana (MIT, Doublevil).
Licence file: `licenses/JmdictFurigana.LICENSE`.

## Deliberately excluded from this distribution

- **CHISE IDS / cjkvi-ids component data** (GPLv2, derived from the CHISE
  project): the `components` section of the lexicon bundle is omitted from
  the distribution. The runtime handles its absence by zero-filling.
- **`place_names_dictionary.tsv`**: not distributed.
- Raw upstream dictionary files (`JMdict_e.gz`, `kanjidic2.xml.gz`,
  UniDic archives, Yomogi TSV, etc.) are build-time inputs and are not
  part of this distribution.
