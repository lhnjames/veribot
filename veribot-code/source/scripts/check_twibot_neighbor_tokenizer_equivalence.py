"""Check the fast active-token matcher against the frozen sklearn analyzer."""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import ijson
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from build_twibot20_neighbor_unigram_incidence import strip_accents


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    with np.load(root / "output/v3_twibot20_sparse_text_evidence/sparse_l1_c1_model.npz",
                 allow_pickle=False) as selected:
        vocabulary = selected["vocabulary"].astype(str)
        active = vocabulary[np.flatnonzero(selected["coefficient"][44:] != 0)]
    pattern = re.compile(r"(?u)\b(?:" + "|".join(map(re.escape, active)) + r")\b")
    analyzer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", vocabulary=list(active)).build_analyzer()
    count = 0
    with zipfile.ZipFile("C:/Users/Lenovo/Downloads/Twibot-20.zip") as zipped:
        for part in ("train", "dev", "test", "support"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for row in ijson.items(stream, "item"):
                    doc = " ".join(str(item or "") for item in row.get("tweet") or [])
                    fast = set(pattern.findall(strip_accents(doc.lower())))
                    standard = set(analyzer(doc)) & set(active)
                    if fast != standard:
                        raise ValueError(f"tokenizer differs at {part} account {row['ID']}: "
                                         f"{sorted(fast ^ standard)[:10]}")
                    count += 1
                    if count % 100 == 0:
                        break
    print("equivalent_source_rows", count)


if __name__ == "__main__":
    main()
