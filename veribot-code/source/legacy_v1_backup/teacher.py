from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI

from .prompting import SYSTEM_PROMPT, teacher_discovery_prompt
from .schema import AccountRecord, EvidencePrediction
from .verifier import EvidenceVerifier


class CachedTeacher:
    def __init__(self, cache_dir: Path, model: str):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.client = OpenAI(base_url=os.environ.get("TEACHER_BASE_URL") or None,
                             api_key=os.environ.get("TEACHER_API_KEY"))

    def complete(self, system: str, user: str) -> str:
        key = hashlib.sha256((self.model + system + user).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists(): return json.loads(path.read_text())["content"]
        result = self.client.chat.completions.create(model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0, response_format={"type": "json_object"})
        content = result.choices[0].message.content
        path.write_text(json.dumps({"model": self.model, "content": content}, ensure_ascii=False))
        return content

    def semantic_judge(self, claim: str, anchors: list[str]) -> bool:
        prompt = "Return JSON {\"supported\":true|false}. Claim:\n" + claim + "\nAnchors:\n" + "\n".join(anchors)
        raw = self.complete("Judge only whether the cited raw evidence entails the claim. Be conservative.", prompt)
        return bool(json.loads(raw).get("supported", False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "deepseek-chat"))
    args = ap.parse_args()
    teacher = CachedTeacher(args.cache, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as src, open(args.output, "a", encoding="utf-8") as dst:
        for line in src:
            record = AccountRecord.model_validate_json(line)
            raw = teacher.complete(SYSTEM_PROMPT, teacher_discovery_prompt(record))
            try: pred = EvidencePrediction.model_validate_json(raw)
            except Exception: continue
            verifier = EvidenceVerifier(semantic_judge=teacher.semantic_judge)
            result = verifier.verify(record, pred)
            target = pred.model_dump()
            teacher_label_matches = pred.label == record.label
            target["label"] = record.label
            target["evidence_status"] = "verified" if result.semantically_supported and teacher_label_matches else "insufficient"
            if target["evidence_status"] == "insufficient": target["anchors"] = []
            dst.write(json.dumps({"record": record.model_dump(), "target": target}, ensure_ascii=False) + "\n")
            dst.flush()


if __name__ == "__main__": main()
