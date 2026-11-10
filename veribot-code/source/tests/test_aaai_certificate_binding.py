"""Reject a changed certificate or a stale computation-code manifest."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/scripts"))

from render_aaai_veribot_evidence_cards import card
from verifiable_bot.aaai_provenance import verify_code_manifest

CERT = ROOT / "research/certificates/aaai_veribot_fox8_profile_class_flip.json"


class CertificateBindingTests(unittest.TestCase):
    def test_modified_certificate_is_not_rendered(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / CERT.name
            check = copy.with_name(copy.stem + "_verification.json")
            shutil.copyfile(CERT, copy)
            shutil.copyfile(CERT.with_name(CERT.stem + "_verification.json"), check)
            payload = json.loads(copy.read_text(encoding="utf-8"))
            payload["selection_rule"] += " altered"
            copy.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "independent readback"):
                card(copy)

    def test_changed_code_hash_is_rejected(self):
        recorded = json.loads(CERT.read_text(encoding="utf-8"))["code_sha256"]
        changed = dict(recorded)
        first = next(iter(changed))
        changed[first] = "0" * 64
        with self.assertRaisesRegex(ValueError, "computation code differs"):
            verify_code_manifest(ROOT, changed)


if __name__ == "__main__":
    unittest.main()
