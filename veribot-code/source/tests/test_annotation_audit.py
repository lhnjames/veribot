import unittest

from source.verifiable_bot.annotation_audit import validate_packet
from source.verifiable_bot.semantic_annotation import CONCEPTS


class AnnotationAuditTests(unittest.TestCase):
    def test_rejects_account_identity_in_blind_metadata(self):
        concept = "templated_text_in_shown_sample"
        row = {"blind_id": "x", "dataset": "twibot20", "concept": concept,
               "question": CONCEPTS[concept]["question"],
               "items": [{"ref": "T1", "text": "hello", "truncated": False}]}
        source = [{"blind_id": "x", "dataset": "twibot20", "account_id": "u1",
                   "source_records": [{"ref": "T1", "locator": "raw[0]"}]}]
        self.assertEqual(validate_packet([row], source)["tasks"], 1)
        row["account_id"] = "u1"
        with self.assertRaisesRegex(ValueError, "unexpected metadata"):
            validate_packet([row], source)


if __name__ == "__main__":
    unittest.main()
