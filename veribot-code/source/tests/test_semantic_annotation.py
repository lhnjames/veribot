import unittest

from source.verifiable_bot.semantic_annotation import (
    blind_id, select_by_hash, trim,
)


class SemanticAnnotationTests(unittest.TestCase):
    def test_selection_ignores_class_label_and_model_score(self):
        rows = [{"uid": f"u{i}", "label": i % 2, "model_score": i / 10}
                for i in range(20)]
        first = [row["uid"] for row in select_by_hash(rows, "uid", 5)]
        for row in rows:
            row["label"] = 1 - row["label"]
            row["model_score"] = 1 - row["model_score"]
        second = [row["uid"] for row in select_by_hash(rows, "uid", 5)]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 5)

    def test_blind_id_and_truncation(self):
        identifier = blind_id("twibot20", "templated_text_in_shown_sample", "secret-id")
        self.assertNotIn("secret-id", identifier)
        self.assertEqual(trim("abcdef", 3), ("abc", True))


if __name__ == "__main__":
    unittest.main()
