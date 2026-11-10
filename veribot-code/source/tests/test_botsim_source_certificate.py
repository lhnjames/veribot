import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from source.verifiable_bot.botsim_native_facts import build
from source.verifiable_bot.botsim_source_audit import audit_all
from source.verifiable_bot.botsim_source_certificate import certificate, sha256


class BotSimSourceCertificateTests(unittest.TestCase):
    def test_recomputes_reply_fact_from_raw_parent(self):
        users = [{"user_id": "a", "name": "Alice"},
                 {"user_id": "b", "name": "Bob"}]
        activity = {
            "a": {"posts": [{"submission_id": "p1", "author_name": "Alice"}],
                  "comment_1": [{"comment_id": "c1", "comment_author_name": "Alice",
                                 "parent_id": "t3_p2"}], "comment_2": []},
            "b": {"posts": [{"submission_id": "p2", "author_name": "Bob"}],
                  "comment_1": [], "comment_2": []},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            users_path = root / "Users.csv"
            with users_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("user_id", "name"))
                writer.writeheader()
                writer.writerows(users)
            activity_path = root / "activity.json"
            activity_path.write_text(json.dumps(activity), encoding="utf-8")
            cache = root / "facts"
            build(users_path, activity_path, cache)
            result = certificate(users_path, activity_path, cache, "a",
                                 "native_reply_out_unique_log1p")
            self.assertEqual(result["raw_count"], 1)
            self.assertEqual(result["source_reply_records"], 1)
            self.assertEqual(result["raw_matching_reply_records"], 1)
            self.assertEqual(result["status"], "verified")
            full = audit_all(users_path, activity_path, cache)
            self.assertEqual(full["verified_fact_cells"], 10)
            self.assertEqual(full["raw_cross_account_reply_comments"], 1)
            traces_path = cache / "reply_sources.jsonl.gz"
            with gzip.open(traces_path, "wt", encoding="utf-8") as stream:
                stream.write("")
            manifest_path = cache / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["reply_sources_sha256"] = sha256(traces_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
                certificate(users_path, activity_path, cache, "a",
                            "native_reply_out_unique_log1p")
            with self.assertRaisesRegex(ValueError, "trace differs from raw comments"):
                audit_all(users_path, activity_path, cache)


if __name__ == "__main__":
    unittest.main()
