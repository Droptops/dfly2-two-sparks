import json
from pathlib import Path
import re
import unittest

import long_tasks
import sanitize_evidence as sanitizer
import verify_evidence as historical_verifier
import verify_suite_evidence as suite_verifier

EVIDENCE = Path(__file__).resolve().parents[2] / "evidence"
FIXTURE = EVIDENCE / "fixtures/long-requests.json"
TOP_K = 512  # the target's sparse selection size at the DFly2 anchor layers


class LongTaskTests(unittest.TestCase):
    def test_text_is_deterministic_and_matches_the_reviewed_naming(self):
        tasks = long_tasks.tasks()
        self.assertEqual(tasks, long_tasks.tasks())
        self.assertEqual([temperature for _, temperature, _ in tasks], [0, 0, 0, .7, .7, .7])
        for name, _, messages in tasks:
            self.assertRegex(name, r"\Achat_(greedy|sampled)_long_[a-z_]+\Z")
            self.assertEqual(messages[0], long_tasks.SYSTEM)
            self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(long_tasks.Lcg(7).next(), long_tasks.Lcg(7).next())
        self.assertNotEqual(long_tasks.service_log(5, 1), long_tasks.service_log(5, 2))

    def test_generated_documents_contain_what_their_questions_ask_about(self):
        tasks = {name: messages for name, _, messages in long_tasks.tasks()}
        log = tasks["chat_greedy_long_log_triage"][1]["content"]
        failures = re.findall(r"req=(R\d+) \S+ status=5\d\d", log)
        self.assertGreaterEqual(len(failures), 3)
        self.assertEqual(len(set(re.findall(r"req=(R\d+)", log))), len(re.findall(r"req=(R\d+)", log)))
        review = tasks["chat_sampled_long_module_review"][1]["content"]
        self.assertEqual(review.count("def percentile("), 1)
        names = re.findall(r"^def (\w+)", review, re.M)
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("rank = int(len(ordered) * q / 100)", review)  # the planted off-by-one
        spec = tasks["chat_sampled_long_spec_answer"][1]["content"]
        self.assertEqual(spec.count("Section 3. Transfers and rate limits."), 1)
        self.assertLess(spec.index("Standard clients 120 requests a minute"), len(spec) // 10)
        history = tasks["chat_sampled_long_history_followup"]
        self.assertIn("multiply by 3 each attempt", history[4]["content"])
        self.assertEqual([m["role"] for m in history[1:]], ["user", "assistant"] * 9 + ["user"])

    def test_published_fixture_is_this_text_and_every_prompt_exceeds_the_selection_size(self):
        document = json.loads(FIXTURE.read_text(encoding="utf-8"))
        sanitizer.validate_requests(document)
        self.assertEqual(document["metadata"]["suite"], "long")
        self.assertEqual(document["metadata"]["tasks"],
                         [{"name": name, "messages": messages} for name, _, messages in long_tasks.tasks()])
        chat = historical_verifier.load(EVIDENCE / "chat-requests.json")["metadata"]
        for key in ("chat_template_sha256", "tokenizer_sha256", "thinking_mode"):
            self.assertEqual(document["metadata"][key], chat[key])
        requests = document["requests"]
        self.assertEqual([r["name"] for r in requests], [name for name, _, _ in long_tasks.tasks()])
        self.assertEqual([r["seed"] for r in requests], list(range(262711, 262717)))
        self.assertEqual([len(r["prompt"]) for r in requests], [1210, 2678, 3863, 1979, 4464, 6892])
        for request in requests:
            self.assertGreater(len(request["prompt"]), 2 * TOP_K)
            self.assertLessEqual(len(request["prompt"]) + request["max_tokens"] + 16, 8192)
            self.assertTrue(all(type(t) is int and 0 <= t < 129280 for t in request["prompt"]))
            self.assertEqual((request["max_tokens"], request["top_p"], request["top_k"]), (256, 1, 20))
        self.assertEqual(suite_verifier.reviewed_fixture_hash("long"), historical_verifier.sha256(FIXTURE))
        self.assertNotIn(b"\r", FIXTURE.read_bytes())


if __name__ == "__main__":
    unittest.main()
