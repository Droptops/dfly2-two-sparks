#!/usr/bin/env python3
"""Fixed target-tokenized inputs for serial, DSpark, and DFly2 comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer


def chat_tasks():
    """Six distinct tasks, including an assistant-history turn and bounded outputs."""
    system = {"role": "system", "content": "Answer accurately and concisely. Follow the requested output format."}
    tasks = [
        ("chat_greedy_merge_intervals", 0, [
            {"role": "user", "content": "Write a Python function merge_intervals(intervals) that merges overlapping or touching closed integer intervals. Input may be unsorted; return a sorted list of tuples without mutating the input. Return [] for empty input. Include three assert tests, one for [(5,7),(1,3),(3,6)] and one for empty input. Output only code."},
        ]),
        ("chat_greedy_latest_sql", 0, [
            {"role": "user", "content": "In SQLite, events has columns event_id INTEGER PRIMARY KEY, device_id TEXT, occurred_at TEXT (ISO8601 UTC), and status TEXT. Write a query returning the latest event for each device, breaking timestamp ties by the largest event_id. Return device_id, occurred_at, and status, sorted by device_id. Then explain in two sentences why ordering only by timestamp is insufficient."},
        ]),
        ("chat_greedy_extract_json", 0, [
            {"role": "user", "content": "Extract this support note as a JSON object with exactly these keys: customer (string), order_id (string), damaged_items (array of strings), requested_action (string), tracking_number (string or null). Use null for absent information; do not invent facts. Note: 'Hi, I am Lena Ortiz. Order ZX-204 arrived today. The blue mug is chipped and the glass lid is cracked. Please replace both pieces. The shipping box looked fine.' Return only JSON."},
        ]),
        ("chat_sampled_binary_search", .7, [
            {"role": "user", "content": "Repair this Python binary search so it returns the index of the FIRST occurrence of target in a sorted list, or -1 if absent, in O(log n). Explain the bug briefly and include tests for duplicates, an empty list, and an absent value.\n\ndef find(a, target):\n    lo, hi = 0, len(a)\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if a[mid] == target:\n            return mid\n        if a[mid] < target:\n            lo = mid\n        else:\n            hi = mid\n    return -1"},
        ]),
        ("chat_sampled_multiturn_edit", .7, [
            {"role": "user", "content": "Write a Python function that normalizes a list of email addresses by stripping whitespace and lowercasing each address."},
            {"role": "assistant", "content": "def normalize_emails(addresses):\n    return [address.strip().lower() for address in addresses]"},
            {"role": "user", "content": "Update that function to drop blank entries and remove duplicates while keeping the order of the first occurrence. Do not change the function name or mutate the input list. Include an assert test using [' A@example.com ', '', 'a@EXAMPLE.com', 'b@example.com']. Output only code."},
        ]),
        ("chat_sampled_explain_cache", .7, [
            {"role": "user", "content": "A teammate proposes caching every successful HTTP GET response forever. In 130 to 170 words, explain two concrete correctness problems with this plan and propose a cache policy for a weather API that updates every ten minutes. Distinguish freshness from eviction and mention how conditional requests help. Do not cite sources or discuss implementation libraries."},
        ]),
    ]
    return [(name, temperature, [system.copy(), *messages]) for name, temperature, messages in tasks]


def chat_requests(model, tok):
    # Lazy import leaves the original suite's dependencies and execution path
    # intact. local_files_only prevents any checkpoint/network lookup.
    from transformers import AutoTokenizer

    template_path = model / "chat_template.jinja"
    template = template_path.read_text(encoding="utf-8")
    renderer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    requests = []
    for index, (name, temperature, messages) in enumerate(chat_tasks()):
        # Pass the actual file explicitly, rather than an embedded/config default.
        # Chat mode spends the bounded output allowance on the requested answer.
        text = renderer.apply_chat_template(messages, chat_template=template, tokenize=False,
                                            add_generation_prompt=True, thinking_mode="chat")
        # The template emits BOS and role markers. Never add another BOS or a
        # tokenizer postprocessor here; use the same target tokenizer.json as
        # the default suite. This also avoids version-specific HF token outputs.
        ids = tok.encode(text, add_special_tokens=False).ids
        if not ids or len(ids) + 256 + 16 > 8192:
            raise ValueError(f"{name} does not fit the validation pool of 8192 tokens")
        requests.append({"name": name, "prompt": ids, "max_tokens": 256,
                         "temperature": temperature, "top_p": 1, "top_k": 20,
                         "seed": 262611 + index})
    metadata = {"suite": "chat", "thinking_mode": "chat",
                "chat_template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
                "tokenizer_sha256": hashlib.sha256((model / "tokenizer.json").read_bytes()).hexdigest(),
                "tasks": [{"name": name, "messages": messages} for name, _, messages in chat_tasks()]}
    return requests, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--suite", choices=("default", "chat"), default="default",
                        help="default preserves the original six inputs; chat uses the target's actual chat template")
    args = parser.parse_args()
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    metadata = None
    if args.suite == "chat":
        requests, metadata = chat_requests(args.model, tok)
    else:
        code = "Write a Python function that returns the prime numbers less than n. Explain the algorithm."
        base = tok.encode("A notebook contains sequential observations of a simple counting task. ", add_special_tokens=False).ids
        requests = []
        for size in (1, 128, 512, 1024):
            ids = [0] + (base * ((size + len(base) - 1) // len(base)))[:size - 1]
            requests.append({"name": f"greedy_{size}", "prompt": ids, "max_tokens": 32,
                             "temperature": 0, "top_p": 1, "top_k": 20, "seed": 262611})
        ids = [0] + tok.encode(code, add_special_tokens=False).ids
        for seed in (262611, 262612):
            requests.append({"name": f"sampled_code_{seed}", "prompt": ids, "max_tokens": 64,
                             "temperature": .7, "top_p": 1, "top_k": 20, "seed": seed})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"requests": requests}
    if metadata is not None:
        payload["metadata"] = metadata
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"requests": len(requests), "maximum_output_tokens": sum(r["max_tokens"] for r in requests),
                      "prompt_lengths": [len(r["prompt"]) for r in requests]}))


if __name__ == "__main__":
    main()
