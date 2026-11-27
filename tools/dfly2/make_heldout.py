#!/usr/bin/env python3
"""Frozen independent prompts; render through the existing target-template builder.

Use --freeze once before inspecting candidate measurements. Render from the
frozen JSON thereafter; never regenerate or tune tasks in response to scores.
This script does not load model weights or use a GPU.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SUITE = "heldout-parity-v1"
SEED = 263801
SYSTEM = "Answer accurately and concisely. Follow the requested output format."


def prompts():
    incident = []
    for i in range(90):
        minute = i * 2
        stage = "normal"
        detail = "Queue depth is 3; completed jobs 120; retry counter is 0."
        if 24 <= i < 42:
            stage = "degraded"
            detail = "Queue depth is 47; completed jobs 82; retry counter is 18."
        elif 42 <= i < 49:
            stage = "recovering"
            detail = "Queue depth is 15; completed jobs 110; retry counter is 4."
        incident.append(f"Minute {minute:03d}: status={stage}. {detail}")
    incident[24] += " A dependency timeout was first reported."
    incident[40] += " Operators lowered retry concurrency; no deployment was rolled back."
    incident[42] += " The dependency became healthy again."

    readings = [f"sensor=S{i:03d}; floor={i % 5}; sample_count={20 + i % 7}; missing={i % 4}; unit=C" for i in range(140)]
    settings = [f"ZONE_{i:03d}: enabled={'false' if i % 9 == 0 else 'true'}, retries={i % 4}, label=zone-{i:03d}" for i in range(110)]
    notes = [f"記録{i:03d}: 担当者は{'青木' if i % 2 == 0 else '森'}。状態は{'完了' if i % 5 else '保留'}。部品番号はP{i:03d}。" for i in range(72)]
    items = [
        ("prose_greedy_idempotent", "prose", 0,
         "In 120 to 150 words, explain to a new developer why retrying a timed-out request can accidentally perform an action twice. Use a document-upload example, distinguish a timeout from a confirmed failure, and describe how an idempotency key helps. State one thing an idempotency key does not solve."),
        ("prose_sampled_tradeoff", "prose", .7,
         "A small volunteer group is choosing between a shared spreadsheet and a custom application for assigning weekly shifts. They have 18 volunteers, two coordinators, no software budget, and frequent last-minute swaps. In 120 to 160 words, recommend an initial approach, explain one concrete drawback, and name a measurable condition that would justify changing it. Do not recommend products."),
        ("prose_greedy_incident", "prose", 0,
         "The following is a synthetic operations log. Write a 100-to-140-word incident summary giving the first degraded minute, the recovery minute, what operators changed, and what the log does not establish about causation. Do not invent a root cause.\n\n" + "\n".join(incident)),
        ("structured_sampled_catalog", "structured", .7,
         "Convert this catalog into JSON with keys items and unresolved. Each item needs sku, name, quantity, and color. Do not infer an absent color; use null. An SKU occurring twice must appear once, with quantities added only when the names and colors match. Put incompatible duplicates in unresolved. Records: A17 | cable | 3 | blue; B02 | clamp | 4 | absent; A17 | cable | 2 | blue; C91 | tray | 1 | gray; C91 | bin | 5 | gray. Output only JSON."),
        ("structured_greedy_schedule", "structured", 0,
         "Three room reservations use half-open intervals [start,end): R1 09:00-09:30; R2 09:30-10:00; R3 09:15-09:45; R4 10:00-10:20. Return JSON with exactly two keys: conflicting_pairs (pairs of reservation IDs, lexicographically sorted) and maximum_simultaneous (integer). Touching endpoints do not overlap. Output only JSON."),
        ("structured_sampled_readings", "structured", .7,
         "Use only this synthetic sensor inventory. Return a JSON array for S003, S028, S077, S119, and S139, in that order. Each object must contain sensor, floor, observed, missing, where observed equals sample_count minus missing. Output only the five objects in the array.\n\n" + "\n".join(readings)),
        ("code_greedy_runlength", "code", 0,
         "Write Python function runs(values) returning a list of (value,count) tuples for consecutive equal values. It must accept any iterable, consume it once, and return [] when empty. Include assert tests for [1,1,2,1], an empty list, and an iterator over ['x','x','y']. Output only code, at most 25 lines."),
        ("code_sampled_leftjoin", "code", .7,
         "SQLite tables are customers(id INTEGER PRIMARY KEY, name TEXT) and orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount_cents INTEGER). Write one query returning every customer's id and name, order_count, and total_cents, including customers with no orders. Sort by customer id. Explain in at most two sentences why COUNT(*) would be wrong for customers with no orders."),
        ("code_greedy_configparse", "code", 0,
         "Write Python function enabled_zones(text) for the exact configuration format below. Ignore blank lines. Return enabled zone IDs whose retries value is at least 2, sorted numerically by ID. Split keys explicitly rather than relying on field order; assume valid input. Output only the function, no expanded result list, at most 25 lines.\n\n" + "\n".join(settings)),
        ("multilingual_sampled_spanish", "multilingual", .7,
         "Responde en español, en 100 a 130 palabras. Una biblioteca quiere sustituir todos sus libros impresos por libros electrónicos para ahorrar espacio. Explica dos ventajas, dos limitaciones concretas y una alternativa gradual. No supongas que todos los lectores tienen conexión a internet o un dispositivo propio."),
        ("multilingual_greedy_french", "multilingual", 0,
         "Réponds uniquement en français. Un atelier annonce : les commandes reçues avant mardi à 12 h sont préparées mercredi ; les commandes reçues après cette limite sont préparées le mercredi suivant. La livraison prend ensuite deux jours ouvrés, sans jours fériés. Léa commande mardi à 11 h et Marc mardi à 14 h. En deux phrases, indique le jour de livraison de chacun et précise si c'est la semaine de commande ou la suivante."),
        ("multilingual_sampled_japanese", "multilingual", .7,
         "以下は架空の部品管理記録です。日本語で、保留件数、完了件数、保留の部品番号の最初の3件と最後の2件を報告してください。記録にない理由は推測しないでください。回答は200字以内にしてください。\n\n" + "\n".join(notes)),
    ]
    return [{"name": name, "category": category, "temperature": temp,
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
            for name, category, temp, prompt in items]


def frozen_document():
    return {"schema": "dfly2-heldout-tasks-v1", "suite": SUITE, "first_seed": SEED,
            "policy": "Frozen before candidate measurements; preserve all tasks and failures; max_tokens=256; top_p=1; top_k=20.",
            "tasks": prompts()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, help="create the immutable, pre-tokenization prompt fixture")
    p.add_argument("--frozen", type=Path, help="render this frozen prompt fixture")
    p.add_argument("--model", type=Path)
    p.add_argument("--public-repo", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.freeze:
        if any((args.frozen, args.model, args.public_repo, args.output)):
            p.error("--freeze is a separate operation")
        payload = (json.dumps(frozen_document(), ensure_ascii=False, indent=2) + "\n").encode()
        with args.freeze.open("xb") as f:
            f.write(payload)
        print(json.dumps({"tasks": 12, "sha256": hashlib.sha256(payload).hexdigest()}))
        return
    if not all((args.frozen, args.model, args.public_repo, args.output)):
        p.error("render requires --frozen --model --public-repo --output")
    raw = args.frozen.read_bytes()
    frozen = json.loads(raw)
    if frozen != frozen_document():
        raise ValueError("frozen tasks differ from the reviewed generator; do not tune them after measurements")
    tools = args.public_repo / "tools/dfly2"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("fixed_target_requests", tools / "make_requests.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    builder.SUITES[SUITE] = (lambda: [(t["name"], t["temperature"], t["messages"]) for t in frozen["tasks"]], SEED)
    requests, metadata = builder.chat_requests(args.model, tok, SUITE)
    metadata["frozen_tasks_sha256"] = hashlib.sha256(raw).hexdigest()
    metadata["categories"] = {t["name"]: t["category"] for t in frozen["tasks"]}
    payload = (json.dumps({"requests": requests, "metadata": metadata}, ensure_ascii=False, indent=2) + "\n").encode()
    with args.output.open("xb") as f:
        f.write(payload)
    print(json.dumps({"requests": len(requests), "sha256": hashlib.sha256(payload).hexdigest(),
                      "frozen_tasks_sha256": metadata["frozen_tasks_sha256"],
                      "prompt_lengths": [len(r["prompt"]) for r in requests],
                      "greedy": sum(r["temperature"] == 0 for r in requests),
                      "sampled": sum(r["temperature"] == .7 for r in requests)}))


if __name__ == "__main__":
    main()
