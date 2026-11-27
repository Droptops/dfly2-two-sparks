# Frozen v5 GPU evaluation

**Performance parity with DSpark was not reached.** The adaptive candidate improved aggregate chat decode by 3.3% over legacy DFly2, but regressed by 7.7% on the longer fixture. It was slower than legacy on 10 of the 12 individual chat/long requests. It remains opt-in; no serving default was switched.

All 39 final GPU runs preserved their serial-reference outputs. On held-out tasks, adaptive DFly2 reached 47.32 estimated decode tok/s versus serial 38.02 and DSpark 71.67. Legacy DFly2 was not measured on that held-out fixture, so this is not evidence of a held-out gain over legacy. Full [Tencent-reference numerical parity](REFERENCE.md) also remains unestablished.

The [build receipt](../evidence/experimental-build-v5.json) and [frozen selection](../evidence/experimental-selection-v5.json) identify this separate v5 evaluation. Complete [chat](../evidence/experimental/benchmark-chat/analysis.json), [longer-input](../evidence/experimental/benchmark-long/analysis.json), and [held-out](../evidence/experimental/heldout/analysis.json) evidence is published. [Historical results](RESULTS.md) remain unchanged.

The v4 sweep is retained but excluded because initial drafting bypassed the selected depth. This report uses the corrected v5 sweep and frozen v5 build.

The predeclared one-repeat v5 chat sweep selected `adaptive` at 48.72 estimated decode tok/s. It selected the highest reported rate among depths 1–5, adaptive scheduling and reference normalization; equal rates retain the earlier sweep variant. Serial and depth zero were controls. That exploratory number is selection evidence, not the final performance estimate.

The controls were frozen before evaluation. Held-out requests were excluded from selection according to the frozen protocol. Chat and long compare five modes in three rotated repeats; held-out compares serial, the frozen candidate and ordinary DSpark in three rotated repeats.

## Frozen candidate controls

```json
{
  "world_size": 2,
  "streams": 1,
  "parallel": 1,
  "pool_tokens": 8192,
  "cuda_graphs": false,
  "profiling": true,
  "drafting": true,
  "fixed_draft_depth": null,
  "experimental": {
    "device_feedback": true,
    "rdma_best": true,
    "detailed_profile": false,
    "adaptive": true,
    "reference_norms": false,
    "copies": false
  }
}
```

These are two-node, one-stream, eager runs with an 8192-token pool. Detailed timing events and capture are disabled; ordinary native summary counters remain enabled.

## Final rates

Rates are median (minimum–maximum), tok/s, over three runs. Decode excludes prefill and the first generated token per request; end-to-end includes prefill. Model loading, launcher preflight and container startup are excluded from these native request timings. Times are rounded by the native runner, and displayed rates are rounded to 0.01 tok/s. Observed ranges are not confidence intervals. The fixtures contain different tasks, so rate differences cannot be attributed to context length alone.

| Fixture | Mode | Estimated decode tok/s | End-to-end tok/s |
| --- | --- | ---: | ---: |
| chat | Serial | 38.60 (38.24–38.76) | 35.41 (35.18–35.54) |
| chat | Legacy DFly2 | 45.45 (45.10–45.47) | 39.64 (38.98–39.86) |
| chat | Adaptive DFly2 | 46.97 (46.17–47.48) | 42.06 (41.34–42.62) |
| chat | DSpark fixed five, copies off | 67.97 (67.30–68.56) | 58.33 (57.81–58.44) |
| chat | Ordinary DSpark | 74.57 (73.48–74.63) | 63.26 (61.91–63.30) |
| long | Serial | 38.41 (37.64–38.45) | 25.75 (24.21–25.87) |
| long | Legacy DFly2 | 55.78 (55.49–55.97) | 32.22 (31.70–32.33) |
| long | Adaptive DFly2 | 51.46 (50.06–52.86) | 31.17 (30.33–31.72) |
| long | DSpark fixed five, copies off | 79.17 (77.90–79.39) | 38.98 (38.12–39.20) |
| long | Ordinary DSpark | 81.61 (81.56–81.67) | 40.13 (39.98–40.13) |
| heldout | Serial | 38.02 (38.01–38.10) | 30.33 (30.31–30.39) |
| heldout | Adaptive DFly2 | 47.32 (44.59–47.68) | 36.02 (33.94–36.07) |
| heldout | Ordinary DSpark | 71.67 (71.42–71.99) | 48.28 (48.25–48.72) |

## Candidate rate changes

Percentage changes compare the reported median rates; positive is faster. Values inherit the underlying 0.01 tok/s rounding.

| Fixture | Baseline | Decode change | End-to-end change |
| --- | --- | ---: | ---: |
| chat | Legacy DFly2 | +3.34% | +6.10% |
| chat | Serial | +21.68% | +18.78% |
| chat | DSpark fixed five, copies off | -30.90% | -27.89% |
| chat | Ordinary DSpark | -37.01% | -33.51% |
| long | Legacy DFly2 | -7.74% | -3.26% |
| long | Serial | +33.98% | +21.05% |
| long | DSpark fixed five, copies off | -35.00% | -20.04% |
| long | Ordinary DSpark | -36.94% | -22.33% |
| heldout | Serial | +24.46% | +18.76% |
| heldout | Ordinary DSpark | -33.98% | -25.39% |

## Per-request candidate changes versus legacy DFly2

Every measured chat and long request is retained, including positive, negative and unchanged results. Held-out has no legacy comparison because that mode was not measured. Negative rate change means slower; positive total-time change means a longer response.

The chat aggregate improvement is not uniform: some requests regress despite its higher aggregate median rate.

chat: candidate median decode was slower on 5/6 requests; median end-to-end was slower on 5/6. long: candidate median decode was slower on 5/6 requests; median end-to-end was slower on 5/6.

| Fixture | Request | Decode rate change | End-to-end rate change | Total-time change |
| --- | --- | ---: | ---: | ---: |
| chat | `chat_greedy_merge_intervals` | -21.90% | -5.75% | +6.10% |
| chat | `chat_greedy_latest_sql` | -2.38% | -1.06% | +1.06% |
| chat | `chat_greedy_extract_json` | -25.88% | -15.34% | +18.10% |
| chat | `chat_sampled_binary_search` | -2.10% | -1.24% | +1.25% |
| chat | `chat_sampled_multiturn_edit` | -12.14% | -7.78% | +8.43% |
| chat | `chat_sampled_explain_cache` | +29.40% | +28.29% | -22.07% |
| long | `chat_greedy_long_log_triage` | -27.03% | -9.23% | +10.17% |
| long | `chat_greedy_long_module_extend` | -1.87% | -0.73% | +0.73% |
| long | `chat_greedy_long_notes_summary` | +3.43% | +2.24% | -2.17% |
| long | `chat_sampled_long_history_followup` | -9.68% | -5.65% | +5.97% |
| long | `chat_sampled_long_module_review` | -3.52% | -1.71% | +1.73% |
| long | `chat_sampled_long_spec_answer` | -2.34% | -0.98% | +0.99% |

## Per-request candidate slowdowns versus serial

The table retains every request where either candidate median rate is lower than serial. Negative rate change means slower; a positive total-time change means longer response time. Small differences can be run-to-run noise.

| Fixture | Request | Decode rate change | End-to-end rate change | Total-time change |
| --- | --- | ---: | ---: | ---: |
| chat | `chat_sampled_explain_cache` | -6.95% | -6.81% | +7.29% |
| heldout | `prose_greedy_idempotent` | -4.38% | -4.93% | +5.17% |
| heldout | `prose_sampled_tradeoff` | -11.22% | -10.56% | +11.82% |

## Retained prefill variation

No slow repeat was discarded. On the longer fixture, serial repeat 1 recorded 17.39 seconds of summed prefill versus 14.93 and 15.11 seconds; its spec-answer prefill was 5.62 seconds versus 4.34 and 4.32. Log-triage prefill also varied: legacy repeat 2 took 2.10 seconds versus 1.68/1.68, fixed-five DSpark repeat 3 took 2.10 versus 1.36/1.32, and candidate repeat 3 took 1.52 versus 1.22/1.17. These observations do not identify their cause.

The candidate's longer-fixture summed-prefill median was lower than legacy (14.90 versus 15.43 seconds), but total request time was worse (37.50 versus 36.28 seconds). Its end-to-end regression remains despite that prefill difference.

## Output and lifecycle checks

All 39 runs matched every serial-reference output token and finish reason for their fixture. All 78 rank exits were zero, with no recorded OOM or runtime error, and owned containers were removed.

| Fixture | Requests per run | Output tokens per run | Runs |
| --- | ---: | ---: | ---: |
| chat | 6 | 959 | 15 |
| long | 6 | 1169 | 15 |
| heldout | 12 | 1449 | 9 |

## Held-out limits

The held-out fixture contains 12 requests, with 84–3018 input tokens. Its bytes match the request hash in the candidate freeze, and its runs began after chat and long evaluation finished. The protocol forbids retuning after selection; recorded chronology cannot independently prove that no human inspected or used held-out content.

One exploratory chat repeat can overfit. These small fixed fixtures and three repeats do not establish the best policy on other tasks, statistical significance, 64K context behavior, concurrent serving, CUDA graph performance, production quality, or Tencent implementation parity. Exact target outputs are correctness checks on these requests, not proof that the drafter's internal values or proposals match the original reference.

## What remains

The device-feedback implementation reduced host readbacks in the separate diagnostic profile, and normalization rounding improved part of the reference discrepancy. Neither result establishes full serving-speed or numerical parity. The observed-cost scheduler needs further investigation before any default change: aggregate chat gains conceal substantial request regressions, and the longer-fixture result is worse than legacy.

The [detailed profiles](EXPERIMENTAL.md#reading-detailed-profiles) identify correction attention and repeated target-head work as substantial remaining costs. Those instrumented intervals are diagnostic, not additional throughput results. Future changes need a new frozen evaluation; these held-out results must not be reused to tune a candidate and then presented as independent confirmation.

## Evidence identities

Candidate freeze time: `2026-11-27T01:16:06.174077+00:00`.

| Export | Analysis SHA256 | Matrix SHA256 |
| --- | --- | --- |
| `benchmark-chat` | `2ca68e4fad04af83a1728edbe1a23dfefe3af6192aa1fc8309d5b87aeee30b6c` | `65a49fae52e03a292c6fe267150798d068f2f036936765ad05d86b72384c027e` |
| `benchmark-long` | `7e4f14d556c75534681d0a2e0cbd0e7a2ed2d684352651fbd1b8f3ca1a8697ce` | `65a49fae52e03a292c6fe267150798d068f2f036936765ad05d86b72384c027e` |
| `heldout` | `bb60dd68fd6e4a42b42a6d97be5f05cffd6ce2795eecfab5dabcedf9315a6b0b` | `7304a3bfe809038f9fa699cbdc37958473c6d3f47b27284bfd8d35ef57131ec7` |
