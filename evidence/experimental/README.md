# Experimental evidence

These bundles are separate from the historical release receipts. Run
`python3 scripts/check.py` from the repository root to verify the staged
bundles' recorded consistency. Offline checks cannot authenticate GPU execution
or reproduce private captured tensors.

- `reference-v3-original` and `reference-v4-rounding` retain the released-weight
  numerical failures and proposal mismatches described in
  [the reference report](../../docs/REFERENCE.md).
- `profile-v4-long` contains six diagnostic runs with additional timing events.
  Those timings are excluded from throughput rankings.
- `sweep-v4-chat` is **disqualified from policy and speed comparisons**. The
  first draft after each prompt bypassed the requested depth selector. The
  requested depth-zero case performed six real draft passes, proposed 30
  tokens and accepted 22. Fixed depths one through four and adaptive mode also
  began with an unselected five-token draft. All final target outputs matched,
  which did not detect this control failure. The bundle retains every run and
  records `initial_draft_depth_bypass`; its verifier suppresses aggregate
  throughput comparisons. The original private attempt records are retained.
- `sweep-v5-chat` contains all nine corrected exploratory runs on the final v5
  public image. Fixed depth zero passes the no-drafting gate; all target output
  token arrays and finish reasons match. One chat repeat selected adaptive at
  48.72 estimated decode tokens/s under the declared rule. This is candidate
  selection evidence, not a final repeated or held-out performance result.
  [The frozen selection receipt](../experimental-selection-v5.json) binds the
  sweep files, effective controls, build identity and held-out fixture hashes.
  [The selection methodology](../../docs/EXPERIMENTAL.md#frozen-v5-candidate-selection)
  describes the separate 39-run evaluation and staged/complete verification.

The initial-depth correction is tested at request admission as well as during
later rounds. The evidence verifier separately requires zero passes and zero
drafted/accepted tokens for an undisqualified fixed-depth-zero run. Corrected
measurements must use a fresh native executable and new run directories.

The final `benchmark-chat`, `benchmark-long` and `heldout` bundles retain all
**39 successful runs**, including slow repeats and every request-level result.
The [results report](../../docs/EXPERIMENTAL_RESULTS.md) records the small chat
aggregate gain, longer-fixture regression, and remaining DSpark gap. The public
selection verifier now passes with `--require-complete`; frozen controls and
identities match throughout. These exact-output checks do not revise the
numerical failures in the separate reference bundles.
