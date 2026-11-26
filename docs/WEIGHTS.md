# DFly2 TP2 weights

`pack_weights.py` prepares the released `AngelSlim/DS-V4.1-Flash-DFly2-B6`
checkpoint for `dfly_weights.zig`. It uses the Python standard library and does
not execute model code. Both ranks keep all 128 routed experts, with each
expert's intermediate channels split in half. Attention heads and output groups
are split between ranks. The correction MLP is also split at its intermediate
dimension. Fusion and normalization tensors remain replicated.

```sh
python3 tools/dfly2/pack_weights.py /path/to/DS-V4.1-Flash-DFly2-B6 /path/to/DFly2-TP2
python3 -m unittest discover -s tools/dfly2 -p test_pack_weights.py -v
tf-dsv41-dfly-weights /path/to/DFly2-TP2 0 /path/to/target-model
tf-dsv41-dfly-weights /path/to/DFly2-TP2 1 /path/to/target-model
```

Run the two upload checks separately on the intended hosts. They allocate only
the draft weights, compare every GPU byte against the packed file, and release
the allocations. They do not start a server or load the main DeepSeek model.

Packing preserves every value's original bytes. Expert codes remain low-nibble
first E2M1 and scales remain E8M0, one exponent per 32 input channels. Both are
labeled `U8` in the packed files so TensorFold can read their raw storage without
adding new core dtypes. No BF16 expansion or requantization occurs. The source
schema has 2,392 tensors; stacking experts produces 106 tensors per rank and
4,529,862,920 tensor bytes per rank. This is weight storage, not a peak runtime
memory measurement.

The converter refuses existing destinations and stages the complete pack before
renaming it into place. `dfly2-pack.json` records SHA-256 digests of every source
shard and rank file. The native loader checks the selected rank's entire file
digest, every tensor dtype and shape, and the manifest format. Both ranks compare
the same manifest identity at startup. The target embedding and output weights
are borrowed from the existing target and are absent from this pack.

Configuration checks cover the B6 architecture and the target's geometry,
compressed-cache source and RoPE. They do not establish quality or useful draft
acceptance when the drafter is paired with the 1017 EXL3 target. That requires
the integration's forward and end-to-end validation.
