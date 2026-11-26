# Build and prepare the public runtime

This is an experimental DFly2 integration, not a complete two-Spark provisioning system. The steps below build from public sources and prepare the native runtime assets. They do not configure the fabric, stop other services, or load a model. Use the separate cluster configuration and validation launcher after preparation.

## Prerequisites and scope

Use ARM64 Linux on a GB10 system, Docker, Git and Python 3.10 or newer. This public repository can be checked out on Windows. Preparing the upstream sources requires a Linux filesystem, including under WSL, because upstream tracks golden-test filenames containing `?`, quotes and colons. Native builds are documented for ARM64 Linux. Docker must support NVIDIA GPU access for later inference; compilation, source tests and asset generation below need no GPU.

Both systems need a working compatible NVIDIA driver, NVIDIA Container Toolkit, ConnectX-7/RoCE connectivity, `/dev/infiniband`, and enough local disk and free unified memory. Driver 580.178.04 was used in the original experiment; this is an observation, not a claim that every driver/platform combination has been qualified. The public recipe identifies NCCL 2.30.7; the locked NVIDIA image supplies CUDA 13.3, cuBLAS, NCCL, libibverbs and Python dependencies. Do not install a second CUDA or NCCL stack over it.

Network access is required for GitHub, the NVIDIA container registry, a Zig download mirror or ziglang.org, and Hugging Face. Registry authentication or model access approval may be required by those providers. Review the licenses of the NVIDIA container, TensorFold, the upstream recipe, the public kernel kit, the target checkpoint and the drafter. This repository does not redistribute model weights or grant access to them. Do not place access tokens in build arguments, image layers or cluster JSON.

Budget approximately 196 GiB for the target, 203 GB for the two Engram shards, 106 GB per target rank cache, the container/build workspace, and the separate DFly2 checkpoint and TP2 pack. Keep Engram and each rank cache on that host's local disk. The target and kit may be copied to both systems or exposed through a correctly mounted read-only shared filesystem; NFS is not required by the native engine. Initial inference validation requires at least 112 GiB available memory on each host. Those disk and memory figures are different quantities.

## Source and image build

Run from this repository's root on Linux:

```bash
python3 scripts/prepare_source.py
python3 scripts/prepare_source.py --check
python3 scripts/build_image.py --jobs 4
```

Preparation requires a fresh `_build` directory. It fetches the exact revisions in `upstream.lock.json`, verifies every recipe patch and this repository's native patch by SHA256, applies the ordered series, and verifies the final Git tree IDs. It leaves failed preparation directories for inspection. Use a new `--destination` if retrying without removing a failed directory. The same directory is passed to `build_image.py --source-dir PATH`.

The serving source is TensorFold `fbbc97132cd004237d9887b8d3185896efaedc5c` plus the complete public recipe patch series at `c7f2b85100f0a7b099c92a4e21825e30d93c9c51`, then `patches/0001-dfly2-native.patch`. A separate inputs-only checkout at `59402086381a42c74d53053443224eec2790b4c5` supplies the rank-cache builder and standalone asset scripts. Its two recipe patches are also applied and checked. It does not replace the serving engine.

The Dockerfile pins the public `nvcr.io/nvidia/pytorch:26.07-py3` image by digest, and Zig 0.17.0 by archive SHA256. It compiles SM121 native kernels, the lane runner, the full native server, and the DFly weight checker. The image includes `/opt/zig/zig`, `/opt/tensorfold/bin`, `/opt/tensorfold/native/bin`, `/opt/tensorfold/tools/inputs`, and a source/build receipt under `/opt/tensorfold/share/build`. A source-tree hash is reproducible; compilation and Docker layer timestamps need not produce the same image ID across separate builds.

The final build layer adds an isolated `/opt/dfly2-tools` Python environment that inherits the pinned image's Torch and other packages. Transformers 5.14.1, tokenizers 0.22.2 and the small Typer dependency set are installed from exact wheel URLs and SHA256 hashes in `build/tool-requirements.txt`, with dependency resolution disabled. The base's tokenizers 0.23.1 is newer than this Transformers release allows; the tool environment shadows it without changing the base installation. `python3` on the image's PATH uses this environment. The build checks scoped package requirements and the actual AutoTokenizer import. This also requires access to `files.pythonhosted.org`; no CUDA or Torch package is installed by this step.

Build once and transfer that image to the other host:

```bash
docker save dfly2-two-sparks:experimental | ssh YOUR_WORKER docker load
```

Replace `YOUR_WORKER` with your existing SSH alias. After configuring both nodes, run the read-only preflight in [RUN.md](RUN.md). It requires matching image-content fingerprints over architecture, OS, variant, the complete image configuration, and ordered RootFS layers. Docker's containerd and overlay2 stores can report different image IDs after transferring the same image, so the launcher compares this content identity. Identical tags alone are insufficient; use the same transferred build on both hosts.

## Host tests and extracting binaries

The preparation helpers have CPU-only integrity/refusal tests:

```bash
python3 -m unittest discover -s scripts -p test_build_tools.py -v
```

Run the host suite without GPU access. The temporary filesystem is important: upstream direct-I/O fault tests intentionally inject an unaligned write that Docker's default overlay filesystem rejects before the expected test path.

```bash
mkdir -p _build/tensorfold/.zig-cache
docker run --rm --network none --cpus 4 --memory 6g \
  --tmpfs /tmp:rw,exec,size=4g,mode=1777 \
  --tmpfs /workspace/.zig-cache:rw,exec,size=4g,mode=1777 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
  -v "$PWD/_build/tensorfold:/workspace:ro" -w /workspace \
  --entrypoint /opt/zig/zig dfly2-two-sparks:experimental \
  build test -j4 --cache-dir /tmp/zig-local-cache
```

Use the exit status and Zig test result to judge success; these fault-injection tests intentionally print some failed subprocess commands. The source checkout is read-only, both compiler caches are in `/tmp`, and tests that explicitly use the working directory's `.zig-cache/tmp` get the separate writable tmpfs mount.

The portable launcher uses explicitly staged binary paths below `model_root`. The example layout uses `/srv/models`; substitute an absolute directory you own everywhere if needed. Extract the matching binaries on **each** node:

```bash
MODEL_ROOT=/srv/models
mkdir -p "$MODEL_ROOT/dfly2-experiment/bin"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$MODEL_ROOT/dfly2-experiment/bin:/out" \
  --entrypoint /bin/sh dfly2-two-sparks:experimental -c \
  'cp /opt/tensorfold/bin/tf-dsv41-lanes /opt/tensorfold/bin/tf-dsv41-dfly-weights /out/'
sha256sum "$MODEL_ROOT/dfly2-experiment/bin/"*
```

No inference process should be using a destination binary while it is replaced. Runtime assets and model data are not included in the image.

## Model downloads

Install the Hugging Face CLI in your own Python environment and authenticate through its supported login flow if required. These commands download substantial data. They do not start the model.

```bash
MODEL_ROOT=/srv/models
hf download Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw \
  --revision 45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b \
  --local-dir "$MODEL_ROOT/target"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --include config.json --include model.safetensors.index.json \
  --include model-00047-of-00048.safetensors \
  --include model-00048-of-00048.safetensors \
  --local-dir "$MODEL_ROOT/engram"
hf download AngelSlim/DS-V4.1-Flash-DFly2-B6 \
  --revision d08e55a6897b1974307ea9771038838de5986e66 \
  --local-dir "$MODEL_ROOT/DFly2-B6"
```

The two Engram shard sizes are 101535150936 and 101537926640 bytes. Their published SHA256 values are respectively `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` and `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd`. Verify those before preparing a new installation. Retain Hugging Face revision metadata and content checks for the target and drafter as well; an existing directory with the expected name is not proof of its contents.

Use the checked target's own tokenizer and chat template when generating validation requests. The pinned public recipe also includes its serving template at `_build/recipe/model/chat_template.jinja`; using a different template changes the inputs and must be recorded. The published smoke/chat measurements must not be treated as interchangeable with differently tokenized requests.

## Kernel kit, RoPE and token map

Create a fresh kit checkout under the model root:

```bash
git init "$MODEL_ROOT/kernel-kit"
git -C "$MODEL_ROOT/kernel-kit" remote add origin \
  https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10.git
git -C "$MODEL_ROOT/kernel-kit" fetch --depth 1 origin \
  abe07c7d509a0199cc7fec15263be2aa52ac652f
git -C "$MODEL_ROOT/kernel-kit" checkout --detach FETCH_HEAD
git -C "$MODEL_ROOT/kernel-kit" rev-parse HEAD
mkdir -p "$MODEL_ROOT/assets"
```

The native kit path is `kernel-kit/kit`, not the repository root. It supplies the existing AOT cubins, EXL3 kernels and Engram constants. Generate its missing RoPE tables, the 1017 routing bias and token map using the **inputs** scripts built into the image:

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$MODEL_ROOT/target:/model:ro" \
  -v "$MODEL_ROOT/kernel-kit/kit:/kit" \
  -v "$MODEL_ROOT/assets:/assets" \
  --entrypoint /bin/sh dfly2-two-sparks:experimental -ec '
    python3 /opt/tensorfold/tools/inputs/make_rope.py /model/config.json /kit
    python3 /opt/tensorfold/tools/inputs/make_bias_vl.py /model /kit/vision/gate_bias_vl.safetensors
    python3 /opt/tensorfold/tools/inputs/make_token_map.py /model /assets/dsv41_token_map.json
  '
python3 scripts/verify_kit.py "$MODEL_ROOT/kernel-kit/kit" \
  --allow-1017-bias --receipt "$MODEL_ROOT/assets/kit-receipt.json"
```

`verify_kit.py` pins the manifest itself and checks every file listed by it. The only explicit exception is the generated `vision/gate_bias_vl.safetensors`, following the public recipe's 1017-specific rule; its actual hash is recorded. This exception does not prove which checkpoint generated that file. The token-map script checks the compressed vocabulary size against the target config. Also compare the resulting token-map SHA256 to `kit/engram.json`'s `token_map_sha256`; a difference requires investigation and recording rather than silently asserting reference equivalence.

All these operations are CPU-only. The pinned inputs RoPE script uses blockwise Torch generation; do not substitute the serving branch's similarly named script, which depends on the separate Python engine. Fetching the public binary kit avoids rebuilding AOT recordings on a GPU.

## Per-node rank cache and drafter pack

On rank 0, with the target available locally or through a mounted share:

```bash
mkdir -p "$MODEL_ROOT/rank-cache"
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$MODEL_ROOT/target:/model:ro" \
  -v "$MODEL_ROOT/rank-cache:/rank-cache" \
  --entrypoint tf-dsv41-rank-cache dfly2-two-sparks:experimental \
  /model /rank-cache 0 2
```

Run the same command on rank 1 with final arguments `1 2`, using that node's local `rank-cache` destination. Each output is approximately 106 GB. The filenames contain the cache-index hash; this hash does **not** identify every weight byte, so do not reuse a rank cache with modified weights merely because its filename matches. Keep both Engram shards local on each node as well.

Pack the DFly2 checkpoint once and make the resulting directory available at the configured absolute path on both nodes:

```bash
python3 tools/dfly2/pack_weights.py "$MODEL_ROOT/DFly2-B6" "$MODEL_ROOT/DFly2-B6-TP2"
```

The pack is additional to the target rank cache. Its native codes and scales are preserved; each rank adds 4,529,862,920 bytes of draft weights plus scratch. The converter refuses an existing destination. Refer to its usage and the weight-format documentation for validation; do not rename a DSpark cache into a DFly2 pack.

## Runtime contract and handoff

Set paths in the cluster JSON to the directories actually prepared above. Both ranks need the same model, kit, token map, pack and binaries; each rank uses its corresponding local target cache and Engram files. The image supplies `libcublasLt.so.13`, `libnccl.so.2` and `libibverbs.so.1`; the NVIDIA runtime supplies host `libcuda.so.1`. The launcher provides host networking/IPC, unlimited pinned-memory lock, `/dev/infiniband` and NVIDIA device access. It needs no private library bind mounts.

Native asset variables are `TF_DS_KIT`, `TENSORFOLD_CUDA_KERNELS` (the kit's `aot` subdirectory), `TF_DS_TOKEN_MAP`, `TF_DS_RANK_CACHE`, `TF_DS_ENGRAM`, and `TF_DS_DFLY2_DIR`. The launcher sets these from the explicit configuration. Fabric interface names, HCA names and GID indices are machine-specific and must be discovered on both hosts. Match those values to the direct ConnectX-7 link; example values are placeholders.

The DFly path remains restricted to TP2, one active text stream, replicated KV and eager execution. Start with the bounded 8192-token validation pool. Successful source preparation or image compilation does not qualify larger contexts, vision, tool serving, concurrent streams or CUDA graphs. Read the validation instructions and current evidence before choosing DFly2 over the existing DSpark backend.

Public upstream references: [recipe Dockerfile](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/c7f2b85100f0a7b099c92a4e21825e30d93c9c51/zig/Dockerfile), [asset preparation](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/c7f2b85100f0a7b099c92a4e21825e30d93c9c51/scripts/prepare.sh), [runtime requirements](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/c7f2b85100f0a7b099c92a4e21825e30d93c9c51/docs/requirements.md), [public kernel kit](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/tree/abe07c7d509a0199cc7fec15263be2aa52ac652f).
