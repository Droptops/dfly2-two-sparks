# The exact source trees and patch bytes are checked by prepare_source.py.
# Build with scripts/build_image.py; no existing custom image is needed.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG ZIG_VERSION
ARG ZIG_SHA256
ARG BUILD_JOBS=4
RUN python3 -c 'import torch, safetensors, PIL, tokenizers' && \
    /usr/local/cuda/bin/nvcc --version && \
    ldconfig -p | grep libibverbs && ldconfig -p | grep libnccl
RUN test "$(uname -m)" = aarch64 && \
    for base in https://zig.linus.dev/zig "https://ziglang.org/download/${ZIG_VERSION}"; do \
      if curl --connect-timeout 20 --max-time 300 --retry 3 --retry-max-time 600 -fsSL \
        "${base}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" -o /tmp/zig.tar.xz && \
        echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c -; then break; fi; \
    done && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && \
    rm /tmp/zig.tar.xz && test "$(/opt/zig/zig version)" = "${ZIG_VERSION}"

COPY tensorfold /build/tensorfold
COPY inputs /build/inputs
COPY source-files.sha256 source-receipt.json upstream.lock.json /build/
RUN cd /build && sha256sum -c source-files.sha256 >/dev/null
RUN cd /build/tensorfold && \
    /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast \
      --prefix /opt/tensorfold -j${BUILD_JOBS} install native && \
    test -x /opt/tensorfold/bin/tf-dsv41-lanes && \
    test -x /opt/tensorfold/bin/tf-dsv41-dfly-weights && \
    test -x /opt/tensorfold/native/bin/tensorfold-native && \
    mkdir -p /opt/tensorfold/share/doc/tensorfold && \
    for f in LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/; fi; \
    done && \
    if [ -e tools/dsv41-zig/ATTRIBUTION.md ]; then \
      cp tools/dsv41-zig/ATTRIBUTION.md /opt/tensorfold/share/doc/tensorfold/DSV41-ATTRIBUTION.md; fi
RUN cd /build/inputs && \
    /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast \
      --prefix /opt/inputs -j${BUILD_JOBS} install && \
    cp /opt/inputs/bin/tf-dsv41-rank-cache /opt/tensorfold/bin/ && \
    mkdir -p /opt/tensorfold/tools/inputs && \
    cp tools/dsv41-zig/inputs/make_token_map.py tools/dsv41-zig/kit/make_rope.py \
      tools/dsv41-zig/kit/make_bias_vl.py /opt/tensorfold/tools/inputs/ && \
    cp LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSE-inputs-branch
RUN mkdir -p /opt/tensorfold/share/build && \
    cp /build/source-receipt.json /build/upstream.lock.json /build/source-files.sha256 /opt/tensorfold/share/build/ && \
    rm -rf /build /opt/inputs /root/.cache/zig
# Chat-request generation uses Transformers. Keep its small, hash-locked tool
# environment separate from the NVIDIA image's packages and native build cache.
COPY tool-requirements.txt verify_tools.py /opt/tensorfold/share/build/
RUN python3 -m venv --without-pip --system-site-packages /opt/dfly2-tools && \
    /opt/dfly2-tools/bin/python -m pip install --disable-pip-version-check --no-cache-dir \
      --require-virtualenv --no-deps --require-hashes \
      -r /opt/tensorfold/share/build/tool-requirements.txt && \
    /opt/dfly2-tools/bin/python /opt/tensorfold/share/build/verify_tools.py
LABEL ai.dfly2.experimental="true" \
      tf.ref="fbbc97132cd004237d9887b8d3185896efaedc5c" \
      tf.inputs="59402086381a42c74d53053443224eec2790b4c5"
ENV PATH=/opt/dfly2-tools/bin:/opt/tensorfold/native/bin:/opt/tensorfold/bin:${PATH}
WORKDIR /workspace
