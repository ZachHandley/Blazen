# syntax=docker/dockerfile:1.7
#
# Blazen CI base image for aarch64-unknown-linux-musl.
#
# Pre-bakes everything `.forgejo/workflows/build-artifacts.yaml` installs at
# runtime for the aarch64 musl matrix slot. Unlike the x86_64 musl image
# (which uses `rust-musl-cross` for a cross-compile toolchain from a glibc
# host), this image builds NATIVELY on alpine arm64: /usr/bin/gcc IS a
# working musl compiler for the target, so we just alias it under the
# short-triple name rustc expects.
#
# Build (on an arm64 host or with qemu):
#   podman build \
#     -f aarch64-unknown-linux-musl.Dockerfile \
#     -t blazen-ci-base:aarch64-unknown-linux-musl-local .
#
# Not intended to be pushed yet; CI workflow adjustments are out of scope.

FROM alpine:3.20

# ---------------------------------------------------------------------------
# System build dependencies. Alpine equivalents of the manylinux yum set plus
# the extras the musl cross image bakes in: patchelf (maturin), libffi-dev
# (cffi/cryptography build.rs), linux-headers (various sys crates), clang+llvm
# (bindgen for llama-cpp-sys-2), file (openssl-src configure probes), xz/tar
# (sccache tarball extraction), bash (rustup-init and uv installers expect a
# real bash rather than ash), perl (openssl-src Configure script).
# ---------------------------------------------------------------------------
RUN set -eux; \
    apk add --no-cache \
        bash \
        curl \
        git \
        gcc \
        g++ \
        musl-dev \
        linux-headers \
        cmake \
        make \
        ccache \
        pkgconf \
        openssl-dev \
        clang-dev \
        llvm-dev \
        perl \
        patchelf \
        libffi-dev \
        file \
        xz \
        tar

# ---------------------------------------------------------------------------
# Linker setup — NOT a cross-toolchain install.
#
# On alpine arm64 the host IS the target: /usr/bin/gcc is a native
# aarch64-linux-musl compiler. rustc's aarch64-unknown-linux-musl target
# defaults to invoking `aarch64-linux-musl-gcc` as the linker, so we just
# alias the native gcc/g++ under that short-triple name. This mirrors what
# `.forgejo/workflows/build-artifacts.yaml` lines 200-213 does at runtime
# today; baking it into the image removes that setup step from every CI run.
# ---------------------------------------------------------------------------
RUN set -eux; \
    ln -sf /usr/bin/gcc /usr/local/bin/aarch64-linux-musl-gcc; \
    ln -sf /usr/bin/g++ /usr/local/bin/aarch64-linux-musl-g++; \
    aarch64-linux-musl-gcc --version | head -n1

# ---------------------------------------------------------------------------
# Rustup + stable toolchain at image-level constants.
#
# Same /opt/{rustup,cargo} layout as the x86_64-gnu Dockerfile so the workflow
# can point PATH at stable absolute paths even when CI overrides CARGO_HOME
# per-job for registry caching.
# ---------------------------------------------------------------------------
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:/opt/rustup/bin:/usr/local/bin:/usr/bin:/bin
RUN set -eux; \
    mkdir -p "${RUSTUP_HOME}" "${CARGO_HOME}/bin"; \
    curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh; \
    sh /tmp/rustup-init.sh -y --no-modify-path --default-toolchain stable --profile minimal; \
    rm /tmp/rustup-init.sh; \
    rustup target add aarch64-unknown-linux-musl; \
    rustc --version; \
    cargo --version

# ---------------------------------------------------------------------------
# sccache v0.14.0 pre-baked at a stable absolute path. Uses the official
# Mozilla aarch64-unknown-linux-musl tarball so the binary is statically
# linked and runs directly on the alpine base without extra deps.
# ---------------------------------------------------------------------------
ARG SCCACHE_VERSION=0.14.0
RUN set -eux; \
    cd /tmp; \
    curl -fsSL -o sccache.tgz \
        "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-aarch64-unknown-linux-musl.tar.gz"; \
    tar -xzf sccache.tgz; \
    cp "sccache-v${SCCACHE_VERSION}-aarch64-unknown-linux-musl/sccache" "${CARGO_HOME}/bin/sccache"; \
    chmod 755 "${CARGO_HOME}/bin/sccache"; \
    rm -rf sccache.tgz "sccache-v${SCCACHE_VERSION}-aarch64-unknown-linux-musl"; \
    "${CARGO_HOME}/bin/sccache" --version

# ---------------------------------------------------------------------------
# uv (Astral) at /usr/local/bin/uv via official installer. Then pre-fetch the
# supported Python versions so `uv python find 3.xx` during CI builds is a
# no-op instead of a download.
#
# Gotcha: uv's managed CPython standalone builds are glibc-linked; on alpine
# (musl) they need a glibc shim to load. `gcompat` provides that shim and is
# installed up front so `uv python install` can execute the downloaded
# interpreters during the prefetch verification uv runs after download.
# ---------------------------------------------------------------------------
RUN set -eux; \
    apk add --no-cache gcompat; \
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh; \
    uv --version; \
    uv python install 3.10 3.11 3.12 3.13 3.14

# ---------------------------------------------------------------------------
# Maturin pre-warmed so `uvx maturin build ...` doesn't cold-resolve on first
# CI invocation. `uv tool install` lays it under /root/.local/share/uv/tools;
# the shim lands on PATH via uv's tool bin dir.
# ---------------------------------------------------------------------------
ARG MATURIN_VERSION=1.13.1
ENV UV_TOOL_BIN_DIR=/usr/local/bin
RUN set -eux; \
    uv tool install "maturin==${MATURIN_VERSION}"; \
    maturin --version

# ---------------------------------------------------------------------------
# Final image-level env. CI can still override CARGO_HOME per-job to keep
# registry caching on the workspace volume; PATH keeps /opt/cargo/bin ahead so
# the pre-baked sccache remains resolvable even when CARGO_HOME is elsewhere.
#
# OPENSSL_STATIC / OPENSSL_VENDORED are set for the musl matrix slot so
# openssl-sys / reqwest vendor and statically link openssl during the build
# (matches the workflow's matrix.openssl-vendored wiring for musl entries).
# ---------------------------------------------------------------------------
ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/opt/rustup/bin:/usr/local/bin:/usr/bin:/bin \
    CARGO_INCREMENTAL=0 \
    CARGO_TERM_COLOR=always \
    OPENSSL_STATIC=1 \
    OPENSSL_VENDORED=1
# NOTE: RUSTC_WRAPPER is intentionally NOT set at the image level. Image-level
# RUSTC_WRAPPER=sccache causes sccache to wrap cc during openssl-src's
# aes-cfb-avx512.s assembler build, which fails with "character following name
# is not '#'" — sccache mishandles .s preprocessing. sccache is still
# pre-installed at /opt/cargo/bin/sccache (on PATH); the workflow can opt in
# per-job by exporting RUSTC_WRAPPER when appropriate.

# ---------------------------------------------------------------------------
# Fail-fast smoke test baked into the image build.
# ---------------------------------------------------------------------------
RUN set -eux; \
    sccache --version; \
    rustc --version; \
    cargo --version; \
    rustup target list --installed | grep -Fx aarch64-unknown-linux-musl; \
    uv --version; \
    maturin --version; \
    aarch64-linux-musl-gcc --version | head -n1

WORKDIR /workspace
