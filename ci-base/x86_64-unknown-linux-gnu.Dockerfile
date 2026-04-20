# syntax=docker/dockerfile:1.7
#
# Blazen CI base image for x86_64-unknown-linux-gnu (manylinux_2_28).
#
# Pre-bakes everything `.forgejo/workflows/build-artifacts.yaml` installs at
# runtime for the x86_64 glibc / manylinux matrix slot, so Forgejo jobs skip
# setup-rust's runtime sccache install (where EXDEV-crossing io.cp/io.rmRF is
# the suspected source of ENOENT) and kick off the actual cargo/maturin build
# immediately.
#
# Build:
#   podman build \
#     -f x86_64-unknown-linux-gnu.Dockerfile \
#     -t blazen-ci-base:x86_64-unknown-linux-gnu-local .
#
# Not intended to be pushed yet; CI workflow adjustments are out of scope.

FROM quay.io/pypa/manylinux_2_28_x86_64

# ---------------------------------------------------------------------------
# System build dependencies (matches the `Install system deps (manylinux)`
# step of .forgejo/workflows/build-artifacts.yaml line 120-122). `file` is
# added because various build.rs crates (openssl-src, llama-cpp-sys-2) call
# `file(1)` during configure probes; the musl Dockerfile includes it and the
# manylinux base image does not ship it.
# ---------------------------------------------------------------------------
RUN set -eux; \
    yum install -y \
        cmake \
        openssl-devel \
        ccache \
        pkg-config \
        curl \
        perl-core \
        perl-Time-Piece \
        clang-devel \
        file; \
    yum clean all; \
    rm -rf /var/cache/yum

# ---------------------------------------------------------------------------
# Rustup + stable toolchain at image-level constants.
#
# The manylinux base image does not ship rustup. CI jobs override CARGO_HOME
# per-job (for registry caching), so we install rustup under /opt/{rustup,
# cargo}, which gives the image stable absolute paths the workflow can point
# PATH at even when CARGO_HOME is redirected elsewhere.
# ---------------------------------------------------------------------------
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:/opt/rustup/bin:/usr/local/bin:/usr/bin:/bin
RUN set -eux; \
    mkdir -p "${RUSTUP_HOME}" "${CARGO_HOME}/bin"; \
    curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh; \
    sh /tmp/rustup-init.sh -y --no-modify-path --default-toolchain stable --profile minimal; \
    rm /tmp/rustup-init.sh; \
    rustup target add x86_64-unknown-linux-gnu; \
    rustc --version; \
    cargo --version

# ---------------------------------------------------------------------------
# sccache v0.14.0 pre-baked at a stable absolute path. We deliberately use
# the `x86_64-unknown-linux-musl` tarball here too: Mozilla's musl build is
# statically linked (confirmed via `readelf -l` — static-pie linked, stripped,
# no INTERP) and runs on glibc hosts fine. Using the musl tarball also keeps
# the sccache binary identical across the musl and gnu base images.
# ---------------------------------------------------------------------------
ARG SCCACHE_VERSION=0.14.0
RUN set -eux; \
    cd /tmp; \
    curl -fsSL -o sccache.tgz \
        "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz"; \
    tar -xzf sccache.tgz; \
    cp "sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache" "${CARGO_HOME}/bin/sccache"; \
    chmod 755 "${CARGO_HOME}/bin/sccache"; \
    rm -rf sccache.tgz "sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl"; \
    "${CARGO_HOME}/bin/sccache" --version

# ---------------------------------------------------------------------------
# uv (Astral) at /usr/local/bin/uv via official installer. Then pre-fetch the
# supported Python versions so `uv python find 3.xx` during CI builds is a
# no-op instead of a download. Note: manylinux_2_28 already ships
# /opt/python/cp3XX-cp3XX/bin/python* interpreters, but the workflow uses
# `uv python find`, so we install uv-managed interpreters to match.
# ---------------------------------------------------------------------------
RUN set -eux; \
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
# Unlike the musl image, OPENSSL_STATIC / OPENSSL_VENDORED are NOT set —
# manylinux_2_28 provides a working openssl-devel via yum, and the workflow
# only sets those env vars for the musl/alpine matrix slots (line 68-69 uses
# `matrix.openssl-vendored`, which is only defined for musl/alpine entries).
# ---------------------------------------------------------------------------
ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/opt/rustup/bin:/usr/local/bin:/usr/bin:/bin \
    CARGO_INCREMENTAL=0 \
    CARGO_TERM_COLOR=always
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
    rustup target list --installed | grep -Fx x86_64-unknown-linux-gnu; \
    uv --version; \
    maturin --version

WORKDIR /workspace
