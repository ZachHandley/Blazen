# syntax=docker/dockerfile:1.7
#
# Blazen CI base image for x86_64-unknown-linux-musl.
#
# Pre-bakes everything `.forgejo/workflows/build-artifacts.yaml` installs at
# runtime for the musl matrix slot, so Forgejo jobs skip setup-rust's runtime
# sccache install (where EXDEV-crossing io.cp/io.rmRF is the suspected source
# of ENOENT) and kick off the actual cargo/maturin build immediately.
#
# Build:
#   podman build \
#     -f x86_64-unknown-linux-musl.Dockerfile \
#     -t blazen-ci-base:x86_64-unknown-linux-musl-local .
#
# Not intended to be pushed yet; CI workflow adjustments are out of scope.

FROM ghcr.io/rust-cross/rust-musl-cross:x86_64-musl

# ---------------------------------------------------------------------------
# System build dependencies (matches the `Install musl build prereqs` step of
# .forgejo/workflows/build-artifacts.yaml).
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        ccache \
        cmake \
        curl \
        file \
        libclang-dev \
        libssl-dev \
        patchelf \
        perl \
        pkg-config \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Short-triple symlinks so rustc's default linker lookup finds the cross
# toolchain (see workflow lines 174-189 for the explanation). The base image
# only ships the long-form `x86_64-unknown-linux-musl-gcc`; Rust's target spec
# defaults to the short form `x86_64-linux-musl-gcc`.
# ---------------------------------------------------------------------------
RUN set -eux; \
    triple="x86_64-unknown-linux-musl"; \
    short_triple="x86_64-linux-musl"; \
    bindir="/usr/local/musl/bin"; \
    test -x "${bindir}/${triple}-gcc"; \
    ln -sf "${bindir}/${triple}-gcc" "${bindir}/${short_triple}-gcc"; \
    ln -sf "${bindir}/${triple}-g++" "${bindir}/${short_triple}-g++"; \
    ln -sf "${bindir}/${triple}-gcc" "/usr/local/bin/${short_triple}-gcc"; \
    ln -sf "${bindir}/${triple}-g++" "/usr/local/bin/${short_triple}-g++"

# ---------------------------------------------------------------------------
# Rustup + stable toolchain at image-level constants.
#
# The base image ships a toolchain under /root/.rustup + /root/.cargo. CI jobs
# override CARGO_HOME per-job (for registry caching), so we reinstall rustup
# under /opt/{rustup,cargo}, which gives the image stable absolute paths the
# workflow can point PATH at even when CARGO_HOME is redirected elsewhere.
# ---------------------------------------------------------------------------
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:/opt/rustup/bin:/usr/local/musl/bin:/usr/local/bin:/usr/bin:/bin
RUN set -eux; \
    mkdir -p "${RUSTUP_HOME}" "${CARGO_HOME}/bin"; \
    curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh; \
    sh /tmp/rustup-init.sh -y --no-modify-path --default-toolchain stable --profile minimal; \
    rm /tmp/rustup-init.sh; \
    rustup target add x86_64-unknown-linux-musl; \
    rustc --version; \
    cargo --version

# ---------------------------------------------------------------------------
# sccache v0.14.0 pre-baked at a stable absolute path. Workflow currently
# pulls this via the setup-rust action at runtime; pre-baking sidesteps
# io.cp + io.rmRF crossing EXDEV boundaries (the suspected ENOENT source).
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
# no-op instead of a download.
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
# ---------------------------------------------------------------------------
ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/opt/rustup/bin:/usr/local/musl/bin:/usr/local/bin:/usr/bin:/bin \
    CARGO_INCREMENTAL=0 \
    CARGO_TERM_COLOR=always \
    OPENSSL_STATIC=1 \
    OPENSSL_VENDORED=1
# NOTE: RUSTC_WRAPPER is intentionally NOT set at the image level. sccache is
# pre-installed at /opt/cargo/bin/sccache (on PATH), but whether to enable it
# is a per-job decision in the workflow. Image-level RUSTC_WRAPPER causes
# sccache to wrap cc during openssl-src/aws-lc-sys assembler builds, which
# breaks .s preprocessing on some targets.

# ---------------------------------------------------------------------------
# Fail-fast smoke test baked into the image build.
# ---------------------------------------------------------------------------
RUN set -eux; \
    sccache --version; \
    rustc --version; \
    cargo --version; \
    rustup target list --installed | grep -Fx x86_64-unknown-linux-musl; \
    uv --version; \
    maturin --version; \
    x86_64-linux-musl-gcc --version | head -n1; \
    x86_64-unknown-linux-musl-gcc --version | head -n1

WORKDIR /workspace
