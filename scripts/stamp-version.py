#!/usr/bin/env python3
"""Cross-platform version stamping for Blazen release CI.

Replaces the per-job `find ... sed -i` dance in .forgejo/workflows/release.yaml.
Works identically on Linux (GNU sed), macOS (BSD sed), and Windows (no sed).

Default mode (no flags) — used by the BUILD jobs (wheels/napi/uniffi/cabi):
    python3 scripts/stamp-version.py <VERSION>

    Stamps `0.0.0-dev` -> VERSION across every workspace `Cargo.toml`
    (excluding `target/`), STRIPS `, registry = "forgejo"` off the root
    `Cargo.toml` inter-crate deps (leaving plain `version` + `path` deps),
    and bumps the hand-written `package.json` files for `blazen-node`,
    `blazen-wasm-sdk`, `blazen-workers-alias`, and
    `blazen-workers-tiktoken-alias`.

    The strip matters: the cross-build environments (notably the musl
    wheel/napi containers) do NOT pick up the repo `.cargo/config.toml`
    that defines the `forgejo` registry, so a `registry = "forgejo"` dep is
    an unknown registry there and the build fails. Path deps build fine
    without it. (Observed: run 674 stripped -> musl green; run 678 kept ->
    all three musl targets failed while gnu/darwin/windows passed.)

Keep-registry mode — used ONLY by the publish-rust job:
    python3 scripts/stamp-version.py <VERSION> --keep-forgejo-registry

    Same as default but KEEPS `, registry = "forgejo"` so `cargo publish
    --registry forgejo` resolves each crate's sibling deps from the private
    Forgejo cargo registry (not crates.io). The publish-rust job runs
    setup-rust, which configures the forgejo registry + token, so the
    registry resolves there.

ZRegistry mode — used ONLY by the publish-rust-zreg job:
    python3 scripts/stamp-version.py <VERSION> --registry-zreg

    Same as default version-stamping, but RE-POINTS every inter-crate dep
    from the `forgejo` registry to `zreg` (ZRegistry,
    registry.blackleafdigital.com). It rewrites `registry = "forgejo"` ->
    `registry = "zreg"` across every workspace Cargo.toml (both the root
    `[workspace.dependencies]` blazen-* entries and the two direct
    blazen-node member deps). With each sibling dep carrying
    `registry = "zreg"` + its `path`, `cargo publish --registry zreg`
    resolves workspace siblings from ZRegistry instead of crates.io.
    ZRegistry has NO publish rate limits, so the whole workspace lands in
    one shot regardless of crates.io's per-account caps. Pass this ONLY
    from the publish-rust-zreg job; build jobs and publish-rust must NOT.

Rename-wasm-pkg mode:
    python3 scripts/stamp-version.py <VERSION> --rename-wasm-pkg

    Operates EXCLUSIVELY on the wasm-pack-generated
    `crates/blazen-wasm-sdk/pkg/package.json`. Sets `name` to
    `@blazen-dev/wasm` and `version` to VERSION while preserving every
    other field. Intended to run AFTER `wasm-pack build`, since wasm-pack
    regenerates that file on every build. Does not touch any Cargo.toml
    or hand-written package.json files (those are stamped earlier in CI
    by the default mode).
"""
import argparse
import json
import pathlib
import re
import sys

PUBLISHED_WASM_NAME = "@blazen-dev/wasm"

# Matches a single-line inline-table dependency on a sibling `blazen-*` crate,
# e.g. `blazen-3d-core = { version = "0.6.1", path = "../blazen-3d-core" }`.
# Group 1 = `name = {`, group 2 = the inline-table body, group 3 = `}`.
_PATH_DEP_RE = re.compile(r"^(\s*blazen-[a-z0-9-]+\s*=\s*\{)([^}\n]*)(\})", re.M)


def _add_forgejo_registry_to_path_deps(text: str) -> str:
    """Add `, registry = "forgejo"` to direct `blazen-* = { ... path = ... }`
    deps that lack a registry (skips `workspace = true` and already-tagged deps).
    """

    def repl(m: "re.Match[str]") -> str:
        head, body, tail = m.group(1), m.group(2), m.group(3)
        if "registry" in body or "workspace" in body or "path" not in body:
            return m.group(0)
        return f'{head}{body.rstrip()}, registry = "forgejo" {tail}'

    return _PATH_DEP_RE.sub(repl, text)


# Matches ONLY cross-repo path deps, i.e. `path = "../X/Y/..."` where there
# are TWO OR MORE path segments after `../`. In-tree sibling deps look like
# `path = "../blazen-memory-valkey"` (a single segment after `../`) and must
# NOT be stripped — those resolve inside the same workspace and stripping
# them creates a "different source paths" cargo error (workspace inheritance
# vs registry-only dep declaration mismatch).
_CROSS_REPO_PATH_RE = re.compile(r',\s*path\s*=\s*"\.\./[^/"]+/[^"]+"')


def stamp_default(
    repo: pathlib.Path,
    version: str,
    keep_forgejo_registry: bool = False,
    registry_zreg: bool = False,
) -> None:
    # 1. Every Cargo.toml: 0.0.0-dev -> VERSION, EXCEPT on lines that carry a
    # CROSS-REPO path dep (`path = "../X/Y/..."` — TWO or more segments after
    # `../`). Those reference an external sibling repo which has its OWN
    # release train and carries a pinned version we must not stamp over.
    #
    # In-tree sibling deps like `path = "../blazen-3d"` (single segment) MUST
    # be stamped — they reference workspace siblings whose versions DO follow
    # the workspace release train.
    for cargo in repo.rglob("Cargo.toml"):
        if "target" in cargo.parts:
            continue
        text = cargo.read_text()
        new_lines = []
        for line in text.split("\n"):
            if _CROSS_REPO_PATH_RE.search(line):
                new_lines.append(line)
            else:
                new_lines.append(line.replace("0.0.0-dev", version))
        new = "\n".join(new_lines)
        if new != text:
            cargo.write_text(new)

    # 2. Root Cargo.toml: STRIP `, registry = "forgejo"` off the inter-crate
    # workspace deps by default — the cross-build envs (musl wheel/napi
    # containers) don't see the repo `.cargo/config.toml` that defines the
    # `forgejo` registry, so the dep is an unknown registry there and the build
    # fails (run 678: all musl targets failed; run 674, which stripped, passed).
    # publish-rust passes --keep-forgejo-registry so `cargo publish --registry
    # forgejo` resolves siblings from Forgejo (it configures the registry+token).
    if registry_zreg:
        # publish-rust-zreg path: RE-POINT every inter-crate dep from the
        # `forgejo` registry to `zreg` (ZRegistry). The workspace deps already
        # carry `version` + `path` + `registry = "forgejo"`; swapping the
        # registry name to `zreg` makes `cargo publish --registry zreg` resolve
        # each sibling from ZRegistry (cargo strips `path` on publish and writes
        # the `version` + `registry` into the uploaded manifest). ZRegistry has
        # no publish rate limits, so the full workspace lands in one shot.
        # Applies to the root `[workspace.dependencies]` blazen-* entries AND
        # the two direct blazen-node member deps that declare their own
        # `registry = "forgejo"`.
        for cargo in repo.rglob("Cargo.toml"):
            if "target" in cargo.parts:
                continue
            text = cargo.read_text()
            new = text.replace('registry = "forgejo"', 'registry = "zreg"')
            if new != text:
                cargo.write_text(new)
    elif not keep_forgejo_registry:
        root_cargo = repo / "Cargo.toml"
        root_text = root_cargo.read_text()
        root_cargo.write_text(root_text.replace(', registry = "forgejo"', ""))
    else:
        # publish-rust path:
        #   (a) STRIP `, path = "../..."` from EVERY Cargo.toml — external
        #       sibling repos are not checked out in the publish CI
        #       sandbox, so `cargo metadata` fails when it tries to read
        #       their manifest. With path stripped, cargo resolves those
        #       deps purely from the registry (the version pin is the
        #       sibling repo's own release version, set in step 1).
        #   (b) Inject `, registry = "forgejo"` into every direct
        #       `blazen-* = { ... path = ... }` inline-table dep that
        #       lacks it (existing behavior — the 37 root
        #       `workspace.dependencies` carry `registry = "forgejo"`,
        #       but inter-crate deps declared DIRECTLY in a member
        #       manifest do NOT — so on publish cargo resolves them
        #       from crates.io and fails).
        for cargo in repo.rglob("Cargo.toml"):
            if "target" in cargo.parts:
                continue
            text = cargo.read_text()
            stripped = _CROSS_REPO_PATH_RE.sub("", text)
            new = _add_forgejo_registry_to_path_deps(stripped)
            if new != text:
                cargo.write_text(new)

    # 3. Node package.json: 0.1.0 -> VERSION (for blazen-node)
    node_pkg = repo / "crates" / "blazen-node" / "package.json"
    node_text = node_pkg.read_text()
    node_pkg.write_text(
        re.sub(r'"version"\s*:\s*"0\.1\.0"', f'"version": "{version}"', node_text, count=1)
    )

    # 4. WASM SDK package.json: 0.0.0-dev -> VERSION
    wasm_pkg = repo / "crates" / "blazen-wasm-sdk" / "package.json"
    if wasm_pkg.exists():
        wasm_text = wasm_pkg.read_text()
        wasm_pkg.write_text(
            re.sub(
                r'"version"\s*:\s*"0\.0\.0-dev"',
                f'"version": "{version}"',
                wasm_text,
                count=1,
            )
        )

    # 5. Workers alias package.json: 0.0.0-dev -> VERSION (both `version` and the
    #    @blazen-dev/blazen-wasm32-wasi dep pin get stamped, since both currently
    #    read "0.0.0-dev").
    alias_pkg = repo / "crates" / "blazen-workers-alias" / "package.json"
    if alias_pkg.exists():
        alias_text = alias_pkg.read_text()
        alias_pkg.write_text(alias_text.replace("0.0.0-dev", version))

    # 5b. Tiktoken workers alias package.json: same treatment — both `version`
    #     and the @blazen-dev/blazen-wasm32-wasi-tiktoken dep pin read
    #     "0.0.0-dev" and get stamped to VERSION. The -tiktoken sidecar it
    #     depends on is published at VERSION by the publish-node job (built from
    #     the version-stamped lean wasm32-wasi subpackage).
    tk_alias_pkg = repo / "crates" / "blazen-workers-tiktoken-alias" / "package.json"
    if tk_alias_pkg.exists():
        tk_alias_text = tk_alias_pkg.read_text()
        tk_alias_pkg.write_text(tk_alias_text.replace("0.0.0-dev", version))


def rename_wasm_pkg(repo: pathlib.Path, version: str) -> None:
    """Rewrite the wasm-pack-generated pkg/package.json with the published name + version."""
    pkg_json_path = repo / "crates" / "blazen-wasm-sdk" / "pkg" / "package.json"
    if not pkg_json_path.exists():
        print(
            f"error: {pkg_json_path} not found. "
            "Run `wasm-pack build crates/blazen-wasm-sdk --target web --release` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    data = json.loads(pkg_json_path.read_text())
    data["name"] = PUBLISHED_WASM_NAME
    data["version"] = version
    # wasm-pack emits 2-space indent + trailing newline; match that.
    pkg_json_path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="stamp-version.py",
        description="Stamp Blazen release versions across workspace manifests.",
    )
    parser.add_argument("version", help="version string (e.g. 1.2.3)")
    parser.add_argument(
        "--rename-wasm-pkg",
        action="store_true",
        help=(
            "Rewrite crates/blazen-wasm-sdk/pkg/package.json (the wasm-pack-generated "
            "manifest) with name=@blazen-dev/wasm and the given version. Run AFTER "
            "`wasm-pack build`. Skips all other stamping."
        ),
    )
    parser.add_argument(
        "--keep-forgejo-registry",
        action="store_true",
        help=(
            "Keep `, registry = \"forgejo\"` on the root Cargo.toml inter-crate deps "
            "(default strips it). Pass this ONLY from the publish-rust job so "
            "`cargo publish --registry forgejo` resolves siblings from Forgejo. "
            "Build jobs must NOT pass it (their cross-build envs lack the forgejo "
            "registry config and would fail to resolve the dep)."
        ),
    )
    parser.add_argument(
        "--registry-zreg",
        action="store_true",
        help=(
            "Re-point every inter-crate dep from `registry = \"forgejo\"` to "
            "`registry = \"zreg\"` (ZRegistry) across every workspace Cargo.toml. "
            "Pass this ONLY from the publish-rust-zreg job so `cargo publish "
            "--registry zreg` resolves siblings from ZRegistry (no rate limits). "
            "Mutually exclusive with --keep-forgejo-registry."
        ),
    )
    args = parser.parse_args()

    if args.registry_zreg and args.keep_forgejo_registry:
        parser.error("--registry-zreg and --keep-forgejo-registry are mutually exclusive")

    repo = pathlib.Path(__file__).resolve().parent.parent

    if args.rename_wasm_pkg:
        rename_wasm_pkg(repo, args.version)
        print(f"renamed wasm pkg/package.json: name={PUBLISHED_WASM_NAME} version={args.version}")
    else:
        stamp_default(
            repo,
            args.version,
            keep_forgejo_registry=args.keep_forgejo_registry,
            registry_zreg=args.registry_zreg,
        )
        if args.registry_zreg:
            disposition = "re-pointed forgejo->zreg"
        elif args.keep_forgejo_registry:
            disposition = "kept"
        else:
            disposition = "stripped"
        print(f"stamped version={args.version} (forgejo registry {disposition})")


if __name__ == "__main__":
    main()
