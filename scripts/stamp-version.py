#!/usr/bin/env python3
"""Cross-platform version stamping for Blazen release CI.

Replaces the per-job `find ... sed -i` dance in .forgejo/workflows/release.yaml.
Works identically on Linux (GNU sed), macOS (BSD sed), and Windows (no sed).

Default mode (no flags):
    python3 scripts/stamp-version.py <VERSION>

    Stamps `0.0.0-dev` -> VERSION across every workspace `Cargo.toml`
    (excluding `target/`), strips `, registry = "forgejo"` from the root
    `Cargo.toml`, and bumps the hand-written `package.json` files for
    `blazen-node`, `blazen-wasm-sdk`, and `blazen-workers-alias`.

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


def stamp_default(repo: pathlib.Path, version: str) -> None:
    # 1. Every Cargo.toml: 0.0.0-dev -> VERSION
    for cargo in repo.rglob("Cargo.toml"):
        if "target" in cargo.parts:
            continue
        text = cargo.read_text()
        new = text.replace("0.0.0-dev", version)
        if new != text:
            cargo.write_text(new)

    # 2. Root Cargo.toml only: strip `, registry = "forgejo"`
    root_cargo = repo / "Cargo.toml"
    root_text = root_cargo.read_text()
    root_cargo.write_text(root_text.replace(', registry = "forgejo"', ""))

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


def rename_wasm_pkg(repo: pathlib.Path, version: str) -> None:
    """Rewrite the wasm-pack-generated pkg/package.json with the published name + version."""
    pkg_json_path = repo / "crates" / "blazen-wasm-sdk" / "pkg" / "package.json"
    if not pkg_json_path.exists():
        print(
            f"error: {pkg_json_path} not found. "
            "Run `wasm-pack build crates/blazen-wasm-sdk --target bundler --release` first.",
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
    args = parser.parse_args()

    repo = pathlib.Path(__file__).resolve().parent.parent

    if args.rename_wasm_pkg:
        rename_wasm_pkg(repo, args.version)
        print(f"renamed wasm pkg/package.json: name={PUBLISHED_WASM_NAME} version={args.version}")
    else:
        stamp_default(repo, args.version)
        print(f"stamped version={args.version}")


if __name__ == "__main__":
    main()
