#!/usr/bin/env python3
"""Cross-platform version stamping for Blazen release CI.

Replaces the per-job `find ... sed -i` dance in .forgejo/workflows/release.yaml.
Works identically on Linux (GNU sed), macOS (BSD sed), and Windows (no sed).
"""
import pathlib
import re
import sys

if len(sys.argv) != 2:
    print("usage: stamp-version.py <VERSION>", file=sys.stderr)
    sys.exit(1)

version = sys.argv[1]
repo = pathlib.Path(__file__).resolve().parent.parent

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
node_pkg.write_text(re.sub(r'"version"\s*:\s*"0\.1\.0"', f'"version": "{version}"', node_text, count=1))

# 4. WASM SDK package.json: 0.0.0-dev -> VERSION
wasm_pkg = repo / "crates" / "blazen-wasm-sdk" / "package.json"
if wasm_pkg.exists():
    wasm_text = wasm_pkg.read_text()
    wasm_pkg.write_text(re.sub(r'"version"\s*:\s*"0\.0\.0-dev"', f'"version": "{version}"', wasm_text, count=1))

print(f"stamped version={version}")
