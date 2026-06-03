#!/usr/bin/env python3
"""Semi-automated PyPI release pruner for the `blazen` project.

PyPI enforces a per-project **size limit** (default 10 GB = total size of all
live files). When `blazen` hits it, `publish-python` fails with
``400 Project size too large``. Two facts shape this tool:

  * **Deleting** old releases reclaims quota; **yanking** does NOT
    (https://docs.pypi.org/project-management/storage-limits/).
  * PyPI exposes **no delete API**. The only automation path is
    `pypi-cleanup` (https://pypi.org/project/pypi-cleanup/), which drives the
    authenticated web UI.

LIMITATION — why this is RUN-IT-YOURSELF, not a CI job:
  PyPI 2FA is mandatory and `pypi-cleanup` has no non-interactive TOTP hook
  (no flag/env for a TOTP seed). So deletion needs a human to type the OTP.
  This script is therefore a *semi-automated* maintenance tool: it computes the
  plan from live PyPI data and shells out to `pypi-cleanup`, which prompts you
  for the one-time code. The standing permanent fix is the PyPI project
  size-limit increase (filed separately at github.com/pypi/support).

What it does:
  1. Reads the live PyPI JSON, sums per-release sizes (PEP 440 ordered).
  2. Picks releases to delete by ONE of:
       * regex mode (default): every version matching --delete-regex
         (default ``^0\\.[1-5]\\.`` → all 0.1.x–0.5.x, keeping 0.6+); or
       * target mode (--target-gb N): delete the OLDEST releases until the
         project is <= N GB, always keeping the newest --keep-recent.
  3. Prints the plan with sizes (dry-run by default). With --do-it it invokes
     ``pypi-cleanup -u <user> -p <pkg> -r <exact-version-regex> --do-it``,
     which authenticates (password via PYPI_CLEANUP_PASSWORD or prompt) and
     asks for your OTP, then deletes exactly the previewed versions.

Usage:
  # Preview the default (delete 0.1.x–0.5.x, keep 0.6+):
  python3 scripts/prune-pypi.py

  # Preview a size target instead (delete oldest until <= 8 GB, keep newest 12):
  python3 scripts/prune-pypi.py --target-gb 8 --keep-recent 12

  # Actually delete (prompts for password+OTP; or set PYPI_CLEANUP_PASSWORD):
  PYPI_CLEANUP_PASSWORD=... python3 scripts/prune-pypi.py --do-it -u <pypi-user>
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

DEFAULT_PACKAGE = "blazen"
DEFAULT_HOST = "https://pypi.org"
# Keep 0.6.x and anything newer; delete the legacy 0.1.x–0.5.x bloat.
DEFAULT_DELETE_REGEX = r"^0\.[1-5]\."

try:  # PEP 440 ordering when available (pypi-cleanup pulls `packaging` in too).
    from packaging.version import Version, InvalidVersion

    def _vkey(v: str):
        try:
            return (0, Version(v))
        except InvalidVersion:
            return (1, v)
except Exception:  # pragma: no cover - fallback if packaging is absent

    def _vkey(v: str):
        nums = [int(x) for x in re.findall(r"\d+", v)]
        return (0, nums, v)


def _human(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def fetch_releases(host: str, package: str) -> "dict[str, int]":
    """Return {version: total_size_bytes} for every release with files."""
    url = f"{host.rstrip('/')}/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    sizes: "dict[str, int]" = {}
    for version, files in data["releases"].items():
        if not files:  # skip versions with no files (already fully deleted)
            continue
        sizes[version] = sum(int(f.get("size", 0)) for f in files)
    return sizes


def select_by_regex(sizes, pattern):
    rx = re.compile(pattern)
    delete = [v for v in sizes if rx.search(v)]
    return delete


def select_by_target(sizes, target_bytes, keep_recent):
    """Delete OLDEST releases until total <= target, never touching the newest
    `keep_recent`. Returns the list of versions to delete."""
    ordered = sorted(sizes, key=_vkey)  # oldest -> newest
    protected = set(ordered[-keep_recent:]) if keep_recent > 0 else set()
    total = sum(sizes.values())
    delete = []
    for v in ordered:  # oldest first
        if total <= target_bytes:
            break
        if v in protected:
            continue
        delete.append(v)
        total -= sizes[v]
    return delete


def main() -> None:
    p = argparse.ArgumentParser(
        prog="prune-pypi.py",
        description="Preview/delete old PyPI releases to stay under the project size limit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--package", default=DEFAULT_PACKAGE, help=f"PyPI project (default: {DEFAULT_PACKAGE})")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"PyPI base URL (default: {DEFAULT_HOST})")
    p.add_argument(
        "--delete-regex",
        default=DEFAULT_DELETE_REGEX,
        help=(
            "Regex of versions to DELETE (default: %(default)r → all 0.1.x–0.5.x). "
            "Ignored when --target-gb is set."
        ),
    )
    p.add_argument(
        "--target-gb",
        type=float,
        default=None,
        help="Size-target mode: delete OLDEST releases until project <= this many GB.",
    )
    p.add_argument(
        "--keep-recent",
        type=int,
        default=12,
        help="Target mode only: always keep this many newest releases (default: 12).",
    )
    p.add_argument(
        "-u",
        "--username",
        default=os.environ.get("PYPI_USERNAME", "zachhandley"),
        help="PyPI username (default: $PYPI_USERNAME or 'zachhandley').",
    )
    p.add_argument(
        "--do-it",
        action="store_true",
        help="Actually delete via pypi-cleanup (prompts for password+OTP). Default is dry-run.",
    )
    args = p.parse_args()

    sizes = fetch_releases(args.host, args.package)
    if not sizes:
        print(f"No releases with files found for {args.package!r}. Nothing to do.")
        return
    total = sum(sizes.values())

    if args.target_gb is not None:
        target_bytes = int(args.target_gb * 1e9)
        delete = select_by_target(sizes, target_bytes, args.keep_recent)
        mode = f"target <= {args.target_gb} GB (keep newest {args.keep_recent})"
    else:
        delete = select_by_regex(sizes, args.delete_regex)
        mode = f"regex {args.delete_regex!r}"

    delete = sorted(set(delete), key=_vkey)
    freed = sum(sizes[v] for v in delete)
    remaining = total - freed
    keep_count = len(sizes) - len(delete)

    print(f"Project : {args.package}  ({args.host})")
    print(f"Mode    : {mode}")
    print(f"Current : {_human(total)} across {len(sizes)} releases")
    print(f"Delete  : {len(delete)} releases, freeing {_human(freed)}")
    print(f"After   : {_human(remaining)} across {keep_count} releases")
    print()
    if not delete:
        print("Nothing matches — no deletions needed.")
        return
    for v in delete:
        print(f"  DELETE {v:14} {_human(sizes[v])}")
    print()

    # Exact-version alternation so pypi-cleanup deletes EXACTLY the previewed set
    # (anchored, dots escaped) — never a broader match than what's shown above.
    exact_regex = "^(" + "|".join(re.escape(v) for v in delete) + ")$"

    if not args.do_it:
        print("DRY RUN — nothing deleted. Re-run with --do-it -u <pypi-user> to apply.")
        print(f"pypi-cleanup regex that would be used:\n  {exact_regex}")
        return

    if not args.username:
        sys.exit("error: --do-it requires -u/--username (your PyPI account name).")
    if not shutil.which("pypi-cleanup"):
        sys.exit(
            "error: pypi-cleanup not found. Install it first:\n"
            "  pip install pypi-cleanup   (or: uv tool install pypi-cleanup)"
        )
    if not os.environ.get("PYPI_CLEANUP_PASSWORD"):
        print(
            "note: PYPI_CLEANUP_PASSWORD not set — pypi-cleanup will prompt for it.\n"
            "      It will also prompt for your TOTP one-time code (mandatory 2FA).",
            file=sys.stderr,
        )

    cmd = [
        "pypi-cleanup",
        "--host", args.host.rstrip("/") + "/",
        "--username", args.username,
        "--package", args.package,
        "--version-regex", exact_regex,
        "--do-it",
    ]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
