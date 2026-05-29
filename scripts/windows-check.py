#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rich>=13.7",
# ]
# ///
"""
windows-check.py — One-shot MSVC build verification driver for Blazen.

Rsyncs the current tree to the MiniWindows SSH host, runs a `cargo build`
on the target that has historically produced the MSVC LNK2038
`'RuntimeLibrary'` mismatch, fetches the log back, and asserts the
mismatch is absent.

    uv run scripts/windows-check.py
    uv run scripts/windows-check.py --host MiniWindows@<ip>
    uv run scripts/windows-check.py --crate blazen-cabi

Modeled on `/home/zach/github/ZLayer/scripts/windows/debug_e2e.py` but
slimmer — Blazen has no HCN lifecycle to manage; we just need
"build + report".
"""

from __future__ import annotations

import argparse
import base64
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

console = Console(highlight=False)

# Probe order: LAN -> Netbird overlay IP -> Netbird DNS. First responder wins.
# Keep these in sync with the reference_ssh_miniwindows memory note.
DEFAULT_HOSTS = [
    "MiniWindows@192.168.68.92",
    "MiniWindows@100.96.69.45",
    "MiniWindows@miniwindows.net.home",
]

# Default target + crate. Override via --target/--crate.
DEFAULT_TARGET = "x86_64-pc-windows-msvc"
DEFAULT_CRATE = "blazen-uniffi"

# Remote path on the cygwin side of MiniWindows. ZLayer uses /cygdrive/c/src/ZLayer;
# we pick a sibling dir so the two repos don't trample each other.
REMOTE_SRC = "/cygdrive/c/src/Blazen"


def section(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/]"))


def die(msg: str, code: int = 2) -> None:
    console.print(f"[bold red]error:[/] {msg}")
    sys.exit(code)


def stream(
    argv: list[str],
    prefix: str,
    *,
    check: bool = True,
    capture: bool = False,
) -> tuple[int, list[str]]:
    """Run argv, stream stdout/stderr line-by-line, return (rc, lines)."""
    captured: list[str] = []
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as e:
        die(f"{argv[0]} not on PATH: {e}")
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if capture:
            captured.append(line)
        console.print(f"[dim]{prefix}[/] {line}")
    rc = proc.wait()
    if check and rc != 0:
        die(f"{prefix} failed (rc={rc})")
    return rc, captured


def ssh_argv(host: str, remote_cmd: str) -> list[str]:
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        # MiniWindows is shared with ZLayer; under contention it can
        # take 10-20s to accept a new SSH connection. 30s gives us
        # headroom without making "host genuinely down" cases too slow.
        "-o", "ConnectTimeout=30",
        # Keep the SSH session alive during long builds — cargo builds
        # of blazen-uniffi can run 30+ min with no stdout for chunks of
        # cmake work. ServerAlive ping every 30s for up to 60 missed
        # responses = 30 min before considering connection dead.
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=60",
        # Disable TCPKeepAlive — relying on application-level
        # ServerAlive instead, which is more reliable through NAT.
        "-o", "TCPKeepAlive=yes",
        host,
        remote_cmd,
    ]


def detect_host(candidates: list[str]) -> str:
    section("1. Detect host")
    for host in candidates:
        console.print(f"  probe [bold]{host}[/] ...")
        try:
            rc = subprocess.run(
                ssh_argv(host, "echo ok"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # Match ConnectTimeout=30 in ssh_argv + a little slack for
                # shell startup.
                timeout=40,
            ).returncode
        except subprocess.TimeoutExpired:
            console.print(f"    [yellow]timeout[/]")
            continue
        if rc == 0:
            console.print(f"    [green]reachable[/] → using [bold]{host}[/]")
            return host
        console.print(f"    [yellow]no response (rc={rc})[/]")
    die(
        "no MiniWindows host reachable. Tried: " + ", ".join(candidates)
        + ". Override with --host MiniWindows@<addr>."
    )
    raise AssertionError("unreachable")  # for type checker


@dataclass
class BuildSpec:
    crate: str
    target: str

    @property
    def label(self) -> str:
        return f"{self.crate}_{self.target}"


def phase_rsync(local_repo: Path, host: str) -> None:
    section("2. Rsync")
    if shutil.which("rsync") is None:
        die("rsync not on PATH")
    src = str(local_repo).rstrip("/") + "/"
    dst = f"{host}:{REMOTE_SRC}/"
    # `target` is excluded — MiniWindows rebuilds from scratch each invocation.
    # That's slow but reliably reproduces clean-build CI behavior. If iteration
    # speed becomes painful, add a `--keep-target` flag.
    # NOTE: no `-z` (compression). Cygwin's gzip layer on the receiver side
    # closes the SSH socket mid-stream with `rsync rc=12 /
    # connection unexpectedly closed`. Plain `-a` works reliably even on
    # multi-GB trees (verified 2026-05-28 against MiniWindows@192.168.68.92).
    argv = [
        "rsync", "-a", "--delete",
        "--exclude=target",
        "--exclude=.git",
        "--exclude=node_modules",
        "--exclude=*.log",
        "--exclude=.claude",
        src, dst,
    ]
    # Cygwin SSH occasionally drops the connection AFTER the transfer finishes
    # (rc=12 / "connection unexpectedly closed"). Retry once before bailing.
    rc, _ = stream(argv, "[rsync]", check=False)
    if rc != 0:
        console.print(f"[yellow]rsync rc={rc} — retrying once[/]")
        rc, _ = stream(argv, "[rsync-retry]", check=False)
        if rc != 0:
            die(f"rsync failed twice (rc={rc})")


def _ps_encoded_cmd(ps_cmd: str) -> str:
    """Base64-UTF-16-LE encode a PowerShell command for `-EncodedCommand`."""
    return base64.b64encode(ps_cmd.encode("utf-16-le")).decode("ascii")


def _ssh_ps(host: str, ps_cmd: str, *, check: bool = True) -> tuple[int, list[str]]:
    """Run a PowerShell command on `host` via SSH using `-EncodedCommand`.

    The encoded-command path bypasses PowerShell's argument parser entirely,
    so embedded backslashes / single-quoted paths round-trip safely.
    """
    remote = " ".join([
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-EncodedCommand", _ps_encoded_cmd(ps_cmd),
    ])
    return stream(ssh_argv(host, remote), "[ssh]", check=check, capture=True)


REMOTE_RUN_DIR = "C:\\src\\Blazen-runs"
REMOTE_LOG = f"{REMOTE_RUN_DIR}\\windows-check.log"
REMOTE_EXIT = f"{REMOTE_RUN_DIR}\\windows-check.exit"


def phase_build(host: str, spec: BuildSpec) -> tuple[int, list[str]]:
    """Launch cargo as a detached PowerShell process on the remote, poll for
    completion via short SSH connections, fetch the log when done.

    Why detached: MiniWindows is shared with ZLayer; long-running in-band
    SSH sessions get killed by TCP-layer disconnects under contention.
    Detaching cargo into its own background powershell.exe + polling via
    short SSH connections survives any number of mid-build disconnects.
    """
    import time as _time

    section(f"3. Launch detached build of {spec.crate} ({spec.target})")

    # The inner PowerShell script that actually runs cargo.
    # `Tee-Object` captures cargo's combined stdout+stderr to a log file
    # while still letting cargo write to its own stdout (which is now a
    # detached console). Writing $LASTEXITCODE to the .exit file is the
    # poll-sentinel.
    inner_ps = (
        f"Set-Location 'C:\\src\\Blazen'\n"
        f"rustup target add {spec.target} 2>&1 | Out-Null\n"
        f"$ErrorActionPreference = 'Continue'\n"
        # 2>&1 redirects native-command stderr into stdout so the Tee
        # captures everything. Without it, link errors land on stderr
        # and never reach the log.
        #
        # --offline: the MiniWindows host intermittently wedges on cargo's
        # `Updating crates.io index` step — cargo sits at ~0 CPU for 40+ min
        # with no output (TCP to index.crates.io is open, but the sparse-index
        # HTTP fetch stalls), so the build never starts. The registry cache and
        # Cargo.lock are rsync'd from the (already-resolved) host, so --offline
        # bypasses the index entirely and goes straight to compiling. If a crate
        # is genuinely missing from the cache, --offline fails fast with a clear
        # message instead of hanging.
        f"cargo build -p {spec.crate} --release --offline --target {spec.target} 2>&1 "
        f"| Tee-Object -FilePath '{REMOTE_LOG}' -Encoding utf8\n"
        f"$LASTEXITCODE | Out-File -Encoding ascii -FilePath '{REMOTE_EXIT}'\n"
    )
    inner_b64 = _ps_encoded_cmd(inner_ps)

    # Launcher script. Clears prior sentinels, then `Start-Process` spawns
    # cargo in a hidden, detached PowerShell. The launcher exits immediately
    # so the SSH connection closes cleanly.
    launcher = (
        f"New-Item -ItemType Directory -Force -Path '{REMOTE_RUN_DIR}' | Out-Null; "
        f"Remove-Item -Force -ErrorAction SilentlyContinue '{REMOTE_LOG}'; "
        f"Remove-Item -Force -ErrorAction SilentlyContinue '{REMOTE_EXIT}'; "
        f"Start-Process -FilePath 'powershell.exe' "
        f"-ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand','{inner_b64}' "
        f"-WindowStyle Hidden; "
        f"Write-Host 'LAUNCHED'"
    )
    rc, lines = _ssh_ps(host, launcher, check=False)
    if rc != 0 or not any("LAUNCHED" in line for line in lines):
        die(f"failed to launch detached build (rc={rc})")

    # Poll. Each iteration is a fresh SSH connection -> short-lived -> NAT
    # eviction can't kill it.
    section("3b. Poll for build completion")
    poll_ps = (
        f"if (Test-Path '{REMOTE_EXIT}') {{ "
        f"  $code = (Get-Content '{REMOTE_EXIT}' -Raw).Trim(); "
        f"  Write-Host \"DONE:$code\" "
        f"}} elseif (Test-Path '{REMOTE_LOG}') {{ "
        f"  $tail = Get-Content '{REMOTE_LOG}' -Tail 1 -ErrorAction SilentlyContinue; "
        f"  Write-Host \"RUNNING:$tail\" "
        f"}} else {{ "
        f"  Write-Host 'STARTING' "
        f"}}"
    )

    BUILD_BUDGET_SEC = 90 * 60   # 90-min total budget
    POLL_INTERVAL_SEC = 20
    LOG_INTERVAL_SEC = 60        # only print "still running" once a minute

    deadline = _time.time() + BUILD_BUDGET_SEC
    last_log = 0.0
    exit_code: int | None = None
    consecutive_ssh_fails = 0

    while _time.time() < deadline:
        rc, lines = _ssh_ps(host, poll_ps, check=False)
        if rc != 0:
            consecutive_ssh_fails += 1
            if consecutive_ssh_fails > 30:
                # 30 * 20s = 10min of unreachable host
                die("MiniWindows unreachable for 10 minutes during poll")
            _time.sleep(POLL_INTERVAL_SEC)
            continue
        consecutive_ssh_fails = 0
        status = next(
            (l.strip() for l in lines if l.strip().startswith(("DONE:", "RUNNING:", "STARTING"))),
            "",
        )
        if status.startswith("DONE:"):
            try:
                exit_code = int(status.split(":", 1)[1].strip())
            except ValueError:
                exit_code = -1
            console.print(f"  [bold green]build finished (exit={exit_code})[/]")
            break
        now = _time.time()
        if now - last_log > LOG_INTERVAL_SEC:
            console.print(f"  [dim]{status[:140]}[/]")
            last_log = now
        _time.sleep(POLL_INTERVAL_SEC)
    else:
        die(f"build did not complete within {BUILD_BUDGET_SEC // 60} minutes")

    # Fetch the log. Big logs need base64 transport.
    section("3c. Fetch remote build log")
    fetch_ps = (
        f"if (Test-Path '{REMOTE_LOG}') {{ "
        f"  [Convert]::ToBase64String([IO.File]::ReadAllBytes('{REMOTE_LOG}')) "
        f"}} else {{ Write-Host 'NO_LOG' }}"
    )
    rc, lines = _ssh_ps(host, fetch_ps, check=False)
    if rc != 0:
        die("failed to fetch remote build log")
    # The base64 output may span many lines (PowerShell wraps long output);
    # join all non-warning lines into one blob.
    b64 = "".join(
        line.strip()
        for line in lines
        if line.strip() and not line.startswith(("**", "[ssh]"))
    )
    if b64.startswith("NO_LOG"):
        die("remote log missing — build never wrote any output")
    try:
        log_bytes = base64.b64decode(b64)
    except Exception as e:
        die(f"failed to decode remote log ({len(b64)} chars): {e}")
    assert exit_code is not None
    log_text = log_bytes.decode("utf-8", errors="replace")
    return exit_code, log_text.splitlines()


def write_log(local_repo: Path, spec: BuildSpec, lines: list[str]) -> Path:
    section("4. Save log")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs_dir = local_repo / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"windows_{ts}_{spec.label}.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"  wrote [bold]{log_path}[/] ({len(lines)} lines)")
    return log_path


def summarize(log_path: Path, build_rc: int) -> int:
    section("5. Summarize")
    content = log_path.read_text(encoding="utf-8", errors="replace")
    # The exact failure mode we're trying to prevent. Match either the LNK2038
    # error code OR the human-readable "RuntimeLibrary" mismatch text.
    mismatch_markers = ["LNK2038", "'RuntimeLibrary'", "MT_StaticRelease", "MD_DynamicRelease"]
    hits: list[str] = []
    for line in content.splitlines():
        if any(m in line for m in mismatch_markers):
            hits.append(line.strip())

    if hits:
        console.print(f"[bold red]LNK2038 / RuntimeLibrary mismatch detected ({len(hits)} matches):[/]")
        for line in hits[:20]:
            console.print(f"  [red]{line}[/]")
        if len(hits) > 20:
            console.print(f"  [dim]... and {len(hits) - 20} more in {log_path}[/]")
        return 1

    if build_rc != 0:
        console.print(f"[bold yellow]Build failed (rc={build_rc}) BUT the CRT mismatch is NOT the cause.[/]")
        console.print(f"  Inspect the full log at [bold]{log_path}[/] for the actual failure.")
        return build_rc

    console.print("[bold green]PASS: CRT alignment confirmed, build succeeded.[/]")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--host",
        default=None,
        help="SSH target (e.g. MiniWindows@1.2.3.4). When omitted, probes "
             "LAN/Netbird candidates in order.",
    )
    p.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Rust target triple (default: {DEFAULT_TARGET})",
    )
    p.add_argument(
        "--crate",
        default=DEFAULT_CRATE,
        help=f"Crate to build with `-p` (default: {DEFAULT_CRATE})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    local_repo = Path(__file__).resolve().parent.parent
    if not (local_repo / "Cargo.toml").is_file():
        die(f"could not locate workspace root from {local_repo}")
    console.print(f"[dim]repo: {local_repo}[/]")

    host = args.host or detect_host(DEFAULT_HOSTS)
    spec = BuildSpec(crate=args.crate, target=args.target)

    phase_rsync(local_repo, host)
    build_rc, build_lines = phase_build(host, spec)
    log_path = write_log(local_repo, spec, build_lines)
    return summarize(log_path, build_rc)


if __name__ == "__main__":
    sys.exit(main())
