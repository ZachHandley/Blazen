# Changelog

## Unreleased

### CI

- **`aarch64-unknown-linux-musl` wheels and napi binary now build.** The
  `rust-musl-cross:aarch64-musl` image ships only long-triple binaries
  (`aarch64-unknown-linux-musl-gcc`), but Rust's target spec defaults the
  linker to the short form (`aarch64-linux-musl-gcc`). Prior attempts to
  override this via `CARGO_TARGET_*_LINKER` in `$GITHUB_ENV` did not
  propagate through `pnpm` / `uvx` to the cargo subprocess on the Forgejo
  runner. The probe step now creates short-triple symlinks in both
  `/usr/local/musl/bin` and `/usr/local/bin` so cargo's default lookup
  succeeds with zero env-var plumbing.
- **`x86_64-pc-windows-msvc` wheels and napi binary now build.**
  `llama-cpp-sys-2`'s `build.rs` unconditionally forwards every `CMAKE_*`
  environment variable as a `-D` flag to cmake. The Windows runner host
  injects `CMAKE_C_COMPILER_LAUNCHER=ccache` / `CMAKE_CXX_COMPILER_LAUNCHER=ccache`
  into the process env from some machine- or user-scope source outside the
  workflow, which then reached cmake, which then made Ninja try to spawn
  `ccache` — failing with `CreateProcess: The system cannot find the file
  specified`. The `Build wheels` and `Build napi binary` steps are now
  split into Windows (PowerShell) and non-Windows (bash) siblings; the
  Windows branch does `Remove-Item Env:CMAKE_*_COMPILER_LAUNCHER` before
  spawning `uvx maturin build` / `pnpm exec napi build`, so child
  processes inherit a clean env and the `-D` flag is never emitted.
- Added `Diagnose inherited env (Windows)` step (permanent) that dumps
  `CMAKE_*` / `*LAUNCHER*` variables at Process, Machine, and User scope
  on every Windows build, so the source of any future launcher leak is
  immediately visible in the log.
