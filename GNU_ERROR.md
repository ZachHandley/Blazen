# manylinux_2_28 GNU wheel failure — cause + options

## The error

```
maturin failed
Caused by: Error ensuring manylinux_2_28 compliance
Caused by: Your library is not manylinux_2_28 compliant because of the
  presence of too-recent versioned symbols:
  ["libstdc++.so.6 offending versions: GLIBCXX_3.4.25"]
```

`build-wheels (x86_64-unknown-linux-gnu)` and `build-wheels (aarch64-unknown-linux-gnu)` only. musl, alpine, macOS, Windows, napi — all fine.

## What's actually wrong (verified)

**maturin's bundled manylinux_2_28 policy is too strict.** I pulled `src/auditwheel/manylinux-policy.json` from `PyO3/maturin` at tags v1.10.0, v1.11.0, v1.12.0, v1.13.0, and v1.13.1. Every released maturin version caps `manylinux_2_28` GLIBCXX at **3.4.24**:

```json
"manylinux_2_28": {
  "x86_64": {
    "GLIBCXX": [..., "3.4.22", "3.4.23", "3.4.24"]
  }
}
```

The actual `manylinux_2_28` spec — and `pypa/auditwheel`'s real policy — allows GLIBCXX up to **3.4.25**, which is what stock GCC 8 (the manylinux_2_28 base image's system compiler) emits for ordinary C++17 code (`__throw_bad_array_new_length`, etc.). The wheel is genuinely manylinux_2_28-compliant; maturin's bundled policy is one micro-version behind. Upstream main has the same cap. There is no maturin upgrade path.

## Why this shows up *now* (best honest read)

The two relevant facts:

1. The `[release] Build-artifacts fixed (protoc) for musl targets` commit (df7c412b) — landed earlier in this session — was the **first** time the manylinux gnu wheel build actually compiled the workspace end-to-end. Before that, blazen-peer's build.rs blew up on missing `protoc` and the build never reached maturin's compliance step.
2. blazen-py's `default = ["local-all"]` (already present at v0.1.150) pulls in llama-cpp-sys-2, whisper-rs-sys, candle-*-sys, piper-sys, mistralrs — all of which compile substantial C++17. C++17 stdlib features generate GLIBCXX_3.4.25 references; that's expected on AlmaLinux 8 / manylinux_2_28.

So: the maturin/policy gap has been there all along, but until protoc was unblocked the gnu manylinux build couldn't get far enough to trip it. **It's not a regression in any of our deps; it's a latent bug in maturin's policy file that becomes visible the moment the build starts producing real C++ object files.**

I did **not** verify that v0.1.150's manylinux gnu wheel actually shipped — given protoc was missing and blazen-peer was already in v0.1.150's tree, my best guess is the prior release either skipped the manylinux gnu artifact or its CI run failed too. I can confirm this by pulling the v0.1.150 wheel package list from forge.blackleafdigital.com if you want hard evidence rather than inference.

## Fix options

| Option | What it does | Cost |
|---|---|---|
| **A. `--auditwheel skip` + run real `auditwheel repair` afterward** | maturin builds the wheel without running its (buggy) audit; we then call the auditwheel binary that ships in the manylinux_2_28 base image to verify against the *correct* policy and bundle external libs as needed. Wheel ends up tagged manylinux_2_28 normally. | ~25 lines of shell added to one workflow step. Pure additive — no compatibility loss. The textbook PyPA pattern for this exact failure mode. |
| **B. Bump compatibility to `manylinux_2_34`** | Change `manylinux: "2_28"` to `"2_34"` in the matrix. maturin's 2_34 policy caps GLIBCXX at 3.4.29, which clears 3.4.25. | One-line workflow change. **But** drops support for RHEL 8 / AlmaLinux 8 / Rocky 8 (glibc 2.28), Ubuntu 18.04/20.04, Debian 10/11, Amazon Linux 2 — a real audience for a Python LLM library. |
| C. Static-link libstdc++ for every C++ dep | Add `-static-libstdc++` link flags so libstdc++ symbols are inlined and never appear as version requirements. | Touches build.rs / CFLAGS for every C++-bound crate (llama-cpp-sys-2, whisper-rs-sys, candle-*-sys, piper-sys, ort-sys). High maintenance burden, easy to break locally. |
| D. Patch maturin's policy.json in the Dockerfile | Rebuild maturin from source with a corrected policy.json. | Forks a vendored maturin; future upgrades have to re-apply the patch. Not worth it. |

## Recommendation

**Option A.** The real auditwheel binary ships in the manylinux_2_28 container at `/opt/python/cp310-cp310/bin/auditwheel`. The pattern is exactly what PyPA documents for this exact maturin/auditwheel disagreement. Doesn't change wheel compatibility, doesn't fork tooling, doesn't lose users. Pseudocode:

```bash
# In the manylinux container only
uvx maturin build --release ... --auditwheel skip ...
auditwheel repair --plat "manylinux_2_28_$(uname -m)" -w wheels-repaired/ wheels/*.whl
mv wheels-repaired/*.whl wheels/
```

If you'd rather just bump to manylinux_2_34 and stop fighting the policy mismatch, say so — that's Option B and it's also reasonable, just a compatibility tradeoff.
