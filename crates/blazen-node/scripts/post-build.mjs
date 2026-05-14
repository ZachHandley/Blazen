// Post-build script: patches the napi-rs-generated `index.js` and
// `index.d.ts` to:
//
//   1. Add type aliases mirroring blazen-llm (`MediaSource`, `ImageSource`).
//   2. Wrap every exported function and class method so thrown napi errors
//      are upgraded to typed JS error classes (see `error-classes.js`).
//   3. Re-export the typed error classes (`BlazenError`, `ProviderError`,
//      ...) from the package's main entry so consumers can do
//      `const { BlazenError } = require('blazen')`.
//   4. Append `.d.ts` declarations for the error classes.
//
// Idempotent: each section is gated by a sentinel marker so re-running
// after another `napi build` only re-applies sections that were wiped
// when the napi-rs generator overwrote the files. A single `napi build`
// run regenerates `index.js` from scratch but leaves `index.d.ts` mostly
// alone -- both cases are handled.

import { appendFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs'

const dtsPath = new URL('../index.d.ts', import.meta.url)
const jsPath = new URL('../index.js', import.meta.url)
const wasiBrowserPath = new URL('../blazen.wasi-browser.js', import.meta.url)
const workersJsPath = new URL('../blazen.workers.js', import.meta.url)
const workersDtsPath = new URL('../blazen.workers.d.ts', import.meta.url)

// ---------------------------------------------------------------------------
// Section 1: type aliases on index.d.ts
// ---------------------------------------------------------------------------

{
  const current = readFileSync(dtsPath, 'utf8')
  const aliases = [
    'export type MediaSource = JsImageSource',
    'export type ImageSource = JsImageSource',
    'export type ContentHandle = JsContentHandle',
    'export type ContentMetadata = JsContentMetadata',
    'export type ContentKind = JsContentKind',
  ]
  const banner = '\n// --- post-build: type aliases mirroring blazen-llm ---\n'
  const missing = aliases.filter((line) => !current.includes(line))
  if (missing.length > 0) {
    const block = `${current.endsWith('\n') ? '' : '\n'}${banner}${missing.join('\n')}\n`
    appendFileSync(dtsPath, block)
  }
}

// ---------------------------------------------------------------------------
// Section 1b: ContentBody / ContentHint helper types referenced by
// `CustomContentStoreOptions` callback signatures (`#[napi(ts_type = ...)]`
// on the Rust side names these types but does not declare them).
// ---------------------------------------------------------------------------

{
  const current = readFileSync(dtsPath, 'utf8')
  const sentinel = '// --- post-build: ContentBody / ContentHint helper types ---'
  if (!current.includes(sentinel)) {
    const decls = [
      '/**',
      ' * JSON-tagged body passed to a custom store\'s `put` callback.',
      ' *',
      ' * Mirrors `blazen_llm::content::ContentBody`. The `stream` variant',
      ' * carries a live `AsyncIterable<Uint8Array>` so chunks flow lazily',
      ' * from Rust into JS without staging the whole payload in memory.',
      ' */',
      'export type ContentBody =',
      '  | { type: \'bytes\'; data: Buffer | Uint8Array | number[] }',
      '  | { type: \'url\'; url: string }',
      '  | { type: \'local_path\'; path: string }',
      '  | { type: \'provider_file\'; provider: string; id: string }',
      '  | { type: \'stream\'; stream: AsyncIterable<Uint8Array>; sizeHint: number | null }',
      '/**',
      ' * Optional hints passed alongside a `ContentBody` into `put`.',
      ' * Mirrors `blazen_llm::content::ContentHint` (every field optional).',
      ' */',
      'export interface ContentHint {',
      '  mimeType?: string | null',
      '  kind?: ContentKind | null',
      '  displayName?: string | null',
      '  byteSize?: number | null',
      '}',
    ]
    const banner = `\n${sentinel}\n`
    const block = `${current.endsWith('\n') ? '' : '\n'}${banner}${decls.join('\n')}\n`
    appendFileSync(dtsPath, block)
  }
}

// ---------------------------------------------------------------------------
// Section 2: append typed-error wrapping + re-exports to index.js
// ---------------------------------------------------------------------------
//
// Strategy: instead of editing every individual `module.exports.X = ...`
// line, we append a block at the END of `index.js` that:
//   - imports `./error-classes.js` (the typed-error shim);
//   - iterates over the current `module.exports` and replaces every
//     function with a wrapper that calls `enrichError` on throw / reject;
//   - for every constructor (class), wraps each enumerable prototype
//     method the same way;
//   - assigns the typed error classes onto `module.exports` so consumers
//     can do `require('blazen').BlazenError`.
//
// Sentinel: `// --- post-build: typed-error wrapping ---`. If present,
// skip re-appending.

{
  const current = readFileSync(jsPath, 'utf8')
  const sentinel = '// --- post-build: typed-error wrapping ---'
  if (!current.includes(sentinel)) {
    const block = `
${sentinel}
;(() => {
  const errorClasses = require('./error-classes.js')
  const { enrichError } = errorClasses

  // Wrap a single function so any thrown error -- sync or async -- is
  // passed through enrichError before reaching the caller.
  const wrap = (fn) => {
    return function blazenWrapped(...args) {
      try {
        const result = fn.apply(this, args)
        if (result && typeof result.then === 'function') {
          return result.then(
            (v) => v,
            (e) => {
              throw enrichError(e)
            },
          )
        }
        return result
      } catch (e) {
        throw enrichError(e)
      }
    }
  }

  // Patch a class's prototype methods in-place. Skips the constructor
  // and any non-function or non-configurable property.
  const patchPrototype = (Cls) => {
    if (typeof Cls !== 'function') return
    const proto = Cls.prototype
    if (!proto || typeof proto !== 'object') return
    for (const key of Object.getOwnPropertyNames(proto)) {
      if (key === 'constructor') continue
      const desc = Object.getOwnPropertyDescriptor(proto, key)
      if (!desc || !desc.configurable) continue
      if (typeof desc.value !== 'function') continue
      try {
        Object.defineProperty(proto, key, {
          ...desc,
          value: wrap(desc.value),
        })
      } catch {
        // Some napi prototypes are frozen; skip gracefully.
      }
    }
  }

  // Wrap every top-level export. Distinguish functions (call wrap) from
  // constructors (call patchPrototype). Heuristic: a constructor has a
  // \`prototype\` object with own properties beyond \`constructor\`. When
  // in doubt we patch the prototype AND wrap the function -- wrapping a
  // constructor call in try/catch is safe (\`new wrappedCtor()\` still
  // works because \`fn.apply(this, args)\` on a constructor called with
  // \`new\` would normally fail, but napi-rs class constructors are
  // exposed as plain factories that do not require \`new\`).
  for (const key of Object.keys(module.exports)) {
    const orig = module.exports[key]
    if (typeof orig !== 'function') continue
    // Skip the typed-error classes we are about to install.
    if (Object.prototype.hasOwnProperty.call(errorClasses, key)) continue
    if (orig.prototype && typeof orig.prototype === 'object') {
      patchPrototype(orig)
    }
    // Only wrap "plain" functions (lowercase first letter convention) and
    // any function whose prototype is empty (i.e. not a class). napi-rs
    // emits classes with PascalCase names; we leave those callable as-is
    // so \`new ClassName(...)\` keeps working but their methods are
    // already patched above.
    //
    // Detect "class-ness" via either prototype methods OR own static
    // properties beyond the built-in function metadata fields. Without the
    // static-property check, a class whose only public surface is a static
    // factory (e.g. \`UpstashBackend.create\`) would slip through and get
    // replaced with a \`wrap()\`-returned function that loses the static
    // method.
    const FUNCTION_BUILTIN_PROPS = new Set([
      'length',
      'name',
      'arguments',
      'caller',
      'prototype',
    ])
    const ownStaticPropNames = Object.getOwnPropertyNames(orig).filter(
      (p) => !FUNCTION_BUILTIN_PROPS.has(p),
    )
    const hasPrototypeMethods =
      orig.prototype &&
      Object.getOwnPropertyNames(orig.prototype).some((p) => p !== 'constructor')
    const isLikelyClass = hasPrototypeMethods || ownStaticPropNames.length > 0
    if (!isLikelyClass) {
      module.exports[key] = wrap(orig)
    }
  }

  // Re-export the typed error classes + enrichError.
  for (const [name, value] of Object.entries(errorClasses)) {
    module.exports[name] = value
  }
})()
`
    appendFileSync(jsPath, block)
  }
}

// ---------------------------------------------------------------------------
// Section 3: append .d.ts declarations for the typed error classes
// ---------------------------------------------------------------------------

{
  const current = readFileSync(dtsPath, 'utf8')
  const sentinel = '// --- post-build: typed error classes ---'
  if (!current.includes(sentinel)) {
    const decls = [
      'export class BlazenError extends Error {}',
      'export class AuthError extends BlazenError {}',
      'export class RateLimitError extends BlazenError {}',
      'export class TimeoutError extends BlazenError {}',
      'export class ValidationError extends BlazenError {}',
      'export class ContentPolicyError extends BlazenError {}',
      'export class UnsupportedError extends BlazenError {}',
      'export class ComputeError extends BlazenError {}',
      'export class MediaError extends BlazenError {}',
      'export class ProviderError extends BlazenError {',
      '  provider: string | null',
      '  status: number | null',
      '  endpoint: string | null',
      '  requestId: string | null',
      '  detail: string | null',
      '  retryAfterMs: number | null',
      '}',
      'export class LlamaCppError extends ProviderError {}',
      'export class LlamaCppInvalidOptionsError extends LlamaCppError {}',
      'export class LlamaCppModelLoadError extends LlamaCppError {}',
      'export class LlamaCppInferenceError extends LlamaCppError {}',
      'export class LlamaCppEngineNotAvailableError extends LlamaCppError {}',
      'export class CandleLlmError extends ProviderError {}',
      'export class CandleLlmInvalidOptionsError extends CandleLlmError {}',
      'export class CandleLlmModelLoadError extends CandleLlmError {}',
      'export class CandleLlmInferenceError extends CandleLlmError {}',
      'export class CandleLlmEngineNotAvailableError extends CandleLlmError {}',
      'export class CandleEmbedError extends ProviderError {}',
      'export class CandleEmbedInvalidOptionsError extends CandleEmbedError {}',
      'export class CandleEmbedModelLoadError extends CandleEmbedError {}',
      'export class CandleEmbedEmbeddingError extends CandleEmbedError {}',
      'export class CandleEmbedEngineNotAvailableError extends CandleEmbedError {}',
      'export class CandleEmbedTaskPanickedError extends CandleEmbedError {}',
      'export class MistralRsError extends ProviderError {}',
      'export class MistralRsInvalidOptionsError extends MistralRsError {}',
      'export class MistralRsInitError extends MistralRsError {}',
      'export class MistralRsInferenceError extends MistralRsError {}',
      'export class MistralRsEngineNotAvailableError extends MistralRsError {}',
      'export class WhisperError extends ProviderError {}',
      'export class WhisperInvalidOptionsError extends WhisperError {}',
      'export class WhisperModelLoadError extends WhisperError {}',
      'export class WhisperTranscriptionError extends WhisperError {}',
      'export class WhisperEngineNotAvailableError extends WhisperError {}',
      'export class WhisperIoError extends WhisperError {}',
      'export class PiperError extends ProviderError {}',
      'export class PiperInvalidOptionsError extends PiperError {}',
      'export class PiperModelLoadError extends PiperError {}',
      'export class PiperSynthesisError extends PiperError {}',
      'export class PiperEngineNotAvailableError extends PiperError {}',
      'export class DiffusionError extends ProviderError {}',
      'export class DiffusionInvalidOptionsError extends DiffusionError {}',
      'export class DiffusionModelLoadError extends DiffusionError {}',
      'export class DiffusionGenerationError extends DiffusionError {}',
      'export class FastEmbedError extends ProviderError {}',
      'export class EmbedUnknownModelError extends FastEmbedError {}',
      'export class EmbedInitError extends FastEmbedError {}',
      'export class EmbedEmbedError extends FastEmbedError {}',
      'export class EmbedMutexPoisonedError extends FastEmbedError {}',
      'export class EmbedTaskPanickedError extends FastEmbedError {}',
      'export class TractError extends ProviderError {}',
      'export class PeerEncodeError extends BlazenError {}',
      'export class PeerTransportError extends BlazenError {}',
      'export class PeerEnvelopeVersionError extends BlazenError {}',
      'export class PeerWorkflowError extends BlazenError {}',
      'export class PeerTlsError extends BlazenError {}',
      'export class PeerUnknownStepError extends BlazenError {}',
      'export class PersistError extends BlazenError {}',
      'export class PromptError extends BlazenError {}',
      'export class PromptMissingVariableError extends PromptError {}',
      'export class PromptNotFoundError extends PromptError {}',
      'export class PromptVersionNotFoundError extends PromptError {}',
      'export class PromptIoError extends PromptError {}',
      'export class PromptYamlError extends PromptError {}',
      'export class PromptJsonError extends PromptError {}',
      'export class PromptValidationError extends PromptError {}',
      'export class MemoryError extends BlazenError {}',
      'export class MemoryNoEmbedderError extends MemoryError {}',
      'export class MemoryElidError extends MemoryError {}',
      'export class MemoryEmbeddingError extends MemoryError {}',
      'export class MemoryNotFoundError extends MemoryError {}',
      'export class MemorySerializationError extends MemoryError {}',
      'export class MemoryIoError extends MemoryError {}',
      'export class MemoryBackendError extends MemoryError {}',
      'export class CacheError extends BlazenError {}',
      'export class DownloadError extends CacheError {}',
      'export class CacheDirError extends CacheError {}',
      'export class IoError extends CacheError {}',
      'export declare function enrichError(err: unknown): unknown',
    ]
    const banner = `\n${sentinel}\n`
    const block = `${current.endsWith('\n') ? '' : '\n'}${banner}${decls.join('\n')}\n`
    appendFileSync(dtsPath, block)
  }
}

// ---------------------------------------------------------------------------
// Section 4: emit blazen.workers.js -- the `blazen/workers` subpath entry
// ---------------------------------------------------------------------------
//
// napi-rs's auto-generated `blazen.wasi-browser.js` does
// `new URL('./blazen.wasm32-wasi.wasm', import.meta.url).href` at module init.
// In workerd-bundled chunks `import.meta.url` is empty, so the URL constructor
// throws `Invalid URL string` before any user code runs. The auto-generated
// `index.js` is also unusable on edge runtimes because of its static-require
// per-platform detection chain.
//
// Consumers on Cloudflare Workers / SvelteKit / Astro+@astrojs/cloudflare are
// expected to `import { ... } from 'blazen/workers'` -- that subpath resolves
// here via the `exports` map in package.json. This file uses wrangler's
// static `import wasm from '...wasm'` form against the wasm32-wasi
// subpackage's bundled .wasm file (subpath exposed via the staging script
// locally and `.forgejo/workflows/release.yaml` in CI). Wrangler / the
// `@cloudflare/vite-plugin` resolve that to a real `WebAssembly.Module` at
// bundle time -- no URL, no fetch, no top-level await, no platform
// detection.
//
// Generated from `blazen.wasi-browser.js` so the ~400 `export const X` lines
// stay in lockstep with whatever napi-rs emits this build. We keep the entire
// exports section byte-identical and only rewrite the init prelude.
//
// Also emits `blazen.workers.d.ts` -- a one-liner `export * from './index'`
// so the TypeScript surface for `blazen/workers` is identical to `blazen`.
//
// Idempotent: re-running rewrites both files. The wasi-browser source is
// regenerated by `napi build` every run so we always want to regenerate from
// it -- no sentinel check.
{
  if (!existsSync(wasiBrowserPath)) {
    // eslint-disable-next-line no-console
    console.warn('[post-build] skipping workers entry: wasi-browser.js missing (no wasm32-wasi target built)')
  } else {
    const wasiBrowser = readFileSync(wasiBrowserPath, 'utf8')

    // Split at the boundary: everything before `export default __napiModule.exports`
    // is napi-rs's init block (we replace it); everything from that line onward
    // is the ~400-line export list (we keep it).
    const splitMarker = 'export default __napiModule.exports'
    const splitIdx = wasiBrowser.indexOf(splitMarker)
    if (splitIdx === -1) {
      throw new Error(
        '[post-build] blazen.wasi-browser.js shape changed: missing `export default __napiModule.exports` marker. Update post-build.mjs Section 4.',
      )
    }
    const exportsSection = wasiBrowser.slice(splitIdx)

    const workersPrelude = `// auto-generated by scripts/post-build.mjs from blazen.wasi-browser.js
// DO NOT EDIT -- regenerated every \`napi build\`.
//
// The \`blazen/workers\` subpath entry for Cloudflare Workers (workerd) and
// any other edge runtime that supports wrangler-style static .wasm imports.
// Consumers do \`import { ... } from 'blazen/workers'\` instead of
// \`from 'blazen'\`; they must also install
// \`@blazen-dev/blazen-wasm32-wasi\` directly (an optional peerDependency on
// the parent \`blazen\` package).
//
// Loads the wasm via wrangler's static \`import wasm from '@scope/pkg/foo.wasm'\`
// form, which both wrangler >=3.15 and \`@cloudflare/vite-plugin\` resolve to
// a real WebAssembly.Module at bundle time. No \`import.meta.url\`, no fetch,
// no top-level await, no per-platform require() chain.

import {
  getDefaultContext as __emnapiGetDefaultContext,
  instantiateNapiModuleSync as __emnapiInstantiateNapiModuleSync,
  WASI as __WASI,
} from '@napi-rs/wasm-runtime'

// Subpath import on the wasm32-wasi subpackage. The subpackage's package.json
// exposes \`./blazen.wasm32-wasi.wasm\` in its exports map (added by the
// release pipeline and by \`stage-local-blazen.sh\` for dev).
import __wasmModule from '@blazen-dev/blazen-wasm32-wasi/blazen.wasm32-wasi.wasm'

// Cloudflare Workers / workerd disallows dynamic code generation
// (\`new Function\`, \`eval\`, \`napi_run_script\`). Blazen's WASI async
// dispatcher (\`crates/blazen-node/src/wasi_async.rs\`) needs a microtask
// scheduler and a setTimeout-based sleeper, normally built via
// \`env.run_script\`. We predefine them here as globals so the Rust
// \`install()\` function can read them via \`globalThis\` instead of
// invoking the JS engine's eval, which workerd blocks.
globalThis.__blazenScheduler = () => {
  Promise.resolve().then(() => globalThis.__blazenDrainAsyncQueue())
}
globalThis.__blazenSleeper = (ms) =>
  new Promise((r) => globalThis.setTimeout(r, ms))

const __wasi = new __WASI({ version: 'preview1' })
const __emnapiContext = __emnapiGetDefaultContext()

const __sharedMemory = new WebAssembly.Memory({
  initial: 4000,
  maximum: 65536,
  shared: true,
})

const {
  instance: __napiInstance,
  module: __wasiModule,
  napiModule: __napiModule,
} = __emnapiInstantiateNapiModuleSync(__wasmModule, {
  context: __emnapiContext,
  asyncWorkPoolSize: 0, // workerd has no Worker class
  wasi: __wasi,
  overwriteImports(importObject) {
    importObject.env = {
      ...importObject.env,
      ...importObject.napi,
      ...importObject.emnapi,
      memory: __sharedMemory,
    }
    return importObject
  },
  beforeInit({ instance }) {
    for (const name of Object.keys(instance.exports)) {
      if (name.startsWith('__napi_register__')) {
        instance.exports[name]()
      }
    }
  },
})

`

    writeFileSync(workersJsPath, workersPrelude + exportsSection)

    // Type surface is identical to the main entry -- re-export everything.
    writeFileSync(
      workersDtsPath,
      "// auto-generated by scripts/post-build.mjs -- DO NOT EDIT.\nexport * from './index'\n",
    )
  }
}

// Verify final state and write a small status line so build logs are
// informative without being noisy.
{
  const finalDts = readFileSync(dtsPath, 'utf8')
  const finalJs = readFileSync(jsPath, 'utf8')
  const has = (haystack, needle) => haystack.includes(needle)
  const status = {
    'd.ts: type aliases': has(finalDts, 'export type MediaSource = JsImageSource'),
    'd.ts: content helpers': has(
      finalDts,
      '// --- post-build: ContentBody / ContentHint helper types ---',
    ),
    'd.ts: error classes': has(finalDts, '// --- post-build: typed error classes ---'),
    'js: error wrapping': has(finalJs, '// --- post-build: typed-error wrapping ---'),
    'js: workers entry': existsSync(workersJsPath),
    'd.ts: workers shim': existsSync(workersDtsPath),
  }
  // eslint-disable-next-line no-console
  console.log(`[post-build] ${JSON.stringify(status)}`)
}
