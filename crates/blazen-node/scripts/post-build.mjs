// Post-build script: patches the napi-rs-generated `index.js` and
// `index.d.ts` to:
//
//   1. Add type aliases mirroring blazen-llm (`MediaSource`, `ImageSource`).
//   1b. Inject ContentBody / ContentHint helper types.
//   3. Append `.d.ts` declarations for the runtime-registered error classes
//      (BlazenError + 70+ subclasses). The classes themselves are constructed
//      at module load by `crates/blazen-node/src/error_classes.rs::register_all_classes`
//      via napi-patched's `register_error_class` API; we just need TypeScript
//      to know they exist.
//   4. Emit the `blazen/workers` subpath entry for Cloudflare workerd.
//
// **Section 2 (JS-side error wrapping)** used to live here. It was deleted
// when napi 3.9.0 + napi-patched's `register_error_class` made it possible
// to construct typed subclasses natively from Rust. Errors emitted by Rust
// via `napi::Error::with_class("ProviderError", reason).with_field(...)`
// arrive on the JS side as proper instances of the registered class with
// structured fields as own properties -- no JS-side enrichError/sentinel
// parsing required.
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
    // Control-plane: napi-rs emits `Js*`-prefixed names for `#[napi(object)]`
    // shapes; mirror them under their unprefixed equivalents so consumers
    // can write `Assignment`, `RunEvent`, etc. directly.
    'export type WorkerCapability = JsWorkerCapability',
    'export type AdmissionMode = JsAdmissionMode',
    'export type Assignment = JsAssignment',
    'export type RunStatus = JsRunStatus',
    'export type RunStateSnapshot = JsRunStateSnapshot',
    'export type RunEvent = JsRunEvent',
    'export type WorkerInfo = JsWorkerInfo',
    'export type MtlsOptions = JsMtlsOptions',
    'export type ClientConnectOptions = JsClientConnectOptions',
    'export type SubmitWorkflowOptions = JsSubmitWorkflowOptions',
    'export type SubscribeAllOptions = JsSubscribeAllOptions',
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
// Section 2b: cjs-module-lexer hints for runtime-registered error classes
// ---------------------------------------------------------------------------
//
// The native module's `module_exports` hook (see
// `crates/blazen-node/src/lib.rs::module_exports`) binds every JS error
// class onto `module.exports` at module load. That's enough for
// `require('blazen').BlazenError` to work, but it does NOT teach Node's
// `cjs-module-lexer` (which statically analyses CJS modules to compute the
// ESM named-export set) that these symbols exist -- so
// `import { BlazenError } from 'blazen'` resolves to `undefined`.
//
// Fix: append literal `module.exports.X = module.exports.X` self-assignments
// for every registered class. At runtime these are no-ops; at parse time the
// lexer detects them and adds the names to the ESM named-export set.
//
// Keep this list in sync with `ERROR_CLASS_HIERARCHY` in
// `crates/blazen-node/src/error_classes.rs`.

{
  const current = readFileSync(jsPath, 'utf8')
  const sentinel = '// --- post-build: cjs-module-lexer hints for native error classes ---'
  if (!current.includes(sentinel)) {
    const classNames = [
      'BlazenError',
      'AuthError', 'RateLimitError', 'TimeoutError', 'ValidationError',
      'ContentPolicyError', 'UnsupportedError', 'ComputeError', 'MediaError',
      'ProviderError',
      'LlamaCppError', 'LlamaCppInvalidOptionsError', 'LlamaCppModelLoadError',
      'LlamaCppInferenceError', 'LlamaCppEngineNotAvailableError', 'LlamaCppAdapterFailedError',
      'CandleLlmError', 'CandleLlmInvalidOptionsError', 'CandleLlmModelLoadError',
      'CandleLlmInferenceError', 'CandleLlmEngineNotAvailableError', 'CandleLlmUnsupportedError',
      'CandleEmbedError', 'CandleEmbedInvalidOptionsError', 'CandleEmbedModelLoadError',
      'CandleEmbedEmbeddingError', 'CandleEmbedEngineNotAvailableError', 'CandleEmbedTaskPanickedError',
      'MistralRsError', 'MistralRsInvalidOptionsError', 'MistralRsInitError',
      'MistralRsInferenceError', 'MistralRsEngineNotAvailableError', 'MistralRsAdapterFailedError',
      'WhisperError', 'WhisperInvalidOptionsError', 'WhisperModelLoadError',
      'WhisperTranscriptionError', 'WhisperEngineNotAvailableError', 'WhisperIoError',
      'PiperError', 'PiperInvalidOptionsError', 'PiperModelLoadError',
      'PiperSynthesisError', 'PiperEngineNotAvailableError',
      'DiffusionError', 'DiffusionInvalidOptionsError', 'DiffusionModelLoadError', 'DiffusionGenerationError',
      'FastEmbedError', 'EmbedUnknownModelError', 'EmbedInitError',
      'EmbedEmbedError', 'EmbedMutexPoisonedError', 'EmbedTaskPanickedError',
      'TractError',
      'PeerEncodeError', 'PeerTransportError', 'PeerEnvelopeVersionError',
      'PeerWorkflowError', 'PeerTlsError', 'PeerUnknownStepError',
      'PersistError',
      'CacheError', 'DownloadError', 'CacheDirError', 'IoError',
      'PromptError', 'PromptMissingVariableError', 'PromptNotFoundError',
      'PromptVersionNotFoundError', 'PromptIoError', 'PromptYamlError',
      'PromptJsonError', 'PromptValidationError',
      'MemoryError', 'MemoryNoEmbedderError', 'MemoryElidError',
      'MemoryEmbeddingError', 'MemoryNotFoundError', 'MemorySerializationError',
      'MemoryIoError', 'MemoryBackendError',
    ]
    const lines = classNames.map((n) => `module.exports.${n} = module.exports.${n}`)
    const block = `\n${sentinel}\n${lines.join('\n')}\n`
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
      'export class LlamaCppAdapterFailedError extends LlamaCppError {}',
      'export class CandleLlmError extends ProviderError {}',
      'export class CandleLlmInvalidOptionsError extends CandleLlmError {}',
      'export class CandleLlmModelLoadError extends CandleLlmError {}',
      'export class CandleLlmInferenceError extends CandleLlmError {}',
      'export class CandleLlmEngineNotAvailableError extends CandleLlmError {}',
      'export class CandleLlmUnsupportedError extends CandleLlmError {}',
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
      'export class MistralRsAdapterFailedError extends MistralRsError {}',
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

// Error-class factory: napi-patched's \`register_error_class\` uses this
// global to build the JS subclass chain when running on hosts that block
// dynamic code generation (i.e. workerd's \`napi_run_script\` is a no-op).
// On Node / wasi-node / browsers, this global is absent and napi-patched
// falls back to evaluating an inline script with the same shape.
globalThis.__blazenErrorClassFactory = (parent, name) =>
  class extends parent {
    constructor(message, props) {
      super(message)
      this.name = name
      if (props) Object.assign(this, props)
    }
  }

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
    'js: error-class esm hints': has(
      finalJs,
      '// --- post-build: cjs-module-lexer hints for native error classes ---',
    ),
    'd.ts: error classes': has(finalDts, '// --- post-build: typed error classes ---'),
    'js: workers entry': existsSync(workersJsPath),
    'd.ts: workers shim': existsSync(workersDtsPath),
  }
  // eslint-disable-next-line no-console
  console.log(`[post-build] ${JSON.stringify(status)}`)
}
