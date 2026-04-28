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

import { appendFileSync, readFileSync, writeFileSync } from 'node:fs'

const dtsPath = new URL('../index.d.ts', import.meta.url)
const jsPath = new URL('../index.js', import.meta.url)

// ---------------------------------------------------------------------------
// Section 1: type aliases on index.d.ts
// ---------------------------------------------------------------------------

{
  const current = readFileSync(dtsPath, 'utf8')
  const aliases = [
    'export type MediaSource = JsImageSource',
    'export type ImageSource = JsImageSource',
  ]
  const banner = '\n// --- post-build: type aliases mirroring blazen-llm ---\n'
  const missing = aliases.filter((line) => !current.includes(line))
  if (missing.length > 0) {
    const block = `${current.endsWith('\n') ? '' : '\n'}${banner}${missing.join('\n')}\n`
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
    const isLikelyClass =
      orig.prototype &&
      Object.getOwnPropertyNames(orig.prototype).some((p) => p !== 'constructor')
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

// Verify final state and write a small status line so build logs are
// informative without being noisy.
{
  const finalDts = readFileSync(dtsPath, 'utf8')
  const finalJs = readFileSync(jsPath, 'utf8')
  const has = (haystack, needle) => haystack.includes(needle)
  const status = {
    'd.ts: type aliases': has(finalDts, 'export type MediaSource = JsImageSource'),
    'd.ts: error classes': has(finalDts, '// --- post-build: typed error classes ---'),
    'js: error wrapping': has(finalJs, '// --- post-build: typed-error wrapping ---'),
  }
  // eslint-disable-next-line no-console
  console.log(`[post-build] ${JSON.stringify(status)}`)
  // Avoid `writeFileSync` if nothing changed -- but the appendFileSync
  // calls above already exited early on idempotency. Nothing else to do.
  void writeFileSync // explicitly unused; left imported for future expansions
}
