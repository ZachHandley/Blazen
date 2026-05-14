package blazen

import (
	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// This file collects the idiomatic Go wrappers around the
// CustomProvider-adjacent UniFFI surface introduced by the typed
// [CustomProvider] pivot:
//
//   - [ApiProtocol] — tagged-union shim around the UniFFI enum
//     `ApiProtocol` (OpenAi / Custom variants).
//   - [OpenAICompatConfig] — wrapper around the UniFFI Record
//     `OpenAiCompatConfig` carrying provider name, base URL, API key, auth
//     method, headers, query params, etc.
//   - [BaseProviderDefaults] — placeholder defaults Record applicable to
//     every provider role.
//   - [CompletionProviderDefaults] — defaults applied to every completion
//     call (system prompt, default tools, default response_format).
//   - Nine role-specific defaults wrappers (Embedding, AudioSpeech,
//     AudioMusic, VoiceCloning, ImageGeneration, ImageUpscale, Video,
//     Transcription, ThreeD, BackgroundRemoval).
//   - [BaseProvider] — clone-with-mutation wrapper around an existing
//     [CompletionModel], exposing builder-style methods for default
//     system prompt, default tools, and default response format.
//   - [CustomProvider] — the typed 16-method interface the host
//     implements to plug a foreign provider into Blazen. Includes the
//     [UnsupportedCustomProvider] base struct callers embed to default
//     every method to a typed [UnsupportedError].
//   - [CustomProviderHandle] — opaque handle returned by the
//     [Ollama] / [LMStudio] / [OpenAICompat] / [CustomProviderFrom]
//     factories. Exposes `AsBase()` for builder chaining and forwards
//     all 16 typed methods to the wrapped provider.
//
// The wrappers mirror the conventions established in providers.go and
// llm.go: empty-string-means-unset for optional strings, PascalCase Go
// names, builder methods on pointer receivers, and idiomatic error
// returns via [wrapErr].

// ---------------------------------------------------------------------
// Re-exports of UniFFI record types used by the CustomProvider interface
// ---------------------------------------------------------------------
//
// The user-facing [CustomProvider] interface (aliased below) is the
// UniFFI-generated interface verbatim. Its method signatures reference
// UniFFI record types (SpeechRequest, AudioResult, ImageRequest, ...).
// Re-exporting those records here lets host implementations live entirely
// inside the public `blazen` package without reaching into the internal
// `internal/uniffi/blazen` package.
//
// Records whose names already exist in [llm.go] or [streaming.go]
// (CompletionRequest, CompletionResponse, EmbeddingResponse, StreamChunk,
// TokenUsage) are exported with a `Provider` prefix so the wire-format
// records carried by the [CustomProvider] callback boundary do not clash
// with the idiomatic Go-side wrapper structs. Inside the alias, Go treats
// the two names as identical — so passing a wire-shaped value to a
// blazen.CustomProvider method works without conversion.

// ProviderCompletionRequest is the wire-format record passed to
// [CustomProvider.Complete] and [CustomProvider.Stream]. Identical to
// the UniFFI `CompletionRequest` record; aliased under the `Provider`
// prefix to avoid colliding with the user-facing [CompletionRequest]
// wrapper struct.
type ProviderCompletionRequest = uniffiblazen.CompletionRequest

// ProviderCompletionResponse is the wire-format record returned by
// [CustomProvider.Complete]. Identical to the UniFFI
// `CompletionResponse` record.
type ProviderCompletionResponse = uniffiblazen.CompletionResponse

// ProviderEmbeddingResponse is the wire-format record returned by
// [CustomProvider.Embed]. Identical to the UniFFI `EmbeddingResponse`
// record.
type ProviderEmbeddingResponse = uniffiblazen.EmbeddingResponse

// ProviderStreamChunk is the wire-format chunk delivered to
// [CompletionStreamSink.OnChunk]. Identical to the UniFFI `StreamChunk`
// record.
type ProviderStreamChunk = uniffiblazen.StreamChunk

// ProviderTokenUsage is the wire-format token-usage record delivered to
// [CompletionStreamSink.OnDone]. Identical to the UniFFI `TokenUsage`
// record.
type ProviderTokenUsage = uniffiblazen.TokenUsage

// SpeechRequest is the input record for [CustomProvider.TextToSpeech].
type SpeechRequest = uniffiblazen.SpeechRequest

// AudioResult is the output record returned by speech / music / SFX
// generation methods on [CustomProvider].
type AudioResult = uniffiblazen.AudioResult

// MusicRequest is the input record for [CustomProvider.GenerateMusic]
// and [CustomProvider.GenerateSfx].
type MusicRequest = uniffiblazen.MusicRequest

// VoiceCloneRequest is the input record for [CustomProvider.CloneVoice].
type VoiceCloneRequest = uniffiblazen.VoiceCloneRequest

// VoiceHandle is the record returned by [CustomProvider.CloneVoice] and
// [CustomProvider.ListVoices], and accepted by [CustomProvider.DeleteVoice].
type VoiceHandle = uniffiblazen.VoiceHandle

// ImageRequest is the input record for [CustomProvider.GenerateImage].
type ImageRequest = uniffiblazen.ImageRequest

// ImageResult is the output record returned by image-generation,
// upscale, and background-removal methods on [CustomProvider].
type ImageResult = uniffiblazen.ImageResult

// UpscaleRequest is the input record for [CustomProvider.UpscaleImage].
type UpscaleRequest = uniffiblazen.UpscaleRequest

// VideoRequest is the input record for [CustomProvider.TextToVideo] and
// [CustomProvider.ImageToVideo].
type VideoRequest = uniffiblazen.VideoRequest

// VideoResult is the output record returned by video-generation methods
// on [CustomProvider].
type VideoResult = uniffiblazen.VideoResult

// TranscriptionRequest is the input record for [CustomProvider.Transcribe].
type TranscriptionRequest = uniffiblazen.TranscriptionRequest

// TranscriptionResult is the output record returned by
// [CustomProvider.Transcribe].
type TranscriptionResult = uniffiblazen.TranscriptionResult

// ThreeDRequest is the input record for [CustomProvider.Generate3d].
type ThreeDRequest = uniffiblazen.ThreeDRequest

// ThreeDResult is the output record returned by [CustomProvider.Generate3d].
type ThreeDResult = uniffiblazen.ThreeDResult

// BackgroundRemovalRequest is the input record for
// [CustomProvider.RemoveBackground].
type BackgroundRemovalRequest = uniffiblazen.BackgroundRemovalRequest

// CompletionStreamSink is the foreign-implemented sink interface the
// streaming engine pushes chunks into. Implementations are expected to
// call [CompletionStreamSink.OnDone] exactly once on success or
// [CompletionStreamSink.OnError] exactly once on failure; after either
// terminal callback fires no further [CompletionStreamSink.OnChunk]
// calls will be made.
//
// Aliased from the UniFFI surface so host implementations of
// [CustomProvider.Stream] can construct a sink type without importing
// the internal uniffi package.
type CompletionStreamSink = uniffiblazen.CompletionStreamSink

// ---------------------------------------------------------------------
// ApiProtocol
// ---------------------------------------------------------------------

// APIProtocolKind enumerates the two ApiProtocol variants exposed by
// the UniFFI surface. Use [ApiProtocol.Kind] to inspect a value.
type APIProtocolKind string

const (
	// APIProtocolKindOpenAI denotes the OpenAI-wire-format variant: the
	// framework handles HTTP transport, SSE parsing, tool calls, and
	// retries against an OpenAI-compatible endpoint described by an
	// [OpenAICompatConfig].
	APIProtocolKindOpenAI APIProtocolKind = "openai"
	// APIProtocolKindCustom denotes the foreign-dispatched variant:
	// every completion call is dispatched to a foreign-implemented
	// [CustomProvider]. No wire-format configuration lives here — the
	// host owns the protocol entirely.
	APIProtocolKindCustom APIProtocolKind = "custom"
)

// ApiProtocol selects how a [CustomProvider] talks to its backend for
// completion calls. Construct via [NewOpenAIProtocol] for an
// OpenAI-compatible endpoint, or [NewCustomProtocol] for fully
// host-defined dispatch.
//
// The zero value is not usable — always construct via one of the
// factory functions.
type ApiProtocol struct {
	inner uniffiblazen.ApiProtocol
}

// NewOpenAIProtocol builds an OpenAI-wire-format protocol from the
// supplied [OpenAICompatConfig].
func NewOpenAIProtocol(config OpenAICompatConfig) *ApiProtocol {
	return &ApiProtocol{
		inner: uniffiblazen.ApiProtocolOpenAi{Config: config.toFFI()},
	}
}

// NewCustomProtocol builds a foreign-dispatched protocol. Completion
// methods are routed to the foreign-implemented [CustomProvider]
// supplied to [CustomProviderFrom]; this constructor carries no
// configuration of its own.
func NewCustomProtocol() *ApiProtocol {
	return &ApiProtocol{inner: uniffiblazen.ApiProtocolCustom{}}
}

// Kind returns [APIProtocolKindOpenAI] or [APIProtocolKindCustom]
// depending on which variant this ApiProtocol holds.
func (p *ApiProtocol) Kind() APIProtocolKind {
	if p == nil {
		return ""
	}
	switch p.inner.(type) {
	case uniffiblazen.ApiProtocolOpenAi:
		return APIProtocolKindOpenAI
	case uniffiblazen.ApiProtocolCustom:
		return APIProtocolKindCustom
	default:
		return ""
	}
}

// Config returns the [OpenAICompatConfig] when [ApiProtocol.Kind]
// returns [APIProtocolKindOpenAI], or nil otherwise. The returned value
// is a copy — mutating it has no effect on the protocol.
func (p *ApiProtocol) Config() *OpenAICompatConfig {
	if p == nil {
		return nil
	}
	v, ok := p.inner.(uniffiblazen.ApiProtocolOpenAi)
	if !ok {
		return nil
	}
	cfg := openAICompatConfigFromFFI(v.Config)
	return &cfg
}

// toFFI returns the underlying UniFFI ApiProtocol value, for forwarding
// to generated factory functions.
func (p *ApiProtocol) toFFI() uniffiblazen.ApiProtocol {
	return p.inner
}

// ---------------------------------------------------------------------
// AuthMethod
// ---------------------------------------------------------------------

// AuthMethodKind enumerates the four AuthMethod variants. Use
// [AuthMethod.Kind] to discriminate.
type AuthMethodKind string

const (
	// AuthMethodKindBearer sends the API key as
	// `Authorization: Bearer <key>` (OpenAI, OpenRouter, Groq, ...).
	AuthMethodKindBearer AuthMethodKind = "bearer"
	// AuthMethodKindAPIKeyHeader sends the key under a custom header
	// name (e.g. `x-api-key`).
	AuthMethodKindAPIKeyHeader AuthMethodKind = "api_key_header"
	// AuthMethodKindAzureAPIKey sends `api-key: <key>` (Azure OpenAI).
	AuthMethodKindAzureAPIKey AuthMethodKind = "azure_api_key"
	// AuthMethodKindKeyPrefix sends `Authorization: Key <key>` (fal.ai).
	AuthMethodKindKeyPrefix AuthMethodKind = "key_prefix"
)

// AuthMethod describes how a [CustomProvider] authenticates with an
// OpenAI-compatible backend. Construct via the AuthMethod* factory
// functions below.
//
// The zero value is not usable — always go through a factory.
type AuthMethod struct {
	inner uniffiblazen.AuthMethod
}

// AuthMethodBearer returns an AuthMethod that sends the API key as
// `Authorization: Bearer <key>`. This is the default for most
// OpenAI-compatible providers.
func AuthMethodBearer() *AuthMethod {
	return &AuthMethod{inner: uniffiblazen.AuthMethodBearer{}}
}

// AuthMethodAPIKeyHeader returns an AuthMethod that sends the API key
// under a custom header name (e.g. `x-api-key`).
func AuthMethodAPIKeyHeader(headerName string) *AuthMethod {
	return &AuthMethod{inner: uniffiblazen.AuthMethodApiKeyHeader{HeaderName: headerName}}
}

// AuthMethodAzureAPIKey returns an AuthMethod that sends the API key as
// `api-key: <key>` (Azure OpenAI).
func AuthMethodAzureAPIKey() *AuthMethod {
	return &AuthMethod{inner: uniffiblazen.AuthMethodAzureApiKey{}}
}

// AuthMethodKeyPrefix returns an AuthMethod that sends the API key as
// `Authorization: Key <key>` (fal.ai).
func AuthMethodKeyPrefix() *AuthMethod {
	return &AuthMethod{inner: uniffiblazen.AuthMethodKeyPrefix{}}
}

// Kind reports which AuthMethod variant this value holds.
func (a *AuthMethod) Kind() AuthMethodKind {
	if a == nil {
		return ""
	}
	switch a.inner.(type) {
	case uniffiblazen.AuthMethodBearer:
		return AuthMethodKindBearer
	case uniffiblazen.AuthMethodApiKeyHeader:
		return AuthMethodKindAPIKeyHeader
	case uniffiblazen.AuthMethodAzureApiKey:
		return AuthMethodKindAzureAPIKey
	case uniffiblazen.AuthMethodKeyPrefix:
		return AuthMethodKindKeyPrefix
	default:
		return ""
	}
}

// HeaderName returns the custom header name when [AuthMethod.Kind] is
// [AuthMethodKindAPIKeyHeader], or "" otherwise.
func (a *AuthMethod) HeaderName() string {
	if a == nil {
		return ""
	}
	v, ok := a.inner.(uniffiblazen.AuthMethodApiKeyHeader)
	if !ok {
		return ""
	}
	return v.HeaderName
}

// toFFI returns the underlying UniFFI AuthMethod value.
func (a *AuthMethod) toFFI() uniffiblazen.AuthMethod {
	return a.inner
}

// authMethodFromFFI lifts a raw UniFFI AuthMethod into the wrapper
// form.
func authMethodFromFFI(m uniffiblazen.AuthMethod) *AuthMethod {
	return &AuthMethod{inner: m}
}

// ---------------------------------------------------------------------
// OpenAICompatConfig
// ---------------------------------------------------------------------

// OpenAICompatConfig configures an OpenAI-compatible provider backend.
// Pass one to [NewOpenAIProtocol] or [OpenAICompat] to build a
// [CustomProviderHandle] backed by an OpenAI-compatible endpoint.
//
// AuthMethod, ExtraHeaders, and QueryParams are nilable — leaving any
// of them nil yields the documented Rust defaults
// ([AuthMethodBearer], empty header map, empty query map).
type OpenAICompatConfig struct {
	// ProviderName is a human-readable label used in logs and model
	// info records (e.g. "openrouter", "lmstudio").
	ProviderName string
	// BaseURL is the API root (e.g. "https://api.openai.com/v1").
	BaseURL string
	// APIKey is the credential. May be empty for backends that don't
	// require authentication (Ollama, LocalAI without auth, ...).
	APIKey string
	// DefaultModel is the model id to use when a request doesn't
	// override it.
	DefaultModel string
	// AuthMethod selects the wire-level credential format. nil means
	// [AuthMethodBearer] (the Rust default).
	AuthMethod *AuthMethod
	// ExtraHeaders is a flat list of HTTP headers added to every
	// request. Use a slice (rather than a map) because the wire format
	// preserves insertion order and the Rust side stores headers as a
	// `Vec<(String, String)>`.
	ExtraHeaders []KeyValue
	// QueryParams is a flat list of query parameters appended to every
	// request URL (e.g. Azure's `api-version`).
	QueryParams []KeyValue
	// SupportsModelListing toggles support for the `/models` listing
	// endpoint when the framework needs to discover models.
	SupportsModelListing bool
}

// KeyValue is a string-string pair used for HTTP headers and query
// parameters in [OpenAICompatConfig]. Mirrors UniFFI's `KeyValue`
// Record one-to-one — re-exported here so callers don't need to import
// the internal package.
type KeyValue struct {
	Key   string
	Value string
}

// keyValueSliceToFFI converts a wrapper slice into its UniFFI form. A
// nil input yields a nil output to preserve round-trip equality.
func keyValueSliceToFFI(in []KeyValue) []uniffiblazen.KeyValue {
	if in == nil {
		return nil
	}
	out := make([]uniffiblazen.KeyValue, len(in))
	for i, kv := range in {
		out[i] = uniffiblazen.KeyValue{Key: kv.Key, Value: kv.Value}
	}
	return out
}

// keyValueSliceFromFFI converts a UniFFI slice into the wrapper form.
func keyValueSliceFromFFI(in []uniffiblazen.KeyValue) []KeyValue {
	if in == nil {
		return nil
	}
	out := make([]KeyValue, len(in))
	for i, kv := range in {
		out[i] = KeyValue{Key: kv.Key, Value: kv.Value}
	}
	return out
}

// toFFI lowers an [OpenAICompatConfig] into the UniFFI record.
// AuthMethod defaults to Bearer when nil.
func (c OpenAICompatConfig) toFFI() uniffiblazen.OpenAiCompatConfig {
	var auth uniffiblazen.AuthMethod = uniffiblazen.AuthMethodBearer{}
	if c.AuthMethod != nil {
		auth = c.AuthMethod.toFFI()
	}
	return uniffiblazen.OpenAiCompatConfig{
		ProviderName:         c.ProviderName,
		BaseUrl:              c.BaseURL,
		ApiKey:               c.APIKey,
		DefaultModel:         c.DefaultModel,
		AuthMethod:           auth,
		ExtraHeaders:         keyValueSliceToFFI(c.ExtraHeaders),
		QueryParams:          keyValueSliceToFFI(c.QueryParams),
		SupportsModelListing: c.SupportsModelListing,
	}
}

// openAICompatConfigFromFFI lifts a UniFFI record into the wrapper
// form.
func openAICompatConfigFromFFI(c uniffiblazen.OpenAiCompatConfig) OpenAICompatConfig {
	return OpenAICompatConfig{
		ProviderName:         c.ProviderName,
		BaseURL:              c.BaseUrl,
		APIKey:               c.ApiKey,
		DefaultModel:         c.DefaultModel,
		AuthMethod:           authMethodFromFFI(c.AuthMethod),
		ExtraHeaders:         keyValueSliceFromFFI(c.ExtraHeaders),
		QueryParams:          keyValueSliceFromFFI(c.QueryParams),
		SupportsModelListing: c.SupportsModelListing,
	}
}

// ---------------------------------------------------------------------
// BaseProviderDefaults (placeholder)
// ---------------------------------------------------------------------

// BaseProviderDefaults carries provider-role-agnostic configuration
// that applies before every request, regardless of role (completion,
// embedding, image, audio, ...).
//
// V1 carries no data fields — provided so future hook fields can be
// added without breaking the public API.
type BaseProviderDefaults struct{}

// NewBaseProviderDefaults returns an empty [BaseProviderDefaults].
// Currently equivalent to passing nil — present for forward
// compatibility.
func NewBaseProviderDefaults() *BaseProviderDefaults {
	return &BaseProviderDefaults{}
}

// toFFI lowers the wrapper into the UniFFI record. The UniFFI side
// carries a reserved boolean — we always send false.
func (d *BaseProviderDefaults) toFFI() uniffiblazen.BaseProviderDefaults {
	return uniffiblazen.BaseProviderDefaults{Reserved: false}
}

// optBaseProviderDefaultsToFFI lowers a nilable wrapper into the
// UniFFI optional record.
func optBaseProviderDefaultsToFFI(d *BaseProviderDefaults) *uniffiblazen.BaseProviderDefaults {
	if d == nil {
		return nil
	}
	v := d.toFFI()
	return &v
}

// baseProviderDefaultsFromFFI lifts a UniFFI optional record into the
// wrapper form. A nil input yields a nil output.
func baseProviderDefaultsFromFFI(d *uniffiblazen.BaseProviderDefaults) *BaseProviderDefaults {
	if d == nil {
		return nil
	}
	return &BaseProviderDefaults{}
}

// ---------------------------------------------------------------------
// CompletionProviderDefaults
// ---------------------------------------------------------------------

// CompletionProviderDefaults holds defaults applied to every
// completion call: a default system prompt, default tools, and a
// default response format.
//
// All fields are optional — leaving a field nil or empty means "no
// default; let the request decide". SystemPrompt, ToolsJSON, and
// ResponseFormatJSON use the empty-string convention from the rest of
// the binding: an empty string is treated as unset.
//
// ToolsJSON is the JSON-encoded list of `ToolDefinition` records; the
// caller is responsible for serialising via [encoding/json] or any
// other marshaller. ResponseFormatJSON is the JSON-encoded
// OpenAI-style `response_format` value.
type CompletionProviderDefaults struct {
	// Base is the role-agnostic defaults block. nil means "no base
	// defaults" — equivalent to an empty [BaseProviderDefaults] for
	// V1.
	Base *BaseProviderDefaults
	// SystemPrompt, when non-empty, is prepended as a system message
	// to every completion request that doesn't already supply one.
	SystemPrompt string
	// ToolsJSON, when non-empty, is the JSON-encoded `[]ToolDefinition`
	// merged into every completion request's tool list.
	// Request-supplied tools win on name collision.
	ToolsJSON string
	// ResponseFormatJSON, when non-empty, is the JSON-encoded
	// OpenAI-style `response_format` value applied when a request
	// lacks one.
	ResponseFormatJSON string
}

// NewCompletionProviderDefaults returns an empty
// [CompletionProviderDefaults]. Equivalent to the Rust
// `CompletionProviderDefaults::default()`.
func NewCompletionProviderDefaults() *CompletionProviderDefaults {
	return &CompletionProviderDefaults{}
}

// toFFI lowers the wrapper into the UniFFI record, translating the
// empty-string convention into FFI optionals.
func (d *CompletionProviderDefaults) toFFI() uniffiblazen.CompletionProviderDefaults {
	if d == nil {
		return uniffiblazen.CompletionProviderDefaults{}
	}
	return uniffiblazen.CompletionProviderDefaults{
		Base:               optBaseProviderDefaultsToFFI(d.Base),
		SystemPrompt:       optString(d.SystemPrompt),
		ToolsJson:          optString(d.ToolsJSON),
		ResponseFormatJson: optString(d.ResponseFormatJSON),
	}
}

// completionProviderDefaultsFromFFI lifts a UniFFI record into the
// wrapper form.
func completionProviderDefaultsFromFFI(d uniffiblazen.CompletionProviderDefaults) *CompletionProviderDefaults {
	return &CompletionProviderDefaults{
		Base:               baseProviderDefaultsFromFFI(d.Base),
		SystemPrompt:       stringOrEmpty(d.SystemPrompt),
		ToolsJSON:          stringOrEmpty(d.ToolsJson),
		ResponseFormatJSON: stringOrEmpty(d.ResponseFormatJson),
	}
}

// ---------------------------------------------------------------------
// Role-specific defaults placeholders
// ---------------------------------------------------------------------
//
// Each role-specific defaults type carries only a [BaseProviderDefaults]
// pointer today. The wrappers exist now so foreign callers can pass them
// through the public surface without churn as additional role-specific
// hook fields are added.

// EmbeddingProviderDefaults holds defaults applied to every embedding
// call. V1 carries only a [BaseProviderDefaults].
type EmbeddingProviderDefaults struct {
	Base *BaseProviderDefaults
}

// NewEmbeddingProviderDefaults returns an empty
// [EmbeddingProviderDefaults].
func NewEmbeddingProviderDefaults() *EmbeddingProviderDefaults {
	return &EmbeddingProviderDefaults{}
}

func (d *EmbeddingProviderDefaults) toFFI() uniffiblazen.EmbeddingProviderDefaults {
	if d == nil {
		return uniffiblazen.EmbeddingProviderDefaults{}
	}
	return uniffiblazen.EmbeddingProviderDefaults{Base: optBaseProviderDefaultsToFFI(d.Base)}
}

// AudioSpeechProviderDefaults holds defaults applied to every
// text-to-speech call. V1 carries only a [BaseProviderDefaults].
type AudioSpeechProviderDefaults struct {
	Base *BaseProviderDefaults
}

// NewAudioSpeechProviderDefaults returns an empty
// [AudioSpeechProviderDefaults].
func NewAudioSpeechProviderDefaults() *AudioSpeechProviderDefaults {
	return &AudioSpeechProviderDefaults{}
}

func (d *AudioSpeechProviderDefaults) toFFI() uniffiblazen.AudioSpeechProviderDefaults {
	if d == nil {
		return uniffiblazen.AudioSpeechProviderDefaults{}
	}
	return uniffiblazen.AudioSpeechProviderDefaults{Base: optBaseProviderDefaultsToFFI(d.Base)}
}

// AudioMusicProviderDefaults holds defaults applied to every music
// generation call. V1 carries only a [BaseProviderDefaults].
type AudioMusicProviderDefaults struct {
	Base *BaseProviderDefaults
}

// NewAudioMusicProviderDefaults returns an empty
// [AudioMusicProviderDefaults].
func NewAudioMusicProviderDefaults() *AudioMusicProviderDefaults {
	return &AudioMusicProviderDefaults{}
}

func (d *AudioMusicProviderDefaults) toFFI() uniffiblazen.AudioMusicProviderDefaults {
	if d == nil {
		return uniffiblazen.AudioMusicProviderDefaults{}
	}
	return uniffiblazen.AudioMusicProviderDefaults{Base: optBaseProviderDefaultsToFFI(d.Base)}
}

// VoiceCloningProviderDefaults holds defaults applied to every voice
// cloning call. V1 carries only a [BaseProviderDefaults].
type VoiceCloningProviderDefaults struct {
	Base *BaseProviderDefaults
}

// NewVoiceCloningProviderDefaults returns an empty
// [VoiceCloningProviderDefaults].
func NewVoiceCloningProviderDefaults() *VoiceCloningProviderDefaults {
	return &VoiceCloningProviderDefaults{}
}

func (d *VoiceCloningProviderDefaults) toFFI() uniffiblazen.VoiceCloningProviderDefaults {
	if d == nil {
		return uniffiblazen.VoiceCloningProviderDefaults{}
	}
	return uniffiblazen.VoiceCloningProviderDefaults{Base: optBaseProviderDefaultsToFFI(d.Base)}
}

// ImageGenerationProviderDefaults holds defaults applied to every
// image-generation call. V1 carries only a [BaseProviderDefaults].
type ImageGenerationProviderDefaults struct {
	Base *BaseProviderDefaults
}

// NewImageGenerationProviderDefaults returns an empty
// [ImageGenerationProviderDefaults].
func NewImageGenerationProviderDefaults() *ImageGenerationProviderDefaults {
	return &ImageGenerationProviderDefaults{}
}

func (d *ImageGenerationProviderDefaults) toFFI() uniffiblazen.ImageGenerationProviderDefaults {
	if d == nil {
		return uniffiblazen.ImageGenerationProviderDefaults{}
	}
	return uniffiblazen.ImageGenerationProviderDefaults{Base: optBaseProviderDefaultsToFFI(d.Base)}
}

// ImageUpscaleProviderDefaults holds defaults applied to every image
// upscale call. V1 carries only a [BaseProviderDefaults].
type ImageUpscaleProviderDefaults struct {
	Base *BaseProviderDefaults
}

// NewImageUpscaleProviderDefaults returns an empty
// [ImageUpscaleProviderDefaults].
func NewImageUpscaleProviderDefaults() *ImageUpscaleProviderDefaults {
	return &ImageUpscaleProviderDefaults{}
}

func (d *ImageUpscaleProviderDefaults) toFFI() uniffiblazen.ImageUpscaleProviderDefaults {
	if d == nil {
		return uniffiblazen.ImageUpscaleProviderDefaults{}
	}
	return uniffiblazen.ImageUpscaleProviderDefaults{Base: optBaseProviderDefaultsToFFI(d.Base)}
}

// VideoProviderDefaults holds defaults applied to every video
// generation call. V1 carries only a [BaseProviderDefaults].
type VideoProviderDefaults struct {
	Base *BaseProviderDefaults
}

// NewVideoProviderDefaults returns an empty [VideoProviderDefaults].
func NewVideoProviderDefaults() *VideoProviderDefaults {
	return &VideoProviderDefaults{}
}

func (d *VideoProviderDefaults) toFFI() uniffiblazen.VideoProviderDefaults {
	if d == nil {
		return uniffiblazen.VideoProviderDefaults{}
	}
	return uniffiblazen.VideoProviderDefaults{Base: optBaseProviderDefaultsToFFI(d.Base)}
}

// TranscriptionProviderDefaults holds defaults applied to every
// speech-to-text call. V1 carries only a [BaseProviderDefaults].
type TranscriptionProviderDefaults struct {
	Base *BaseProviderDefaults
}

// NewTranscriptionProviderDefaults returns an empty
// [TranscriptionProviderDefaults].
func NewTranscriptionProviderDefaults() *TranscriptionProviderDefaults {
	return &TranscriptionProviderDefaults{}
}

func (d *TranscriptionProviderDefaults) toFFI() uniffiblazen.TranscriptionProviderDefaults {
	if d == nil {
		return uniffiblazen.TranscriptionProviderDefaults{}
	}
	return uniffiblazen.TranscriptionProviderDefaults{Base: optBaseProviderDefaultsToFFI(d.Base)}
}

// ThreeDProviderDefaults holds defaults applied to every 3D-generation
// call. V1 carries only a [BaseProviderDefaults].
type ThreeDProviderDefaults struct {
	Base *BaseProviderDefaults
}

// NewThreeDProviderDefaults returns an empty [ThreeDProviderDefaults].
func NewThreeDProviderDefaults() *ThreeDProviderDefaults {
	return &ThreeDProviderDefaults{}
}

func (d *ThreeDProviderDefaults) toFFI() uniffiblazen.ThreeDProviderDefaults {
	if d == nil {
		return uniffiblazen.ThreeDProviderDefaults{}
	}
	return uniffiblazen.ThreeDProviderDefaults{Base: optBaseProviderDefaultsToFFI(d.Base)}
}

// BackgroundRemovalProviderDefaults holds defaults applied to every
// background-removal call. V1 carries only a [BaseProviderDefaults].
type BackgroundRemovalProviderDefaults struct {
	Base *BaseProviderDefaults
}

// NewBackgroundRemovalProviderDefaults returns an empty
// [BackgroundRemovalProviderDefaults].
func NewBackgroundRemovalProviderDefaults() *BackgroundRemovalProviderDefaults {
	return &BackgroundRemovalProviderDefaults{}
}

func (d *BackgroundRemovalProviderDefaults) toFFI() uniffiblazen.BackgroundRemovalProviderDefaults {
	if d == nil {
		return uniffiblazen.BackgroundRemovalProviderDefaults{}
	}
	return uniffiblazen.BackgroundRemovalProviderDefaults{Base: optBaseProviderDefaultsToFFI(d.Base)}
}

// ---------------------------------------------------------------------
// BaseProvider
// ---------------------------------------------------------------------

// BaseProvider wraps a [CompletionModel] with a
// [CompletionProviderDefaults] block that is applied before every
// completion call.
//
// On the Rust side the underlying handle is `Arc<RwLock<...>>`, so the
// builder methods clone-with-mutation: each call returns a fresh
// BaseProvider that shares the inner model but carries the new
// defaults. The original is unchanged and safe to keep using.
//
// Aliased from the UniFFI surface so callers obtain BaseProvider
// handles directly from [CustomProviderHandle.AsBase] without paying
// for an idiomatic wrapper that would have to mirror every builder
// method one-to-one. Call [BaseProvider.Destroy] when finished to
// release the underlying native handle; UniFFI also installs a
// finalizer as a safety net.
type BaseProvider = uniffiblazen.BaseProvider

// NewBaseProviderFromCompletionModel wraps an existing
// [CompletionModel] with an empty [CompletionProviderDefaults]. Use
// the builder methods on the returned [BaseProvider] to attach
// defaults afterwards.
//
// The supplied [CompletionModel] continues to be usable independently;
// the returned [BaseProvider] holds its own reference to the inner
// Rust handle.
func NewBaseProviderFromCompletionModel(model *CompletionModel) (*BaseProvider, error) {
	ensureInit()
	if model == nil || model.inner == nil {
		return nil, &ValidationError{Message: "completion model is nil or closed"}
	}
	return uniffiblazen.BaseProviderFromCompletionModel(model.inner), nil
}

// NewBaseProviderWithDefaults wraps an existing [CompletionModel] with
// explicit [CompletionProviderDefaults] in a single step. Equivalent
// to [NewBaseProviderFromCompletionModel] followed by
// [BaseProvider.WithDefaults], but avoids the intermediate handle.
func NewBaseProviderWithDefaults(model *CompletionModel, defaults *CompletionProviderDefaults) (*BaseProvider, error) {
	ensureInit()
	if model == nil || model.inner == nil {
		return nil, &ValidationError{Message: "completion model is nil or closed"}
	}
	var ffi uniffiblazen.CompletionProviderDefaults
	if defaults != nil {
		ffi = defaults.toFFI()
	}
	return uniffiblazen.BaseProviderWithCompletionDefaults(model.inner, ffi), nil
}

// CompletionProviderDefaultsOf returns a snapshot of the
// currently-configured completion defaults on a [BaseProvider]. The
// returned value is a copy — mutating it has no effect on the
// provider; use the [BaseProvider.WithDefaults] / [BaseProvider.WithSystemPrompt] /
// [BaseProvider.WithToolsJson] / [BaseProvider.WithResponseFormatJson]
// builders to change defaults.
//
// Companion to the typed [BaseProvider.Defaults] method on the
// UniFFI handle: this helper lifts the raw FFI defaults into the
// idiomatic Go-side [CompletionProviderDefaults] struct.
func CompletionProviderDefaultsOf(p *BaseProvider) *CompletionProviderDefaults {
	if p == nil {
		return nil
	}
	return completionProviderDefaultsFromFFI(p.Defaults())
}

// ---------------------------------------------------------------------
// CustomProvider
// ---------------------------------------------------------------------

// CustomProvider is the foreign-implementable provider interface
// host applications wire into Blazen. Implementors supply the 16
// typed methods (`Complete`, `Stream`, `Embed`, plus 13 compute /
// media methods) plus the `ProviderId` accessor, then pass the
// implementation to [CustomProviderFrom] to obtain a
// [CustomProviderHandle] usable wherever Blazen expects a provider.
//
// Host implementations should embed [UnsupportedCustomProvider] for
// the methods they don't support — the embedded struct's default
// methods return a typed [UnsupportedError] (UniFFI's
// `BlazenError::Unsupported`). Override only the methods your
// provider actually implements:
//
//	type MyProvider struct {
//	    blazen.UnsupportedCustomProvider
//	}
//
//	func (p *MyProvider) ProviderId() string { return "my-provider" }
//
//	func (p *MyProvider) Complete(req blazen.ProviderCompletionRequest) (blazen.ProviderCompletionResponse, error) {
//	    // ...
//	}
//
//	handle := blazen.CustomProviderFrom(&MyProvider{})
//
// The interface is aliased verbatim from the UniFFI-generated surface;
// all method signatures use the wire-format records re-exported in
// this file (Provider*, SpeechRequest, AudioResult, ...).
type CustomProvider = uniffiblazen.CustomProvider

// CustomProviderHandle is the opaque handle returned by the
// [Ollama] / [LMStudio] / [OpenAICompat] / [CustomProviderFrom]
// factories. Forwards all 16 typed methods to the wrapped provider,
// applying any per-instance completion defaults attached via
// [BaseProvider]-style builders before each completion call.
//
// Use [CustomProviderHandle.AsBase] to obtain the paired
// [BaseProvider] handle for builder-style chaining of system prompt,
// tool list, or response format defaults.
//
// Aliased verbatim from the UniFFI-generated surface so callers can
// reach the full method set (Complete, Stream, Embed, TextToSpeech,
// GenerateMusic, GenerateSfx, CloneVoice, ListVoices, DeleteVoice,
// GenerateImage, UpscaleImage, TextToVideo, ImageToVideo, Transcribe,
// Generate3d, RemoveBackground, plus the WithSystemPrompt /
// WithToolsJson / WithResponseFormatJson builders) without paying for
// a hand-written wrapper.
type CustomProviderHandle = uniffiblazen.CustomProviderHandle

// CustomProviderFrom wraps a foreign-implemented [CustomProvider]
// into a [CustomProviderHandle].
//
// The returned handle holds an internal adapter that translates each
// typed method call into a UniFFI-level dispatch back into the
// supplied provider. Pass the handle wherever Blazen expects a
// provider — its `AsBase()` returns a [BaseProvider] that plugs into
// agents, workflow steps, and any API that takes a [CompletionModel].
//
// Equivalent to the upstream `custom_provider_from_foreign` factory.
func CustomProviderFrom(provider CustomProvider) *CustomProviderHandle {
	ensureInit()
	return uniffiblazen.CustomProviderFromForeign(provider)
}

// Ollama builds a [CustomProviderHandle] for a local Ollama server.
//
// Equivalent to [OpenAICompat] with `base_url = http://{host}:{port}/v1`
// and no API key. `host` defaults to `"localhost"` (pass the empty
// string to use the default); `port` defaults to 11434 (pass 0 to use
// the default).
func Ollama(host string, port uint16, model string) *CustomProviderHandle {
	ensureInit()
	var hostPtr *string
	if host != "" {
		hostPtr = &host
	}
	var portPtr *uint16
	if port != 0 {
		portPtr = &port
	}
	return uniffiblazen.Ollama(model, hostPtr, portPtr)
}

// LMStudio builds a [CustomProviderHandle] for an LM Studio server.
//
// Equivalent to [OpenAICompat] with `base_url = http://{host}:{port}/v1`
// and no API key. `host` defaults to `"localhost"` (pass the empty
// string to use the default); `port` defaults to 1234 (pass 0 to use
// the default).
func LMStudio(host string, port uint16, model string) *CustomProviderHandle {
	ensureInit()
	var hostPtr *string
	if host != "" {
		hostPtr = &host
	}
	var portPtr *uint16
	if port != 0 {
		portPtr = &port
	}
	return uniffiblazen.LmStudio(model, hostPtr, portPtr)
}

// OpenAICompat builds a [CustomProviderHandle] for an arbitrary
// OpenAI-compatible backend (vLLM, llama.cpp's server, TGI, hosted
// OpenAI-compat services, ...).
//
// providerID is the stable identifier surfaced via
// [CustomProviderHandle.ProviderId]; config selects the base URL,
// model, auth method, headers, and query parameters of the upstream
// endpoint.
func OpenAICompat(providerID string, config *OpenAICompatConfig) *CustomProviderHandle {
	ensureInit()
	var cfg uniffiblazen.OpenAiCompatConfig
	if config != nil {
		cfg = config.toFFI()
	} else {
		cfg = uniffiblazen.OpenAiCompatConfig{
			ProviderName:         providerID,
			AuthMethod:           uniffiblazen.AuthMethodBearer{},
			ExtraHeaders:         nil,
			QueryParams:          nil,
			SupportsModelListing: false,
		}
	}
	return uniffiblazen.OpenaiCompat(providerID, cfg)
}

// ---------------------------------------------------------------------
// UnsupportedCustomProvider
// ---------------------------------------------------------------------

// UnsupportedCustomProvider is the default-embeddable base that turns
// every [CustomProvider] method into a typed [UnsupportedError]. Host
// implementations embed it to override only the capabilities they
// actually support:
//
//	type SpeechOnly struct {
//	    blazen.UnsupportedCustomProvider
//	}
//
//	func (p *SpeechOnly) ProviderId() string { return "speech-only" }
//
//	func (p *SpeechOnly) TextToSpeech(req blazen.SpeechRequest) (blazen.AudioResult, error) {
//	    // real implementation here
//	}
//
//	// Complete / Stream / Embed / GenerateMusic / ... all fall through
//	// to the embedded UnsupportedCustomProvider defaults and return a
//	// typed UnsupportedError.
//
// `ProviderId()` returns the literal string `"unsupported"` so
// implementations that forget to override it still produce a stable,
// non-empty value in logs. Override it on every concrete provider.
//
// The zero value is ready to use. No initialisation or Close is
// required — UnsupportedCustomProvider holds no native resources.
type UnsupportedCustomProvider struct{}

// unsupported is the canonical UniFFI BlazenError returned by every
// default method on [UnsupportedCustomProvider]. It is built lazily on
// each call so the wrapped Rust error captures the failing method
// name in its message.
func unsupportedMethod(name string) error {
	return uniffiblazen.NewBlazenErrorUnsupported(
		"custom provider method `" + name + "` is not implemented",
	)
}

// ProviderId returns the literal string `"unsupported"`. Host
// implementations should override this on every concrete provider to
// supply a stable identifier for logs and metrics.
func (UnsupportedCustomProvider) ProviderId() string { return "unsupported" }

// Complete returns a typed [UnsupportedError]. Override this method to
// implement non-streaming chat completion.
func (UnsupportedCustomProvider) Complete(_ ProviderCompletionRequest) (ProviderCompletionResponse, error) {
	return ProviderCompletionResponse{}, unsupportedMethod("complete")
}

// Stream returns a typed [UnsupportedError]. Override this method to
// implement streaming chat completion.
//
// Implementors are expected to call `sink.OnDone` exactly once on
// successful streams or `sink.OnError` exactly once on failure;
// returning an error from `Stream` itself is reserved for the case
// where the initial request conversion fails before any chunks are
// produced.
func (UnsupportedCustomProvider) Stream(_ ProviderCompletionRequest, _ CompletionStreamSink) error {
	return unsupportedMethod("stream")
}

// Embed returns a typed [UnsupportedError]. Override this method to
// implement embedding generation.
func (UnsupportedCustomProvider) Embed(_ []string) (ProviderEmbeddingResponse, error) {
	return ProviderEmbeddingResponse{}, unsupportedMethod("embed")
}

// TextToSpeech returns a typed [UnsupportedError]. Override this
// method to implement text-to-speech synthesis.
func (UnsupportedCustomProvider) TextToSpeech(_ SpeechRequest) (AudioResult, error) {
	return AudioResult{}, unsupportedMethod("text_to_speech")
}

// GenerateMusic returns a typed [UnsupportedError]. Override this
// method to implement music generation.
func (UnsupportedCustomProvider) GenerateMusic(_ MusicRequest) (AudioResult, error) {
	return AudioResult{}, unsupportedMethod("generate_music")
}

// GenerateSfx returns a typed [UnsupportedError]. Override this method
// to implement sound-effect generation.
func (UnsupportedCustomProvider) GenerateSfx(_ MusicRequest) (AudioResult, error) {
	return AudioResult{}, unsupportedMethod("generate_sfx")
}

// CloneVoice returns a typed [UnsupportedError]. Override this method
// to implement voice cloning from a reference sample.
func (UnsupportedCustomProvider) CloneVoice(_ VoiceCloneRequest) (VoiceHandle, error) {
	return VoiceHandle{}, unsupportedMethod("clone_voice")
}

// ListVoices returns a typed [UnsupportedError]. Override this method
// to expose the list of voices known to the provider.
func (UnsupportedCustomProvider) ListVoices() ([]VoiceHandle, error) {
	return nil, unsupportedMethod("list_voices")
}

// DeleteVoice returns a typed [UnsupportedError]. Override this method
// to remove a previously-cloned voice.
func (UnsupportedCustomProvider) DeleteVoice(_ VoiceHandle) error {
	return unsupportedMethod("delete_voice")
}

// GenerateImage returns a typed [UnsupportedError]. Override this
// method to implement text-to-image generation.
func (UnsupportedCustomProvider) GenerateImage(_ ImageRequest) (ImageResult, error) {
	return ImageResult{}, unsupportedMethod("generate_image")
}

// UpscaleImage returns a typed [UnsupportedError]. Override this
// method to implement image upscaling.
func (UnsupportedCustomProvider) UpscaleImage(_ UpscaleRequest) (ImageResult, error) {
	return ImageResult{}, unsupportedMethod("upscale_image")
}

// TextToVideo returns a typed [UnsupportedError]. Override this method
// to implement text-to-video generation.
func (UnsupportedCustomProvider) TextToVideo(_ VideoRequest) (VideoResult, error) {
	return VideoResult{}, unsupportedMethod("text_to_video")
}

// ImageToVideo returns a typed [UnsupportedError]. Override this
// method to implement image-to-video generation.
func (UnsupportedCustomProvider) ImageToVideo(_ VideoRequest) (VideoResult, error) {
	return VideoResult{}, unsupportedMethod("image_to_video")
}

// Transcribe returns a typed [UnsupportedError]. Override this method
// to implement speech-to-text transcription.
func (UnsupportedCustomProvider) Transcribe(_ TranscriptionRequest) (TranscriptionResult, error) {
	return TranscriptionResult{}, unsupportedMethod("transcribe")
}

// Generate3d returns a typed [UnsupportedError]. Override this method
// to implement 3D-model generation.
func (UnsupportedCustomProvider) Generate3d(_ ThreeDRequest) (ThreeDResult, error) {
	return ThreeDResult{}, unsupportedMethod("generate_3d")
}

// RemoveBackground returns a typed [UnsupportedError]. Override this
// method to implement image background removal.
func (UnsupportedCustomProvider) RemoveBackground(_ BackgroundRemovalRequest) (ImageResult, error) {
	return ImageResult{}, unsupportedMethod("remove_background")
}

// Compile-time assertion that UnsupportedCustomProvider satisfies the
// full [CustomProvider] interface. If a method is added to the UniFFI
// trait surface and we forget to mirror it here, this assignment will
// fail to compile and the omission surfaces at build time.
var _ CustomProvider = UnsupportedCustomProvider{}
