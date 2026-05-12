package blazen

import (
	uniffiblazen "github.com/zorpxinc/blazen-go/internal/uniffi/blazen"
)

// This file collects the provider factory wrappers exposed by blazen-go.
//
// The generated UniFFI surface exposes each provider as a free constructor
// returning the opaque CompletionModel / EmbeddingModel handle. The
// wrappers here add three Go-flavoured affordances on top:
//
//   - Idiomatic naming (OpenAI/XAI/LlamaCpp/MistralRs/DeepSeek/FastEmbed)
//     instead of UniFFI's PascalCase-of-snake (Openai, Xai, ...).
//   - Empty-string-means-unset for optional *string parameters so the
//     common case stays positional and avoids the take-address dance.
//   - A single ensureInit() call so callers never have to remember to
//     initialise the native runtime; the underlying Init() is idempotent
//     and gated by sync.Once.
//
// Providers whose models are not chat completers (diffusion image models,
// Whisper STT, Piper / Fal TTS) live in compute.go because their public
// surface is the speech/image API rather than CompletionModel /
// EmbeddingModel.

// optUint32 returns a pointer to v, or nil when v is zero. Matches the
// empty-string convention used by [optString] for the numeric optional
// parameters that local-runtime providers accept (context length, GPU
// layer count, embedding dimensions, batch size).
func optUint32(v uint32) *uint32 {
	if v == 0 {
		return nil
	}
	return &v
}

// ---------------------------------------------------------------------
// HTTP completion providers
// ---------------------------------------------------------------------

// NewAnthropicCompletion creates a [CompletionModel] backed by the
// Anthropic Messages API.
//
// model selects the model id (e.g. "claude-3-5-sonnet-latest"); pass ""
// for the provider default. baseURL overrides the API root; pass "" for
// https://api.anthropic.com.
func NewAnthropicCompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewAnthropicCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewAzureCompletion creates a [CompletionModel] backed by an Azure
// OpenAI deployment.
//
// resourceName and deploymentName identify the Azure resource and the
// deployed model respectively. apiVersion pins the REST API version;
// pass "" to use the binding's built-in default.
func NewAzureCompletion(apiKey, resourceName, deploymentName, apiVersion string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewAzureCompletionModel(apiKey, resourceName, deploymentName, optString(apiVersion))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewBedrockCompletion creates a [CompletionModel] backed by AWS Bedrock.
//
// region is the AWS region (e.g. "us-east-1"). model selects the
// foundation model id; pass "" for the provider default. baseURL
// overrides the Bedrock endpoint and is primarily useful for VPC
// endpoints; pass "" for the standard regional endpoint.
func NewBedrockCompletion(apiKey, region, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewBedrockCompletionModel(apiKey, region, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewCohereCompletion creates a [CompletionModel] backed by the Cohere
// Chat API.
//
// model selects the model id (e.g. "command-r-plus"); pass "" for the
// provider default. baseURL overrides the API root; pass "" for the
// standard Cohere endpoint.
func NewCohereCompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewCohereCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewDeepSeekCompletion creates a [CompletionModel] backed by the
// DeepSeek chat API.
//
// model selects the model id (e.g. "deepseek-chat"); pass "" for the
// provider default. baseURL overrides the API root; pass "" for the
// standard DeepSeek endpoint.
func NewDeepSeekCompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewDeepseekCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewFireworksCompletion creates a [CompletionModel] backed by
// Fireworks AI.
//
// model selects the model id; pass "" for the provider default. baseURL
// overrides the API root; pass "" for the standard Fireworks endpoint.
func NewFireworksCompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewFireworksCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewGeminiCompletion creates a [CompletionModel] backed by Google's
// Gemini API.
//
// model selects the model id (e.g. "gemini-1.5-pro"); pass "" for the
// provider default. baseURL overrides the API root; pass "" for the
// standard Google endpoint.
func NewGeminiCompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewGeminiCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewGroqCompletion creates a [CompletionModel] backed by Groq's
// OpenAI-compatible API.
//
// model selects the model id (e.g. "llama-3.1-70b-versatile"); pass ""
// for the provider default. baseURL overrides the API root; pass "" for
// the standard Groq endpoint.
func NewGroqCompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewGroqCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewMistralCompletion creates a [CompletionModel] backed by the
// Mistral AI API.
//
// model selects the model id (e.g. "mistral-large-latest"); pass "" for
// the provider default. baseURL overrides the API root; pass "" for the
// standard Mistral endpoint.
func NewMistralCompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewMistralCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewOpenAICompletion creates a [CompletionModel] backed by the OpenAI
// chat completions API.
//
// model selects the model id (e.g. "gpt-4o"); pass "" for the provider
// default. baseURL overrides the API root; pass "" for
// https://api.openai.com.
func NewOpenAICompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewOpenaiCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewOpenAICompatCompletion creates a [CompletionModel] for an arbitrary
// OpenAI-compatible chat completions endpoint (Ollama, vLLM, LM Studio,
// LocalAI, ...).
//
// All four parameters are required: providerName labels the upstream for
// telemetry, baseURL is the API root, apiKey is the bearer credential
// (use a placeholder string when the upstream does not require one),
// and model selects the model id.
func NewOpenAICompatCompletion(providerName, baseURL, apiKey, model string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewOpenaiCompatCompletionModel(providerName, baseURL, apiKey, model)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewOpenRouterCompletion creates a [CompletionModel] backed by
// OpenRouter, an aggregator that proxies dozens of upstream providers
// behind a single OpenAI-compatible API.
//
// model selects the OpenRouter model slug (e.g.
// "anthropic/claude-3.5-sonnet"); pass "" for the provider default.
// baseURL overrides the API root; pass "" for
// https://openrouter.ai/api/v1.
func NewOpenRouterCompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewOpenrouterCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewPerplexityCompletion creates a [CompletionModel] backed by the
// Perplexity AI sonar API.
//
// model selects the model id (e.g. "sonar-pro"); pass "" for the
// provider default. baseURL overrides the API root; pass "" for the
// standard Perplexity endpoint.
func NewPerplexityCompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewPerplexityCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewTogetherCompletion creates a [CompletionModel] backed by Together
// AI's OpenAI-compatible API.
//
// model selects the model id; pass "" for the provider default. baseURL
// overrides the API root; pass "" for the standard Together endpoint.
func NewTogetherCompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewTogetherCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// NewXAICompletion creates a [CompletionModel] backed by xAI's Grok
// API (OpenAI-compatible).
//
// model selects the model id (e.g. "grok-2-latest"); pass "" for the
// provider default. baseURL overrides the API root; pass "" for the
// standard xAI endpoint.
func NewXAICompletion(apiKey, model, baseURL string) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewXaiCompletionModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// FalCompletionOpts configures [NewFalCompletion].
//
// APIKey is required. Model, BaseURL, and Endpoint use the
// empty-string-means-unset convention. Enterprise toggles the
// fal.ai enterprise routing path. AutoRouteModality enables
// automatic modality detection (text/image/video) for the chosen
// endpoint.
type FalCompletionOpts struct {
	APIKey            string // required: fal.ai key
	Model             string // "" for the provider default
	BaseURL           string // "" for https://fal.run
	Endpoint          string // "" for the default per-model endpoint
	Enterprise        bool   // true to use fal.ai enterprise routing
	AutoRouteModality bool   // true to auto-detect text/image/video
}

// NewFalCompletion creates a [CompletionModel] backed by fal.ai.
//
// fal.ai exposes both LLM endpoints and image / video generation
// endpoints behind the same API; this factory targets the text-chat
// surface. See compute.go for the image / video / TTS factories that
// share the same backend.
func NewFalCompletion(opts FalCompletionOpts) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewFalCompletionModel(
		opts.APIKey,
		optString(opts.Model),
		optString(opts.BaseURL),
		optString(opts.Endpoint),
		opts.Enterprise,
		opts.AutoRouteModality,
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// ---------------------------------------------------------------------
// Local-runtime completion providers
// ---------------------------------------------------------------------

// CandleCompletionOpts configures [NewCandleCompletion].
//
// ModelID is required and identifies the Hugging Face repo (e.g.
// "meta-llama/Llama-3-8B"). Device is "" for auto-select, otherwise
// "cpu", "cuda:0", "metal", etc. Quantization is "" for none, otherwise
// a backend-specific tag (e.g. "q4_0"). Revision pins a git revision.
// ContextLength is zero for the model default, otherwise the desired
// context window in tokens.
type CandleCompletionOpts struct {
	ModelID       string // required: Hugging Face repo id
	Device        string // "" = auto
	Quantization  string // "" = none
	Revision      string // "" = HEAD
	ContextLength uint32 // 0 = model default
}

// NewCandleCompletion creates a [CompletionModel] running locally via
// the Candle inference runtime (CPU/CUDA/Metal).
func NewCandleCompletion(opts CandleCompletionOpts) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewCandleCompletionModel(
		opts.ModelID,
		optString(opts.Device),
		optString(opts.Quantization),
		optString(opts.Revision),
		optUint32(opts.ContextLength),
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// LlamaCppCompletionOpts configures [NewLlamaCppCompletion].
//
// ModelPath is required and points at a GGUF file on the local
// filesystem. Device is "" for auto-select. Quantization is informational
// (the GGUF file itself encodes the quantization). ContextLength is
// zero for the GGUF default. NGpuLayers is the number of transformer
// layers to offload to the GPU (zero for CPU-only).
type LlamaCppCompletionOpts struct {
	ModelPath     string // required: path to a .gguf file
	Device        string // "" = auto
	Quantization  string // "" = inferred from the GGUF header
	ContextLength uint32 // 0 = GGUF default
	NGpuLayers    uint32 // 0 = CPU-only
}

// NewLlamaCppCompletion creates a [CompletionModel] running locally via
// the llama.cpp runtime against a GGUF model file.
func NewLlamaCppCompletion(opts LlamaCppCompletionOpts) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewLlamacppCompletionModel(
		opts.ModelPath,
		optString(opts.Device),
		optString(opts.Quantization),
		optUint32(opts.ContextLength),
		optUint32(opts.NGpuLayers),
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// MistralRsCompletionOpts configures [NewMistralRsCompletion].
//
// ModelID is required and identifies the Hugging Face repo. Device is
// "" for auto-select. Quantization is a backend-specific tag.
// ContextLength is zero for the model default. Vision enables the
// multimodal (vision) execution path for models that support it.
type MistralRsCompletionOpts struct {
	ModelID       string // required: Hugging Face repo id
	Device        string // "" = auto
	Quantization  string // "" = none
	ContextLength uint32 // 0 = model default
	Vision        bool   // true to enable vision path
}

// NewMistralRsCompletion creates a [CompletionModel] running locally via
// the mistral.rs inference runtime.
func NewMistralRsCompletion(opts MistralRsCompletionOpts) (*CompletionModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewMistralrsCompletionModel(
		opts.ModelID,
		optString(opts.Device),
		optString(opts.Quantization),
		optUint32(opts.ContextLength),
		opts.Vision,
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCompletionModel(inner), nil
}

// ---------------------------------------------------------------------
// Embedding providers
// ---------------------------------------------------------------------

// NewOpenAIEmbedding creates an [EmbeddingModel] backed by the OpenAI
// embeddings API.
//
// model selects the model id (e.g. "text-embedding-3-small"); pass ""
// for the provider default. baseURL overrides the API root; pass "" for
// https://api.openai.com.
func NewOpenAIEmbedding(apiKey, model, baseURL string) (*EmbeddingModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewOpenaiEmbeddingModel(apiKey, optString(model), optString(baseURL))
	if err != nil {
		return nil, wrapErr(err)
	}
	return newEmbeddingModel(inner), nil
}

// CandleEmbeddingOpts configures [NewCandleEmbedding].
//
// ModelID identifies the Hugging Face repo (e.g.
// "sentence-transformers/all-MiniLM-L6-v2"); pass "" for the binding's
// built-in default. Device is "" for auto-select. Revision pins a git
// revision.
type CandleEmbeddingOpts struct {
	ModelID  string // "" = binding default
	Device   string // "" = auto
	Revision string // "" = HEAD
}

// NewCandleEmbedding creates an [EmbeddingModel] running locally via the
// Candle inference runtime.
func NewCandleEmbedding(opts CandleEmbeddingOpts) (*EmbeddingModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewCandleEmbeddingModel(
		optString(opts.ModelID),
		optString(opts.Device),
		optString(opts.Revision),
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newEmbeddingModel(inner), nil
}

// FalEmbeddingOpts configures [NewFalEmbedding].
//
// APIKey is required. Model is "" for the provider default. Dimensions
// is zero for the model's native output size; otherwise it requests a
// truncated embedding of the given dimensionality (Matryoshka-style),
// where supported by the model.
type FalEmbeddingOpts struct {
	APIKey     string // required: fal.ai key
	Model      string // "" = provider default
	Dimensions uint32 // 0 = native output size
}

// NewFalEmbedding creates an [EmbeddingModel] backed by fal.ai's
// embedding endpoints.
func NewFalEmbedding(opts FalEmbeddingOpts) (*EmbeddingModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewFalEmbeddingModel(
		opts.APIKey,
		optString(opts.Model),
		optUint32(opts.Dimensions),
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newEmbeddingModel(inner), nil
}

// FastEmbedEmbeddingOpts configures [NewFastEmbedEmbedding].
//
// ModelName is "" for the binding's built-in default (a small
// English-only model). MaxBatchSize is zero for the binding default,
// otherwise the maximum number of inputs per batch. ShowDownloadProgress
// is a tri-state — nil leaves the default, &true logs progress, &false
// suppresses it.
type FastEmbedEmbeddingOpts struct {
	ModelName            string // "" = binding default
	MaxBatchSize         uint32 // 0 = binding default
	ShowDownloadProgress *bool  // nil = leave default
}

// NewFastEmbedEmbedding creates an [EmbeddingModel] running locally via
// the FastEmbed ONNX-based inference runtime.
func NewFastEmbedEmbedding(opts FastEmbedEmbeddingOpts) (*EmbeddingModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewFastembedEmbeddingModel(
		optString(opts.ModelName),
		optUint32(opts.MaxBatchSize),
		opts.ShowDownloadProgress,
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newEmbeddingModel(inner), nil
}

// TractEmbeddingOpts configures [NewTractEmbedding].
//
// ModelName is "" for the binding's built-in default. MaxBatchSize is
// zero for the binding default. ShowDownloadProgress is a tri-state —
// nil leaves the default, &true logs progress, &false suppresses it.
type TractEmbeddingOpts struct {
	ModelName            string // "" = binding default
	MaxBatchSize         uint32 // 0 = binding default
	ShowDownloadProgress *bool  // nil = leave default
}

// NewTractEmbedding creates an [EmbeddingModel] running locally via the
// Tract pure-Rust ONNX inference runtime.
func NewTractEmbedding(opts TractEmbeddingOpts) (*EmbeddingModel, error) {
	ensureInit()
	inner, err := uniffiblazen.NewTractEmbeddingModel(
		optString(opts.ModelName),
		optUint32(opts.MaxBatchSize),
		opts.ShowDownloadProgress,
	)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newEmbeddingModel(inner), nil
}
