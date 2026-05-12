package blazen

import (
	"errors"
	"fmt"

	uniffiblazen "github.com/zorpxinc/blazen-go/internal/uniffi/blazen"
)

// Error is the common interface implemented by every typed error returned
// from the blazen-go surface. Use errors.As to recover the concrete variant:
//
//	var rl *blazen.RateLimitError
//	if errors.As(err, &rl) {
//	    // honour rl.RetryAfterMs
//	}
type Error interface {
	error
	// blazenError is unexported to keep this interface closed — only the
	// concrete error structs defined in this package may implement Error.
	blazenError()
}

// AuthError represents an authentication or credentials failure
// (missing API key, invalid token, etc.).
type AuthError struct {
	Message string
}

func (e *AuthError) Error() string { return "auth: " + e.Message }
func (e *AuthError) blazenError()  {}

// RateLimitError represents a rate-limit response from a provider.
// RetryAfterMs is set when the provider returned a Retry-After hint.
type RateLimitError struct {
	Message      string
	RetryAfterMs uint64
}

func (e *RateLimitError) Error() string { return "rate limit: " + e.Message }
func (e *RateLimitError) blazenError()  {}

// TimeoutError represents an operation that timed out before the provider
// responded. ElapsedMs is the wall-clock time that elapsed before the abort.
type TimeoutError struct {
	Message   string
	ElapsedMs uint64
}

func (e *TimeoutError) Error() string {
	return fmt.Sprintf("timeout: %s (elapsed %d ms)", e.Message, e.ElapsedMs)
}
func (e *TimeoutError) blazenError() {}

// ValidationError represents an input validation failure (bad schema,
// missing required field, malformed JSON, etc.).
type ValidationError struct {
	Message string
}

func (e *ValidationError) Error() string { return "validation: " + e.Message }
func (e *ValidationError) blazenError()  {}

// ContentPolicyError represents a content-policy refusal by the provider
// (safety filter triggered, etc.).
type ContentPolicyError struct {
	Message string
}

func (e *ContentPolicyError) Error() string { return "content policy: " + e.Message }
func (e *ContentPolicyError) blazenError()  {}

// UnsupportedError represents an operation that is unsupported on the
// current platform / build / provider.
type UnsupportedError struct {
	Message string
}

func (e *UnsupportedError) Error() string { return "unsupported: " + e.Message }
func (e *UnsupportedError) blazenError()  {}

// ComputeError represents a compute backend failure (CPU/GPU/accelerator
// error, OOM, etc.).
type ComputeError struct {
	Message string
}

func (e *ComputeError) Error() string { return "compute: " + e.Message }
func (e *ComputeError) blazenError()  {}

// MediaError represents a media-handling failure (decode/encode/transcode).
type MediaError struct {
	Message string
}

func (e *MediaError) Error() string { return "media: " + e.Message }
func (e *MediaError) blazenError()  {}

// ProviderError represents a provider/backend failure. Kind identifies the
// specific backend and failure mode (e.g. "OpenAIHttp", "LlamaCppModelLoad").
// Provider / Status / Endpoint / RequestID / Detail / RetryAfterMs surface
// the same structured fields the Node binding's ProviderError class carries.
type ProviderError struct {
	Kind         string
	Message      string
	Provider     string
	Status       uint32
	Endpoint     string
	RequestID    string
	Detail       string
	RetryAfterMs uint64
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("provider %s: %s", e.Kind, e.Message)
}
func (e *ProviderError) blazenError() {}

// WorkflowError represents a workflow-execution failure (step panic,
// deadlock, missing context, etc.).
type WorkflowError struct {
	Message string
}

func (e *WorkflowError) Error() string { return "workflow: " + e.Message }
func (e *WorkflowError) blazenError()  {}

// ToolError represents a tool/function-call failure during LLM agent
// execution.
type ToolError struct {
	Message string
}

func (e *ToolError) Error() string { return "tool: " + e.Message }
func (e *ToolError) blazenError()  {}

// PeerError represents a distributed peer-to-peer failure. Kind is one of
// "Encode", "Transport", "EnvelopeVersion", "Workflow", "Tls",
// "UnknownStep".
type PeerError struct {
	Kind    string
	Message string
}

func (e *PeerError) Error() string {
	return fmt.Sprintf("peer %s: %s", e.Kind, e.Message)
}
func (e *PeerError) blazenError() {}

// PersistError represents a persistence-layer failure (redb / valkey
// checkpoint store).
type PersistError struct {
	Message string
}

func (e *PersistError) Error() string { return "persist: " + e.Message }
func (e *PersistError) blazenError()  {}

// PromptError represents a prompt-template failure. Kind is one of
// "MissingVariable", "NotFound", "VersionNotFound", "Io", "Yaml", "Json",
// "Validation".
type PromptError struct {
	Kind    string
	Message string
}

func (e *PromptError) Error() string {
	return fmt.Sprintf("prompt %s: %s", e.Kind, e.Message)
}
func (e *PromptError) blazenError() {}

// MemoryError represents a memory-subsystem failure. Kind is one of
// "NoEmbedder", "Elid", "Embedding", "NotFound", "Serialization", "Io",
// "Backend".
type MemoryError struct {
	Kind    string
	Message string
}

func (e *MemoryError) Error() string {
	return fmt.Sprintf("memory %s: %s", e.Kind, e.Message)
}
func (e *MemoryError) blazenError() {}

// CacheError represents a model-cache / download failure. Kind is one of
// "Download", "CacheDir", "Io".
type CacheError struct {
	Kind    string
	Message string
}

func (e *CacheError) Error() string {
	return fmt.Sprintf("cache %s: %s", e.Kind, e.Message)
}
func (e *CacheError) blazenError() {}

// CancelledError represents an operation that was cancelled (typically
// because the caller's context.Context was cancelled).
type CancelledError struct{}

func (e *CancelledError) Error() string { return "cancelled" }
func (e *CancelledError) blazenError()  {}

// InternalError is the fallback for errors that don't fit any other
// variant. It should be rare in practice.
type InternalError struct {
	Message string
}

func (e *InternalError) Error() string { return "internal: " + e.Message }
func (e *InternalError) blazenError()  {}

// wrapErr converts an error returned by the uniffi-generated bindings into
// the corresponding concrete Go error type defined in this package.
//
// Non-Blazen errors (including nil) pass through unchanged.
func wrapErr(err error) error {
	if err == nil {
		return nil
	}
	var be *uniffiblazen.BlazenError
	if !errors.As(err, &be) {
		return err
	}
	switch v := be.Variant.(type) {
	case uniffiblazen.BlazenErrorAuth:
		return &AuthError{Message: v.Message}
	case uniffiblazen.BlazenErrorRateLimit:
		var ra uint64
		if v.RetryAfterMs != nil {
			ra = *v.RetryAfterMs
		}
		return &RateLimitError{Message: v.Message, RetryAfterMs: ra}
	case uniffiblazen.BlazenErrorTimeout:
		return &TimeoutError{Message: v.Message, ElapsedMs: v.ElapsedMs}
	case uniffiblazen.BlazenErrorValidation:
		return &ValidationError{Message: v.Message}
	case uniffiblazen.BlazenErrorContentPolicy:
		return &ContentPolicyError{Message: v.Message}
	case uniffiblazen.BlazenErrorUnsupported:
		return &UnsupportedError{Message: v.Message}
	case uniffiblazen.BlazenErrorCompute:
		return &ComputeError{Message: v.Message}
	case uniffiblazen.BlazenErrorMedia:
		return &MediaError{Message: v.Message}
	case uniffiblazen.BlazenErrorProvider:
		out := &ProviderError{Kind: v.Kind, Message: v.Message}
		if v.Provider != nil {
			out.Provider = *v.Provider
		}
		if v.Status != nil {
			out.Status = *v.Status
		}
		if v.Endpoint != nil {
			out.Endpoint = *v.Endpoint
		}
		if v.RequestId != nil {
			out.RequestID = *v.RequestId
		}
		if v.Detail != nil {
			out.Detail = *v.Detail
		}
		if v.RetryAfterMs != nil {
			out.RetryAfterMs = *v.RetryAfterMs
		}
		return out
	case uniffiblazen.BlazenErrorWorkflow:
		return &WorkflowError{Message: v.Message}
	case uniffiblazen.BlazenErrorTool:
		return &ToolError{Message: v.Message}
	case uniffiblazen.BlazenErrorPeer:
		return &PeerError{Kind: v.Kind, Message: v.Message}
	case uniffiblazen.BlazenErrorPersist:
		return &PersistError{Message: v.Message}
	case uniffiblazen.BlazenErrorPrompt:
		return &PromptError{Kind: v.Kind, Message: v.Message}
	case uniffiblazen.BlazenErrorMemory:
		return &MemoryError{Kind: v.Kind, Message: v.Message}
	case uniffiblazen.BlazenErrorCache:
		return &CacheError{Kind: v.Kind, Message: v.Message}
	case uniffiblazen.BlazenErrorCancelled:
		return &CancelledError{}
	case uniffiblazen.BlazenErrorInternal:
		return &InternalError{Message: v.Message}
	default:
		return &InternalError{Message: be.Error()}
	}
}

// unwrapToValidation converts a Go error to a uniffi BlazenError::Validation
// for use inside foreign-implemented trait method bodies that must surface
// failures back into the Rust runtime.
func unwrapToValidation(err error) *uniffiblazen.BlazenError {
	return &uniffiblazen.BlazenError{
		Variant: uniffiblazen.BlazenErrorValidation{Message: err.Error()},
	}
}
