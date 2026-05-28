package blazen

import (
	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// Media is a multimodal attachment carried alongside a [ChatMessage].
//
// Kind is one of "image", "audio", or "video". MimeType is the IANA
// media type (e.g. "image/png", "audio/wav"). DataBase64 is the raw
// bytes base64-encoded; an empty string is permitted when the provider
// accepts a URL referenced elsewhere in the message body.
type Media struct {
	Kind       string
	MimeType   string
	DataBase64 string
}

// ChatMessage is a single message in a chat conversation.
//
// Role is "system", "user", "assistant", or "tool". Content is the
// text payload and is empty when the message carries only tool calls
// or media. MediaParts attaches multimodal inputs. ToolCalls is set
// when an assistant message requests tool invocations. ToolCallID and
// Name use the empty string to mean "unset"; provide a non-empty value
// only when the message references a prior tool call or carries a
// participant name.
type ChatMessage struct {
	Role       string
	Content    string
	MediaParts []Media
	ToolCalls  []ToolCall
	ToolCallID string
	Name       string
}

// Tool is a function definition the model may invoke during a chat
// completion. ParametersJSON is a JSON Schema string describing the
// tool's input arguments.
type Tool struct {
	Name           string
	Description    string
	ParametersJSON string
}

// ToolCall is a tool invocation requested by the model. ArgumentsJSON
// is the JSON-encoded arguments object — parse it with [encoding/json]
// (or any JSON library) on the receiving side.
type ToolCall struct {
	ID            string
	Name          string
	ArgumentsJSON string
}

// TokenUsage reports the token counts associated with a completion or
// embedding response. Providers that do not surface every counter
// leave the corresponding field zero.
type TokenUsage struct {
	PromptTokens      uint64
	CompletionTokens  uint64
	TotalTokens       uint64
	CachedInputTokens uint64
	ReasoningTokens   uint64
}

// ModelRequest is a provider-agnostic chat completion request.
//
// Temperature, MaxTokens, and TopP retain pointer types so callers can
// distinguish an explicit zero from "use the provider default" — pass
// nil to defer to the provider. Model, ResponseFormatJSON, and System
// use the empty-string convention: an empty string means "unset", any
// non-empty value is forwarded to the provider.
//
// ResponseFormatJSON, when set, is a JSON Schema string constraining
// the model's output. System, when set, is prepended as a system-role
// message before Messages.
type ModelRequest struct {
	Messages           []ChatMessage
	Tools              []Tool
	Temperature        *float64
	MaxTokens          *uint32
	TopP               *float64
	Model              string
	ResponseFormatJSON string
	System             string
}

// ModelResponse is the outcome of a non-streaming chat completion.
//
// Content is the empty string when the provider returned no text (e.g.
// the model emitted only tool calls). FinishReason is the empty string
// when the provider did not report one.
type ModelResponse struct {
	Content      string
	ToolCalls    []ToolCall
	FinishReason string
	Model        string
	Usage        TokenUsage
}

// EmbeddingResponse is the outcome of an embedding request.
// Embeddings[i] is the vector produced for the i-th input string.
type EmbeddingResponse struct {
	Embeddings [][]float64
	Model      string
	Usage      TokenUsage
}

// optString returns a pointer to s, or nil when s is empty. It is the
// canonical helper for translating the wrapper's empty-string-means-unset
// convention into the FFI's *string optionals.
func optString(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// stringOrEmpty returns *p, or the empty string when p is nil. Inverse
// of [optString] for converting FFI optionals back into wrapper types.
func stringOrEmpty(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

// toFFI converts the wrapper Media into its FFI counterpart.
func (m Media) toFFI() uniffiblazen.Media {
	return uniffiblazen.Media{
		Kind:       m.Kind,
		MimeType:   m.MimeType,
		DataBase64: m.DataBase64,
	}
}

// mediaFromFFI converts an FFI Media into the wrapper form.
func mediaFromFFI(m uniffiblazen.Media) Media {
	return Media{
		Kind:       m.Kind,
		MimeType:   m.MimeType,
		DataBase64: m.DataBase64,
	}
}

// mediaSliceToFFI converts a wrapper Media slice into its FFI form.
// A nil input yields a nil output to preserve round-trip equality.
func mediaSliceToFFI(in []Media) []uniffiblazen.Media {
	if in == nil {
		return nil
	}
	out := make([]uniffiblazen.Media, len(in))
	for i, m := range in {
		out[i] = m.toFFI()
	}
	return out
}

// mediaSliceFromFFI converts an FFI Media slice into the wrapper form.
func mediaSliceFromFFI(in []uniffiblazen.Media) []Media {
	if in == nil {
		return nil
	}
	out := make([]Media, len(in))
	for i, m := range in {
		out[i] = mediaFromFFI(m)
	}
	return out
}

// toFFI converts the wrapper Tool into its FFI counterpart, renaming
// ParametersJSON to the generated bindgen's ParametersJson slot.
func (t Tool) toFFI() uniffiblazen.Tool {
	return uniffiblazen.Tool{
		Name:           t.Name,
		Description:    t.Description,
		ParametersJson: t.ParametersJSON,
	}
}

// toolFromFFI converts an FFI Tool into the wrapper form.
func toolFromFFI(t uniffiblazen.Tool) Tool {
	return Tool{
		Name:           t.Name,
		Description:    t.Description,
		ParametersJSON: t.ParametersJson,
	}
}

// toolSliceToFFI converts a wrapper Tool slice into its FFI form.
func toolSliceToFFI(in []Tool) []uniffiblazen.Tool {
	if in == nil {
		return nil
	}
	out := make([]uniffiblazen.Tool, len(in))
	for i, t := range in {
		out[i] = t.toFFI()
	}
	return out
}

// toolSliceFromFFI converts an FFI Tool slice into the wrapper form.
func toolSliceFromFFI(in []uniffiblazen.Tool) []Tool {
	if in == nil {
		return nil
	}
	out := make([]Tool, len(in))
	for i, t := range in {
		out[i] = toolFromFFI(t)
	}
	return out
}

// toFFI converts the wrapper ToolCall into its FFI counterpart,
// renaming ID to Id and ArgumentsJSON to ArgumentsJson.
func (tc ToolCall) toFFI() uniffiblazen.ToolCall {
	return uniffiblazen.ToolCall{
		Id:            tc.ID,
		Name:          tc.Name,
		ArgumentsJson: tc.ArgumentsJSON,
	}
}

// toolCallFromFFI converts an FFI ToolCall into the wrapper form.
func toolCallFromFFI(tc uniffiblazen.ToolCall) ToolCall {
	return ToolCall{
		ID:            tc.Id,
		Name:          tc.Name,
		ArgumentsJSON: tc.ArgumentsJson,
	}
}

// toolCallSliceToFFI converts a wrapper ToolCall slice into its FFI
// form.
func toolCallSliceToFFI(in []ToolCall) []uniffiblazen.ToolCall {
	if in == nil {
		return nil
	}
	out := make([]uniffiblazen.ToolCall, len(in))
	for i, tc := range in {
		out[i] = tc.toFFI()
	}
	return out
}

// toolCallSliceFromFFI converts an FFI ToolCall slice into the wrapper
// form.
func toolCallSliceFromFFI(in []uniffiblazen.ToolCall) []ToolCall {
	if in == nil {
		return nil
	}
	out := make([]ToolCall, len(in))
	for i, tc := range in {
		out[i] = toolCallFromFFI(tc)
	}
	return out
}

// toFFI converts the wrapper ChatMessage into its FFI counterpart,
// translating the empty-string ToolCallID/Name convention into the
// FFI's optional-string fields.
func (m ChatMessage) toFFI() uniffiblazen.ChatMessage {
	return uniffiblazen.ChatMessage{
		Role:       m.Role,
		Content:    m.Content,
		MediaParts: mediaSliceToFFI(m.MediaParts),
		ToolCalls:  toolCallSliceToFFI(m.ToolCalls),
		ToolCallId: optString(m.ToolCallID),
		Name:       optString(m.Name),
	}
}

// chatMessageFromFFI converts an FFI ChatMessage into the wrapper form.
func chatMessageFromFFI(m uniffiblazen.ChatMessage) ChatMessage {
	return ChatMessage{
		Role:       m.Role,
		Content:    m.Content,
		MediaParts: mediaSliceFromFFI(m.MediaParts),
		ToolCalls:  toolCallSliceFromFFI(m.ToolCalls),
		ToolCallID: stringOrEmpty(m.ToolCallId),
		Name:       stringOrEmpty(m.Name),
	}
}

// chatMessageSliceToFFI converts a wrapper ChatMessage slice into its
// FFI form.
func chatMessageSliceToFFI(in []ChatMessage) []uniffiblazen.ChatMessage {
	if in == nil {
		return nil
	}
	out := make([]uniffiblazen.ChatMessage, len(in))
	for i, m := range in {
		out[i] = m.toFFI()
	}
	return out
}

// toFFI converts the wrapper TokenUsage into its FFI counterpart. The
// field names are already idiomatic Go on the generated side, so this
// is a straight copy.
func (u TokenUsage) toFFI() uniffiblazen.TokenUsage {
	return uniffiblazen.TokenUsage{
		PromptTokens:      u.PromptTokens,
		CompletionTokens:  u.CompletionTokens,
		TotalTokens:       u.TotalTokens,
		CachedInputTokens: u.CachedInputTokens,
		ReasoningTokens:   u.ReasoningTokens,
	}
}

// tokenUsageFromFFI converts an FFI TokenUsage into the wrapper form.
func tokenUsageFromFFI(u uniffiblazen.TokenUsage) TokenUsage {
	return TokenUsage{
		PromptTokens:      u.PromptTokens,
		CompletionTokens:  u.CompletionTokens,
		TotalTokens:       u.TotalTokens,
		CachedInputTokens: u.CachedInputTokens,
		ReasoningTokens:   u.ReasoningTokens,
	}
}

// toFFI converts the wrapper ModelRequest into its FFI
// counterpart. Pointer-typed numeric fields are passed through as-is so
// callers retain the ability to distinguish "explicitly zero" from
// "unset"; string-typed Model/ResponseFormatJSON/System are translated
// from the wrapper's empty-string convention into FFI optionals.
func (r ModelRequest) toFFI() uniffiblazen.ModelRequest {
	return uniffiblazen.ModelRequest{
		Messages:           chatMessageSliceToFFI(r.Messages),
		Tools:              toolSliceToFFI(r.Tools),
		Temperature:        r.Temperature,
		MaxTokens:          r.MaxTokens,
		TopP:               r.TopP,
		Model:              optString(r.Model),
		ResponseFormatJson: optString(r.ResponseFormatJSON),
		System:             optString(r.System),
	}
}

// completionResponseFromFFI converts an FFI ModelResponse into the
// wrapper form.
func completionResponseFromFFI(r uniffiblazen.ModelResponse) ModelResponse {
	return ModelResponse{
		Content:      r.Content,
		ToolCalls:    toolCallSliceFromFFI(r.ToolCalls),
		FinishReason: r.FinishReason,
		Model:        r.Model,
		Usage:        tokenUsageFromFFI(r.Usage),
	}
}

// embeddingResponseFromFFI converts an FFI EmbeddingResponse into the
// wrapper form. The nested float64 slices are passed by reference; the
// FFI guarantees a freshly allocated outer slice per call so aliasing
// is safe.
func embeddingResponseFromFFI(r uniffiblazen.EmbeddingResponse) EmbeddingResponse {
	return EmbeddingResponse{
		Embeddings: r.Embeddings,
		Model:      r.Model,
		Usage:      tokenUsageFromFFI(r.Usage),
	}
}

