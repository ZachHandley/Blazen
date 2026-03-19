# fal.ai API Reference for Rust Provider Implementation

> Researched and compiled: March 18, 2026
> Primary documentation: https://fal.ai/docs/documentation
> API Reference: https://fal.ai/docs/model-apis/model-endpoints
> OpenAPI spec: https://fal.ai/docs/api-reference/platform-apis/openapi/v1.json

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Base URLs & Domains](#2-base-urls--domains)
3. [Model Addressing](#3-model-addressing)
4. [Execution Modes Overview](#4-execution-modes-overview)
5. [Synchronous Execution (fal.run)](#5-synchronous-execution-falrun)
6. [Queue API (Async Execution)](#6-queue-api-async-execution)
7. [Streaming Status (SSE)](#7-streaming-status-sse)
8. [WebSocket API](#8-websocket-api)
9. [Webhooks](#9-webhooks)
10. [Platform APIs](#10-platform-apis)
11. [Image Generation Models](#11-image-generation-models)
12. [Video Generation Models](#12-video-generation-models)
13. [Audio/Speech Models](#13-audiospeech-models)
14. [Upscaling & Utility Models](#14-upscaling--utility-models)
15. [Pricing & Billing](#15-pricing--billing)
16. [Error Handling](#16-error-handling)
17. [File Storage & Media URLs](#17-file-storage--media-urls)
18. [Request Headers Reference](#18-request-headers-reference)
19. [Implementation Notes for Rust](#19-implementation-notes-for-rust)

---

## 1. Authentication

**Source:** https://fal.ai/docs/reference/platform-apis/authentication

### API Key Format

Keys are formatted as `key_id:key_secret` (e.g., `abc123def456:sk_live_abc123def456xyz789`).

### Authorization Header

```
Authorization: Key <key_id>:<key_secret>
```

The prefix is literally the word `Key` followed by a space, then the full API key string. This is NOT Bearer auth.

### Key Scopes

| Scope | Use Cases |
|-------|-----------|
| `API` | Model inference, model discovery, pricing, analytics |
| `Admin` | Key management, sensitive account data |

### Key Management Dashboard

Generate keys at: https://fal.ai/dashboard/keys

### Programmatic Key Management

**Create Key:**
```
POST https://api.fal.ai/v1/keys
Authorization: Key <admin_key>
Content-Type: application/json

{"alias": "Production Key"}
```

Response (201):
```json
{
  "key_id": "abc123def456",
  "key_secret": "sk_live_abc123def456xyz789",
  "key": "abc123def456:sk_live_abc123def456xyz789"
}
```

Note: `key_secret` is returned ONLY at creation time. Store it immediately.

**List Keys:**
```
GET https://api.fal.ai/v1/keys?limit=50&expand=creator_info
Authorization: Key <admin_key>
```

Response (200):
```json
{
  "keys": [
    {
      "key_id": "string",
      "alias": "string",
      "scope": "API",
      "created_at": "ISO8601",
      "creator_nickname": "string",
      "creator_email": "string"
    }
  ],
  "next_cursor": "string|null",
  "has_more": true
}
```

### Auth Error Responses

| Status | Meaning |
|--------|---------|
| 401 | Missing key, wrong key, revoked key, missing `Key` prefix |
| 403 | Insufficient scope (e.g., API key used for Admin endpoint) |

---

## 2. Base URLs & Domains

| Domain | Purpose |
|--------|---------|
| `https://fal.run` | Synchronous execution |
| `https://queue.fal.run` | Queue-based async execution |
| `wss://fal.run` | Real-time WebSocket (model-native `/realtime` endpoint) |
| `wss://ws.fal.run` | HTTP-over-WebSocket (any model) |
| `https://api.fal.ai` | Platform APIs (models list, pricing, keys, billing) |

---

## 3. Model Addressing

Models use hierarchical IDs: `{owner}/{model-name}[/{variant}][/{path}]`

Examples:
- `fal-ai/flux/schnell`
- `fal-ai/flux/dev`
- `fal-ai/flux-2-pro`
- `fal-ai/kling-video/v2.1/pro/image-to-video`
- `fal-ai/minimax/video-01`
- `fal-ai/whisper`
- `fal-ai/chatterbox/text-to-speech`

URL construction:
- Sync: `https://fal.run/{model_id}`
- Queue submit: `https://queue.fal.run/{model_id}`
- WebSocket: `wss://fal.run/{model_id}/realtime`

---

## 4. Execution Modes Overview

| Mode | Base URL | Behavior | Use Case |
|------|----------|----------|----------|
| **Synchronous** (`run`) | `fal.run` | Blocks until complete, no queue, no retries | Prototyping, fast models |
| **Queue Submit** (`submit`) | `queue.fal.run` | Returns immediately with `request_id`, poll/stream for result | Production workloads |
| **Queue Subscribe** | `queue.fal.run` | SDK wraps submit+poll into blocking call | Convenience |
| **Streaming** | `fal.run/{id}/stream` | SSE for progressive output (e.g., LLM tokens) | Language models |
| **Real-time** | `wss://fal.run/{id}/realtime` | WebSocket, bypasses queue, sub-100ms latency | Interactive apps |

---

## 5. Synchronous Execution (fal.run)

**Endpoint:**
```
POST https://fal.run/{model_id}
```

**Headers:**
```
Authorization: Key <api_key>
Content-Type: application/json
```

**Request Body:** Model-specific JSON payload.

**Response:** Model-specific JSON result directly (no queue wrapper).

**Error Response Format:**
```json
{
  "detail": "Human-readable error message",
  "error_type": "machine_readable_type"
}
```

Also returns `X-Fal-Error-Type` header.

**Limitations:**
- No automatic retries
- No queue durability
- Subject to HTTP timeout limits

---

## 6. Queue API (Async Execution)

**Source:** https://fal.ai/docs/model-apis/model-endpoints/queue

This is the primary execution mode for production. Full lifecycle: Submit -> Poll/Stream -> Get Result.

### 6.1 Submit Request

```
POST https://queue.fal.run/{model_id}
Authorization: Key <api_key>
Content-Type: application/json

{
  "prompt": "a sunset over mountains",
  ...model-specific params
}
```

**Response (202 Accepted):**
```json
{
  "request_id": "764cabcf-b745-4b3e-ae38-1200304cf45b",
  "response_url": "https://queue.fal.run/{model_id}/requests/{request_id}/response",
  "status_url": "https://queue.fal.run/{model_id}/requests/{request_id}/status",
  "cancel_url": "https://queue.fal.run/{model_id}/requests/{request_id}/cancel",
  "queue_position": 0
}
```

### 6.2 Check Status (Poll)

```
GET https://queue.fal.run/{model_id}/requests/{request_id}/status
GET https://queue.fal.run/{model_id}/requests/{request_id}/status?logs=1
Authorization: Key <api_key>
```

**Response when IN_QUEUE (200):**
```json
{
  "status": "IN_QUEUE",
  "request_id": "764cabcf-...",
  "queue_position": 2,
  "response_url": "https://queue.fal.run/.../response"
}
```

**Response when IN_PROGRESS (200):**
```json
{
  "status": "IN_PROGRESS",
  "request_id": "764cabcf-...",
  "response_url": "https://queue.fal.run/.../response",
  "logs": [
    {
      "message": "Loading model weights...",
      "timestamp": "2026-02-17T10:30:01.123Z"
    },
    {
      "message": "Generating image...",
      "timestamp": "2026-02-17T10:30:02.456Z"
    }
  ]
}
```

**Response when COMPLETED (200):**
```json
{
  "status": "COMPLETED",
  "request_id": "764cabcf-...",
  "response_url": "https://queue.fal.run/.../response",
  "logs": [
    {"message": "Done.", "timestamp": "2026-02-17T10:30:05.789Z"}
  ],
  "metrics": {
    "inference_time": 3.42
  }
}
```

**Response when COMPLETED with error:**
```json
{
  "status": "COMPLETED",
  "request_id": "764cabcf-...",
  "response_url": "https://queue.fal.run/.../response",
  "error": "Human-readable error message",
  "error_type": "machine_readable_error_code"
}
```

### 6.3 Status Values

| Status | Meaning |
|--------|---------|
| `IN_QUEUE` | Waiting for an available runner. Has `queue_position`. |
| `IN_PROGRESS` | Runner is actively processing. Has optional `logs`. |
| `COMPLETED` | Done. May have succeeded or failed. Check `error` field. Has `metrics`. |

### 6.4 Get Result

```
GET https://queue.fal.run/{model_id}/requests/{request_id}
Authorization: Key <api_key>
```

Or use the `response_url` from the submit response:
```
GET https://queue.fal.run/{model_id}/requests/{request_id}/response
Authorization: Key <api_key>
```

**Response (200):** Model-specific JSON. Example for image generation:
```json
{
  "images": [
    {
      "url": "https://v3.fal.media/files/rabbit/abc123.png",
      "width": 1024,
      "height": 768,
      "content_type": "image/jpeg"
    }
  ],
  "timings": {
    "inference": 5.2
  },
  "seed": 12345,
  "has_nsfw_concepts": [false],
  "prompt": "a sunset over mountains"
}
```

### 6.5 Cancel Request

```
PUT https://queue.fal.run/{model_id}/requests/{request_id}/cancel
Authorization: Key <api_key>
```

**Responses:**

| HTTP Status | Body | Meaning |
|-------------|------|---------|
| 202 | `{"status": "CANCELLATION_REQUESTED"}` | Cancel queued; may still complete if already running |
| 400 | `{"status": "ALREADY_COMPLETED"}` | Request finished before cancel arrived |
| 404 | `{"status": "NOT_FOUND"}` | Request ID does not exist |

### 6.6 Timing & Metrics

The COMPLETED status includes:
- `metrics.inference_time` -- seconds the runner spent processing

The result body (from Get Result) may include:
- `timings` object -- model-specific timing breakdown (e.g., `{"inference": 5.2}`)

**Not reported in API responses:** queue wait time, startup/cold-start time, network latency, cost.

### 6.7 Queue Guarantees

- Requests are never dropped; no queue size limit
- Auto-scales runners based on demand
- Automatic retries: up to 10 times (for 503, 504, connection errors)
- Retries respect `X-Fal-Request-Timeout` deadline
- Retries can be disabled with `X-Fal-No-Retry: 1` header

---

## 7. Streaming Status (SSE)

```
GET https://queue.fal.run/{model_id}/requests/{request_id}/status/stream
GET https://queue.fal.run/{model_id}/requests/{request_id}/status/stream?logs=1
Authorization: Key <api_key>
```

**Response:** `Content-Type: text/event-stream`

Each SSE event contains a JSON status object with the same format as the polling endpoint. Connection stays open until status is `COMPLETED`.

Example SSE stream:
```
data: {"status":"IN_QUEUE","request_id":"abc...","queue_position":2,"response_url":"..."}

data: {"status":"IN_PROGRESS","request_id":"abc...","response_url":"...","logs":[...]}

data: {"status":"COMPLETED","request_id":"abc...","response_url":"...","metrics":{"inference_time":3.42}}
```

---

## 8. WebSocket API

### 8.1 Real-Time WebSocket (Model-Native)

**Source:** https://fal.ai/docs/model-apis/real-time/quickstart

Only available for models with explicit real-time support (e.g., `fal-ai/fast-lcm-diffusion`, `fal-ai/fast-turbo-diffusion`).

**URL:** `wss://fal.run/{model_id}/realtime`

- Bypasses queue entirely, connects directly to a runner
- Sub-100ms latency
- Uses binary msgpack serialization by default (JSON also supported)
- Bidirectional: send request JSON, receive result JSON

### 8.2 HTTP-over-WebSocket (Any Model)

**Source:** https://fal.ai/docs/model-apis/model-endpoints/websockets

**URL:** `wss://ws.fal.run/{model_id}`

Protocol is a four-step sequence:

1. **Client sends:** JSON request payload (same as HTTP POST body)
2. **Server sends start metadata:**
   ```json
   {
     "type": "start",
     "request_id": "<uuid>",
     "status": 200,
     "headers": {"content-type": "application/json", ...}
   }
   ```
3. **Server streams response:** Binary chunks or JSON objects based on Content-Type
4. **Server sends end metadata:**
   ```json
   {
     "type": "end",
     "request_id": "<uuid>",
     "status": 200,
     "time_to_first_byte_seconds": 0.42
   }
   ```

---

## 9. Webhooks

**Source:** https://fal.ai/docs/model-apis/model-endpoints/webhooks

### Submitting with Webhook

Pass webhook URL as query parameter:
```
POST https://queue.fal.run/{model_id}?fal_webhook=https://your-server.com/webhook
Authorization: Key <api_key>
Content-Type: application/json

{...model inputs...}
```

Returns the same 202 submit response with `request_id`.

### Webhook Callback Format

fal sends a POST to your webhook URL when processing completes:

```json
{
  "request_id": "string",
  "gateway_request_id": "string",
  "status": "OK",
  "payload": {
    "images": [...],
    ...model-specific result
  }
}
```

Error callback:
```json
{
  "request_id": "string",
  "gateway_request_id": "string",
  "status": "ERROR",
  "error": "Invalid status code: 422",
  "payload": {"detail": [{"loc": ["body", "prompt"], "msg": "field required"}]}
}
```

### Webhook Headers (for verification)

| Header | Description |
|--------|-------------|
| `X-Fal-Webhook-Request-Id` | Unique request identifier |
| `X-Fal-Webhook-User-Id` | Your user ID |
| `X-Fal-Webhook-Timestamp` | Unix epoch seconds |
| `X-Fal-Webhook-Signature` | ED25519 signature (hex format) |

### Webhook Verification

Verify using ED25519 signatures from JWKS endpoint: `https://rest.fal.ai/.well-known/jwks.json`

Process:
1. Verify timestamp is within +/-5 minutes
2. Compute SHA-256 hash of request body
3. Concatenate headers with hash using newlines
4. Verify signature against JWKS public key

### Retry Behavior

- Initial delivery timeout: 15 seconds
- Retries: up to 10 times over 2 hours
- Design handlers as idempotent (use `request_id` for dedup)

---

## 10. Platform APIs

Base URL: `https://api.fal.ai`

### 10.1 List/Search Models

```
GET https://api.fal.ai/v1/models
Authorization: Key <api_key>  (optional, gives higher rate limits)
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | integer (min: 1) | Max items returned |
| `cursor` | string | Pagination token |
| `endpoint_id` | string or array | Specific model ID(s), 1-50 |
| `q` | string | Free-text search |
| `category` | string | Filter: `text-to-image`, `image-to-video`, etc. |
| `status` | enum | `active` or `deprecated` |
| `expand` | string or array | `openapi-3.0`, `enterprise_status` |

**Response (200):**
```json
{
  "models": [
    {
      "endpoint_id": "fal-ai/flux/dev",
      "metadata": {
        "display_name": "FLUX.1 [dev]",
        "category": "text-to-image",
        "description": "...",
        "status": "active",
        "tags": ["pro"],
        "updated_at": "2026-01-15T...",
        "is_favorited": null,
        "thumbnail_url": "https://...",
        "thumbnail_animated_url": "https://...",
        "model_url": "https://fal.ai/models/fal-ai/flux/dev",
        "github_url": "https://...",
        "license_type": "commercial",
        "date": "2024-08-01T...",
        "group": {"key": "flux", "label": "FLUX"},
        "highlighted": true,
        "kind": "inference",
        "training_endpoint_ids": [],
        "inference_endpoint_ids": [],
        "stream_url": "...",
        "duration_estimate": 0.5,
        "pinned": false
      },
      "openapi": { ... }
    }
  ],
  "next_cursor": "Mg==",
  "has_more": true
}
```

### 10.2 Get Model Pricing

```
GET https://api.fal.ai/v1/models/pricing?endpoint_id=fal-ai/flux/dev
Authorization: Key <api_key>
```

Accepts 1-50 endpoint IDs (comma-separated or repeated param).

**Response (200):**
```json
{
  "prices": [
    {
      "endpoint_id": "fal-ai/flux/dev",
      "unit_price": 0.025,
      "unit": "image",
      "currency": "USD"
    }
  ],
  "next_cursor": null,
  "has_more": false
}
```

**Unit values:** `"image"`, `"video"`, or provider-specific GPU unit strings.

### 10.3 Estimate Cost

```
POST https://api.fal.ai/v1/models/pricing/estimate
Authorization: Key <api_key>
Content-Type: application/json
```

**Two estimation modes:**

Mode 1 -- Historical API Price (based on past usage patterns):
```json
{
  "estimate_type": "historical_api_price",
  "endpoints": {
    "fal-ai/flux/dev": {"call_quantity": 100},
    "fal-ai/flux/schnell": {"call_quantity": 50}
  }
}
```

Mode 2 -- Unit Price (multiply unit price by count):
```json
{
  "estimate_type": "unit_price",
  "endpoints": {
    "fal-ai/flux/dev": {"unit_quantity": 50},
    "fal-ai/flux-pro": {"unit_quantity": 25}
  }
}
```

**Response (200):**
```json
{
  "estimate_type": "historical_api_price",
  "total_cost": 3.75,
  "currency": "USD"
}
```

### 10.4 Standard Error Format (Platform APIs)

```json
{
  "error": {
    "type": "validation_error",
    "message": "...",
    "docs_url": "https://...",
    "request_id": "..."
  }
}
```

Status codes: 400 (validation), 401 (auth), 403 (forbidden), 404 (not found), 429 (rate limit), 500 (server error).

---

## 11. Image Generation Models

### 11.1 FLUX.1 [schnell] -- Fast Generation

**Endpoint:** `fal-ai/flux/schnell`
**Pricing:** $0.003/megapixel (billed rounding up to nearest MP)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text prompt |
| `image_size` | enum or object | `landscape_4_3` | `square_hd`, `square`, `portrait_4_3`, `portrait_16_9`, `landscape_4_3`, `landscape_16_9` OR `{"width": int, "height": int}` |
| `num_inference_steps` | integer | 4 | Range: 1-12 |
| `seed` | integer | null | Reproducibility |
| `guidance_scale` | float | 3.5 | Range: 1-20 |
| `num_images` | integer | 1 | Range: 1-4 |
| `enable_safety_checker` | boolean | true | NSFW filter |
| `output_format` | enum | `jpeg` | `jpeg`, `png` |
| `acceleration` | enum | `none` | `none`, `regular`, `high` |
| `sync_mode` | boolean | false | Return as data URI |

**Output:**
```json
{
  "images": [
    {
      "url": "https://v3.fal.media/files/...",
      "width": 1024,
      "height": 768,
      "content_type": "image/jpeg"
    }
  ],
  "timings": {"inference": 1.2},
  "seed": 12345,
  "has_nsfw_concepts": [false],
  "prompt": "..."
}
```

### 11.2 FLUX.1 [dev] -- High Quality

**Endpoint:** `fal-ai/flux/dev`
**Pricing:** $0.025/megapixel

Same parameters as schnell except:
- `num_inference_steps` default: 28 (higher quality)

### 11.3 FLUX.2 [pro] -- Premium

**Endpoint:** `fal-ai/flux-2-pro`
**Pricing:** $0.03 for first megapixel, $0.015 per extra MP

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text prompt |
| `image_size` | enum or object | `landscape_4_3` | Same as FLUX.1 + custom `{"width": int, "height": int}` (max 14,142px) |
| `seed` | integer | null | Reproducibility |
| `safety_tolerance` | enum | `2` | Scale 1-5 (1=strictest, 5=most permissive) |
| `enable_safety_checker` | boolean | true | NSFW filter |
| `output_format` | enum | `jpeg` | `jpeg`, `png` |
| `sync_mode` | boolean | false | Return as data URI |

**Output:**
```json
{
  "images": [
    {
      "url": "string",
      "content_type": "string",
      "file_name": "string",
      "file_size": 123456,
      "width": 1024,
      "height": 1024
    }
  ],
  "seed": 42
}
```

### 11.4 Other FLUX Variants

| Endpoint ID | Description |
|-------------|-------------|
| `fal-ai/flux-2-flex` | Flexible editing with customizable steps |
| `fal-ai/flux-lora` | FLUX.1 with LoRA fine-tuning support |
| `fal-ai/flux-kontext-lora` | Kontext model with LoRA |

### 11.5 Other Image Models

| Endpoint ID | Description |
|-------------|-------------|
| `fal-ai/nano-banana-2` | Text-to-image |
| `fal-ai/nano-banana-pro/edit` | Image editing |
| `fal-ai/recraft/v4/pro/text-to-image` | Designer-focused |
| `fal-ai/qwen-image-2/text-to-image` | Strong typography |
| `fal-ai/fast-sdxl` | Stable Diffusion XL |

---

## 12. Video Generation Models

### 12.1 Kling 2.1 Pro (Image-to-Video)

**Endpoint:** `fal-ai/kling-video/v2.1/pro/image-to-video`
**Pricing:** $0.49/5s video, $0.90/10s, $0.098/extra second

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Motion/scene description |
| `image_url` | string | **required** | Source image URL |
| `duration` | enum | `"5"` | `"3"` to `"15"` (seconds, as string) |
| `negative_prompt` | string | `"blur, distort, and low quality"` | Elements to avoid |
| `cfg_scale` | float | 0.5 | Range: 0-1 |
| `tail_image_url` | string | null | End frame image |

**Output:**
```json
{
  "video": {
    "url": "https://...",
    "file_name": "output.mp4",
    "file_size": 1234567,
    "content_type": "video/mp4"
  }
}
```

### 12.2 MiniMax (Hailuo) Video 01

**Endpoint:** `fal-ai/minimax/video-01`
**Pricing:** ~5 credits per request

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Video description |
| `prompt_optimizer` | boolean | true | Auto-optimize prompt |

**Output:**
```json
{
  "video": {
    "url": "https://..."
  }
}
```

### 12.3 Other Video Models

| Endpoint ID | Description | Pricing |
|-------------|-------------|---------|
| `fal-ai/veo3.1` | Google Veo 3.1 | -- |
| `fal-ai/sora-2/text-to-video` | OpenAI Sora 2 | $0.30-0.50/s |
| `fal-ai/kling-video/v3/pro/image-to-video` | Kling 3.0 Pro | -- |
| `fal-ai/ltx-2-19b/image-to-video` | LTX with audio | -- |
| `fal-ai/ltx-2.3/text-to-video/fast` | Fast LTX | -- |
| `fal-ai/wan/v2.2-a14b/text-to-video` | Wan 2.5 | $0.05/s |
| `fal-ai/minimax/hailuo-02/pro/image-to-video` | Hailuo 02 Pro | -- |
| `fal-ai/minimax/hailuo-2.3/pro/image-to-video` | Hailuo 2.3 Pro | $0.49/video |

---

## 13. Audio/Speech Models

### 13.1 Whisper (Speech-to-Text)

**Endpoint:** `fal-ai/whisper`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_url` | string | **required** | URL of audio (mp3, mp4, mpeg, mpga, m4a, wav, webm) |
| `task` | enum | `transcribe` | `transcribe`, `translate` |
| `language` | enum | null | 98 language codes (auto-detected if null) |
| `diarize` | boolean | false | Speaker diarization |
| `chunk_level` | enum | `segment` | `none`, `segment`, `word` |
| `version` | string | `3` | Whisper version |
| `batch_size` | integer | 64 | Range: 1-64 |
| `prompt` | string | `""` | Generation prompt |
| `num_speakers` | integer | null | Auto-detected if null |

**Output:**
```json
{
  "text": "Full transcription text",
  "chunks": [
    {
      "text": "segment text",
      "timestamp": [0.0, 2.5],
      "speaker": "SPEAKER_00"
    }
  ],
  "inferred_languages": ["en"],
  "diarization_segments": [...]
}
```

### 13.2 Text-to-Speech Models

| Endpoint ID | Description | Pricing |
|-------------|-------------|---------|
| `fal-ai/chatterbox/text-to-speech` | 23 languages, voice cloning | $0.025/1000 chars |
| `fal-ai/f5-tts` | Voice cloning with reference audio | -- |
| `fal-ai/elevenlabs/tts/eleven-v3` | ElevenLabs integration | -- |
| `fal-ai/index-tts-2/text-to-speech` | Index TTS 2.0 | -- |
| `fal-ai/minimax/speech-02-hd` | HD speech generation | -- |
| `fal-ai/qwen-3-tts/voice-design/1.7b` | Custom voice design | -- |
| `beatoven/music-generation` | Royalty-free music | -- |
| `beatoven/sound-effect-generation` | Sound effects | -- |

---

## 14. Upscaling & Utility Models

### 14.1 ESRGAN (Image Upscaling)

**Endpoint:** `fal-ai/esrgan`
**Pricing:** $0.00111/compute second

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_url` | string | **required** | Input image URL |
| `scale` | float | 2 | Range: 1-8 |
| `tile` | integer | 0 | Tile size for memory (e.g., 200, 400) |
| `face` | boolean | false | Face-specific optimization |
| `model` | enum | `RealESRGAN_x4plus` | See variants below |
| `output_format` | enum | `png` | `png`, `jpeg` |

**Model Variants:** `RealESRGAN_x4plus`, `RealESRGAN_x2plus`, `RealESRGAN_x4plus_anime_6B`, `RealESRGAN_x4_v3`, `RealESRGAN_x4_wdn_v3`, `RealESRGAN_x4_anime_v3`

**Output:**
```json
{
  "image": {
    "url": "string",
    "content_type": "image/png",
    "file_name": "string",
    "file_size": 123456,
    "width": 2048,
    "height": 2048
  }
}
```

### 14.2 Other Utility Models

| Endpoint ID | Description |
|-------------|-------------|
| `fal-ai/bria/background/remove` | Background removal |
| `fal-ai/topaz/upscale/image` | Topaz upscaling |
| `fal-ai/x-ailab/nsfw` | Content moderation/NSFW detection |

---

## 15. Pricing & Billing

**Source:** https://fal.ai/pricing

### Billing Model

- **Pay-per-use** with output-based pricing
- No idle charges, no hidden fees for API calls, storage, or CDN
- No payment required when app is idle

### Billing Units by Model Type

| Type | Billing Unit | Example |
|------|-------------|---------|
| Image | Per megapixel (rounded up) | FLUX schnell: $0.003/MP |
| Image | Per image (flat rate) | FLUX.2 Pro: $0.03/image |
| Video | Per second of output | Wan 2.5: $0.05/s |
| Video | Per video (flat rate) | Hailuo 2.3: $0.28-$0.49/video |
| Audio/TTS | Per 1,000 characters | Chatterbox: $0.025/1K chars |
| Compute | Per compute second | ESRGAN: $0.00111/s |

### Cost Is NOT Returned in API Responses

There is no per-request cost field in queue status or result responses. Use the Platform API pricing endpoints to estimate costs:
- `GET /v1/models/pricing` for unit prices
- `POST /v1/models/pricing/estimate` for cost estimation

### GPU Compute Rates (for custom serverless apps)

| GPU | Hourly Rate | Per-Second |
|-----|-------------|------------|
| H100 (80GB) | $1.89/hr | $0.0005/s |
| H200 (141GB) | $2.10/hr | $0.0006/s |
| A100 (40GB) | $0.99/hr | $0.0003/s |
| B200 (184GB) | Custom | Contact sales |

### Select Model Pricing (as of March 2026)

| Model | Price |
|-------|-------|
| FLUX.1 schnell | $0.003/MP |
| FLUX.1 dev | $0.025/MP |
| FLUX.2 Pro | $0.03 first MP + $0.015/extra MP |
| Flux Kontext Pro | $0.04/image |
| Seedream V4 | $0.03/image |
| Kling 2.5 Turbo Pro | $0.07/s video |
| Sora 2 Pro (720p) | $0.30/s |
| Sora 2 Pro (1080p) | $0.50/s |
| Wan 2.5 | $0.05/s |
| Veo 3 | $0.40/s |
| Kokoro TTS | $0.02/1K chars |
| Chatterbox | $0.025/1K chars |

---

## 16. Error Handling

**Source:** https://fal.ai/docs/model-apis/errors

### Error Response Formats

**Model-level errors** (validation, content policy, etc.):
```json
{
  "detail": [
    {
      "loc": ["body", "prompt"],
      "msg": "field required",
      "type": "missing",
      "url": "https://docs.fal.ai/errors/#missing",
      "ctx": {},
      "input": {}
    }
  ]
}
```

**Request-level errors** (timeouts, runner issues):
```json
{
  "detail": "Request exceeded allowed processing time",
  "error_type": "request_timeout"
}
```

Also returned as `X-Fal-Error-Type` HTTP header.

### Model-Level Error Types

| error_type | HTTP Status | Retryable | Description |
|------------|-------------|-----------|-------------|
| `internal_server_error` | 500 | Maybe | Unexpected server issue |
| `generation_timeout` | 504 | Maybe | Operation exceeded time limit |
| `downstream_service_error` | 400 | Maybe | External service problem |
| `downstream_service_unavailable` | 500 | Maybe | Third-party service unavailable |
| `content_policy_violation` | 422 | No | Input flagged (NSFW, hate, illegal) |
| `no_media_generated` | 422 | No | Model produced no output |
| `image_too_small` | 422 | No | Below minimum dimensions (ctx: `min_height`, `min_width`) |
| `image_too_large` | 422 | No | Exceeds maximum dimensions (ctx: `max_height`, `max_width`) |
| `image_load_error` | 422 | No | Failed to load/process image |
| `file_download_error` | 422 | No | Failed to download URL |
| `face_detection_error` | 422 | No | No face detected |
| `file_too_large` | 422 | No | Exceeds max file size (ctx: `max_size`) |
| `greater_than` | 422 | No | Value not > threshold (ctx: `gt`) |
| `greater_than_equal` | 422 | No | Value not >= threshold (ctx: `ge`) |
| `less_than` | 422 | No | Value not < threshold (ctx: `lt`) |
| `less_than_equal` | 422 | No | Value not <= threshold (ctx: `le`) |
| `multiple_of` | 422 | No | Not a multiple of (ctx: `multiple_of`) |
| `sequence_too_short` | 422 | No | Too few items (ctx: `min_length`) |
| `sequence_too_long` | 422 | No | Too many items (ctx: `max_length`) |
| `one_of` | 422 | No | Not an allowed value (ctx: `expected`) |
| `feature_not_supported` | 422 | No | Unsupported parameter combination |
| `invalid_archive` | 422 | No | Unreadable archive (ctx: `supported_extensions`) |
| `archive_file_count_below_minimum` | 422 | No | Too few files in archive |
| `archive_file_count_exceeds_maximum` | 422 | No | Too many files in archive |
| `audio_duration_too_long` | 422 | No | Audio exceeds max duration |
| `audio_duration_too_short` | 422 | No | Audio below min duration |
| `unsupported_audio_format` | 422 | No | Bad audio format (ctx: `supported_formats`) |
| `unsupported_image_format` | 422 | No | Bad image format (ctx: `supported_formats`) |
| `unsupported_video_format` | 422 | No | Bad video format (ctx: `supported_formats`) |
| `video_duration_too_long` | 422 | No | Video exceeds max duration |
| `video_duration_too_short` | 422 | No | Video below min duration |

### Request-Level Error Types

| error_type | HTTP Status | Retryable | Description |
|------------|-------------|-----------|-------------|
| `request_timeout` | 504 | Yes | Exceeded processing time |
| `startup_timeout` | 504 | Yes | Runner didn't start in time |
| `runner_scheduling_failure` | 503 | Yes | No runner available |
| `runner_connection_timeout` | 503 | Yes | Connection to runner timed out |
| `runner_disconnected` | 503 | Yes | Runner disconnected unexpectedly |
| `runner_connection_refused` | 503 | Yes | Runner rejected connection |
| `runner_connection_error` | 503 | Yes | General connectivity issue |
| `runner_incomplete_response` | 502 | Yes | Partial response received |
| `runner_server_error` | 500 | Yes | Runner internal error |
| `client_disconnected` | 499 | No | Client closed connection early |
| `client_cancelled` | 499 | No | Client-initiated cancellation |
| `bad_request` | 400 | No | Malformed request |
| `internal_error` | 500 | No | Unexpected platform error |

---

## 17. File Storage & Media URLs

### Output Media URLs

Generated media is hosted on fal's CDN. URL formats observed:
- `https://v3.fal.media/files/{random}/{hash}.{ext}`
- `https://fal.media/files/{random}/{hash}.{ext}`

URLs are publicly accessible (no auth needed to download).

### Media Expiration

URLs are subject to expiration settings. Control via:
- `X-Fal-Object-Lifecycle-Preference` header on submit

### File Upload

For inputs requiring URLs (e.g., `image_url`), you can:
1. Use any publicly accessible URL
2. Upload via fal's storage API (SDK: `fal.storage.upload(file)`)
3. Use base64 data URIs for small files

---

## 18. Request Headers Reference

All optional headers for queue submit requests:

| Header | Values | Description |
|--------|--------|-------------|
| `Authorization` | `Key <api_key>` | **Required.** Authentication. |
| `Content-Type` | `application/json` | **Required.** Request body format. |
| `X-Fal-Request-Timeout` | integer (seconds) | Server-side processing deadline |
| `X-Fal-Runner-Hint` | string | Session affinity hint (route to same runner) |
| `X-Fal-Queue-Priority` | `"normal"` or `"low"` | Queue priority |
| `X-Fal-Store-IO` | `"0"` | Disable payload storage |
| `X-Fal-No-Retry` | `"1"`, `"true"`, `"yes"` | Disable automatic retries |
| `X-Fal-Object-Lifecycle-Preference` | string | Control media URL expiration |

---

## 19. Implementation Notes for Rust

### Recommended Architecture

1. **Primary flow:** Queue API (submit -> poll/stream -> get result)
   - POST to `queue.fal.run/{model}` to submit
   - GET status endpoint to poll (or use SSE stream)
   - GET result endpoint when COMPLETED
   - Use `request_id` for tracking

2. **Auth:** Simple header: `Authorization: Key {api_key}`

3. **Polling strategy:**
   - Check `queue_position` from IN_QUEUE status to estimate wait
   - Use SSE streaming (`/status/stream`) for efficient waiting instead of polling
   - Exponential backoff if polling: start at 500ms, cap at 5s

4. **Error handling:**
   - COMPLETED status can mean success OR failure -- always check for `error` field
   - Retryable errors: all 5xx request-level errors
   - Non-retryable: all 422 model-level errors
   - Parse both `detail` array format (model errors) and flat `detail` string format (request errors)

5. **Key types to model in Rust:**

```rust
// Submit response
struct QueueSubmitResponse {
    request_id: String,
    response_url: String,
    status_url: String,
    cancel_url: String,
    queue_position: u32,
}

// Status response (polymorphic on status field)
enum QueueStatus {
    InQueue {
        request_id: String,
        queue_position: u32,
        response_url: String,
    },
    InProgress {
        request_id: String,
        response_url: String,
        logs: Option<Vec<LogEntry>>,
    },
    Completed {
        request_id: String,
        response_url: String,
        logs: Option<Vec<LogEntry>>,
        metrics: Option<Metrics>,
        error: Option<String>,
        error_type: Option<String>,
    },
}

struct LogEntry {
    message: String,
    timestamp: String, // ISO-8601
}

struct Metrics {
    inference_time: Option<f64>, // seconds
}

// Image output (common across image models)
struct ImageOutput {
    images: Vec<Image>,
    timings: Option<serde_json::Value>,
    seed: Option<i64>,
    has_nsfw_concepts: Option<Vec<bool>>,
    prompt: Option<String>,
}

struct Image {
    url: String,
    width: u32,
    height: u32,
    content_type: String,
    file_name: Option<String>,
    file_size: Option<u64>,
}

// Video output
struct VideoOutput {
    video: VideoFile,
}

struct VideoFile {
    url: String,
    content_type: Option<String>,
    file_name: Option<String>,
    file_size: Option<u64>,
}

// Webhook callback
struct WebhookPayload {
    request_id: String,
    gateway_request_id: String,
    status: String, // "OK" or "ERROR"
    payload: Option<serde_json::Value>,
    error: Option<String>,
    payload_error: Option<String>,
}

// Cancel response
struct CancelResponse {
    status: String, // "CANCELLATION_REQUESTED", "ALREADY_COMPLETED", "NOT_FOUND"
}

// Platform API: Model info
struct ModelInfo {
    endpoint_id: String,
    metadata: Option<ModelMetadata>,
    openapi: Option<serde_json::Value>,
}

struct ModelMetadata {
    display_name: String,
    category: String,
    description: String,
    status: String, // "active", "deprecated"
    tags: Vec<String>,
    updated_at: String,
    license_type: String, // "commercial", "research", "private"
    kind: Option<String>, // "inference", "training"
    duration_estimate: Option<f64>,
    // ... other fields
}

// Platform API: Pricing
struct PriceInfo {
    endpoint_id: String,
    unit_price: f64,
    unit: String, // "image", "video", etc.
    currency: String, // "USD"
}
```

6. **HTTP client considerations:**
   - SSE support needed for `/status/stream`
   - WebSocket support optional (for real-time models)
   - All responses are JSON (Content-Type: application/json)
   - Media downloads are direct HTTP GET (no auth needed)
   - Consider connection pooling for high-throughput polling

7. **Rate limiting:**
   - Platform API (api.fal.ai): authenticated requests get higher limits
   - Model API (fal.run / queue.fal.run): no documented rate limits, but auto-scaling may queue
   - Handle 429 responses with exponential backoff

---

## Source URLs

All information was gathered on March 18, 2026 from these pages:

- Main docs: https://fal.ai/docs/documentation
- Queue API: https://fal.ai/docs/model-apis/model-endpoints/queue
- Sync requests: https://fal.ai/docs/model-apis/model-endpoints/synchronous-requests
- Webhooks: https://fal.ai/docs/model-apis/model-endpoints/webhooks
- WebSocket: https://fal.ai/docs/model-apis/model-endpoints/websockets
- Real-time: https://fal.ai/docs/model-apis/real-time/quickstart
- Authentication: https://fal.ai/docs/reference/platform-apis/authentication
- Error reference: https://fal.ai/docs/model-apis/errors
- Model search API: https://fal.ai/docs/platform-apis/v1/models
- Pricing API: https://fal.ai/docs/platform-apis/v1/models/pricing
- Cost estimate API: https://fal.ai/docs/platform-apis/v1/models/pricing/estimate
- Key management: https://fal.ai/docs/platform-apis/v1/keys/create
- FLUX schnell API: https://fal.ai/models/fal-ai/flux/schnell/api
- FLUX dev API: https://fal.ai/models/fal-ai/flux/dev/api
- FLUX.2 Pro: https://fal.ai/models/fal-ai/flux-2-pro
- ESRGAN API: https://fal.ai/models/fal-ai/esrgan/api
- Whisper API: https://fal.ai/models/fal-ai/whisper/api
- Kling video: https://fal.ai/models/fal-ai/kling-video/v2.1/pro/image-to-video
- MiniMax video: https://fal.ai/models/fal-ai/minimax/video-01/api
- Model explorer: https://fal.ai/explore/models
- Pricing page: https://fal.ai/pricing
- OpenAPI spec: https://fal.ai/docs/api-reference/platform-apis/openapi/v1.json
