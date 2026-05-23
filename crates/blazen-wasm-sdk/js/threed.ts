// SPDX-License-Identifier: MPL-2.0
//
// `js/threed.ts` — JS-side shim for the Blazen 3D pipeline.
//
// `blazen-3d`'s `Compat3dProvider` HTTP client is built on `reqwest` and
// does not target `wasm32-unknown-unknown` cleanly, so the WASM SDK
// exposes the request / result records as `#[wasm_bindgen]` Tsify mirrors
// (see `src/threed.rs`) and this shim takes care of the actual HTTP
// transport on top of the browser `fetch()` API.
//
// The wire format mirrors `crates/blazen-3d/src/backends/compat.rs`
// verbatim:
//
//   * `POST {baseUrl}/v1/3d/{texturize,rig,refine,animate}`
//     (compat.rs:28-33, 295-298)
//   * `multipart/form-data` body with:
//       - `mesh.glb` part — the input mesh as `application/octet-stream`
//         (compat.rs:208-213, `mesh_part`).
//       - `request.json` part — the request struct serialised as JSON,
//         `application/json` (compat.rs:216-222, `request_json_part`).
//         Binary fields on the request (`reference_image`, `driving_video`,
//         `bvh_motion`) are stripped from the JSON and uploaded as
//         separate parts named after the field (compat.rs:11-19,
//         312-346 / 480-520).
//   * Optional `Authorization: Bearer <apiKey>` header (compat.rs:122-127).
//   * Response: JSON with base64-encoded payloads under `*_b64` keys
//     (compat.rs:21-25, 140-197) — for the texturize response, the
//     optional `pbr_maps` bundle uses `albedo_png_b64`, `normal_png_b64`,
//     `roughness_png_b64`, `metallic_png_b64` (compat.rs:160-168).
//
// The default per-request timeout is 10 minutes, matching the native
// `DEFAULT_TIMEOUT` (compat.rs:62). 3D inference can run for minutes
// (texturize: tens of seconds, animate: minutes), hence the generous
// default.

import type {
  ThreedAnimateRequest,
  ThreedAnimateResult,
  ThreedPbrMaps,
  ThreedRefineRequest,
  ThreedRefineResult,
  ThreedRefineStats,
  ThreedRigRequest,
  ThreedRigResult,
  ThreedTexturizeRequest,
  ThreedTexturizeResult,
} from "../pkg/blazen_wasm_sdk";

// ---------------------------------------------------------------------------
// Provider configuration
// ---------------------------------------------------------------------------

/**
 * Configuration for {@link Compat3dProvider}. Mirrors
 * `Compat3dProvider::new` + `with_api_key` + `with_timeout` in
 * `crates/blazen-3d/src/backends/compat.rs`.
 */
export interface Compat3dProviderConfig {
  /** Upstream service root (e.g. `"https://3d.example.com"`). */
  baseUrl: string;
  /** Optional bearer token, attached as `Authorization: Bearer <key>`. */
  apiKey?: string;
  /**
   * Per-request timeout in milliseconds. Defaults to 600_000 ms
   * (10 minutes), matching `DEFAULT_TIMEOUT` in
   * `crates/blazen-3d/src/backends/compat.rs` (line 62).
   */
  timeoutMs?: number;
  /**
   * Optional custom `fetch` implementation — handy for tests, edge
   * runtimes, or call-site instrumentation. Defaults to the global
   * `fetch`.
   */
  fetchFn?: typeof fetch;
}

const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000;
const GLB_MIME = "model/gltf-binary";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/**
 * Thrown by every {@link Compat3dProvider} method on a non-2xx response
 * or a malformed response body. Mirrors the
 * `Backend(format!("http {status}: ..."))` mapping in
 * `crates/blazen-3d/src/backends/compat.rs` (lines 279-284).
 */
export class Compat3dError extends Error {
  /** HTTP status code, or `0` for network / decode failures. */
  readonly status: number;
  /** Pipeline stage that produced the error. */
  readonly stage: "texturize" | "rig" | "refine" | "animate";

  constructor(
    stage: Compat3dError["stage"],
    status: number,
    message: string,
  ) {
    super(`compat3d ${stage} failed (http ${status}): ${message}`);
    this.name = "Compat3dError";
    this.status = status;
    this.stage = stage;
  }
}

// ---------------------------------------------------------------------------
// Internal multipart / fetch helpers
// ---------------------------------------------------------------------------

type Stage = Compat3dError["stage"];

function bytesToBlob(bytes: Uint8Array | number[], mime: string): Blob {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return new Blob([buf], { type: mime });
}

/**
 * Base64-decode a string into a `Uint8Array`. Matches the native
 * `B64.decode(...)` calls in `crates/blazen-3d/src/backends/compat.rs`
 * (e.g. line 353 for `textured_glb_b64`).
 */
function decodeB64(b64: string): Uint8Array {
  // `atob` is available in browsers and modern Node (>= 16). Use it
  // rather than pulling in a base64 polyfill.
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

interface MultipartInput {
  meshBytes: Uint8Array | number[];
  /** JSON body for the `request.json` part. Binary fields must already be stripped. */
  requestJson: Record<string, unknown>;
  /** Side-channel binary parts named after the field they came from. */
  binaryParts?: Array<{ name: string; bytes: Uint8Array | number[] }>;
}

function buildMultipart(input: MultipartInput): FormData {
  const form = new FormData();
  // `mesh.glb` part — `application/octet-stream`, file_name `mesh.glb`.
  // Compat3dProvider::mesh_part: crates/blazen-3d/src/backends/compat.rs:208-213
  form.append(
    "mesh.glb",
    bytesToBlob(input.meshBytes, "application/octet-stream"),
    "mesh.glb",
  );
  // `request.json` part — `application/json`, file_name `request.json`.
  // Compat3dProvider::request_json_part: crates/blazen-3d/src/backends/compat.rs:216-222
  const jsonBlob = new Blob([JSON.stringify(input.requestJson)], {
    type: "application/json",
  });
  form.append("request.json", jsonBlob, "request.json");
  // Side-channel binary parts. `application/octet-stream`, file_name
  // is the part name itself.
  // Compat3dProvider::binary_part: crates/blazen-3d/src/backends/compat.rs:226-230
  if (input.binaryParts) {
    for (const { name, bytes } of input.binaryParts) {
      form.append(name, bytesToBlob(bytes, "application/octet-stream"), name);
    }
  }
  return form;
}

async function postMultipart(
  config: Compat3dProviderConfig,
  stage: Stage,
  endpoint: string,
  form: FormData,
): Promise<unknown> {
  const baseUrl = config.baseUrl.replace(/\/+$/, "");
  const url = `${baseUrl}${endpoint}`;
  const headers: Record<string, string> = {};
  if (config.apiKey) {
    headers["Authorization"] = `Bearer ${config.apiKey}`;
  }
  const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const fetchFn = config.fetchFn ?? fetch;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let resp: Response;
  try {
    resp = await fetchFn(url, {
      method: "POST",
      body: form,
      headers,
      signal: controller.signal,
    });
  } catch (err) {
    throw new Compat3dError(
      stage,
      0,
      `network: ${err instanceof Error ? err.message : String(err)}`,
    );
  } finally {
    clearTimeout(timer);
  }

  const text = await resp.text();
  if (!resp.ok) {
    // Mirrors `cap(text, 4 * 1024)` in compat.rs:282 — keep the
    // upstream body bounded so a chatty server can't blow up the log.
    const body = text.length > 4096 ? `${text.slice(0, 4096)}...[truncated]` : text;
    throw new Compat3dError(stage, resp.status, body);
  }
  if (text.length === 0) {
    throw new Compat3dError(stage, resp.status, "upstream returned empty body");
  }
  try {
    return JSON.parse(text) as unknown;
  } catch (err) {
    throw new Compat3dError(
      stage,
      resp.status,
      `response decode: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

function optionalB64(s: string | null | undefined): Uint8Array | undefined {
  if (s === null || s === undefined) {
    return undefined;
  }
  return decodeB64(s);
}

// ---------------------------------------------------------------------------
// Compat3dProvider
// ---------------------------------------------------------------------------

/**
 * HTTP-proxy client for the Blazen 3D pipeline. Mirrors the native
 * `Compat3dProvider` (`crates/blazen-3d/src/backends/compat.rs`) so the
 * WASM SDK can drive the same upstream service from the browser.
 *
 * ```ts
 * import init, { ThreedTexturizeRequest } from "@blazen-dev/wasm";
 * import { Compat3dProvider } from "@blazen-dev/wasm/js/threed";
 *
 * await init();
 *
 * const provider = new Compat3dProvider({
 *   baseUrl: "https://3d.example.com",
 *   apiKey: import.meta.env.BLAZEN_3D_KEY,
 * });
 *
 * const req: ThreedTexturizeRequest = {
 *   prompt: "weathered bronze",
 *   pbr: true,
 *   resolution: 1024,
 * };
 * const result = await provider.texturize(meshGlbBytes, req);
 * // result.texturedGlb is a Uint8Array of GLB bytes.
 * ```
 */
export class Compat3dProvider {
  private readonly config: Compat3dProviderConfig;

  constructor(config: Compat3dProviderConfig) {
    this.config = { ...config };
  }

  /**
   * Apply or generate a texture/material for an existing 3D mesh.
   * Endpoint: `POST /v1/3d/texturize`
   * (see `crates/blazen-3d/src/backends/compat.rs` line 295).
   */
  async texturize(
    meshGlb: Uint8Array | number[],
    request: ThreedTexturizeRequest,
  ): Promise<ThreedTexturizeResult> {
    // Strip the optional binary field out of the JSON body so it
    // travels as a side-channel multipart part instead of
    // base64-bloating the JSON. Mirrors compat.rs:312-327.
    const { referenceImage, ...rest } = request;
    const binaryParts: MultipartInput["binaryParts"] = [];
    if (referenceImage) {
      binaryParts.push({ name: "reference_image", bytes: referenceImage });
    }
    const form = buildMultipart({
      meshBytes: meshGlb,
      requestJson: rest,
      binaryParts,
    });
    const wire = (await postMultipart(
      this.config,
      "texturize",
      "/v1/3d/texturize",
      form,
    )) as TexturizeResultWire;
    const texturedGlb = decodeB64(wire.textured_glb_b64);
    if (texturedGlb.length === 0) {
      throw new Compat3dError(
        "texturize",
        200,
        "upstream returned empty textured_glb_b64",
      );
    }
    let pbrMaps: ThreedPbrMaps | undefined;
    if (wire.pbr_maps) {
      // Mirrors compat.rs:360-378.
      const m = wire.pbr_maps;
      pbrMaps = {
        albedoPng: decodeB64(m.albedo_png_b64),
        normalPng: optionalB64(m.normal_png_b64),
        roughnessPng: optionalB64(m.roughness_png_b64),
        metallicPng: optionalB64(m.metallic_png_b64),
      };
    }
    return {
      texturedGlb,
      mimeType: wire.mime_type ?? GLB_MIME,
      pbrMaps,
    };
  }

  /**
   * Auto-rig a 3D mesh.
   * Endpoint: `POST /v1/3d/rig`
   * (see `crates/blazen-3d/src/backends/compat.rs` line 296).
   */
  async rig(
    meshGlb: Uint8Array | number[],
    request: ThreedRigRequest,
  ): Promise<ThreedRigResult> {
    const form = buildMultipart({
      meshBytes: meshGlb,
      requestJson: request as unknown as Record<string, unknown>,
    });
    const wire = (await postMultipart(
      this.config,
      "rig",
      "/v1/3d/rig",
      form,
    )) as RigResultWire;
    const riggedGlb = decodeB64(wire.rigged_glb_b64);
    if (riggedGlb.length === 0) {
      throw new Compat3dError(
        "rig",
        200,
        "upstream returned empty rigged_glb_b64",
      );
    }
    return {
      riggedGlb,
      mimeType: wire.mime_type ?? GLB_MIME,
      boneNames: wire.bone_names ?? [],
    };
  }

  /**
   * Refine a 3D mesh: decimate, fill holes, unwrap UVs, retopologize, smooth.
   * Endpoint: `POST /v1/3d/refine`
   * (see `crates/blazen-3d/src/backends/compat.rs` line 297).
   */
  async refine(
    meshGlb: Uint8Array | number[],
    request: ThreedRefineRequest,
  ): Promise<ThreedRefineResult> {
    const form = buildMultipart({
      meshBytes: meshGlb,
      requestJson: request as unknown as Record<string, unknown>,
    });
    const wire = (await postMultipart(
      this.config,
      "refine",
      "/v1/3d/refine",
      form,
    )) as RefineResultWire;
    const refinedGlb = decodeB64(wire.refined_glb_b64);
    if (refinedGlb.length === 0) {
      throw new Compat3dError(
        "refine",
        200,
        "upstream returned empty refined_glb_b64",
      );
    }
    // The native `RefineStats` ships through JSON verbatim — its
    // field names are already snake_case on the wire, so we map to
    // the camelCase TS shape here.
    const stats: ThreedRefineStats = {
      inputTriCount: wire.stats.input_tri_count,
      outputTriCount: wire.stats.output_tri_count,
      uvChartCount:
        wire.stats.uv_chart_count === null ? undefined : wire.stats.uv_chart_count,
    };
    return {
      refinedGlb,
      mimeType: wire.mime_type ?? GLB_MIME,
      stats,
    };
  }

  /**
   * Animate a rigged 3D mesh from a prompt, BVH clip, or driving video.
   * Endpoint: `POST /v1/3d/animate`
   * (see `crates/blazen-3d/src/backends/compat.rs` line 298).
   */
  async animate(
    riggedGlb: Uint8Array | number[],
    request: ThreedAnimateRequest,
  ): Promise<ThreedAnimateResult> {
    // Strip binary fields out of the JSON body so they travel as
    // side-channel multipart parts. Mirrors compat.rs:480-495.
    const { drivingVideo, bvhMotion, ...rest } = request;
    const binaryParts: MultipartInput["binaryParts"] = [];
    if (drivingVideo) {
      binaryParts.push({ name: "driving_video", bytes: drivingVideo });
    }
    if (bvhMotion) {
      binaryParts.push({ name: "bvh_motion", bytes: bvhMotion });
    }
    const form = buildMultipart({
      meshBytes: riggedGlb,
      requestJson: rest,
      binaryParts,
    });
    const wire = (await postMultipart(
      this.config,
      "animate",
      "/v1/3d/animate",
      form,
    )) as AnimateResultWire;
    const animatedGlb = decodeB64(wire.animated_glb_b64);
    if (animatedGlb.length === 0) {
      throw new Compat3dError(
        "animate",
        200,
        "upstream returned empty animated_glb_b64",
      );
    }
    return {
      animatedGlb,
      mimeType: wire.mime_type ?? GLB_MIME,
      durationSeconds: wire.duration_seconds,
      fps: wire.fps,
    };
  }
}

// ---------------------------------------------------------------------------
// Wire shapes
// ---------------------------------------------------------------------------
//
// These are the snake_case `*Wire` deserialization shapes from
// `crates/blazen-3d/src/backends/compat.rs` lines 140-197. Kept private
// so callers always see the camelCase Tsify shapes; the snake_case
// renaming happens inside each provider method.

interface PbrMapsWire {
  albedo_png_b64: string;
  normal_png_b64?: string | null;
  roughness_png_b64?: string | null;
  metallic_png_b64?: string | null;
}

interface TexturizeResultWire {
  textured_glb_b64: string;
  mime_type?: string;
  pbr_maps?: PbrMapsWire | null;
}

interface RigResultWire {
  rigged_glb_b64: string;
  mime_type?: string;
  bone_names?: string[];
}

interface RefineResultWire {
  refined_glb_b64: string;
  mime_type?: string;
  stats: {
    input_tri_count: number;
    output_tri_count: number;
    uv_chart_count?: number | null;
  };
}

interface AnimateResultWire {
  animated_glb_b64: string;
  mime_type?: string;
  duration_seconds: number;
  fps: number;
}
