/// <reference types="@cloudflare/workers-types" />

/**
 * Cloudflare Worker fronting the static Astro site at blazen.dev. Adds two
 * KV-backed pricing routes; everything else falls through to the Workers
 * Static Assets binding so the site keeps serving normally.
 *
 * Routes:
 *   GET  /api/pricing.json          → bulk catalog (the entire models map)
 *   GET  /api/pricing/model/:id     → one model's PricingEntry, or 404
 *   POST /api/pricing/upload        → replace the catalog (Bearer-auth'd)
 *
 * Storage layout in PRICING_KV:
 *   - "bulk"          → the entire `pricing.json` document as a string
 *   - "model:<id>"    → JSON of one PricingEntry (per-model O(1) lookup)
 *
 * The upload endpoint writes both the bulk blob AND fans out per-model keys
 * in one batch so /api/pricing.json and /api/pricing/model/:id are coherent.
 * Downstream readers (Blazen bindings calling refresh_default()) hit the
 * bulk endpoint at startup; spot lookups hit the per-model endpoint and
 * fall back fast on miss.
 */

interface Env {
    ASSETS: Fetcher;
    PRICING_KV: KVNamespace;
    PRICING_UPLOAD_TOKEN: string;
}

interface PricingEntry {
    input_per_million?: number;
    output_per_million?: number;
    per_image?: number;
    per_second?: number;
}

interface PricingFile {
    $schema_version?: number;
    $generated_at?: string;
    $source?: string;
    models: Record<string, PricingEntry>;
}

const KV_BULK_KEY = "bulk";
const KV_MODEL_PREFIX = "model:";
const CACHE_HEADERS_PUBLIC = {
    "content-type": "application/json; charset=utf-8",
    // 5 min edge cache + revalidate. KV reads are already <10ms cold, but
    // edge caching lets common reads avoid the KV hop entirely.
    "cache-control": "public, max-age=300, stale-while-revalidate=86400",
};

export default {
    async fetch(request: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
        const url = new URL(request.url);
        const { pathname } = url;

        if (pathname === "/api/pricing.json" && request.method === "GET") {
            return handleBulkGet(env);
        }
        if (pathname.startsWith("/api/pricing/model/") && request.method === "GET") {
            const id = decodeURIComponent(pathname.slice("/api/pricing/model/".length));
            return handleModelGet(env, id);
        }
        if (pathname === "/api/pricing/upload" && request.method === "POST") {
            return handleUpload(request, env);
        }
        // 405 for unsupported methods on the API namespace so misuse fails
        // loudly rather than silently falling through to the static site.
        if (pathname === "/api/pricing.json" || pathname.startsWith("/api/pricing/")) {
            return new Response("method not allowed", { status: 405 });
        }

        return env.ASSETS.fetch(request);
    },
};

async function handleBulkGet(env: Env): Promise<Response> {
    const blob = await env.PRICING_KV.get(KV_BULK_KEY);
    if (blob === null) {
        return new Response(
            JSON.stringify({ $schema_version: 1, models: {} }),
            { status: 200, headers: CACHE_HEADERS_PUBLIC },
        );
    }
    return new Response(blob, { status: 200, headers: CACHE_HEADERS_PUBLIC });
}

async function handleModelGet(env: Env, modelId: string): Promise<Response> {
    if (modelId.length === 0 || modelId.length > 200) {
        return new Response("invalid model id", { status: 400 });
    }
    const entry = await env.PRICING_KV.get(KV_MODEL_PREFIX + modelId);
    if (entry === null) {
        return new Response("not found", { status: 404 });
    }
    return new Response(entry, { status: 200, headers: CACHE_HEADERS_PUBLIC });
}

async function handleUpload(request: Request, env: Env): Promise<Response> {
    const auth = request.headers.get("authorization");
    const expected = `Bearer ${env.PRICING_UPLOAD_TOKEN}`;
    if (!env.PRICING_UPLOAD_TOKEN || auth !== expected) {
        return new Response("unauthorized", { status: 401 });
    }

    let parsed: PricingFile;
    try {
        parsed = await request.json<PricingFile>();
    } catch {
        return new Response("invalid json", { status: 400 });
    }
    if (!parsed?.models || typeof parsed.models !== "object") {
        return new Response("missing models map", { status: 400 });
    }

    const bulkBody = JSON.stringify(parsed);
    // Write bulk first so a partial fan-out failure still leaves the bulk
    // endpoint internally consistent. Per-model writes are best-effort and
    // happen in parallel; KV's eventual consistency means readers may briefly
    // see new bulk + old per-model entries — acceptable for pricing.
    await env.PRICING_KV.put(KV_BULK_KEY, bulkBody);

    const writes = Object.entries(parsed.models).map(([id, entry]) =>
        env.PRICING_KV.put(KV_MODEL_PREFIX + id, JSON.stringify(entry)),
    );
    await Promise.all(writes);

    return new Response(
        JSON.stringify({ ok: true, count: Object.keys(parsed.models).length }),
        { status: 200, headers: { "content-type": "application/json" } },
    );
}
