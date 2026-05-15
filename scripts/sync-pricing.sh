#!/usr/bin/env bash
# Fetch the latest model pricing from upstream catalogs and write
# crates/blazen-llm/data/pricing.json.
#
# Sources, in increasing authority:
#   1. https://models.dev/api.json           — community catalog, ~50 providers
#   2. https://openrouter.ai/api/v1/models   — live, includes prompt/completion pricing per token
#   3. https://api.together.xyz/v1/models    — live, includes input/output per million
#
# Live provider endpoints win where they overlap with models.dev. Keys are
# normalized (lowercase + strip provider prefix + strip date/version suffix)
# to match crates/blazen-llm/src/pricing.rs::normalize_model_id, so the file
# is a drop-in replacement for the runtime HashMap and lookups resolve
# date-pinned snapshots back to their base model.
#
# Invoked daily by .forgejo/workflows/check-pricing.yaml, which then pushes
# the result to the `models-data` branch and uploads it to the blazen.dev
# Cloudflare Worker. The auto-tag job in ci.yaml pulls models-data → main
# before each release so published artifacts ship with the latest snapshot.
#
# Usage: ./scripts/sync-pricing.sh
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT="crates/blazen-llm/data/pricing.json"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Each upstream curl is best-effort: if one source is unreachable we still
# want a refresh from the others rather than failing the whole workflow.
# `|| true` keeps the script alive; the jq stage emits {} for missing files.
curl -fsSL --max-time 30 https://models.dev/api.json         -o "$TMP/models.dev.json"  || echo "[]" > "$TMP/models.dev.json"
curl -fsSL --max-time 30 https://openrouter.ai/api/v1/models -o "$TMP/openrouter.json" || echo '{"data":[]}' > "$TMP/openrouter.json"
curl -fsSL --max-time 30 https://api.together.xyz/v1/models  -o "$TMP/together.json"   || echo '[]' > "$TMP/together.json"

# Normalize a model id the same way crates/blazen-llm/src/pricing.rs does:
#   - lowercase
#   - strip everything up to the last '/' (openrouter prefix style)
#   - strip trailing '-YYYYMMDD' or '-YYYY-MM-DD' date suffix
#   - strip trailing '-vN[.M[:K]]' version suffix (bedrock style)
# Implemented as a jq function so all three pipelines reuse it.
read -r -d '' JQ_NORMALIZE <<'JQ' || true
def normalize:
    ascii_downcase
    | (if test("/") then sub(".*/"; "") else . end)
    | sub("-[0-9]{8}$"; "")
    | sub("-[0-9]{4}-[0-9]{2}-[0-9]{2}$"; "")
    | sub("-v[0-9]+(\\.[0-9]+)?(:[0-9]+)?$"; "");
JQ

# ---- 1. models.dev (lowest priority) ---------------------------------------
# Schema: { "<provider_id>": { "models": { "<model_id>": { "cost": { "input": <usd/million>, "output": <usd/million>, ... }, ... }, ... }, ... } }
# Image / audio / video pricing on models.dev varies by model; only token
# pricing is reliably present, so that's what we project here.
jq --argjson empty '{}' '
    '"$JQ_NORMALIZE"'
    [
        (to_entries[] | .value.models // {} | to_entries[] |
            select(.value.cost != null) |
            {
                key: (.key | normalize),
                value: {
                    input_per_million:  (.value.cost.input  // 0),
                    output_per_million: (.value.cost.output // 0)
                }
            }
        )
    ] | from_entries
' "$TMP/models.dev.json" > "$TMP/from-models.dev.json"

# ---- 2. OpenRouter (mid priority) ------------------------------------------
# Schema: { "data": [ { "id": "openai/gpt-4o", "pricing": { "prompt": "<usd/TOKEN>", "completion": "<usd/TOKEN>" } } ] }
# Note: prompt/completion are strings of USD per *single* token, so we
# multiply by 1_000_000 to convert to our per-million convention.
jq '
    '"$JQ_NORMALIZE"'
    [
        .data[]? |
        select(.pricing.prompt? != null and .pricing.completion? != null) |
        {
            key: (.id | normalize),
            value: {
                input_per_million:  ((.pricing.prompt    | tonumber) * 1000000),
                output_per_million: ((.pricing.completion | tonumber) * 1000000)
            }
        }
    ] | from_entries
' "$TMP/openrouter.json" > "$TMP/from-openrouter.json"

# ---- 3. Together (highest priority) ----------------------------------------
# Schema: bare array [ { "id": "...", "pricing": { "input": <usd/million>, "output": <usd/million> } } ]
jq '
    '"$JQ_NORMALIZE"'
    [
        .[]? |
        select(.pricing.input? != null and .pricing.output? != null) |
        {
            key: (.id | normalize),
            value: {
                input_per_million:  (.pricing.input  | tonumber),
                output_per_million: (.pricing.output | tonumber)
            }
        }
    ] | from_entries
' "$TMP/together.json" > "$TMP/from-together.json"

# ---- Merge (Together > OpenRouter > models.dev) ----------------------------
GENERATED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

jq -n \
    --slurpfile a "$TMP/from-models.dev.json" \
    --slurpfile b "$TMP/from-openrouter.json" \
    --slurpfile c "$TMP/from-together.json" \
    --arg generated_at "$GENERATED_AT" \
    '{
        "$schema_version": 1,
        "$generated_at": $generated_at,
        "$source": "models.dev + openrouter.ai + together.xyz",
        "models": ($a[0] + $b[0] + $c[0])
    }' > "$OUTPUT.new"

# Sanity check: must contain at least the historical baseline (17 entries).
# Anything less means upstream is having a very bad day and we'd rather
# preserve the existing committed file than overwrite it with garbage.
COUNT=$(jq '.models | length' "$OUTPUT.new")
if [ "$COUNT" -lt 17 ]; then
    echo "ERROR: only ${COUNT} models in merged catalog; refusing to overwrite ${OUTPUT}" >&2
    echo "Upstream sources may be down. Existing file is preserved." >&2
    rm -f "$OUTPUT.new"
    exit 1
fi

mv "$OUTPUT.new" "$OUTPUT"
echo "Wrote ${COUNT} model entries to ${OUTPUT}"
