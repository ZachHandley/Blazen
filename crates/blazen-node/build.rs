//! Build script that generates NAPI mirror types from `blazen-llm` source files.
//!
//! For every `pub struct` / `pub enum` that derives `Serialize` (and is not in
//! the exclusion list), we emit:
//!
//! - A `#[napi(object)]` mirror struct prefixed with `Js`
//! - `From<JsFoo> for CoreFoo` and `From<CoreFoo> for JsFoo` impls
//! - For simple enums (no data variants): a `#[napi(string_enum)]` mirror
//!
//! The output lands in `$OUT_DIR/napi_mirrors.rs`.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::{env, fs};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Attribute, Fields, Item, Type, parse_file};

extern crate napi_build;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Types that are hand-maintained and should NOT be generated.
const EXCLUDED: &[&str] = &[
    "ChatMessage",
    "CompletionRequest",
    "CompletionResponse",
    "StreamChunk",
    "FinishReason",
    "Artifact",
    "ResponseFormat",
    "MediaType",
];

/// Source files (relative to this crate's directory).
const SOURCE_FILES: &[&str] = &[
    "../blazen-llm/src/compute/requests.rs",
    "../blazen-llm/src/compute/results.rs",
    "../blazen-llm/src/compute/job.rs",
    "../blazen-llm/src/types/usage.rs",
    "../blazen-llm/src/types/tool.rs",
    "../blazen-llm/src/types/provider_options.rs",
    "../blazen-llm/src/media.rs",
];

fn main() {
    // Rerun if any source file changes.
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    for src in SOURCE_FILES {
        let abs = Path::new(&manifest_dir).join(src);
        println!("cargo:rerun-if-changed={}", abs.display());
    }

    let excluded: HashSet<&str> = EXCLUDED.iter().copied().collect();

    // Pass 1: collect all struct definitions so we can resolve flatten fields.
    let mut all_structs: HashMap<String, Vec<RawField>> = HashMap::new();
    let mut all_items: Vec<ParsedItem> = Vec::new();

    for src in SOURCE_FILES {
        let abs = Path::new(&manifest_dir).join(src);
        let content = fs::read_to_string(&abs)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", abs.display()));
        let syntax = parse_file(&content)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", abs.display()));

        for item in &syntax.items {
            match item {
                Item::Struct(s) if is_pub(&s.vis) && has_serialize(&s.attrs) => {
                    let name = s.ident.to_string();
                    if excluded.contains(name.as_str()) {
                        continue;
                    }
                    let fields = extract_raw_fields(&s.fields);
                    all_structs.insert(name.clone(), fields.clone());
                    all_items.push(ParsedItem::Struct { name, fields });
                }
                Item::Enum(e) if is_pub(&e.vis) && has_serialize(&e.attrs) => {
                    let name = e.ident.to_string();
                    if excluded.contains(name.as_str()) {
                        continue;
                    }
                    // Check if it's a simple enum (no data variants).
                    let is_simple = e.variants.iter().all(|v| matches!(v.fields, Fields::Unit));
                    if is_simple {
                        let variant_names: Vec<String> =
                            e.variants.iter().map(|v| v.ident.to_string()).collect();
                        all_items.push(ParsedItem::SimpleEnum {
                            name,
                            variants: variant_names,
                        });
                    }
                    // Enums with data variants are skipped (too complex for auto-gen).
                }
                _ => {}
            }
        }
    }

    // Pass 2: generate code.
    let mut tokens = TokenStream::new();

    // Preamble.
    tokens.extend(quote! {
        use napi_derive::napi;
    });

    for item in &all_items {
        match item {
            ParsedItem::Struct { name, fields } => {
                // Resolve flattened fields for the napi struct (inline them).
                let flat_fields = resolve_flat_fields(fields, &all_structs);
                tokens.extend(gen_struct_mirror(name, &flat_fields));
                tokens.extend(gen_from_impls(name, fields, &all_structs));
            }
            ParsedItem::SimpleEnum { name, variants } => {
                tokens.extend(gen_enum_mirror(name, variants));
            }
        }
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("napi_mirrors.rs");
    fs::write(&out_path, tokens.to_string())
        .unwrap_or_else(|e| panic!("Failed to write {}: {e}", out_path.display()));

    napi_build::setup();
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Tracks `#[serde(default)]` vs `#[serde(default = "func")]`.
#[derive(Clone, Debug, PartialEq, Eq)]
enum SerdeDefault {
    /// No `#[serde(default)]` attribute.
    None,
    /// `#[serde(default)]` (uses `Default::default()`).
    Default,
    /// `#[serde(default = "some_func")]`.
    Func(String),
}

/// A raw field as parsed from the source, before flatten resolution.
#[derive(Clone, Debug)]
struct RawField {
    name: String,
    ty: FieldType,
    is_flatten: bool,
    serde_default: SerdeDefault,
}

/// A flat field for the napi mirror struct (flattens already resolved).
#[derive(Clone, Debug)]
struct FlatField {
    name: String,
    ty: FieldType,
    serde_default: SerdeDefault,
}

#[derive(Clone, Debug)]
enum FieldType {
    String,
    Bool,
    U8,
    U32,
    U64,
    F32,
    F64,
    I32,
    I64,
    Usize,
    JsonValue,
    DateTimeUtc,
    /// `MediaType` is excluded from codegen but referenced by `MediaOutput`.
    /// We represent it as a MIME `String` in the napi layer.
    MediaTypeRef,
    Option(Box<FieldType>),
    Vec(Box<FieldType>),
    Named(String),
}

#[derive(Clone, Debug)]
enum ParsedItem {
    Struct { name: String, fields: Vec<RawField> },
    SimpleEnum { name: String, variants: Vec<String> },
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn is_pub(vis: &syn::Visibility) -> bool {
    matches!(vis, syn::Visibility::Public(_))
}

fn has_serialize(attrs: &[Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("derive") {
            let mut found = false;
            let _ = attr.parse_nested_meta(|meta| {
                let path_str = meta
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                if path_str == "Serialize" || path_str.ends_with("::Serialize") {
                    found = true;
                }
                Ok(())
            });
            if found {
                return true;
            }
        }
    }
    false
}

fn has_flatten_attr(attrs: &[Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("serde") {
            let mut found = false;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("flatten") {
                    found = true;
                }
                Ok(())
            });
            if found {
                return true;
            }
        }
    }
    false
}

fn extract_raw_fields(fields: &Fields) -> Vec<RawField> {
    match fields {
        Fields::Named(named) => named
            .named
            .iter()
            .filter(|f| is_pub(&f.vis))
            .map(|f| {
                let name = f.ident.as_ref().unwrap().to_string();
                let ty = parse_type(&f.ty);
                let is_flatten = has_flatten_attr(&f.attrs);
                let serde_default = parse_serde_default(&f.attrs);
                RawField {
                    name,
                    ty,
                    is_flatten,
                    serde_default,
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Check for `#[serde(default)]` or `#[serde(default = "func_name")]`.
fn parse_serde_default(attrs: &[Attribute]) -> SerdeDefault {
    for attr in attrs {
        if attr.path().is_ident("serde") {
            let mut result = SerdeDefault::None;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("default") {
                    if let Ok(lit) = meta
                        .value()
                        .and_then(syn::parse::ParseBuffer::parse::<syn::LitStr>)
                    {
                        result = SerdeDefault::Func(lit.value());
                    } else {
                        result = SerdeDefault::Default;
                    }
                }
                Ok(())
            });
            if result != SerdeDefault::None {
                return result;
            }
        }
    }
    SerdeDefault::None
}

fn parse_type(ty: &Type) -> FieldType {
    match ty {
        Type::Path(tp) => {
            let segments: Vec<_> = tp.path.segments.iter().collect();

            // serde_json::Value
            if segments.len() == 2
                && segments[0].ident == "serde_json"
                && segments[1].ident == "Value"
            {
                return FieldType::JsonValue;
            }

            // DateTime<Utc> or chrono::DateTime<Utc>
            if segments.last().is_some_and(|s| s.ident == "DateTime") {
                return FieldType::DateTimeUtc;
            }

            let last = segments.last().unwrap();
            let ident_str = last.ident.to_string();

            match ident_str.as_str() {
                "String" => FieldType::String,
                "bool" => FieldType::Bool,
                "u8" => FieldType::U8,
                "u32" => FieldType::U32,
                "u64" => FieldType::U64,
                "i32" => FieldType::I32,
                "i64" => FieldType::I64,
                "f32" => FieldType::F32,
                "f64" => FieldType::F64,
                "usize" => FieldType::Usize,
                "Value" if segments.len() == 1 => FieldType::JsonValue,
                "MediaType" => FieldType::MediaTypeRef,
                "Option" => {
                    let inner = extract_generic_arg(last);
                    FieldType::Option(Box::new(inner))
                }
                "Vec" => {
                    let inner = extract_generic_arg(last);
                    FieldType::Vec(Box::new(inner))
                }
                other => FieldType::Named(other.to_owned()),
            }
        }
        _ => FieldType::Named("Unknown".to_owned()),
    }
}

fn extract_generic_arg(segment: &syn::PathSegment) -> FieldType {
    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(syn::GenericArgument::Type(ty)) = args.args.first()
    {
        return parse_type(ty);
    }
    FieldType::Named("Unknown".to_owned())
}

// ---------------------------------------------------------------------------
// Flatten resolution
// ---------------------------------------------------------------------------

/// Recursively inline flattened fields to produce a flat list for the napi struct.
fn resolve_flat_fields(
    fields: &[RawField],
    all_structs: &HashMap<String, Vec<RawField>>,
) -> Vec<FlatField> {
    let mut result = Vec::new();
    for field in fields {
        if field.is_flatten
            && let FieldType::Named(ref type_name) = field.ty
            && let Some(inner_fields) = all_structs.get(type_name)
        {
            let resolved = resolve_flat_fields(inner_fields, all_structs);
            result.extend(resolved);
        } else {
            // Can't resolve flatten or not flattened -- keep it as a regular field.
            result.push(FlatField {
                name: field.name.clone(),
                ty: field.ty.clone(),
                serde_default: field.serde_default.clone(),
            });
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Code generation helpers
// ---------------------------------------------------------------------------

fn to_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;
    for (i, c) in s.chars().enumerate() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else if i == 0 {
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(c.to_ascii_lowercase());
    }
    result
}

fn needs_js_name(field_name: &str) -> bool {
    field_name.contains('_')
}

/// The NAPI-compatible type tokens.
fn napi_type_tokens(ft: &FieldType) -> TokenStream {
    match ft {
        FieldType::String | FieldType::DateTimeUtc | FieldType::MediaTypeRef => {
            quote! { String }
        }
        FieldType::Bool => quote! { bool },
        FieldType::U8 | FieldType::U32 => quote! { u32 },
        FieldType::U64 | FieldType::F32 | FieldType::F64 => quote! { f64 },
        FieldType::I32 => quote! { i32 },
        FieldType::I64 | FieldType::Usize => quote! { i64 },
        FieldType::JsonValue => quote! { serde_json::Value },
        FieldType::Option(inner) => {
            let inner_ty = napi_type_tokens(inner);
            quote! { Option<#inner_ty> }
        }
        FieldType::Vec(inner) => {
            let inner_ty = napi_type_tokens(inner);
            quote! { Vec<#inner_ty> }
        }
        FieldType::Named(name) => {
            let js_name = format_ident!("Js{name}");
            quote! { #js_name }
        }
    }
}

/// Expression to convert a core value to a napi value.
fn core_to_napi(ft: &FieldType, expr: TokenStream) -> TokenStream {
    match ft {
        FieldType::String
        | FieldType::Bool
        | FieldType::U32
        | FieldType::I32
        | FieldType::I64
        | FieldType::F64
        | FieldType::JsonValue => expr,
        FieldType::U8 => quote! { u32::from(#expr) },
        FieldType::U64 => quote! { #expr as f64 },
        FieldType::F32 => quote! { f64::from(#expr) },
        FieldType::Usize => quote! { #expr as i64 },
        FieldType::DateTimeUtc => quote! { #expr.to_rfc3339() },
        FieldType::MediaTypeRef => quote! { #expr.mime().to_owned() },
        FieldType::Option(inner) => {
            let inner_conv = core_to_napi(inner, quote! { __v });
            quote! { #expr.map(|__v| #inner_conv) }
        }
        FieldType::Vec(inner) => {
            let inner_conv = core_to_napi(inner, quote! { __v });
            quote! { #expr.into_iter().map(|__v| #inner_conv).collect() }
        }
        FieldType::Named(_) => {
            quote! { #expr.into() }
        }
    }
}

/// Expression to convert a napi value to a core value.
fn napi_to_core(ft: &FieldType, expr: TokenStream) -> TokenStream {
    match ft {
        FieldType::String
        | FieldType::Bool
        | FieldType::U32
        | FieldType::I32
        | FieldType::I64
        | FieldType::F64
        | FieldType::JsonValue => expr,
        FieldType::U8 => {
            quote! { { #[allow(clippy::cast_possible_truncation)] let __r = #expr as u8; __r } }
        }
        FieldType::U64 => {
            quote! { { #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] let __r = #expr as u64; __r } }
        }
        FieldType::F32 => {
            quote! { { #[allow(clippy::cast_possible_truncation)] let __r = #expr as f32; __r } }
        }
        FieldType::Usize => {
            quote! { { #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] let __r = #expr as usize; __r } }
        }
        FieldType::DateTimeUtc => {
            quote! {
                chrono::DateTime::parse_from_rfc3339(&#expr)
                    .unwrap_or_else(|_| chrono::DateTime::parse_from_rfc3339("1970-01-01T00:00:00Z").unwrap())
                    .with_timezone(&chrono::Utc)
            }
        }
        FieldType::MediaTypeRef => {
            quote! { blazen_llm::media::MediaType::from_mime(&#expr) }
        }
        FieldType::Option(inner) => {
            let inner_conv = napi_to_core(inner, quote! { __v });
            quote! { #expr.map(|__v| #inner_conv) }
        }
        FieldType::Vec(inner) => {
            let inner_conv = napi_to_core(inner, quote! { __v });
            quote! { #expr.into_iter().map(|__v| #inner_conv).collect() }
        }
        FieldType::Named(_) => {
            quote! { #expr.into() }
        }
    }
}

// ---------------------------------------------------------------------------
// Struct mirror generation
// ---------------------------------------------------------------------------

/// Returns `true` when a serde-defaulted field needs wrapping in `Option` for napi.
/// Fields that are already `Option<T>` do not need wrapping.
fn needs_option_wrap(f: &FlatField) -> bool {
    f.serde_default != SerdeDefault::None && !matches!(f.ty, FieldType::Option(_))
}

fn gen_struct_mirror(name: &str, flat_fields: &[FlatField]) -> TokenStream {
    let js_name = format_ident!("Js{name}");

    let field_defs: Vec<TokenStream> = flat_fields
        .iter()
        .map(|f| {
            let field_ident = format_ident!("{}", f.name);
            let base_ty = napi_type_tokens(&f.ty);
            // Wrap in Option if the field has a serde default but is not already Optional.
            let field_ty = if needs_option_wrap(f) {
                quote! { Option<#base_ty> }
            } else {
                base_ty
            };

            if needs_js_name(&f.name) {
                let js_field_name = to_camel_case(&f.name);
                quote! {
                    #[napi(js_name = #js_field_name)]
                    pub #field_ident: #field_ty,
                }
            } else {
                quote! {
                    pub #field_ident: #field_ty,
                }
            }
        })
        .collect();

    quote! {
        #[napi(object)]
        pub struct #js_name {
            #(#field_defs)*
        }
    }
}

// ---------------------------------------------------------------------------
// From impl generation (handles flatten nesting)
// ---------------------------------------------------------------------------

fn gen_from_impls(
    name: &str,
    raw_fields: &[RawField],
    all_structs: &HashMap<String, Vec<RawField>>,
) -> TokenStream {
    let js_ident = format_ident!("Js{name}");
    let core_path = core_type_path(name);

    // --- From<CoreType> for JsType ---
    // We need to read from the core struct (possibly through .flatten_field.subfield).
    let core_to_js_fields = gen_core_to_js_fields(raw_fields, &quote! { val }, all_structs);

    // --- From<JsType> for CoreType ---
    let js_to_core_fields = gen_js_to_core_fields(raw_fields, all_structs);

    quote! {
        impl From<#core_path> for #js_ident {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            fn from(val: #core_path) -> Self {
                Self {
                    #(#core_to_js_fields)*
                }
            }
        }

        impl From<#js_ident> for #core_path {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            fn from(val: #js_ident) -> Self {
                Self {
                    #(#js_to_core_fields)*
                }
            }
        }
    }
}

/// Generate field assignments for `From<CoreType> for JsType`.
/// For flattened fields, we access through the nested path (e.g., `val.base.model`).
fn gen_core_to_js_fields(
    raw_fields: &[RawField],
    access_prefix: &TokenStream,
    all_structs: &HashMap<String, Vec<RawField>>,
) -> Vec<TokenStream> {
    let mut result = Vec::new();
    for field in raw_fields {
        if field.is_flatten
            && let FieldType::Named(ref type_name) = field.ty
            && let Some(inner_fields) = all_structs.get(type_name)
        {
            let field_ident = format_ident!("{}", field.name);
            let new_prefix = quote! { #access_prefix.#field_ident };
            let inner = gen_core_to_js_fields(inner_fields, &new_prefix, all_structs);
            result.extend(inner);
            continue;
        }
        let field_ident = format_ident!("{}", field.name);
        let conv = core_to_napi(&field.ty, quote! { #access_prefix.#field_ident });
        // Wrap in Some() when the napi mirror uses Option for a serde-defaulted field.
        let wrap =
            field.serde_default != SerdeDefault::None && !matches!(field.ty, FieldType::Option(_));
        let rhs = if wrap {
            quote! { Some(#conv) }
        } else {
            conv
        };
        result.push(quote! { #field_ident: #rhs, });
    }
    result
}

/// Generate field assignments for `From<JsType> for CoreType`.
/// For flattened fields, we reconstruct the nested struct from flat JS fields.
fn gen_js_to_core_fields(
    raw_fields: &[RawField],
    all_structs: &HashMap<String, Vec<RawField>>,
) -> Vec<TokenStream> {
    let mut result = Vec::new();
    for field in raw_fields {
        let field_ident = format_ident!("{}", field.name);
        if field.is_flatten
            && let FieldType::Named(ref type_name) = field.ty
            && let Some(inner_fields) = all_structs.get(type_name)
        {
            // Recursively build the nested struct.
            let inner_assignments = gen_js_to_core_fields(inner_fields, all_structs);
            let inner_path = core_type_path(type_name);
            result.push(quote! {
                #field_ident: #inner_path {
                    #(#inner_assignments)*
                },
            });
            continue;
        }
        // For serde-defaulted, non-Option fields we unwrap the JS Option with
        // the matching default value.
        if field.serde_default != SerdeDefault::None && !matches!(field.ty, FieldType::Option(_)) {
            let unwrap_suffix = unwrap_suffix_tokens(&field.ty, &field.serde_default);
            if is_identity_conversion(&field.ty) {
                // No conversion needed -- just unwrap with the default.
                result.push(quote! {
                    #field_ident: val.#field_ident #unwrap_suffix,
                });
            } else {
                let default_val = explicit_default_tokens(&field.ty, &field.serde_default);
                let inner_conv = napi_to_core(&field.ty, quote! { __v });
                result.push(quote! {
                    #field_ident: val.#field_ident.map_or(#default_val, |__v| #inner_conv),
                });
            }
            continue;
        }
        let conv = napi_to_core(&field.ty, quote! { val.#field_ident });
        result.push(quote! { #field_ident: #conv, });
    }
    result
}

/// Returns `true` when `napi_to_core` for this type is an identity (no conversion).
fn is_identity_conversion(ty: &FieldType) -> bool {
    matches!(
        ty,
        FieldType::String
            | FieldType::Bool
            | FieldType::U32
            | FieldType::I32
            | FieldType::I64
            | FieldType::F64
            | FieldType::JsonValue
    )
}

/// Produce the `.unwrap_or(...)` / `.unwrap_or_default()` suffix for a
/// serde-defaulted field (identity-conversion path).
fn unwrap_suffix_tokens(ty: &FieldType, sd: &SerdeDefault) -> TokenStream {
    // Only `default_true` needs an explicit value; everything else can use
    // `unwrap_or_default()` which clippy prefers.
    match (ty, sd) {
        (FieldType::Bool, SerdeDefault::Func(f)) if f == "default_true" => {
            quote! { .unwrap_or(true) }
        }
        _ => quote! { .unwrap_or_default() },
    }
}

/// Produce an explicit default-value expression for the `map_or` path
/// (non-identity conversions).
fn explicit_default_tokens(ty: &FieldType, sd: &SerdeDefault) -> TokenStream {
    match (ty, sd) {
        (FieldType::Bool, SerdeDefault::Func(f)) if f == "default_true" => quote! { true },
        (FieldType::Bool, _) => quote! { false },
        _ => {
            let core_ty = core_field_type_tokens(ty);
            quote! { <#core_ty as Default>::default() }
        }
    }
}

/// Tokens for the core Rust type (used in `map_or` default expressions).
fn core_field_type_tokens(ft: &FieldType) -> TokenStream {
    match ft {
        FieldType::String => quote! { String },
        FieldType::Bool => quote! { bool },
        FieldType::U8 => quote! { u8 },
        FieldType::U32 => quote! { u32 },
        FieldType::U64 => quote! { u64 },
        FieldType::I32 => quote! { i32 },
        FieldType::I64 => quote! { i64 },
        FieldType::F32 => quote! { f32 },
        FieldType::F64 => quote! { f64 },
        FieldType::Usize => quote! { usize },
        FieldType::JsonValue => quote! { serde_json::Value },
        _ => {
            // Fallback -- use the napi representation.
            napi_type_tokens(ft)
        }
    }
}

// ---------------------------------------------------------------------------
// Enum mirror generation
// ---------------------------------------------------------------------------

fn gen_enum_mirror(name: &str, variants: &[String]) -> TokenStream {
    let js_name = format_ident!("Js{name}");
    let core_path = core_type_path(name);

    let variant_defs: Vec<TokenStream> = variants
        .iter()
        .map(|v| {
            let v_ident = format_ident!("{v}");
            let v_lower = to_snake_case(v);
            quote! {
                #[napi(value = #v_lower)]
                #v_ident,
            }
        })
        .collect();

    let core_to_js_arms: Vec<TokenStream> = variants
        .iter()
        .map(|v| {
            let v_ident = format_ident!("{v}");
            quote! { #core_path::#v_ident => #js_name::#v_ident, }
        })
        .collect();

    let js_to_core_arms: Vec<TokenStream> = variants
        .iter()
        .map(|v| {
            let v_ident = format_ident!("{v}");
            quote! { #js_name::#v_ident => #core_path::#v_ident, }
        })
        .collect();

    quote! {
        #[napi(string_enum)]
        pub enum #js_name {
            #(#variant_defs)*
        }

        impl From<#core_path> for #js_name {
            fn from(val: #core_path) -> Self {
                match val {
                    #(#core_to_js_arms)*
                }
            }
        }

        impl From<#js_name> for #core_path {
            fn from(val: #js_name) -> Self {
                match val {
                    #(#js_to_core_arms)*
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Core type path mapping
// ---------------------------------------------------------------------------

fn core_type_path(name: &str) -> TokenStream {
    let ident = format_ident!("{name}");

    match name {
        // Compute request types
        "ImageRequest"
        | "UpscaleRequest"
        | "VideoRequest"
        | "SpeechRequest"
        | "MusicRequest"
        | "TranscriptionRequest"
        | "ThreeDRequest"
        | "BackgroundRemovalRequest" => {
            quote! { blazen_llm::compute::requests::#ident }
        }

        // Compute result types
        "ImageResult"
        | "VideoResult"
        | "AudioResult"
        | "TranscriptionSegment"
        | "TranscriptionResult"
        | "ThreeDResult" => {
            quote! { blazen_llm::compute::results::#ident }
        }

        // Compute job types
        "JobHandle" | "JobStatus" | "ComputeRequest" | "ComputeResult" => {
            quote! { blazen_llm::compute::job::#ident }
        }

        // Provider option types
        "ProviderOptions" | "FalOptions" | "FalLlmEndpointKind" | "AzureOptions"
        | "BedrockOptions" => {
            quote! { blazen_llm::types::provider_options::#ident }
        }

        // Media types
        "MediaOutput" | "GeneratedImage" | "GeneratedVideo" | "GeneratedAudio"
        | "Generated3DModel" => {
            quote! { blazen_llm::media::#ident }
        }

        // Usage, tool, and fallback types
        _ => {
            quote! { blazen_llm::types::#ident }
        }
    }
}
