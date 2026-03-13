//! `Blazen` procedural macros.
//!
//! Provides derive macros and attribute macros for the `Blazen` framework,
//! including `#[derive(Event)]` and `#[step]` attribute macros.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    DeriveInput, FnArg, GenericArgument, ItemFn, PathArguments, ReturnType, Token, Type, TypePath,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

/// Derive macro that generates a `blazen_events::Event` trait implementation.
///
/// The struct must also derive `Clone` and `Serialize` (from serde) for the
/// generated implementation to compile at the call site.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Debug, Clone, Serialize, Deserialize, Event)]
/// pub struct AnalyzeEvent {
///     pub text: String,
///     pub score: f64,
/// }
/// ```
///
/// This generates an `impl blazen_events::Event for AnalyzeEvent` with:
/// - `event_type()` returning a unique `&'static str` based on `module_path!()` and the struct name
/// - `event_type_id()` delegating to `event_type()`
/// - `as_any()` for downcasting
/// - `clone_boxed()` for trait-object cloning
/// - `to_json()` for serde serialization
#[proc_macro_derive(Event)]
pub fn derive_event(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics blazen_events::Event for #name #ty_generics #where_clause {
            fn event_type() -> &'static str {
                concat!(module_path!(), "::", stringify!(#name))
            }

            fn event_type_id(&self) -> &'static str {
                <Self as blazen_events::Event>::event_type()
            }

            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }

            fn clone_boxed(&self) -> Box<dyn blazen_events::AnyEvent> {
                Box::new(self.clone())
            }

            fn to_json(&self) -> ::serde_json::Value {
                ::serde_json::to_value(self).unwrap_or(::serde_json::Value::Null)
            }
        }
    };

    TokenStream::from(expanded)
}

// ---------------------------------------------------------------------------
// #[step] attribute macro
// ---------------------------------------------------------------------------

/// Arguments for the `#[step(...)]` attribute.
///
/// Currently supports:
/// - `#[step]` -- no arguments
/// - `#[step(emits = [EventA, EventB])]` -- explicit list of emitted event types
///   (useful when the step returns `StepOutput` directly)
struct StepAttr {
    emits: Vec<Type>,
}

/// A bracketed, comma-separated list of types: `[TypeA, TypeB]`
struct BracketedTypes {
    types: Vec<Type>,
}

impl Parse for BracketedTypes {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let content;
        syn::bracketed!(content in input);
        let types: Punctuated<Type, Token![,]> =
            content.parse_terminated(Type::parse, Token![,])?;
        Ok(BracketedTypes {
            types: types.into_iter().collect(),
        })
    }
}

impl Parse for StepAttr {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(StepAttr { emits: Vec::new() });
        }

        let mut emits = Vec::new();

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            if ident == "emits" {
                input.parse::<Token![=]>()?;
                let bracketed: BracketedTypes = input.parse()?;
                emits = bracketed.types;
            } else {
                return Err(syn::Error::new(ident.span(), "unknown step attribute"));
            }

            // optional trailing comma between attributes
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(StepAttr { emits })
    }
}

/// Returns true if the type path's last segment is `Context`.
fn is_context_type(ty: &Type) -> bool {
    if let Type::Path(TypePath { path, .. }) = ty
        && let Some(seg) = path.segments.last()
    {
        return seg.ident == "Context";
    }
    false
}

/// Returns true if the type path's last segment is `StepOutput`.
fn is_step_output_type(ty: &Type) -> bool {
    if let Type::Path(TypePath { path, .. }) = ty
        && let Some(seg) = path.segments.last()
    {
        return seg.ident == "StepOutput";
    }
    false
}

/// Extract the first generic argument from `Result<T, E>` -- returns T.
fn extract_result_ok_type(ty: &Type) -> Option<&Type> {
    if let Type::Path(TypePath { path, .. }) = ty
        && let Some(seg) = path.segments.last()
        && seg.ident == "Result"
        && let PathArguments::AngleBracketed(ref args) = seg.arguments
    {
        for arg in &args.args {
            if let GenericArgument::Type(inner_ty) = arg {
                return Some(inner_ty);
            }
        }
    }
    None
}

/// `#[step]` attribute macro that generates a step registration function
/// alongside the original async function.
///
/// # Usage
///
/// ```rust,ignore
/// #[step]
/// async fn analyze(event: StartEvent, ctx: Context) -> Result<AnalyzeEvent, WorkflowError> {
///     // ...
/// }
/// ```
///
/// This generates an `analyze_registration()` function returning a
/// `blazen_core::StepRegistration` that wraps the original function
/// with proper type-erasure and downcasting.
///
/// When the return type is `Result<StepOutput, WorkflowError>`, the macro
/// does not wrap the output in `StepOutput::Single` and instead returns
/// it directly. In this case you can optionally annotate emitted types:
///
/// ```rust,ignore
/// #[step(emits = [BuyEvent, SellEvent])]
/// async fn decide(event: AnalysisEvent, ctx: Context) -> Result<StepOutput, WorkflowError> {
///     // ...
/// }
/// ```
#[proc_macro_attribute]
pub fn step(attr: TokenStream, item: TokenStream) -> TokenStream {
    let step_attr = parse_macro_input!(attr as StepAttr);
    let input_fn = parse_macro_input!(item as ItemFn);

    match step_impl(&step_attr, &input_fn) {
        Ok(ts) => ts,
        Err(err) => err.to_compile_error().into(),
    }
}

fn step_impl(step_attr: &StepAttr, input_fn: &ItemFn) -> syn::Result<TokenStream> {
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let registration_fn_name = format_ident!("{}_registration", fn_name);

    // ---------------------------------------------------------------
    // 1. Extract the event parameter type (first non-Context param).
    // ---------------------------------------------------------------
    let mut event_type: Option<&Type> = None;

    for arg in &input_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            let ty = &*pat_type.ty;
            if !is_context_type(ty) {
                event_type = Some(ty);
                break;
            }
        }
    }

    let event_type = event_type.ok_or_else(|| {
        syn::Error::new_spanned(
            &input_fn.sig,
            "#[step] function must have at least one non-Context parameter (the event type)",
        )
    })?;

    // ---------------------------------------------------------------
    // 2. Extract the return type from Result<T, ...>.
    // ---------------------------------------------------------------
    let return_type = match &input_fn.sig.output {
        ReturnType::Type(_, ty) => ty.as_ref(),
        ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                &input_fn.sig,
                "#[step] function must return Result<T, WorkflowError>",
            ));
        }
    };

    let ok_type = extract_result_ok_type(return_type).ok_or_else(|| {
        syn::Error::new_spanned(
            return_type,
            "#[step] function must return Result<T, WorkflowError>",
        )
    })?;

    let returns_step_output = is_step_output_type(ok_type);

    // ---------------------------------------------------------------
    // 3. Generate the emits metadata.
    // ---------------------------------------------------------------
    let emits_tokens = if !step_attr.emits.is_empty() {
        let emit_types = &step_attr.emits;
        quote! {
            vec![#( <#emit_types as blazen_events::Event>::event_type() ),*]
        }
    } else if returns_step_output {
        // No metadata available for StepOutput returns without explicit emits.
        quote! { vec![] }
    } else {
        // The ok_type IS the concrete event type.
        quote! { vec![<#ok_type as blazen_events::Event>::event_type()] }
    };

    // ---------------------------------------------------------------
    // 4. Generate the handler closure body.
    // ---------------------------------------------------------------
    let handler_body = if returns_step_output {
        // The function already returns StepOutput, so just pass through.
        quote! {
            let typed_event = event
                .as_any()
                .downcast_ref::<#event_type>()
                .ok_or_else(|| blazen_core::WorkflowError::EventDowncastFailed {
                    expected: <#event_type as blazen_events::Event>::event_type(),
                    got: event.event_type_id().to_string(),
                })?
                .clone();
            #fn_name(typed_event, ctx).await
        }
    } else {
        // Wrap the concrete event return in StepOutput::Single.
        quote! {
            let typed_event = event
                .as_any()
                .downcast_ref::<#event_type>()
                .ok_or_else(|| blazen_core::WorkflowError::EventDowncastFailed {
                    expected: <#event_type as blazen_events::Event>::event_type(),
                    got: event.event_type_id().to_string(),
                })?
                .clone();
            let result = #fn_name(typed_event, ctx).await?;
            Ok(blazen_core::StepOutput::Single(Box::new(result)))
        }
    };

    // ---------------------------------------------------------------
    // 5. Emit the original function + the registration function.
    // ---------------------------------------------------------------
    let expanded = quote! {
        #input_fn

        /// Auto-generated step registration for [`#fn_name`].
        fn #registration_fn_name() -> blazen_core::StepRegistration {
            blazen_core::StepRegistration {
                name: #fn_name_str.to_string(),
                accepts: vec![<#event_type as blazen_events::Event>::event_type()],
                emits: #emits_tokens,
                handler: ::std::sync::Arc::new(
                    |event: Box<dyn blazen_events::AnyEvent>,
                     ctx: blazen_core::Context|
                     -> ::std::pin::Pin<Box<dyn ::std::future::Future<
                        Output = ::std::result::Result<
                            blazen_core::StepOutput,
                            blazen_core::WorkflowError,
                        >,
                    > + Send>> {
                        Box::pin(async move {
                            #handler_body
                        })
                    },
                ),
                max_concurrency: 1,
            }
        }
    };

    Ok(TokenStream::from(expanded))
}
