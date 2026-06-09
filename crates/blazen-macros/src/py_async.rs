//! `#[py_async]` attribute macro.
//!
//! Applied as an **outer attribute on an `impl` block** containing
//! `#[pymethods]`. Walks every `async fn` item inside; for each one,
//! replaces it in-place with **two** Rust sibling methods so that
//! `#[pymethods]` (which expands after this) sees only synchronous
//! signatures:
//!
//! 1. A **sync** Python method with the original name. Body: clones
//!    `self` and runs the user's async body to completion via
//!    `crate::convert::block_on_context` (which releases the GIL).
//! 2. An **async** Python method named `a<orig>` (e.g. `set` → `aset`,
//!    `get` → `aget`). Body: clones `self` and hands the future to
//!    `pyo3_async_runtimes::tokio::future_into_py`, returning an
//!    asyncio awaitable.
//!
//! Non-`async fn` items inside the impl block pass through unchanged.
//!
//! ## Usage
//!
//! ```ignore
//! #[py_async]
//! #[gen_stub_pymethods]
//! #[pymethods]
//! impl PyContext {
//!     async fn set_bytes(&self, key: String, data: Vec<u8>) -> PyResult<()> {
//!         let inner = self.inner.clone();
//!         inner.set_bytes(&key, data).await;
//!         Ok(())
//!     }
//! }
//! ```
//!
//! Order matters: `#[py_async]` must appear **above** `#[pymethods]` so
//! it expands first.
//!
//! ## Constraints on each `async fn`
//!
//! - `&self` receiver required for methods; the implementing struct
//!   must be `Clone`.
//! - Return type must be `PyResult<T>` for any non-`()` output (it is
//!   required by `pyo3_async_runtimes::tokio::future_into_py`'s
//!   `Output = PyResult<T>` bound — the same bound applies to the sync
//!   form for consistency).
//! - No `py: Python<'_>` argument; the macro injects `py` into both
//!   wrappers. Post-await GIL work re-attaches via `pyo3::Python::attach`.
//! - Borrowed args (`&Bound<'_, PyAny>`, `&str`, `&[u8]`) cannot cross
//!   the await point. Use owned types in the signature (`Py<PyAny>`,
//!   `String`, `Vec<u8>`).
//!
//! ## Factory variant
//!
//! An `async fn` annotated with `#[py_async_factory]` (a sub-attribute
//! the macro recognises and strips) instead emits:
//!
//! - A sync `#[new]` constructor.
//! - An async `aopen` classmethod returning an awaitable.
//!
//! Used for provider constructors whose model-load body is heavy.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    FnArg, GenericParam, Ident, ImplItem, ImplItemFn, ItemImpl, LifetimeParam, PathArguments,
    ReturnType, Type, parse_macro_input, parse_quote, visit_mut::VisitMut,
};

pub(crate) fn py_async_impl(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as ItemImpl);
    match expand_impl(&mut input) {
        Ok(()) => quote! { #input }.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_impl(input: &mut ItemImpl) -> syn::Result<()> {
    let mut new_items: Vec<ImplItem> = Vec::with_capacity(input.items.len() * 2);

    for item in std::mem::take(&mut input.items) {
        match item {
            ImplItem::Fn(method) if method.sig.asyncness.is_some() => {
                let pair = expand_async_method(&method)?;
                new_items.extend(pair);
            }
            other => new_items.push(other),
        }
    }

    input.items = new_items;
    Ok(())
}

fn expand_async_method(method: &ImplItemFn) -> syn::Result<Vec<ImplItem>> {
    let is_factory = method
        .attrs
        .iter()
        .any(|a| a.path().is_ident("py_async_factory"));
    let is_static = method
        .attrs
        .iter()
        .any(|a| a.path().is_ident("py_async_static"));

    if is_factory && is_static {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[py_async_factory] and #[py_async_static] are mutually exclusive",
        ));
    }

    // Reject explicit `Python<'_>` arg — we inject `py`.
    for arg in &method.sig.inputs {
        if let FnArg::Typed(pat_type) = arg
            && is_python_type(&pat_type.ty)
        {
            return Err(syn::Error::new_spanned(
                pat_type,
                "#[py_async] methods must not take a `Python<'_>` argument; \
                 the macro injects `py` into both generated wrappers",
            ));
        }
    }

    // Rewrite `self` → `__this` in the body.
    let mut body = method.block.clone();
    let mut rewriter = SelfRewriter;
    rewriter.visit_block_mut(&mut body);

    if is_factory {
        expand_factory(method, &body)
    } else if is_static {
        expand_static(method, &body)
    } else {
        expand_method(method, &body)
    }
}

fn expand_method(method: &ImplItemFn, body: &syn::Block) -> syn::Result<Vec<ImplItem>> {
    let receiver = method
        .sig
        .inputs
        .iter()
        .find(|a| matches!(a, FnArg::Receiver(_)));
    if receiver.is_none() {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[py_async] (non-factory) requires `&self`",
        ));
    }

    let other_args: Vec<&FnArg> = method
        .sig
        .inputs
        .iter()
        .filter(|a| matches!(a, FnArg::Typed(_)))
        .collect();

    let name = &method.sig.ident;
    let aname = format_ident!("a{}", name);

    let preserved_attrs: Vec<&syn::Attribute> = method
        .attrs
        .iter()
        .filter(|a| !a.path().is_ident("py_async") && !a.path().is_ident("py_async_factory"))
        .collect();

    let ret_ty = match &method.sig.output {
        ReturnType::Default => quote! { pyo3::PyResult<()> },
        ReturnType::Type(_, ty) => quote! { #ty },
    };
    let body_ts = quote! { #body };

    // Sync wrapper signature: receiver + `py: Python<'_>` + other args.
    let sync_args = quote! { &self, py: pyo3::Python<'_>, #(#other_args),* };
    let async_args = quote! { &self, py: pyo3::Python<'py>, #(#other_args),* };

    // Generics: async wrapper needs a `'py` lifetime.
    let (sync_generics, sync_where) = (
        method.sig.generics.params.iter().collect::<Vec<_>>(),
        method.sig.generics.where_clause.as_ref(),
    );
    let sync_generics_ts = if sync_generics.is_empty() {
        quote! {}
    } else {
        quote! { < #(#sync_generics),* > }
    };

    let mut async_gen = method.sig.generics.clone();
    let lt: LifetimeParam = parse_quote! { 'py };
    async_gen.params.insert(0, GenericParam::Lifetime(lt));
    let async_params: Vec<_> = async_gen.params.iter().collect();
    let async_generics_ts = quote! { < #(#async_params),* > };
    let async_where = async_gen.where_clause.as_ref();

    let vis = &method.vis;

    let sync_fn: ImplItem = parse_quote! {
        #(#preserved_attrs)*
        #vis fn #name #sync_generics_ts ( #sync_args ) -> #ret_ty #sync_where {
            let __this: Self = ::std::clone::Clone::clone(self);
            crate::convert::block_on_context(py, async move {
                let __ret: #ret_ty = (async move #body_ts).await;
                __ret
            })
        }
    };

    let async_fn: ImplItem = parse_quote! {
        #(#preserved_attrs)*
        #vis fn #aname #async_generics_ts ( #async_args )
            -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>>
            #async_where
        {
            let __this: Self = ::std::clone::Clone::clone(self);
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let __ret: #ret_ty = (async move #body_ts).await;
                __ret
            })
        }
    };

    Ok(vec![sync_fn, async_fn])
}

/// Static-method variant: `async fn name(args...) -> PyResult<Self>`
/// (typically). Emits a sync `#[staticmethod]` and an async
/// `#[staticmethod]` named `a<name>`. The async wrapper wraps the body's
/// return value (which must be `Self`) in `Py::new` so it can be returned
/// as a Python object from `future_into_py`.
fn expand_static(method: &ImplItemFn, body: &syn::Block) -> syn::Result<Vec<ImplItem>> {
    if method
        .sig
        .inputs
        .iter()
        .any(|a| matches!(a, FnArg::Receiver(_)))
    {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[py_async_static] takes no `self`",
        ));
    }

    let sync_name = method.sig.ident.clone();
    let aname = format_ident!("a{}", sync_name);

    let other_args: Vec<&FnArg> = method.sig.inputs.iter().collect();

    let preserved_attrs: Vec<&syn::Attribute> = method
        .attrs
        .iter()
        .filter(|a| {
            !a.path().is_ident("py_async")
                && !a.path().is_ident("py_async_static")
                && !a.path().is_ident("py_async_factory")
                && !a.path().is_ident("staticmethod")
        })
        .collect();

    let ret_ty = match &method.sig.output {
        ReturnType::Default => quote! { pyo3::PyResult<Self> },
        ReturnType::Type(_, ty) => quote! { #ty },
    };
    let body_ts = quote! { #body };

    let mut async_gen = method.sig.generics.clone();
    let lt: LifetimeParam = parse_quote! { 'py };
    async_gen.params.insert(0, GenericParam::Lifetime(lt));
    let async_params: Vec<_> = async_gen.params.iter().collect();
    let async_generics_ts = quote! { < #(#async_params),* > };
    let async_where = async_gen.where_clause.as_ref();

    let sync_generics: Vec<_> = method.sig.generics.params.iter().collect();
    let sync_generics_ts = if sync_generics.is_empty() {
        quote! {}
    } else {
        quote! { < #(#sync_generics),* > }
    };
    let sync_where = method.sig.generics.where_clause.as_ref();

    let vis = &method.vis;

    let sync_fn: ImplItem = parse_quote! {
        #(#preserved_attrs)*
        #[staticmethod]
        #vis fn #sync_name #sync_generics_ts ( py: pyo3::Python<'_>, #(#other_args),* )
            -> #ret_ty
            #sync_where
        {
            crate::convert::block_on_context(py, async move {
                let __ret: #ret_ty = (async move #body_ts).await;
                __ret
            })
        }
    };

    let async_fn: ImplItem = parse_quote! {
        #(#preserved_attrs)*
        #[staticmethod]
        #vis fn #aname #async_generics_ts ( py: pyo3::Python<'py>, #(#other_args),* )
            -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>>
            #async_where
        {
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let __ret: #ret_ty = (async move #body_ts).await;
                let __built = __ret?;
                pyo3::Python::attach(|py| pyo3::Py::new(py, __built))
            })
        }
    };

    Ok(vec![sync_fn, async_fn])
}

fn expand_factory(method: &ImplItemFn, body: &syn::Block) -> syn::Result<Vec<ImplItem>> {
    if method
        .sig
        .inputs
        .iter()
        .any(|a| matches!(a, FnArg::Receiver(_)))
    {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[py_async_factory] takes no `self`",
        ));
    }

    let other_args: Vec<&FnArg> = method.sig.inputs.iter().collect();

    let sync_name = method.sig.ident.clone();
    let aname = format_ident!("aopen");

    let preserved_attrs: Vec<&syn::Attribute> = method
        .attrs
        .iter()
        .filter(|a| {
            !a.path().is_ident("py_async")
                && !a.path().is_ident("py_async_factory")
                && !a.path().is_ident("new")
        })
        .collect();

    let ret_ty = match &method.sig.output {
        ReturnType::Default => quote! { pyo3::PyResult<Self> },
        ReturnType::Type(_, ty) => quote! { #ty },
    };
    let body_ts = quote! { #body };

    let mut async_gen = method.sig.generics.clone();
    let lt: LifetimeParam = parse_quote! { 'py };
    async_gen.params.insert(0, GenericParam::Lifetime(lt));
    let async_params: Vec<_> = async_gen.params.iter().collect();
    let async_generics_ts = quote! { < #(#async_params),* > };
    let async_where = async_gen.where_clause.as_ref();

    let sync_generics: Vec<_> = method.sig.generics.params.iter().collect();
    let sync_generics_ts = if sync_generics.is_empty() {
        quote! {}
    } else {
        quote! { < #(#sync_generics),* > }
    };
    let sync_where = method.sig.generics.where_clause.as_ref();

    let vis = &method.vis;

    let sync_fn: ImplItem = parse_quote! {
        #(#preserved_attrs)*
        #[new]
        #vis fn #sync_name #sync_generics_ts ( py: pyo3::Python<'_>, #(#other_args),* )
            -> #ret_ty
            #sync_where
        {
            crate::convert::block_on_context(py, async move {
                let __ret: #ret_ty = (async move #body_ts).await;
                __ret
            })
        }
    };

    let async_fn: ImplItem = parse_quote! {
        #(#preserved_attrs)*
        #[classmethod]
        #vis fn #aname #async_generics_ts (
            _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
            py: pyo3::Python<'py>,
            #(#other_args),*
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>>
            #async_where
        {
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let __ret: #ret_ty = (async move #body_ts).await;
                let __built = __ret?;
                pyo3::Python::attach(|py| pyo3::Py::new(py, __built))
            })
        }
    };

    Ok(vec![sync_fn, async_fn])
}

fn is_python_type(ty: &Type) -> bool {
    if let Type::Path(tp) = ty {
        tp.path.segments.last().is_some_and(|s| s.ident == "Python")
    } else {
        false
    }
}

/// Rewrites `self` identifiers in expression position to `__this` so the
/// body can be moved into an `async move` block without retaining a
/// `&self` borrow.
struct SelfRewriter;

impl VisitMut for SelfRewriter {
    fn visit_expr_path_mut(&mut self, node: &mut syn::ExprPath) {
        if node.qself.is_none() && node.path.is_ident("self") {
            let span = node.path.segments[0].ident.span();
            node.path = syn::Path::from(Ident::new("__this", span));
            if let Some(seg) = node.path.segments.first_mut() {
                seg.arguments = PathArguments::None;
            }
        }
        syn::visit_mut::visit_expr_path_mut(self, node);
    }
}
