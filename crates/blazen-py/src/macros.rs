/// Macro to generate boilerplate Python wrapper types for compute result structs.
///
/// Each generated type follows the same pattern: a frozen `#[pyclass]` with a
/// `pub(crate) inner` field and `#[getter]` methods that delegate to it.
///
/// # Field kinds
///
/// - `clone(ReturnType)` -- calls `.clone()` on the inner field (for `String`,
///   `Option<String>`, etc.)
/// - `copy(ReturnType)` -- copies the inner field directly (for `f64`, `bool`,
///   `Option<f64>`, etc.)
/// - `wrap(PyWrapperType)` -- wraps the inner field in a newtype wrapper via
///   `PyWrapperType { inner: self.inner.field.clone() }`
/// - `vec_wrap(PyWrapperType)` -- maps a `Vec<RustType>` into
///   `Vec<PyWrapperType>` via `.iter().map(...)`
/// - `json` -- converts a `serde_json::Value` field to a Python dict via
///   `crate::convert::json_to_py`
///
/// # Optional repr
///
/// Append `repr: "format string", field1, field2, ...` to generate a custom
/// `__repr__` method.  The format string is passed to `format!()` and each
/// field ident is expanded to `self.inner.<field>`.
/// If omitted a default `TypeName(...)` repr is generated.
macro_rules! py_result_type {
    // -----------------------------------------------------------------------
    // Main entry point -- default repr
    // -----------------------------------------------------------------------
    (
        $(#[$meta:meta])*
        $py_name:literal, $py_ident:ident, $rust_type:ty,
        fields: {
            $( $(#[$field_meta:meta])* $field_name:ident : $field_kind:ident $( ( $($field_args:tt)* ) )? ),* $(,)?
        } $(,)?
    ) => {
        py_result_type!(
            @build [ $(#[$meta])* ]
            $py_name, $py_ident, [ $rust_type ],
            @repr [],
            [ $( $(#[$field_meta])* $field_name : $field_kind $( ( $($field_args)* ) )? ; )* ],
            []
        );
    };

    // -----------------------------------------------------------------------
    // Main entry point -- custom repr
    // -----------------------------------------------------------------------
    (
        $(#[$meta:meta])*
        $py_name:literal, $py_ident:ident, $rust_type:ty,
        fields: {
            $( $(#[$field_meta:meta])* $field_name:ident : $field_kind:ident $( ( $($field_args:tt)* ) )? ),* $(,)?
        },
        repr: $repr_fmt:literal, $( $repr_field:ident ),+ $(,)?
    ) => {
        py_result_type!(
            @build [ $(#[$meta])* ]
            $py_name, $py_ident, [ $rust_type ],
            @repr [ $repr_fmt $( $repr_field )+ ],
            [ $( $(#[$field_meta])* $field_name : $field_kind $( ( $($field_args)* ) )? ; )* ],
            []
        );
    };

    // -----------------------------------------------------------------------
    // @build terminal -- default repr (empty @repr [])
    // -----------------------------------------------------------------------
    (
        @build [ $(#[$meta:meta])* ]
        $py_name:literal, $py_ident:ident, [ $rust_type:ty ],
        @repr [],
        [],
        [ $($acc:tt)* ]
    ) => {
        #[gen_stub_pyclass]
        #[pyclass(name = $py_name, frozen)]
        $(#[$meta])*
        pub struct $py_ident {
            pub(crate) inner: $rust_type,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $py_ident {
            $($acc)*

            fn __repr__(&self) -> String {
                format!(concat!($py_name, "(...)"))
            }
        }
    };

    // -----------------------------------------------------------------------
    // @build terminal -- custom repr
    // -----------------------------------------------------------------------
    (
        @build [ $(#[$meta:meta])* ]
        $py_name:literal, $py_ident:ident, [ $rust_type:ty ],
        @repr [ $repr_fmt:literal $( $repr_field:ident )+ ],
        [],
        [ $($acc:tt)* ]
    ) => {
        #[gen_stub_pyclass]
        #[pyclass(name = $py_name, frozen)]
        $(#[$meta])*
        pub struct $py_ident {
            pub(crate) inner: $rust_type,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $py_ident {
            $($acc)*

            fn __repr__(&self) -> String {
                format!($repr_fmt, $( self.inner.$repr_field ),+ )
            }
        }
    };

    // -----------------------------------------------------------------------
    // @build -- clone field
    // -----------------------------------------------------------------------
    (
        @build [ $(#[$meta:meta])* ]
        $py_name:literal, $py_ident:ident, [ $rust_type:ty ],
        @repr [ $($repr:tt)* ],
        [ $(#[$field_meta:meta])* $field_name:ident : clone ( $ret_ty:ty ) ; $($rest:tt)* ],
        [ $($acc:tt)* ]
    ) => {
        py_result_type!(
            @build [ $(#[$meta])* ]
            $py_name, $py_ident, [ $rust_type ],
            @repr [ $($repr)* ],
            [ $($rest)* ],
            [
                $($acc)*
                $(#[$field_meta])*
                #[getter]
                fn $field_name(&self) -> $ret_ty {
                    self.inner.$field_name.clone()
                }
            ]
        );
    };

    // -----------------------------------------------------------------------
    // @build -- copy field
    // -----------------------------------------------------------------------
    (
        @build [ $(#[$meta:meta])* ]
        $py_name:literal, $py_ident:ident, [ $rust_type:ty ],
        @repr [ $($repr:tt)* ],
        [ $(#[$field_meta:meta])* $field_name:ident : copy ( $ret_ty:ty ) ; $($rest:tt)* ],
        [ $($acc:tt)* ]
    ) => {
        py_result_type!(
            @build [ $(#[$meta])* ]
            $py_name, $py_ident, [ $rust_type ],
            @repr [ $($repr)* ],
            [ $($rest)* ],
            [
                $($acc)*
                $(#[$field_meta])*
                #[getter]
                fn $field_name(&self) -> $ret_ty {
                    self.inner.$field_name
                }
            ]
        );
    };

    // -----------------------------------------------------------------------
    // @build -- wrap field (single nested wrapper type)
    // -----------------------------------------------------------------------
    (
        @build [ $(#[$meta:meta])* ]
        $py_name:literal, $py_ident:ident, [ $rust_type:ty ],
        @repr [ $($repr:tt)* ],
        [ $(#[$field_meta:meta])* $field_name:ident : wrap ( $py_wrapper:ident ) ; $($rest:tt)* ],
        [ $($acc:tt)* ]
    ) => {
        py_result_type!(
            @build [ $(#[$meta])* ]
            $py_name, $py_ident, [ $rust_type ],
            @repr [ $($repr)* ],
            [ $($rest)* ],
            [
                $($acc)*
                $(#[$field_meta])*
                #[getter]
                fn $field_name(&self) -> $py_wrapper {
                    $py_wrapper { inner: self.inner.$field_name.clone() }
                }
            ]
        );
    };

    // -----------------------------------------------------------------------
    // @build -- vec_wrap field (Vec<RustType> -> Vec<PyWrapper>)
    // -----------------------------------------------------------------------
    (
        @build [ $(#[$meta:meta])* ]
        $py_name:literal, $py_ident:ident, [ $rust_type:ty ],
        @repr [ $($repr:tt)* ],
        [ $(#[$field_meta:meta])* $field_name:ident : vec_wrap ( $py_wrapper:ident ) ; $($rest:tt)* ],
        [ $($acc:tt)* ]
    ) => {
        py_result_type!(
            @build [ $(#[$meta])* ]
            $py_name, $py_ident, [ $rust_type ],
            @repr [ $($repr)* ],
            [ $($rest)* ],
            [
                $($acc)*
                $(#[$field_meta])*
                #[getter]
                fn $field_name(&self) -> Vec<$py_wrapper> {
                    self.inner
                        .$field_name
                        .iter()
                        .map(|x| $py_wrapper { inner: x.clone() })
                        .collect()
                }
            ]
        );
    };

    // -----------------------------------------------------------------------
    // @build -- json field (serde_json::Value -> Python dict)
    // -----------------------------------------------------------------------
    (
        @build [ $(#[$meta:meta])* ]
        $py_name:literal, $py_ident:ident, [ $rust_type:ty ],
        @repr [ $($repr:tt)* ],
        [ $(#[$field_meta:meta])* $field_name:ident : json ; $($rest:tt)* ],
        [ $($acc:tt)* ]
    ) => {
        py_result_type!(
            @build [ $(#[$meta])* ]
            $py_name, $py_ident, [ $rust_type ],
            @repr [ $($repr)* ],
            [ $($rest)* ],
            [
                $($acc)*
                $(#[$field_meta])*
                #[getter]
                #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
                fn $field_name(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                    crate::convert::json_to_py(py, &self.inner.$field_name)
                }
            ]
        );
    };
}
