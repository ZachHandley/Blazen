//! Python wrapper for [`blazen_llm::content::ContentKind`].

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass_enum, gen_stub_pymethods};

use blazen_llm::content::ContentKind;

use crate::error::BlazenPyError;

/// Taxonomy of multimodal content kinds.
///
/// Mirrors [`blazen_llm::content::ContentKind`]. Used by tool-input
/// declarations and [`ContentStore`](super::store::PyContentStore) routing.
///
/// Example:
///     >>> ContentKind.Image
///     >>> ContentKind.from_str("image")
///     >>> ContentKind.from_mime("image/png")
#[gen_stub_pyclass_enum]
#[pyclass(name = "ContentKind", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyContentKind {
    Image,
    Audio,
    Video,
    Document,
    ThreeDModel,
    Cad,
    Archive,
    Font,
    Code,
    Data,
    Other,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyContentKind {
    /// The canonical short name (matches the JSON / serde tag).
    #[getter]
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn name_str(&self) -> &'static str {
        ContentKind::from(*self).as_str()
    }

    /// Parse a content kind from its canonical wire name (e.g. `"image"`,
    /// `"three_d_model"`, `"cad"`). Unknown names raise `ValueError`.
    #[staticmethod]
    fn from_str(value: &str) -> PyResult<Self> {
        match value {
            "image" => Ok(Self::Image),
            "audio" => Ok(Self::Audio),
            "video" => Ok(Self::Video),
            "document" => Ok(Self::Document),
            "three_d_model" => Ok(Self::ThreeDModel),
            "cad" => Ok(Self::Cad),
            "archive" => Ok(Self::Archive),
            "font" => Ok(Self::Font),
            "code" => Ok(Self::Code),
            "data" => Ok(Self::Data),
            "other" => Ok(Self::Other),
            other => Err(BlazenPyError::InvalidArgument(format!(
                "unknown content kind: '{other}'"
            ))
            .into()),
        }
    }

    /// Map a MIME type to a content kind. Unknown MIME types resolve to
    /// `ContentKind.Other`.
    #[staticmethod]
    fn from_mime(mime: &str) -> Self {
        ContentKind::from_mime(mime).into()
    }

    /// Map a filename extension (without leading dot) to a content kind.
    /// Matching is case-insensitive. Unknown extensions resolve to
    /// `ContentKind.Other`.
    #[staticmethod]
    fn from_extension(ext: &str) -> Self {
        ContentKind::from_extension(ext).into()
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn __repr__(&self) -> String {
        format!("ContentKind.{}", self.name_str())
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn __str__(&self) -> &'static str {
        self.name_str()
    }
}

impl From<PyContentKind> for ContentKind {
    fn from(p: PyContentKind) -> Self {
        match p {
            PyContentKind::Image => Self::Image,
            PyContentKind::Audio => Self::Audio,
            PyContentKind::Video => Self::Video,
            PyContentKind::Document => Self::Document,
            PyContentKind::ThreeDModel => Self::ThreeDModel,
            PyContentKind::Cad => Self::Cad,
            PyContentKind::Archive => Self::Archive,
            PyContentKind::Font => Self::Font,
            PyContentKind::Code => Self::Code,
            PyContentKind::Data => Self::Data,
            PyContentKind::Other => Self::Other,
        }
    }
}

impl From<ContentKind> for PyContentKind {
    fn from(k: ContentKind) -> Self {
        match k {
            ContentKind::Image => Self::Image,
            ContentKind::Audio => Self::Audio,
            ContentKind::Video => Self::Video,
            ContentKind::Document => Self::Document,
            ContentKind::ThreeDModel => Self::ThreeDModel,
            ContentKind::Cad => Self::Cad,
            ContentKind::Archive => Self::Archive,
            ContentKind::Font => Self::Font,
            ContentKind::Code => Self::Code,
            ContentKind::Data => Self::Data,
            // `ContentKind` is `#[non_exhaustive]`; future variants fall
            // through to `Other` until we add explicit Python mappings.
            _ => Self::Other,
        }
    }
}
