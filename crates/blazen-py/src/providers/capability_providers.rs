//! Per-capability provider classes for user subclassing.
//!
//! Each class exposes one media-generation capability. Users subclass
//! and override the async methods to plug in their own backends.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::compute::request_types::{
    PyBackgroundRemovalRequest, PyImageRequest, PyMusicRequest, PySpeechRequest, PyThreeDRequest,
    PyUpscaleRequest, PyVideoRequest, PyVoiceCloneRequest,
};

// ---------------------------------------------------------------------------
// Helper macro
// ---------------------------------------------------------------------------

/// Declares the struct + `#[pymethods]` impl in one shot so we never try
/// to expand a macro *inside* `#[pymethods]` (which PyO3 forbids).
///
/// `$extra_methods` is a brace-delimited block of additional method items
/// that are spliced into the impl alongside the common constructor/getters.
macro_rules! capability_provider {
    (
        $(#[$meta:meta])*
        $py_name:literal, $struct_name:ident,
        extra { $($extra:tt)* }
    ) => {
        #[gen_stub_pyclass]
        #[pyclass(name = $py_name, subclass)]
        $(#[$meta])*
        pub struct $struct_name {
            config: blazen_llm::ProviderConfig,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $struct_name {
            // -- constructor --------------------------------------------------

            /// `__new__` accepting arbitrary positional/keyword args so Python
            /// subclasses can use any `__init__` signature. Real configuration
            /// happens in `__init__` below.
            #[new]
            #[pyo3(signature = (*_args, **_kwargs))]
            fn new(
                _args: &Bound<'_, pyo3::types::PyTuple>,
                _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
            ) -> Self {
                Self {
                    config: blazen_llm::ProviderConfig::default(),
                }
            }

            /// Subclass-friendly `__init__`. Mirrors the documented constructor
            /// keyword signature and populates ``self.config`` so a Python
            /// subclass that calls `super().__init__(provider_id=...)` sees the
            /// values it passed (without falling through to `object.__init__`,
            /// which would raise ``TypeError``).
            #[pyo3(signature = (*, provider_id=None, base_url=None, pricing=None, memory_estimate_bytes=None))]
            fn __init__(
                &mut self,
                provider_id: Option<String>,
                base_url: Option<String>,
                pricing: Option<PyRef<'_, crate::types::pricing::PyModelPricing>>,
                memory_estimate_bytes: Option<u64>,
            ) {
                self.config = blazen_llm::ProviderConfig {
                    provider_id,
                    base_url,
                    memory_estimate_bytes,
                    pricing: pricing.map(|p| p.inner.clone()),
                    ..Default::default()
                };
            }

            // -- common getters -----------------------------------------------

            /// The provider identifier.
            #[getter]
            fn provider_id(&self) -> Option<&str> {
                self.config.provider_id.as_deref()
            }

            /// The base URL, if set.
            #[getter]
            fn base_url(&self) -> Option<&str> {
                self.config.base_url.as_deref()
            }

            /// Estimated memory footprint in bytes (host RAM if on CPU, GPU VRAM otherwise), if set.
            #[getter]
            fn memory_estimate_bytes(&self) -> Option<u64> {
                self.config.memory_estimate_bytes
            }

            // -- capability-specific methods ----------------------------------

            $($extra)*
        }
    };
}

// ---------------------------------------------------------------------------
// 1. TTSProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for text-to-speech providers.
    ///
    /// Subclass and override ``text_to_speech()`` to implement a custom TTS backend.
    "TTSProvider", TTSProvider,
    extra {
        /// Synthesize speech from text.
        #[pyo3(signature = (request))]
        fn text_to_speech(&self, request: PyRef<'_, PySpeechRequest>) -> PyResult<Py<PyAny>> {
            let _ = request;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override text_to_speech()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 2. MusicProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for music generation providers.
    ///
    /// Subclass and override ``generate_music()`` and ``generate_sfx()`` to
    /// implement a custom music/SFX backend.
    "MusicProvider", MusicProvider,
    extra {
        /// Generate music from a prompt.
        #[pyo3(signature = (request))]
        fn generate_music(&self, request: PyRef<'_, PyMusicRequest>) -> PyResult<Py<PyAny>> {
            let _ = request;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override generate_music()",
            ))
        }

        /// Generate a sound effect from a prompt.
        #[pyo3(signature = (request))]
        fn generate_sfx(&self, request: PyRef<'_, PyMusicRequest>) -> PyResult<Py<PyAny>> {
            let _ = request;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override generate_sfx()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 3. ImageProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for image generation providers.
    ///
    /// Subclass and override ``generate_image()`` and ``upscale_image()`` to
    /// implement a custom image backend.
    "ImageProvider", ImageProvider,
    extra {
        /// Generate an image from a prompt.
        #[pyo3(signature = (request))]
        fn generate_image(&self, request: PyRef<'_, PyImageRequest>) -> PyResult<Py<PyAny>> {
            let _ = request;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override generate_image()",
            ))
        }

        /// Upscale an existing image.
        #[pyo3(signature = (request))]
        fn upscale_image(&self, request: PyRef<'_, PyUpscaleRequest>) -> PyResult<Py<PyAny>> {
            let _ = request;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override upscale_image()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 4. VideoProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for video generation providers.
    ///
    /// Subclass and override ``text_to_video()`` and ``image_to_video()`` to
    /// implement a custom video backend.
    "VideoProvider", VideoProvider,
    extra {
        /// Generate a video from a text prompt.
        #[pyo3(signature = (request))]
        fn text_to_video(&self, request: PyRef<'_, PyVideoRequest>) -> PyResult<Py<PyAny>> {
            let _ = request;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override text_to_video()",
            ))
        }

        /// Generate a video from an image (image-to-video).
        #[pyo3(signature = (request))]
        fn image_to_video(&self, request: PyRef<'_, PyVideoRequest>) -> PyResult<Py<PyAny>> {
            let _ = request;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override image_to_video()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 5. ThreeDProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for 3D model generation providers.
    ///
    /// Subclass and override ``generate_3d()`` to implement a custom 3D backend.
    "ThreeDProvider", ThreeDProvider,
    extra {
        /// Generate a 3D model from a prompt or image.
        #[pyo3(signature = (request))]
        fn generate_3d(&self, request: PyRef<'_, PyThreeDRequest>) -> PyResult<Py<PyAny>> {
            let _ = request;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override generate_3d()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 6. BackgroundRemovalProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for background removal providers.
    ///
    /// Subclass and override ``remove_background()`` to implement a custom
    /// background-removal backend.
    "BackgroundRemovalProvider", BackgroundRemovalProvider,
    extra {
        /// Remove the background from an image.
        #[pyo3(signature = (request))]
        fn remove_background(
            &self,
            request: PyRef<'_, PyBackgroundRemovalRequest>,
        ) -> PyResult<Py<PyAny>> {
            let _ = request;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override remove_background()",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// 7. VoiceProvider
// ---------------------------------------------------------------------------

capability_provider! {
    /// Base class for voice cloning providers.
    ///
    /// Subclass and override ``clone_voice()``, ``list_voices()``, and
    /// ``delete_voice()`` to implement a custom voice-cloning backend.
    "VoiceProvider", VoiceProvider,
    extra {
        /// Clone a voice from audio samples.
        #[pyo3(signature = (request))]
        fn clone_voice(&self, request: PyRef<'_, PyVoiceCloneRequest>) -> PyResult<Py<PyAny>> {
            let _ = request;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override clone_voice()",
            ))
        }

        /// List all available voices.
        fn list_voices(&self) -> PyResult<Py<PyAny>> {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override list_voices()",
            ))
        }

        /// Delete a previously-cloned voice.
        #[pyo3(signature = (voice))]
        fn delete_voice(
            &self,
            voice: PyRef<'_, crate::compute::result_types::PyVoiceHandle>,
        ) -> PyResult<Py<PyAny>> {
            let _ = voice;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "subclass must override delete_voice()",
            ))
        }
    }
}
