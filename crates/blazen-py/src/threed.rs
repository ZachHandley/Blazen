//! Python bindings for the [`blazen_3d`] crate's HTTP-proxy backend.
//!
//! Exposes the four 3D-pipeline capability traits — texturizer, rigger,
//! refiner, animator — as Python classes wrapping the
//! [`Compat3dProvider`] HTTP-proxy backend that forwards each stage to
//! a configurable upstream service over `multipart/form-data`.
//!
//! # Surface
//!
//! * Request types: :class:`TexturizeRequest`, :class:`RigRequest`,
//!   :class:`RefineRequest`, :class:`AnimateRequest`.
//! * Result types: :class:`TexturizeResult` (with :class:`PbrMaps`),
//!   :class:`RigResult`, :class:`RefineResult` (with :class:`RefineStats`),
//!   :class:`AnimateResult`.
//! * Provider: :class:`Compat3dProvider` with async methods
//!   :meth:`texturize`, :meth:`rig`, :meth:`refine`, :meth:`animate`.

use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_3d::backends::compat::Compat3dProvider;
use blazen_3d::{
    AnimateRequest, AnimateResult, Animator3dBackend, PbrMaps, RefineRequest, RefineResult,
    RefineStats, Refiner3dBackend, RigRequest, RigResult, Rigger3dBackend, TexturizeRequest,
    TexturizeResult, Texturizer3dBackend,
};

// ---------------------------------------------------------------------------
// PbrMaps
// ---------------------------------------------------------------------------

/// Bundle of PBR (physically-based rendering) material maps produced by
/// a texturizer backend.
#[gen_stub_pyclass]
#[pyclass(name = "PbrMaps", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyPbrMaps {
    pub(crate) inner: PbrMaps,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPbrMaps {
    /// Base-color / diffuse texture as PNG bytes. Always present.
    #[getter]
    fn albedo_png<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.albedo_png)
    }

    /// Tangent-space normal map as PNG bytes, if produced.
    #[getter]
    fn normal_png<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .normal_png
            .as_deref()
            .map(|b| PyBytes::new(py, b))
    }

    /// Linear roughness map as PNG bytes, if produced.
    #[getter]
    fn roughness_png<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .roughness_png
            .as_deref()
            .map(|b| PyBytes::new(py, b))
    }

    /// Linear metallic map as PNG bytes, if produced.
    #[getter]
    fn metallic_png<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .metallic_png
            .as_deref()
            .map(|b| PyBytes::new(py, b))
    }

    fn __repr__(&self) -> String {
        "PbrMaps(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// RefineStats
// ---------------------------------------------------------------------------

/// Summary statistics emitted by a :meth:`Compat3dProvider.refine` call.
#[gen_stub_pyclass]
#[pyclass(name = "RefineStats", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyRefineStats {
    pub(crate) inner: RefineStats,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRefineStats {
    /// Triangle count of the input mesh.
    #[getter]
    fn input_tri_count(&self) -> u32 {
        self.inner.input_tri_count
    }

    /// Triangle count of the output (refined) mesh.
    #[getter]
    fn output_tri_count(&self) -> u32 {
        self.inner.output_tri_count
    }

    /// When UV unwrapping ran, the number of UV charts the unwrap produced.
    /// ``None`` when UV unwrapping did not run for this call.
    #[getter]
    fn uv_chart_count(&self) -> Option<u32> {
        self.inner.uv_chart_count
    }

    fn __repr__(&self) -> String {
        format!(
            "RefineStats(input_tri_count={}, output_tri_count={}, uv_chart_count={:?})",
            self.inner.input_tri_count, self.inner.output_tri_count, self.inner.uv_chart_count,
        )
    }
}

// ---------------------------------------------------------------------------
// TexturizeRequest
// ---------------------------------------------------------------------------

/// Request parameters for :meth:`Compat3dProvider.texturize`.
#[gen_stub_pyclass]
#[pyclass(name = "TexturizeRequest", from_py_object)]
#[derive(Clone)]
pub struct PyTexturizeRequest {
    pub(crate) inner: TexturizeRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTexturizeRequest {
    /// Construct a new texturize request.
    ///
    /// Args:
    ///     prompt: Text-guided texture prompt (e.g. ``"weathered bronze"``).
    ///     reference_image: Optional PNG/JPEG bytes used as a style anchor.
    ///     style: Backend-specific style preset (``"stylized"``, ``"realistic"``, ...).
    ///     resolution: Target square texture resolution in pixels.
    ///     pbr: ``True`` to request a full PBR material bundle.
    #[new]
    #[pyo3(signature = (*, prompt=None, reference_image=None, style=None, resolution=None, pbr=false))]
    fn new(
        prompt: Option<String>,
        reference_image: Option<Vec<u8>>,
        style: Option<String>,
        resolution: Option<u32>,
        pbr: bool,
    ) -> Self {
        Self {
            inner: TexturizeRequest {
                prompt,
                reference_image,
                style,
                resolution,
                pbr,
            },
        }
    }

    #[getter]
    fn prompt(&self) -> Option<String> {
        self.inner.prompt.clone()
    }

    #[getter]
    fn reference_image<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .reference_image
            .as_deref()
            .map(|b| PyBytes::new(py, b))
    }

    #[getter]
    fn style(&self) -> Option<String> {
        self.inner.style.clone()
    }

    #[getter]
    fn resolution(&self) -> Option<u32> {
        self.inner.resolution
    }

    #[getter]
    fn pbr(&self) -> bool {
        self.inner.pbr
    }

    fn __repr__(&self) -> String {
        format!(
            "TexturizeRequest(prompt={:?}, style={:?}, resolution={:?}, pbr={})",
            self.inner.prompt, self.inner.style, self.inner.resolution, self.inner.pbr,
        )
    }
}

// ---------------------------------------------------------------------------
// TexturizeResult
// ---------------------------------------------------------------------------

/// Result of a successful :meth:`Compat3dProvider.texturize` call.
#[gen_stub_pyclass]
#[pyclass(name = "TexturizeResult", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyTexturizeResult {
    pub(crate) inner: TexturizeResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTexturizeResult {
    /// GLB bytes with the new texture (and PBR maps if any) embedded.
    #[getter]
    fn textured_glb<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.textured_glb)
    }

    /// MIME type of :attr:`textured_glb`; always ``"model/gltf-binary"``.
    #[getter]
    fn mime_type(&self) -> String {
        self.inner.mime_type.clone()
    }

    /// Optional out-of-band PBR map bundle. Duplicates of the maps
    /// embedded in :attr:`textured_glb` when present.
    #[getter]
    fn pbr_maps(&self) -> Option<PyPbrMaps> {
        self.inner
            .pbr_maps
            .as_ref()
            .map(|m| PyPbrMaps { inner: m.clone() })
    }

    fn __repr__(&self) -> String {
        format!(
            "TexturizeResult(bytes={}, mime_type={:?}, pbr_maps={})",
            self.inner.textured_glb.len(),
            self.inner.mime_type,
            self.inner.pbr_maps.is_some(),
        )
    }
}

// ---------------------------------------------------------------------------
// RigRequest
// ---------------------------------------------------------------------------

/// Request parameters for :meth:`Compat3dProvider.rig`.
#[gen_stub_pyclass]
#[pyclass(name = "RigRequest", from_py_object)]
#[derive(Clone)]
pub struct PyRigRequest {
    pub(crate) inner: RigRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRigRequest {
    /// Construct a new rig request.
    ///
    /// Args:
    ///     template: Target rig template (``"humanoid"``, ``"quadruped"``, ``"auto"``).
    ///     skin: ``True`` to apply skin-weight painting after armature placement.
    ///     pose_hint: Optional pose hint (``"t-pose"``, ``"a-pose"``, or backend JSON).
    #[new]
    #[pyo3(signature = (*, template=None, skin=true, pose_hint=None))]
    fn new(template: Option<String>, skin: bool, pose_hint: Option<String>) -> Self {
        Self {
            inner: RigRequest {
                template,
                skin,
                pose_hint,
            },
        }
    }

    #[getter]
    fn template(&self) -> Option<String> {
        self.inner.template.clone()
    }

    #[getter]
    fn skin(&self) -> bool {
        self.inner.skin
    }

    #[getter]
    fn pose_hint(&self) -> Option<String> {
        self.inner.pose_hint.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RigRequest(template={:?}, skin={}, pose_hint={:?})",
            self.inner.template, self.inner.skin, self.inner.pose_hint,
        )
    }
}

// ---------------------------------------------------------------------------
// RigResult
// ---------------------------------------------------------------------------

/// Result of a successful :meth:`Compat3dProvider.rig` call.
#[gen_stub_pyclass]
#[pyclass(name = "RigResult", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyRigResult {
    pub(crate) inner: RigResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRigResult {
    /// GLB bytes with the new armature (and skin weights, if requested) embedded.
    #[getter]
    fn rigged_glb<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.rigged_glb)
    }

    /// MIME type of :attr:`rigged_glb`; always ``"model/gltf-binary"``.
    #[getter]
    fn mime_type(&self) -> String {
        self.inner.mime_type.clone()
    }

    /// Names of bones in the produced armature, in depth-first traversal order.
    #[getter]
    fn bone_names(&self) -> Vec<String> {
        self.inner.bone_names.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RigResult(bytes={}, mime_type={:?}, bones={})",
            self.inner.rigged_glb.len(),
            self.inner.mime_type,
            self.inner.bone_names.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// RefineRequest
// ---------------------------------------------------------------------------

/// Request parameters for :meth:`Compat3dProvider.refine`.
#[gen_stub_pyclass]
#[pyclass(name = "RefineRequest", from_py_object)]
#[derive(Clone)]
pub struct PyRefineRequest {
    pub(crate) inner: RefineRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRefineRequest {
    /// Construct a new refine request.
    ///
    /// Args:
    ///     decimate_target_tris: Decimate the mesh towards this triangle count.
    ///     fill_holes: ``True`` to fill holes via screened poisson reconstruction.
    ///     unwrap_uvs: ``True`` to compute a new UV unwrap.
    ///     retopologize: ``True`` to retopologize the mesh.
    ///     smooth_iterations: Laplacian / Taubin smoothing iteration count.
    #[new]
    #[pyo3(signature = (*, decimate_target_tris=None, fill_holes=false, unwrap_uvs=false, retopologize=false, smooth_iterations=None))]
    fn new(
        decimate_target_tris: Option<u32>,
        fill_holes: bool,
        unwrap_uvs: bool,
        retopologize: bool,
        smooth_iterations: Option<u32>,
    ) -> Self {
        Self {
            inner: RefineRequest {
                decimate_target_tris,
                fill_holes,
                unwrap_uvs,
                retopologize,
                smooth_iterations,
            },
        }
    }

    #[getter]
    fn decimate_target_tris(&self) -> Option<u32> {
        self.inner.decimate_target_tris
    }

    #[getter]
    fn fill_holes(&self) -> bool {
        self.inner.fill_holes
    }

    #[getter]
    fn unwrap_uvs(&self) -> bool {
        self.inner.unwrap_uvs
    }

    #[getter]
    fn retopologize(&self) -> bool {
        self.inner.retopologize
    }

    #[getter]
    fn smooth_iterations(&self) -> Option<u32> {
        self.inner.smooth_iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "RefineRequest(decimate_target_tris={:?}, fill_holes={}, unwrap_uvs={}, retopologize={}, smooth_iterations={:?})",
            self.inner.decimate_target_tris,
            self.inner.fill_holes,
            self.inner.unwrap_uvs,
            self.inner.retopologize,
            self.inner.smooth_iterations,
        )
    }
}

// ---------------------------------------------------------------------------
// RefineResult
// ---------------------------------------------------------------------------

/// Result of a successful :meth:`Compat3dProvider.refine` call.
#[gen_stub_pyclass]
#[pyclass(name = "RefineResult", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyRefineResult {
    pub(crate) inner: RefineResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRefineResult {
    /// GLB bytes with the requested refinement passes applied.
    #[getter]
    fn refined_glb<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.refined_glb)
    }

    /// MIME type of :attr:`refined_glb`; always ``"model/gltf-binary"``.
    #[getter]
    fn mime_type(&self) -> String {
        self.inner.mime_type.clone()
    }

    /// Before/after statistics for the refinement run.
    #[getter]
    fn stats(&self) -> PyRefineStats {
        PyRefineStats {
            inner: self.inner.stats.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RefineResult(bytes={}, mime_type={:?})",
            self.inner.refined_glb.len(),
            self.inner.mime_type,
        )
    }
}

// ---------------------------------------------------------------------------
// AnimateRequest
// ---------------------------------------------------------------------------

/// Request parameters for :meth:`Compat3dProvider.animate`.
#[gen_stub_pyclass]
#[pyclass(name = "AnimateRequest", from_py_object)]
#[derive(Clone)]
pub struct PyAnimateRequest {
    pub(crate) inner: AnimateRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAnimateRequest {
    /// Construct a new animate request.
    ///
    /// Args:
    ///     prompt: Text-guided motion prompt (e.g. ``"walks forward and waves"``).
    ///     driving_video: Optional MP4 bytes for video-driven motion transfer.
    ///     bvh_motion: Optional BVH motion-capture clip bytes.
    ///     duration_seconds: Requested animation duration in seconds.
    ///     fps: Requested animation framerate.
    ///     loop_animation: ``True`` to mark the produced animation as a seamless loop.
    #[new]
    #[pyo3(signature = (*, prompt=None, driving_video=None, bvh_motion=None, duration_seconds=None, fps=None, loop_animation=false))]
    fn new(
        prompt: Option<String>,
        driving_video: Option<Vec<u8>>,
        bvh_motion: Option<Vec<u8>>,
        duration_seconds: Option<f32>,
        fps: Option<u32>,
        loop_animation: bool,
    ) -> Self {
        Self {
            inner: AnimateRequest {
                prompt,
                driving_video,
                bvh_motion,
                duration_seconds,
                fps,
                loop_animation,
            },
        }
    }

    #[getter]
    fn prompt(&self) -> Option<String> {
        self.inner.prompt.clone()
    }

    #[getter]
    fn driving_video<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .driving_video
            .as_deref()
            .map(|b| PyBytes::new(py, b))
    }

    #[getter]
    fn bvh_motion<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .bvh_motion
            .as_deref()
            .map(|b| PyBytes::new(py, b))
    }

    #[getter]
    fn duration_seconds(&self) -> Option<f32> {
        self.inner.duration_seconds
    }

    #[getter]
    fn fps(&self) -> Option<u32> {
        self.inner.fps
    }

    #[getter]
    fn loop_animation(&self) -> bool {
        self.inner.loop_animation
    }

    fn __repr__(&self) -> String {
        format!(
            "AnimateRequest(prompt={:?}, duration_seconds={:?}, fps={:?}, loop_animation={})",
            self.inner.prompt,
            self.inner.duration_seconds,
            self.inner.fps,
            self.inner.loop_animation,
        )
    }
}

// ---------------------------------------------------------------------------
// AnimateResult
// ---------------------------------------------------------------------------

/// Result of a successful :meth:`Compat3dProvider.animate` call.
#[gen_stub_pyclass]
#[pyclass(name = "AnimateResult", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyAnimateResult {
    pub(crate) inner: AnimateResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAnimateResult {
    /// GLB bytes with the animation track(s) embedded.
    #[getter]
    fn animated_glb<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.animated_glb)
    }

    /// MIME type of :attr:`animated_glb`; always ``"model/gltf-binary"``.
    #[getter]
    fn mime_type(&self) -> String {
        self.inner.mime_type.clone()
    }

    /// Actual produced duration in seconds (may differ from the request).
    #[getter]
    fn duration_seconds(&self) -> f32 {
        self.inner.duration_seconds
    }

    /// Actual produced framerate in frames per second (may differ from the request).
    #[getter]
    fn fps(&self) -> u32 {
        self.inner.fps
    }

    fn __repr__(&self) -> String {
        format!(
            "AnimateResult(bytes={}, mime_type={:?}, duration_seconds={}, fps={})",
            self.inner.animated_glb.len(),
            self.inner.mime_type,
            self.inner.duration_seconds,
            self.inner.fps,
        )
    }
}

// ---------------------------------------------------------------------------
// Compat3dProvider
// ---------------------------------------------------------------------------

/// HTTP-proxy backend that implements all four 3D-pipeline capability
/// traits against a configurable upstream service.
///
/// For every stage, this provider POSTs a ``multipart/form-data`` request
/// with the mesh GLB and a JSON request body to
/// ``{base_url}/v1/3d/{texturize,rig,refine,animate}``, and decodes a
/// base64-wrapped JSON response into the corresponding result class.
///
/// Example:
///     >>> provider = Compat3dProvider("https://my-3d-server.example.com", api_key="...")
///     >>> result = await provider.texturize(mesh_glb, TexturizeRequest(prompt="bronze", pbr=True))
#[gen_stub_pyclass]
#[pyclass(name = "Compat3dProvider", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCompat3dProvider {
    inner: Arc<Compat3dProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCompat3dProvider {
    /// Construct a new HTTP-proxy provider.
    ///
    /// Args:
    ///     base_url: Upstream base URL (e.g. ``"https://3d.example.com"``).
    ///     api_key: Optional bearer token for ``Authorization: Bearer ...``.
    ///     timeout_secs: Optional per-request timeout in seconds (default 600).
    #[new]
    #[pyo3(signature = (base_url, api_key=None, timeout_secs=None))]
    fn new(base_url: String, api_key: Option<String>, timeout_secs: Option<u64>) -> Self {
        let mut provider = Compat3dProvider::new(base_url);
        if let Some(key) = api_key {
            provider = provider.with_api_key(key);
        }
        if let Some(secs) = timeout_secs {
            provider = provider.with_timeout(Duration::from_secs(secs));
        }
        Self {
            inner: Arc::new(provider),
        }
    }

    /// Apply or generate a texture/material for an existing 3D mesh.
    ///
    /// Args:
    ///     mesh_glb: Input mesh as GLB or OBJ bytes.
    ///     request: A :class:`TexturizeRequest` describing the desired texture.
    ///
    /// Returns:
    ///     A :class:`TexturizeResult` with the textured GLB and optional PBR maps.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, TexturizeResult]", imports = ("typing",)))]
    fn texturize<'py>(
        &self,
        py: Python<'py>,
        mesh_glb: Vec<u8>,
        request: PyTexturizeRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let provider = Arc::clone(&self.inner);
        let req = request.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = provider
                .texturize(&mesh_glb, req)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok::<PyTexturizeResult, PyErr>(PyTexturizeResult { inner: result })
        })
    }

    /// Auto-rig a 3D mesh, producing a GLB with skeletal armature and
    /// (optionally) skin weights embedded.
    ///
    /// Args:
    ///     mesh_glb: Input mesh as GLB bytes.
    ///     request: A :class:`RigRequest` describing the desired rig.
    ///
    /// Returns:
    ///     A :class:`RigResult` with the rigged GLB and bone names.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, RigResult]", imports = ("typing",)))]
    fn rig<'py>(
        &self,
        py: Python<'py>,
        mesh_glb: Vec<u8>,
        request: PyRigRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let provider = Arc::clone(&self.inner);
        let req = request.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = provider
                .rig(&mesh_glb, req)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok::<PyRigResult, PyErr>(PyRigResult { inner: result })
        })
    }

    /// Refine a 3D mesh: decimate, fill holes, unwrap UVs, retopologize, smooth.
    ///
    /// Args:
    ///     mesh_glb: Input mesh as GLB bytes.
    ///     request: A :class:`RefineRequest` describing the passes to apply.
    ///
    /// Returns:
    ///     A :class:`RefineResult` with the refined GLB and statistics.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, RefineResult]", imports = ("typing",)))]
    fn refine<'py>(
        &self,
        py: Python<'py>,
        mesh_glb: Vec<u8>,
        request: PyRefineRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let provider = Arc::clone(&self.inner);
        let req = request.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = provider
                .refine(&mesh_glb, req)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok::<PyRefineResult, PyErr>(PyRefineResult { inner: result })
        })
    }

    /// Animate a rigged 3D mesh from a text prompt, motion-capture clip,
    /// or driving video.
    ///
    /// Args:
    ///     rigged_glb: Rigged mesh as GLB bytes (output of :meth:`rig`).
    ///     request: An :class:`AnimateRequest` describing the desired motion.
    ///
    /// Returns:
    ///     An :class:`AnimateResult` with the animated GLB.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, AnimateResult]", imports = ("typing",)))]
    fn animate<'py>(
        &self,
        py: Python<'py>,
        rigged_glb: Vec<u8>,
        request: PyAnimateRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let provider = Arc::clone(&self.inner);
        let req = request.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = provider
                .animate(&rigged_glb, req)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok::<PyAnimateResult, PyErr>(PyAnimateResult { inner: result })
        })
    }

    fn __repr__(&self) -> String {
        "Compat3dProvider(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register all 3D-pipeline classes onto the parent Python module.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyPbrMaps>()?;
    parent.add_class::<PyRefineStats>()?;
    parent.add_class::<PyTexturizeRequest>()?;
    parent.add_class::<PyTexturizeResult>()?;
    parent.add_class::<PyRigRequest>()?;
    parent.add_class::<PyRigResult>()?;
    parent.add_class::<PyRefineRequest>()?;
    parent.add_class::<PyRefineResult>()?;
    parent.add_class::<PyAnimateRequest>()?;
    parent.add_class::<PyAnimateResult>()?;
    parent.add_class::<PyCompat3dProvider>()?;
    Ok(())
}
