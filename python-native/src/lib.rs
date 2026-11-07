//! Python boundary for real and complex sample-fed RSRS sessions.

use std::{mem, sync::OnceLock};

use bempp_octree::{Octree, Point};
use bempp_rsrs::rsrs::{
    args::{RsrsArgs, RsrsOptions},
    rsrs_cycle::{PrecomputedAdvance, PrecomputedRsrsSession, Rsrs},
    rsrs_factors::{
        base_factors::BaseFactorOptions,
        commutative_factors::RsrsFactors,
        rsrs_operator::RsrsFactorsImpl,
    },
    sketch::PrecomputedSampleBlock,
};
use mpi::{environment::Threading, topology::SimpleCommunicator};
use num_complex::Complex64;
use numpy::{ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
use rlst::{Side, TransMode};

static MPI_INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();

fn ensure_mpi() -> Result<(), String> {
    MPI_INITIALIZED
        .get_or_init(|| {
            if let Some((universe, provided)) = mpi::initialize_with_threading(Threading::Multiple)
            {
                if provided < Threading::Serialized {
                    return Err(format!(
                        "MPI provides {provided:?}; RSRS Python requires at least Serialized"
                    ));
                }
                mem::forget(universe);
            }
            Ok(())
        })
        .clone()
}

fn row_major_f64(values: PyReadonlyArray2<'_, f64>) -> PyResult<(usize, usize, Vec<f64>)> {
    let view = values.as_array();
    let (rows, columns) = view.dim();
    Ok((rows, columns, view.iter().copied().collect()))
}

fn row_major_c64(
    values: PyReadonlyArray2<'_, Complex64>,
) -> PyResult<(usize, usize, Vec<Complex64>)> {
    let view = values.as_array();
    let (rows, columns) = view.dim();
    Ok((rows, columns, view.iter().copied().collect()))
}

/// First native session boundary. It owns the octree-derived RSRS state and
/// accepts Python-generated sample rows without retaining NumPy memory.
#[pyclass]
struct RsrsSessionF64 {
    state: Option<PrecomputedRsrsSession<f64>>,
}

/// Completed RSRS factors with no dependency on the consumed session.
#[pyclass]
struct RsrsFactorsF64 {
    factors: RsrsFactors<f64>,
}

#[pyclass]
struct RsrsSessionC64 {
    state: Option<PrecomputedRsrsSession<Complex64>>,
}

#[pyclass]
struct RsrsFactorsC64 {
    factors: RsrsFactors<Complex64>,
}

#[pymethods]
impl RsrsSessionF64 {
    #[new]
    #[pyo3(signature = (points, options_json, max_tree_depth=16, leaf_size=None))]
    fn new(
        points: PyReadonlyArrayDyn<'_, f64>,
        options_json: &str,
        max_tree_depth: usize,
        leaf_size: Option<usize>,
    ) -> PyResult<Self> {
        ensure_mpi().map_err(PyRuntimeError::new_err)?;
        let coordinates = points.as_array();
        if coordinates.ndim() != 2 || coordinates.shape()[1] != 3 {
            return Err(PyRuntimeError::new_err(format!(
                "points must have shape (n, 3), got {:?}",
                coordinates.shape()
            )));
        }
        let point_count = coordinates.shape()[0];
        if point_count == 0 {
            return Err(PyRuntimeError::new_err("points must not be empty"));
        }
        let octree_points = coordinates
            .outer_iter()
            .enumerate()
            .map(|(index, row)| Point::new([row[0], row[1], row[2]], index))
            .collect::<Vec<_>>();
        let communicator = SimpleCommunicator::self_comm();
        let max_leaf_points = leaf_size.unwrap_or_else(|| {
            point_count
                .div_ceil(
                    8usize
                        .saturating_pow(max_tree_depth.saturating_sub(1) as u32)
                        .max(1),
                )
                .max(1)
        });
        let octree = Octree::new(
            &octree_points,
            max_tree_depth,
            max_leaf_points,
            &communicator,
        );
        let args: RsrsArgs<f64> = serde_json::from_str(options_json)
            .map_err(|error| PyRuntimeError::new_err(format!("invalid RSRS options: {error}")))?;
        let options = RsrsOptions::new(Some(args));
        Ok(Self {
            state: Some(PrecomputedRsrsSession::new(Rsrs::new(
                &octree,
                options,
                point_count,
            ))),
        })
    }

    #[getter]
    fn dimension(&self) -> PyResult<usize> {
        Ok(self.state()?.rsrs().dimension())
    }

    #[getter]
    fn y_sample_rows(&self) -> PyResult<usize> {
        Ok(self.state()?.rsrs().y_sample_rows())
    }

    #[getter]
    fn z_sample_rows(&self) -> PyResult<usize> {
        Ok(self.state()?.rsrs().z_sample_rows())
    }

    #[getter]
    fn requires_z(&self) -> PyResult<bool> {
        Ok(self.state()?.rsrs().requires_z_samples())
    }

    #[pyo3(signature = (y_test, y_sketch, z_test=None, z_sketch=None))]
    fn add_samples(
        &mut self,
        y_test: PyReadonlyArray2<'_, f64>,
        y_sketch: PyReadonlyArray2<'_, f64>,
        z_test: Option<PyReadonlyArray2<'_, f64>>,
        z_sketch: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<()> {
        let (y_rows, y_dim, y_test) = row_major_f64(y_test)?;
        let (sketch_rows, sketch_dim, y_sketch) = row_major_f64(y_sketch)?;
        if (y_rows, y_dim) != (sketch_rows, sketch_dim) {
            return Err(PyRuntimeError::new_err("Y test/sketch shapes differ"));
        }
        let y = PrecomputedSampleBlock {
            test: &y_test,
            sketch: &y_sketch,
            rows: y_rows,
            dim: y_dim,
        };
        let z_storage = match (z_test, z_sketch) {
            (Some(test), Some(sketch)) => {
                let (rows, dim, test) = row_major_f64(test)?;
                let (sketch_rows, sketch_dim, sketch) = row_major_f64(sketch)?;
                if (rows, dim) != (sketch_rows, sketch_dim) {
                    return Err(PyRuntimeError::new_err("Z test/sketch shapes differ"));
                }
                Some((test, sketch, rows, dim))
            }
            (None, None) => None,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "Z test and sketch must be supplied together",
                ))
            }
        };
        let z = z_storage
            .as_ref()
            .map(|(test, sketch, rows, dim)| PrecomputedSampleBlock {
                test,
                sketch,
                rows: *rows,
                dim: *dim,
            });
        self.state_mut()?
            .rsrs_mut()
            .append_precomputed_samples(y, z)
            .map_err(PyRuntimeError::new_err)
    }

    /// Advance one factorisation level, pause for exact sample deficits, or
    /// finish diagonal extraction.
    fn advance<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new_bound(py);
        match self
            .state_mut()?
            .advance()
            .map_err(PyRuntimeError::new_err)?
        {
            PrecomputedAdvance::NeedsSamples(request) => {
                result.set_item("status", "needs_samples")?;
                result.set_item("y_rows", request.y_rows)?;
                result.set_item("z_rows", request.z_rows)?;
                result.set_item("total_required", request.total_required)?;
                result.set_item("level", request.level)?;
                result.set_item("limiting_box", request.limiting_box)?;
            }
            PrecomputedAdvance::Progressed { completed_level } => {
                result.set_item("status", "progressed")?;
                result.set_item("completed_level", completed_level)?;
            }
            PrecomputedAdvance::Complete => {
                result.set_item("status", "complete")?;
            }
        }
        Ok(result)
    }

    /// Serialize quantities intrinsic to RSRS before the session is consumed.
    fn report_json(&self) -> PyResult<String> {
        let rsrs = self.state()?.rsrs();
        let stats = &rsrs.stats;
        serde_json::to_string(&serde_json::json!({
            "dimension": rsrs.dimension(),
            "active_samples": rsrs.active_samples,
            "y_sample_rows": rsrs.y_sample_rows(),
            "z_sample_rows": rsrs.z_sample_rows(),
            "ranks": stats.ranks,
            "box_sizes": stats.box_sizes,
            "near_field_sizes": stats.near_field_sizes,
            "decomposed_boxes_per_level": stats.dec_boxes_per_level,
            "id_times_ms": stats.id_times,
            "lu_times_ms": stats.lu_times,
            "update_times_ms": stats.update_times,
            "extraction_time_ms": stats.extraction_time,
            "total_factorization_time_ms": stats.total_elapsed_time_wo_sampling,
            "index_calculation_ms": stats.index_calculation,
            "sorting_near_field_ms": stats.sorting_near_field,
            "residual_calculation_ms": stats.residual_calculation,
            "level_effort": stats.level_effort,
            "max_sample_buffer_bytes": stats.max_sample_buffer_bytes,
            "max_factor_bytes": stats.max_factor_bytes,
            "max_accounted_factorization_bytes": stats.max_accounted_factorization_bytes,
        }))
        .map_err(|error| PyRuntimeError::new_err(format!("cannot serialize RSRS report: {error}")))
    }

    /// Consume a completed factorisation and return an independently owned
    /// factor object. The session cannot be used after this call.
    fn take_factors(&mut self) -> PyResult<RsrsFactorsF64> {
        let state = self
            .state
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("RSRS session was already consumed"))?;
        let factors = state.into_factors().map_err(PyRuntimeError::new_err)?;
        Ok(RsrsFactorsF64 { factors })
    }
}

impl RsrsSessionF64 {
    fn state(&self) -> PyResult<&PrecomputedRsrsSession<f64>> {
        self.state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("RSRS session was already consumed"))
    }

    fn state_mut(&mut self) -> PyResult<&mut PrecomputedRsrsSession<f64>> {
        self.state
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("RSRS session was already consumed"))
    }
}

#[pymethods]
impl RsrsFactorsF64 {
    #[getter]
    fn dimension(&self) -> usize {
        self.factors.dim()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        "float64"
    }

    #[pyo3(signature = (values, transpose=false, inverse=false))]
    fn apply_vector<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
        transpose: bool,
        inverse: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let input = values.as_slice()?;
        if input.len() != self.dimension() {
            return Err(PyRuntimeError::new_err(format!(
                "vector has length {}, expected {}",
                input.len(),
                self.dimension()
            )));
        }
        let options = BaseFactorOptions {
            inv: inverse,
            trans: TransMode::NoTrans,
            trans_target: false,
        };
        let side = if transpose { Side::Right } else { Side::Left };
        let mut output = vec![0.0; input.len()];
        self.factors.matvec(input, &mut output, side, &options);
        Ok(output.into_pyarray_bound(py))
    }

    #[pyo3(signature = (values, transpose=false, inverse=false))]
    fn apply_matrix<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray2<'py, f64>,
        transpose: bool,
        inverse: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let view = values.as_array();
        let (rows, columns) = view.dim();
        if rows != self.dimension() {
            return Err(PyRuntimeError::new_err(format!(
                "matrix has {rows} rows, expected {}",
                self.dimension()
            )));
        }
        let options = BaseFactorOptions {
            inv: inverse,
            trans: TransMode::NoTrans,
            trans_target: false,
        };
        let side = if transpose { Side::Right } else { Side::Left };
        let mut output = Array2::zeros((rows, columns));
        for column in 0..columns {
            let input = view.column(column).iter().copied().collect::<Vec<_>>();
            let mut result = vec![0.0; rows];
            self.factors.matvec(&input, &mut result, side, &options);
            output
                .column_mut(column)
                .iter_mut()
                .zip(result.iter())
                .for_each(|(target, source)| *target = *source);
        }
        Ok(output.into_pyarray_bound(py))
    }
}

#[pymethods]
impl RsrsSessionC64 {
    #[new]
    #[pyo3(signature = (points, options_json, max_tree_depth=16, leaf_size=None))]
    fn new(
        points: PyReadonlyArrayDyn<'_, f64>,
        options_json: &str,
        max_tree_depth: usize,
        leaf_size: Option<usize>,
    ) -> PyResult<Self> {
        ensure_mpi().map_err(PyRuntimeError::new_err)?;
        let coordinates = points.as_array();
        if coordinates.ndim() != 2 || coordinates.shape()[1] != 3 {
            return Err(PyRuntimeError::new_err(format!(
                "points must have shape (n, 3), got {:?}",
                coordinates.shape()
            )));
        }
        let point_count = coordinates.shape()[0];
        if point_count == 0 {
            return Err(PyRuntimeError::new_err("points must not be empty"));
        }
        let octree_points = coordinates
            .outer_iter()
            .enumerate()
            .map(|(index, row)| Point::new([row[0], row[1], row[2]], index))
            .collect::<Vec<_>>();
        let communicator = SimpleCommunicator::self_comm();
        let max_leaf_points = leaf_size.unwrap_or_else(|| {
            point_count
                .div_ceil(
                    8usize
                        .saturating_pow(max_tree_depth.saturating_sub(1) as u32)
                        .max(1),
                )
                .max(1)
        });
        let octree = Octree::new(
            &octree_points,
            max_tree_depth,
            max_leaf_points,
            &communicator,
        );
        let args: RsrsArgs<Complex64> = serde_json::from_str(options_json)
            .map_err(|error| PyRuntimeError::new_err(format!("invalid RSRS options: {error}")))?;
        let options = RsrsOptions::<Complex64>::new(Some(args));
        Ok(Self {
            state: Some(PrecomputedRsrsSession::new(Rsrs::new(
                &octree,
                options,
                point_count,
            ))),
        })
    }

    #[getter]
    fn dimension(&self) -> PyResult<usize> {
        Ok(self.state()?.rsrs().dimension())
    }

    #[getter]
    fn requires_z(&self) -> PyResult<bool> {
        Ok(self.state()?.rsrs().requires_z_samples())
    }

    #[pyo3(signature = (y_test, y_sketch, z_test=None, z_sketch=None))]
    fn add_samples(
        &mut self,
        y_test: PyReadonlyArray2<'_, Complex64>,
        y_sketch: PyReadonlyArray2<'_, Complex64>,
        z_test: Option<PyReadonlyArray2<'_, Complex64>>,
        z_sketch: Option<PyReadonlyArray2<'_, Complex64>>,
    ) -> PyResult<()> {
        let (y_rows, y_dim, y_test) = row_major_c64(y_test)?;
        let (sketch_rows, sketch_dim, y_sketch) = row_major_c64(y_sketch)?;
        if (y_rows, y_dim) != (sketch_rows, sketch_dim) {
            return Err(PyRuntimeError::new_err("Y test/sketch shapes differ"));
        }
        let y = PrecomputedSampleBlock {
            test: &y_test,
            sketch: &y_sketch,
            rows: y_rows,
            dim: y_dim,
        };
        let z_storage = match (z_test, z_sketch) {
            (Some(test), Some(sketch)) => {
                let (rows, dim, mut test) = row_major_c64(test)?;
                let (sketch_rows, sketch_dim, mut sketch) = row_major_c64(sketch)?;
                if (rows, dim) != (sketch_rows, sketch_dim) {
                    return Err(PyRuntimeError::new_err("Z test/sketch shapes differ"));
                }
                // Python supplies the natural algebraic-transpose samples
                // Omega_z and A^T Omega_z. SketchData(Trans) stores both rows
                // conjugated internally, matching Rust's fill_array route.
                test.iter_mut().for_each(|value| *value = value.conj());
                sketch.iter_mut().for_each(|value| *value = value.conj());
                Some((test, sketch, rows, dim))
            }
            (None, None) => None,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "Z test and sketch must be supplied together",
                ))
            }
        };
        let z = z_storage
            .as_ref()
            .map(|(test, sketch, rows, dim)| PrecomputedSampleBlock {
                test,
                sketch,
                rows: *rows,
                dim: *dim,
            });
        self.state_mut()?
            .rsrs_mut()
            .append_precomputed_samples(y, z)
            .map_err(PyRuntimeError::new_err)
    }

    fn advance<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new_bound(py);
        match self
            .state_mut()?
            .advance()
            .map_err(PyRuntimeError::new_err)?
        {
            PrecomputedAdvance::NeedsSamples(request) => {
                result.set_item("status", "needs_samples")?;
                result.set_item("y_rows", request.y_rows)?;
                result.set_item("z_rows", request.z_rows)?;
                result.set_item("total_required", request.total_required)?;
                result.set_item("level", request.level)?;
                result.set_item("limiting_box", request.limiting_box)?;
            }
            PrecomputedAdvance::Progressed { completed_level } => {
                result.set_item("status", "progressed")?;
                result.set_item("completed_level", completed_level)?;
            }
            PrecomputedAdvance::Complete => result.set_item("status", "complete")?,
        }
        Ok(result)
    }

    fn report_json(&self) -> PyResult<String> {
        let rsrs = self.state()?.rsrs();
        let stats = &rsrs.stats;
        serde_json::to_string(&serde_json::json!({
            "dimension": rsrs.dimension(),
            "active_samples": rsrs.active_samples,
            "y_sample_rows": rsrs.y_sample_rows(),
            "z_sample_rows": rsrs.z_sample_rows(),
            "ranks": stats.ranks,
            "box_sizes": stats.box_sizes,
            "near_field_sizes": stats.near_field_sizes,
            "decomposed_boxes_per_level": stats.dec_boxes_per_level,
            "id_times_ms": stats.id_times,
            "lu_times_ms": stats.lu_times,
            "update_times_ms": stats.update_times,
            "extraction_time_ms": stats.extraction_time,
            "total_factorization_time_ms": stats.total_elapsed_time_wo_sampling,
            "index_calculation_ms": stats.index_calculation,
            "sorting_near_field_ms": stats.sorting_near_field,
            "residual_calculation_ms": stats.residual_calculation,
            "level_effort": stats.level_effort,
            "max_sample_buffer_bytes": stats.max_sample_buffer_bytes,
            "max_factor_bytes": stats.max_factor_bytes,
            "max_accounted_factorization_bytes": stats.max_accounted_factorization_bytes,
        }))
        .map_err(|error| PyRuntimeError::new_err(format!("cannot serialize RSRS report: {error}")))
    }

    fn take_factors(&mut self) -> PyResult<RsrsFactorsC64> {
        let state = self
            .state
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("RSRS session was already consumed"))?;
        let factors = state.into_factors().map_err(PyRuntimeError::new_err)?;
        Ok(RsrsFactorsC64 { factors })
    }
}

impl RsrsSessionC64 {
    fn state(&self) -> PyResult<&PrecomputedRsrsSession<Complex64>> {
        self.state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("RSRS session was already consumed"))
    }

    fn state_mut(&mut self) -> PyResult<&mut PrecomputedRsrsSession<Complex64>> {
        self.state
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("RSRS session was already consumed"))
    }
}

#[pymethods]
impl RsrsFactorsC64 {
    #[getter]
    fn dimension(&self) -> usize {
        self.factors.dim()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        "complex128"
    }

    #[pyo3(signature = (values, transpose=false, inverse=false))]
    fn apply_vector<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, Complex64>,
        transpose: bool,
        inverse: bool,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let input = values.as_slice()?;
        if input.len() != self.dimension() {
            return Err(PyRuntimeError::new_err(format!(
                "vector has length {}, expected {}",
                input.len(),
                self.dimension()
            )));
        }
        let options = BaseFactorOptions {
            inv: inverse,
            trans: TransMode::NoTrans,
            trans_target: false,
        };
        let side = if transpose { Side::Right } else { Side::Left };
        let mut output = vec![Complex64::new(0.0, 0.0); input.len()];
        self.factors.matvec(input, &mut output, side, &options);
        Ok(output.into_pyarray_bound(py))
    }

    #[pyo3(signature = (values, transpose=false, inverse=false))]
    fn apply_matrix<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray2<'py, Complex64>,
        transpose: bool,
        inverse: bool,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let view = values.as_array();
        let (rows, columns) = view.dim();
        if rows != self.dimension() {
            return Err(PyRuntimeError::new_err(format!(
                "matrix has {rows} rows, expected {}",
                self.dimension()
            )));
        }
        let options = BaseFactorOptions {
            inv: inverse,
            trans: TransMode::NoTrans,
            trans_target: false,
        };
        let side = if transpose { Side::Right } else { Side::Left };
        let mut output = Array2::zeros((rows, columns));
        for column in 0..columns {
            let input = view.column(column).iter().copied().collect::<Vec<_>>();
            let mut result = vec![Complex64::new(0.0, 0.0); rows];
            self.factors.matvec(&input, &mut result, side, &options);
            output
                .column_mut(column)
                .iter_mut()
                .zip(result.iter())
                .for_each(|(target, source)| *target = *source);
        }
        Ok(output.into_pyarray_bound(py))
    }
}

#[pymodule]
fn _bempp_rsrs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RsrsSessionF64>()?;
    module.add_class::<RsrsFactorsF64>()?;
    module.add_class::<RsrsSessionC64>()?;
    module.add_class::<RsrsFactorsC64>()?;
    Ok(())
}
