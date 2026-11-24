use std::{mem, sync::OnceLock};

use bempp_octree::{Octree, Point};
use mpi::{environment::Threading, topology::SimpleCommunicator};
use num_complex::Complex64;
use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rlst::{dense::array::DynArray, RlstScalar};
use rsrs_core::dense_multilevel::{dense_levels_from_plan, DenseMultilevelFactorization};
use rsrs_core::{
    configuration::{
        IdAlgorithm, LeastSquaresAlgorithm, NullSpaceAlgorithm, RankSelection, RsrsConfiguration,
    },
    elementary_factor::FactorMode,
    factorized_operator::DenseFactorizedOperator,
    interpolative_decomposition::IdMethod,
    regularized_least_squares::{Regularization, RegularizedLeastSquaresMethod},
    sampled_id::{InMemorySampleProvider, SampleBlock, ZMode},
    sampled_multilevel::SampledMultilevelFactorization,
    sampled_options::{
        ExtractionOptions, NullificationMethod, PivotOptions, SampledFactorizationOptions,
    },
    scheduling::FactorizationPlan,
};
use rsrs_octree::{
    fixed_rank_plan_from_octree_with_budget_options, tolerance_plan_from_octree_with_options,
};
use rsrs_samples::{generate_complex, generate_real, GeneratorSpec, TestDistribution, TestFamily};
#[cfg(feature = "hdf5-store")]
use rsrs_samples::{
    hdf5_store::{Hdf5SketchCache, SketchBlock},
    ZMode as CacheZMode,
};

static MPI_INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();

fn ensure_mpi() -> Result<(), String> {
    MPI_INITIALIZED
        .get_or_init(|| {
            if let Some((universe, provided)) = mpi::initialize_with_threading(Threading::Multiple)
            {
                if provided < Threading::Serialized {
                    return Err(format!("MPI provides {provided:?}; Serialized is required"));
                }
                mem::forget(universe);
            }
            Ok(())
        })
        .clone()
}

fn plan(
    points: PyReadonlyArray2<'_, f64>,
    options: &RsrsConfiguration,
) -> PyResult<FactorizationPlan> {
    ensure_mpi().map_err(PyRuntimeError::new_err)?;
    let coordinates = points.as_array();
    if coordinates.shape().len() != 2 || coordinates.shape()[1] != 3 {
        return Err(PyRuntimeError::new_err("points must have shape (n, 3)"));
    }
    let octree_points = coordinates
        .outer_iter()
        .enumerate()
        .map(|(index, row)| Point::new([row[0], row[1], row[2]], index))
        .collect::<Vec<_>>();
    let communicator = SimpleCommunicator::self_comm();
    let tree = Octree::new(
        &octree_points,
        options.tree.maximum_depth,
        options.tree.effective_leaf_size(),
        &communicator,
    );
    Ok(match options.rank {
        RankSelection::Fixed { rank } => fixed_rank_plan_from_octree_with_budget_options(
            &tree,
            options.tree.minimum_level,
            rank,
            options.sampling.oversampling,
            options.sampling.minimum_rows,
            options.sampling.fixed_rank_mode,
            options.batching_strategy,
        ),
        selection => tolerance_plan_from_octree_with_options(
            &tree,
            options.tree.minimum_level,
            selection,
            options.parent_rank_policy,
            options.batching_strategy,
        ),
    })
}

fn id_method(options: &RsrsConfiguration) -> IdMethod<f64> {
    match options.id_algorithm {
        IdAlgorithm::Rrqr => IdMethod::Rrqr,
        IdAlgorithm::Srrqr => IdMethod::Srrqr {
            coefficient_bound: options.srrqr_coefficient_bound,
        },
    }
}

fn least_squares(algorithm: LeastSquaresAlgorithm) -> RegularizedLeastSquaresMethod {
    match algorithm {
        LeastSquaresAlgorithm::Lu => RegularizedLeastSquaresMethod::Lu,
        LeastSquaresAlgorithm::Cholesky => RegularizedLeastSquaresMethod::Cholesky,
        LeastSquaresAlgorithm::Svd => RegularizedLeastSquaresMethod::Svd,
    }
}

fn array<Item>(source: PyReadonlyArray2<'_, Item>) -> DynArray<Item, 2>
where
    Item: RlstScalar + numpy::Element,
{
    let view = source.as_array();
    let (rows, columns) = view.dim();
    let mut result = DynArray::<Item, 2>::from_shape([rows, columns]);
    for row in 0..rows {
        for column in 0..columns {
            result[[row, column]] = view[[row, column]];
        }
    }
    result
}

#[cfg(feature = "hdf5-store")]
fn row_major_array<Item: RlstScalar>(
    values: Vec<Item>,
    rows: usize,
    columns: usize,
) -> DynArray<Item, 2> {
    assert_eq!(values.len(), rows * columns);
    let mut result = DynArray::<Item, 2>::from_shape([rows, columns]);
    for row in 0..rows {
        for column in 0..columns {
            result[[row, column]] = values[row * columns + column];
        }
    }
    result
}

#[cfg(feature = "hdf5-store")]
trait Hdf5Scalar: RlstScalar {
    fn read_hdf5(cache: &Hdf5SketchCache, rows: usize) -> Result<SampleBlock<Self>, String>;
}

#[cfg(feature = "hdf5-store")]
impl Hdf5Scalar for f64 {
    fn read_hdf5(cache: &Hdf5SketchCache, rows: usize) -> Result<SampleBlock<Self>, String> {
        match cache
            .read_block(0, rows)
            .map_err(|error| error.to_string())?
        {
            SketchBlock::Real {
                y_test,
                y_sketch,
                z_test,
                z_sketch,
                rows,
                columns,
            } => Ok(SampleBlock {
                y_test: row_major_array(y_test, rows, columns),
                y_sketch: row_major_array(y_sketch, rows, columns),
                z_test: z_test.map(|values| row_major_array(values, rows, columns)),
                z_sketch: z_sketch.map(|values| row_major_array(values, rows, columns)),
            }),
            SketchBlock::Complex { .. } => Err("complex cache opened by a real session".into()),
        }
    }
}

#[cfg(feature = "hdf5-store")]
impl Hdf5Scalar for Complex64 {
    fn read_hdf5(cache: &Hdf5SketchCache, rows: usize) -> Result<SampleBlock<Self>, String> {
        match cache
            .read_block(0, rows)
            .map_err(|error| error.to_string())?
        {
            SketchBlock::Complex {
                y_test,
                y_sketch,
                z_test,
                z_sketch,
                rows,
                columns,
            } => Ok(SampleBlock {
                y_test: row_major_array(y_test, rows, columns),
                y_sketch: row_major_array(y_sketch, rows, columns),
                z_test: z_test.map(|values| row_major_array(values, rows, columns)),
                z_sketch: z_sketch.map(|values| row_major_array(values, rows, columns)),
            }),
            SketchBlock::Real { .. } => Err("real cache opened by a complex session".into()),
        }
    }
}

fn numpy_array<'py, Item: RlstScalar + numpy::Element>(
    py: Python<'py>,
    source: &DynArray<Item, 2>,
) -> Bound<'py, PyArray2<Item>> {
    let values = (0..source.shape()[0])
        .flat_map(|row| (0..source.shape()[1]).map(move |column| source[[row, column]]))
        .collect::<Vec<_>>();
    Array2::from_shape_vec((source.shape()[0], source.shape()[1]), values)
        .expect("shape is valid")
        .into_pyarray_bound(py)
}

trait CanonicalTestScalar: RlstScalar {
    fn generate_tests(
        seed: u64,
        family: TestFamily,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<Self>, String>;
}

impl CanonicalTestScalar for f64 {
    fn generate_tests(
        seed: u64,
        family: TestFamily,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<Self>, String> {
        generate_real(
            &GeneratorSpec::new(seed, family, TestDistribution::RealStandardNormal),
            0,
            rows,
            columns,
        )
        .map_err(|error| error.to_string())
    }
}

impl CanonicalTestScalar for Complex64 {
    fn generate_tests(
        seed: u64,
        family: TestFamily,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<Self>, String> {
        generate_complex(
            &GeneratorSpec::new(
                seed,
                family,
                TestDistribution::CircularComplexStandardNormal,
            ),
            0,
            rows,
            columns,
        )
        .map_err(|error| error.to_string())
    }
}

fn canonical_test_array<Item: CanonicalTestScalar>(
    seed: u64,
    family: TestFamily,
    rows: usize,
    columns: usize,
) -> Result<DynArray<Item, 2>, String> {
    let values = Item::generate_tests(seed, family, rows, columns)?;
    let mut result = DynArray::<Item, 2>::from_shape([rows, columns]);
    for row in 0..rows {
        for column in 0..columns {
            result[[row, column]] = values[row * columns + column];
        }
    }
    Ok(result)
}

macro_rules! binding {
    ($session:ident, $factors:ident, $backend:ident, $item:ty, $real:ty) => {
        enum $backend {
            Sampled(SampledMultilevelFactorization<$item>),
            Dense(DenseFactorizedOperator<$item>),
        }

        impl $backend {
            fn apply(
                &self,
                values: &DynArray<$item, 2>,
                mode: FactorMode,
                inverse: bool,
            ) -> Result<DynArray<$item, 2>, String> {
                match self {
                    Self::Sampled(factors) => factors
                        .apply(values, mode, inverse)
                        .map_err(|error| error.to_string()),
                    Self::Dense(factors) => factors
                        .apply(values, mode, inverse)
                        .map_err(|error| error.to_string()),
                }
            }
        }

        #[pyclass]
        struct $session {
            plan: FactorizationPlan,
            options: RsrsConfiguration,
        }

        #[pyclass]
        struct $factors {
            factors: $backend,
            report_json: String,
        }

        impl $session {
            fn factorize_block(
                &self,
                block: SampleBlock<$item>,
                z_mode: ZMode,
            ) -> PyResult<$factors> {
                let rows = block.rows();
                let provider = InMemorySampleProvider::new(block, z_mode)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                let regularization =
                    |configuration: &rsrs_core::configuration::RegularizationConfiguration| {
                        Regularization {
                            mode: configuration.mode,
                            relative: configuration.relative as $real,
                            absolute_floor: configuration.absolute_floor as $real,
                        }
                    };
                let nullification = match self.options.null_space.algorithm {
                    NullSpaceAlgorithm::Projection => NullificationMethod::Projection {
                        solver: least_squares(self.options.null_space.projection_solver),
                        regularization: regularization(&self.options.null_space.regularization),
                    },
                    NullSpaceAlgorithm::Qr => NullificationMethod::Qr {
                        tolerance: self.options.null_space.tolerance as $real,
                    },
                    NullSpaceAlgorithm::Svd => NullificationMethod::Svd {
                        tolerance: self.options.null_space.tolerance as $real,
                    },
                };
                let numerics = SampledFactorizationOptions {
                    nullification,
                    near_extraction: ExtractionOptions {
                        solver: least_squares(self.options.near_extraction.solver),
                        regularization: regularization(
                            &self.options.near_extraction.regularization,
                        ),
                    },
                    diagonal_extraction: ExtractionOptions {
                        solver: least_squares(self.options.diagonal_extraction.solver),
                        regularization: regularization(
                            &self.options.diagonal_extraction.regularization,
                        ),
                    },
                    near_pivot: PivotOptions {
                        method: self.options.near_pivot.method,
                        shift: self.options.near_pivot.shift as $real,
                    },
                    diagonal_pivot: PivotOptions {
                        method: self.options.diagonal_pivot.method,
                        shift: self.options.diagonal_pivot.shift as $real,
                    },
                };
                let (factors, report) =
                    SampledMultilevelFactorization::factorize_planned_with_report_and_options(
                        &provider,
                        rows,
                        &self.plan,
                        id_method(&self.options),
                        numerics,
                        self.options.threads.rayon_threads,
                        self.options.symmetry,
                    )
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                let report_json = serde_json::to_string_pretty(&report)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                Ok($factors {
                    factors: $backend::Sampled(factors),
                    report_json,
                })
            }

            #[cfg(feature = "hdf5-store")]
            fn selected_sample_rows(&self, available: usize) -> PyResult<usize> {
                let selected = self
                    .options
                    .sampling
                    .rows
                    .unwrap_or(self.plan.required_sample_rows);
                if selected < self.plan.required_sample_rows {
                    return Err(PyRuntimeError::new_err(format!(
                        "configured sample prefix {selected} is smaller than the required {} rows",
                        self.plan.required_sample_rows
                    )));
                }
                if selected > available {
                    return Err(PyRuntimeError::new_err(format!(
                        "RSRS requires {selected} sample rows but the cache has {available} committed rows"
                    )));
                }
                Ok(selected)
            }
        }

        #[pymethods]
        impl $session {
            #[new]
            fn new(points: PyReadonlyArray2<'_, f64>, options_json: &str) -> PyResult<Self> {
                let options: RsrsConfiguration = serde_json::from_str(options_json)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                let plan = plan(points, &options)?;
                Ok(Self { plan, options })
            }

            #[getter]
            fn dimension(&self) -> usize {
                self.plan.dimension
            }

            #[getter]
            fn required_sample_rows(&self) -> usize {
                self.plan.required_sample_rows
            }

            #[pyo3(signature = (y_test, y_sketch, z_test=None, z_sketch=None))]
            fn factorize(
                &self,
                py: Python<'_>,
                y_test: PyReadonlyArray2<'_, $item>,
                y_sketch: PyReadonlyArray2<'_, $item>,
                z_test: Option<PyReadonlyArray2<'_, $item>>,
                z_sketch: Option<PyReadonlyArray2<'_, $item>>,
            ) -> PyResult<$factors> {
                let block = SampleBlock {
                    y_test: array(y_test),
                    y_sketch: array(y_sketch),
                    z_test: z_test.map(array),
                    z_sketch: z_sketch.map(array),
                };
                py.allow_threads(move || self.factorize_block(block, ZMode::Transpose))
            }

            /// Factorize from stored sketches and deterministic generator
            /// recipes. Dense test matrices are regenerated in Rust and never
            /// need to be written to the cache.
            #[pyo3(signature = (y_sketch, y_seed, z_sketch=None, z_seed=None, z_adjoint=false))]
            fn factorize_sketches(
                &self,
                py: Python<'_>,
                y_sketch: PyReadonlyArray2<'_, $item>,
                y_seed: u64,
                z_sketch: Option<PyReadonlyArray2<'_, $item>>,
                z_seed: Option<u64>,
                z_adjoint: bool,
            ) -> PyResult<$factors> {
                if z_sketch.is_some() != z_seed.is_some() {
                    return Err(PyRuntimeError::new_err(
                        "z_sketch and z_seed must either both be present or both be absent",
                    ));
                }
                let y_sketch = array(y_sketch);
                let rows = y_sketch.shape()[0];
                let columns = y_sketch.shape()[1];
                let y_test = canonical_test_array::<$item>(y_seed, TestFamily::Y, rows, columns)
                    .map_err(PyRuntimeError::new_err)?;
                let (z_test, z_sketch) = match (z_seed, z_sketch) {
                    (Some(seed), Some(sketch)) => {
                        let sketch = array(sketch);
                        if sketch.shape() != [rows, columns] {
                            return Err(PyRuntimeError::new_err(
                                "Z sketch shape must match the Y sketch shape",
                            ));
                        }
                        let test =
                            canonical_test_array::<$item>(seed, TestFamily::Z, rows, columns)
                                .map_err(PyRuntimeError::new_err)?;
                        (Some(test), Some(sketch))
                    }
                    (None, None) => (None, None),
                    _ => unreachable!(),
                };
                let block = SampleBlock {
                    y_test,
                    y_sketch,
                    z_test,
                    z_sketch,
                };
                let z_mode = if z_adjoint {
                    ZMode::Adjoint
                } else {
                    ZMode::Transpose
                };
                py.allow_threads(move || self.factorize_block(block, z_mode))
            }

            /// Dense strong recursive skeletonization using the same octree
            /// plan and factor object as sampled RSRS.
            fn factorize_dense(
                &self,
                py: Python<'_>,
                matrix: PyReadonlyArray2<'_, $item>,
            ) -> PyResult<$factors> {
                let matrix = array(matrix);
                if matrix.shape() != [self.plan.dimension, self.plan.dimension] {
                    return Err(PyRuntimeError::new_err(format!(
                        "dense matrix shape {:?} does not match [{}, {}]",
                        matrix.shape(),
                        self.plan.dimension,
                        self.plan.dimension
                    )));
                }
                py.allow_threads(move || {
                    let started = std::time::Instant::now();
                    let levels = dense_levels_from_plan(&self.plan)
                        .map_err(PyRuntimeError::new_err)?;
                    let (dense, transformed) = DenseMultilevelFactorization::factorize(
                        &matrix,
                        &levels,
                        id_method(&self.options),
                    )
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                    let cycles = dense.cycles().len();
                    let active_dofs = dense.active_indices().len();
                    let factors = DenseFactorizedOperator::new(dense, &transformed)
                        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                    let report_json = serde_json::to_string_pretty(&serde_json::json!({
                        "backend": "dense_srs",
                        "dimension": self.plan.dimension,
                        "cycles": cycles,
                        "final_active_dofs": active_dofs,
                        "factorization_seconds": started.elapsed().as_secs_f64(),
                    }))
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                    Ok($factors {
                        factors: $backend::Dense(factors),
                        report_json,
                    })
                })
            }

            /// Open a normalized sketch-only HDF5 cache directly in Rust.
            /// Only the configured/required prefix is loaded; no NumPy copy of
            /// the cache is created in the Python process.
            #[cfg(feature = "hdf5-store")]
            #[pyo3(signature = (path, quick_validate=true))]
            fn factorize_hdf5(
                &self,
                py: Python<'_>,
                path: &str,
                quick_validate: bool,
            ) -> PyResult<$factors> {
                let cache = Hdf5SketchCache::open(path)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                if cache.metadata().operator_dimension != self.plan.dimension {
                    return Err(PyRuntimeError::new_err(format!(
                        "cache dimension {} differs from session dimension {}",
                        cache.metadata().operator_dimension,
                        self.plan.dimension
                    )));
                }
                if quick_validate {
                    cache
                        .validate_quick()
                        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                }
                let rows = self.selected_sample_rows(cache.committed_rows())?;
                let z_mode = match cache.metadata().z_mode.unwrap_or(CacheZMode::Transpose) {
                    CacheZMode::Transpose => ZMode::Transpose,
                    CacheZMode::Adjoint => ZMode::Adjoint,
                };
                py.allow_threads(move || {
                    let block = <$item as Hdf5Scalar>::read_hdf5(&cache, rows)
                        .map_err(PyRuntimeError::new_err)?;
                    self.factorize_block(block, z_mode)
                })
            }
        }

        #[pymethods]
        impl $factors {
            #[getter]
            fn report_json(&self) -> &str {
                &self.report_json
            }

            #[pyo3(signature = (values, transpose=false, adjoint=false, inverse=false))]
            fn matmat<'py>(
                &self,
                py: Python<'py>,
                values: PyReadonlyArray2<'_, $item>,
                transpose: bool,
                adjoint: bool,
                inverse: bool,
            ) -> PyResult<Bound<'py, PyArray2<$item>>> {
                if transpose && adjoint {
                    return Err(PyRuntimeError::new_err(
                        "transpose and adjoint are mutually exclusive",
                    ));
                }
                let mode = if adjoint {
                    FactorMode::Adjoint
                } else if transpose {
                    FactorMode::Transpose
                } else {
                    FactorMode::NoTrans
                };
                let values = array(values);
                let result = py
                    .allow_threads(|| self.factors.apply(&values, mode, inverse))
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                Ok(numpy_array(py, &result))
            }
        }
    };
}

binding!(RsrsSessionF64, RsrsFactorsF64, RsrsBackendF64, f64, f64);
binding!(
    RsrsSessionC64,
    RsrsFactorsC64,
    RsrsBackendC64,
    Complex64,
    f64
);

#[pymodule]
fn _bempp_rsrs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RsrsSessionF64>()?;
    module.add_class::<RsrsSessionC64>()?;
    module.add_class::<RsrsFactorsF64>()?;
    module.add_class::<RsrsFactorsC64>()?;
    Ok(())
}
