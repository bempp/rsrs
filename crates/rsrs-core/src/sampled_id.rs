//! Provider-neutral sampled interpolative decomposition for RSRS.
//!
//! Samples are stored row-wise: each row is one test vector and its matching
//! sketch.  The implementation deliberately knows nothing about HDF5; cache
//! readers and live samplers implement [`SampleProvider`].

use crate::{
    configuration::MatrixSymmetry,
    interpolative_decomposition::{compute_candidate_id, IdMethod, IdRankSelection},
    regularized_least_squares::{
        solve_regularized_least_squares, Regularization, RegularizedLeastSquaresMethod,
        RegularizedLeastSquaresReport,
    },
    sampled_options::NullificationMethod,
};
use rlst::{
    dense::{
        array::DynArray,
        linalg::lapack::{
            interpolative_decomposition::{IdDecomposition, MatrixId},
            null_space::NullSpaceMethod,
        },
    },
    dot_blas, EvaluateArray, Gemm, RlstScalar,
};
use std::{error::Error, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZMode {
    Transpose,
    Adjoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdCombination {
    YOnly,
    ZOnly,
    Concat,
}

/// One owned transfer block. Arrays are row-major in the conceptual sense:
/// `(sample, operator_coordinate)`; `DynArray` owns the actual storage.
pub struct SampleBlock<Item: RlstScalar> {
    pub y_test: DynArray<Item, 2>,
    pub y_sketch: DynArray<Item, 2>,
    pub z_test: Option<DynArray<Item, 2>>,
    pub z_sketch: Option<DynArray<Item, 2>>,
}

impl<Item: RlstScalar> SampleBlock<Item> {
    pub fn rows(&self) -> usize {
        self.y_test.shape()[0]
    }
    pub fn dimension(&self) -> usize {
        self.y_test.shape()[1]
    }

    pub fn validate(&self) -> Result<(), SampleError> {
        let shape = self.y_test.shape();
        if self.y_sketch.shape() != shape {
            return Err(SampleError::InvalidShape(format!(
                "Y test and sketch arrays must have shape {shape:?}"
            )));
        }
        match (&self.z_test, &self.z_sketch) {
            (Some(test), Some(sketch)) if test.shape() == shape && sketch.shape() == shape => {}
            (None, None) => {}
            (Some(test), Some(sketch)) => {
                return Err(SampleError::InvalidShape(format!(
                    "Z test and sketch arrays must have shape {shape:?}, got {:?} and {:?}",
                    test.shape(),
                    sketch.shape()
                )))
            }
            _ => {
                return Err(SampleError::InvalidShape(
                    "Z test and sketch must either both be present or both be absent".into(),
                ))
            }
        }
        Ok(())
    }

    pub fn has_z(&self) -> bool {
        self.z_test.is_some()
    }
}

/// Streaming contract shared by in-memory, live Python, and HDF5 adapters.
pub trait SampleProvider<Item: RlstScalar> {
    type Error: Error + Send + Sync + 'static;

    fn available_rows(&self) -> usize;
    fn dimension(&self) -> usize;
    fn z_mode(&self) -> ZMode;
    fn read_block(&self, first_row: usize, rows: usize) -> Result<SampleBlock<Item>, Self::Error>;

    /// Borrow an already resident sample block when available. Algorithms use
    /// this only when its row count exactly matches the requested prefix. Cache
    /// providers may keep the default and continue to stream owned blocks.
    fn resident_block(&self) -> Option<&SampleBlock<Item>> {
        None
    }
}

/// Simple provider useful for live samplers and tests. Production cache
/// adapters can stream the same block shape without retaining all rows.
pub struct InMemorySampleProvider<Item: RlstScalar> {
    block: SampleBlock<Item>,
    z_mode: ZMode,
}

impl<Item: RlstScalar> InMemorySampleProvider<Item> {
    pub fn new(block: SampleBlock<Item>, z_mode: ZMode) -> Result<Self, SampleError> {
        block.validate()?;
        Ok(Self { block, z_mode })
    }
}

impl<Item: RlstScalar> SampleProvider<Item> for InMemorySampleProvider<Item> {
    type Error = SampleError;

    fn available_rows(&self) -> usize {
        self.block.rows()
    }
    fn dimension(&self) -> usize {
        self.block.dimension()
    }
    fn z_mode(&self) -> ZMode {
        self.z_mode
    }

    fn read_block(&self, first_row: usize, rows: usize) -> Result<SampleBlock<Item>, Self::Error> {
        if first_row.saturating_add(rows) > self.available_rows() {
            return Err(SampleError::InsufficientRows {
                requested: first_row.saturating_add(rows),
                available: self.available_rows(),
            });
        }
        let columns = self.dimension();
        let copy = |source: &DynArray<Item, 2>| {
            let mut result = DynArray::<Item, 2>::from_shape([rows, columns]);
            for row in 0..rows {
                for column in 0..columns {
                    result[[row, column]] = source[[first_row + row, column]];
                }
            }
            result
        };
        Ok(SampleBlock {
            y_test: copy(&self.block.y_test),
            y_sketch: copy(&self.block.y_sketch),
            z_test: self.block.z_test.as_ref().map(copy),
            z_sketch: self.block.z_sketch.as_ref().map(copy),
        })
    }

    fn resident_block(&self) -> Option<&SampleBlock<Item>> {
        Some(&self.block)
    }
}

#[derive(Debug)]
pub enum SampleError {
    InvalidShape(String),
    MissingZ,
    InsufficientRows { requested: usize, available: usize },
    Provider(String),
}

impl fmt::Display for SampleError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(message) | Self::Provider(message) => out.write_str(message),
            Self::MissingZ => {
                out.write_str("general nonsymmetric RSRS requires Z test and sketch samples")
            }
            Self::InsufficientRows {
                requested,
                available,
            } => write!(
                out,
                "sampled ID needs {requested} rows but the provider has {available}"
            ),
        }
    }
}

impl Error for SampleError {}

pub struct SampledIdResult<Item: MatrixId + Gemm> {
    pub id: IdDecomposition<Item>,
    pub nullified_panel: DynArray<Item, 2>,
    pub y_projection: Option<RegularizedLeastSquaresReport<Item::Real>>,
    pub z_projection: Option<RegularizedLeastSquaresReport<Item::Real>>,
}

/// Near interactions recovered from matching tests and sketches.
///
/// `target_by_near[(i,j)] = A[target_i, near_j]` and
/// `near_by_target[(j,i)] = A[near_j, target_i]`.
pub struct SampledNearBlocks<Item: MatrixId + Gemm> {
    pub target_by_near: DynArray<Item, 2>,
    pub near_by_target: DynArray<Item, 2>,
    pub y_extraction: RegularizedLeastSquaresReport<Item::Real>,
    pub z_extraction: RegularizedLeastSquaresReport<Item::Real>,
}

/// Recover both orientations of a nonsymmetric near block entirely from a
/// provider. This is the refactored equivalent of historical `LuLstSq` block
/// extraction and requires no entry access to the original operator.
pub fn extract_sampled_near_blocks<Item, Provider>(
    provider: &Provider,
    sample_rows: usize,
    target_indices: &[usize],
    near_indices: &[usize],
    regularization: Regularization<Item::Real>,
    least_squares: RegularizedLeastSquaresMethod,
) -> Result<SampledNearBlocks<Item>, SampleError>
where
    Item: MatrixId + Gemm,
    Item::Real: Into<Item>,
    Provider: SampleProvider<Item>,
{
    extract_sampled_near_blocks_with_symmetry(
        provider,
        sample_rows,
        target_indices,
        near_indices,
        regularization,
        least_squares,
        MatrixSymmetry::General,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn extract_sampled_near_blocks_with_symmetry<Item, Provider>(
    provider: &Provider,
    sample_rows: usize,
    target_indices: &[usize],
    near_indices: &[usize],
    regularization: Regularization<Item::Real>,
    least_squares: RegularizedLeastSquaresMethod,
    symmetry: MatrixSymmetry,
) -> Result<SampledNearBlocks<Item>, SampleError>
where
    Item: MatrixId + Gemm,
    Item::Real: Into<Item>,
    Provider: SampleProvider<Item>,
{
    if provider.available_rows() < sample_rows {
        return Err(SampleError::InsufficientRows {
            requested: sample_rows,
            available: provider.available_rows(),
        });
    }
    let resident = provider
        .resident_block()
        .filter(|block| block.rows() == sample_rows);
    let owned = if resident.is_none() {
        Some(
            provider
                .read_block(0, sample_rows)
                .map_err(|error| SampleError::Provider(error.to_string()))?,
        )
    } else {
        None
    };
    let block = resident.unwrap_or_else(|| owned.as_ref().expect("owned sample block is present"));
    block.validate()?;
    let y_test_near = select_columns(&block.y_test, near_indices, false);
    let y_sketch_target = select_columns(&block.y_sketch, target_indices, false);
    let (y_near_by_target, y_extraction) = solve_regularized_least_squares(
        &y_test_near,
        &y_sketch_target,
        regularization,
        least_squares,
    )
    .map_err(|error| SampleError::Provider(error.to_string()))?;
    let mut target_by_near =
        DynArray::<Item, 2>::from_shape([target_indices.len(), near_indices.len()]);
    for near in 0..near_indices.len() {
        for target in 0..target_indices.len() {
            target_by_near[[target, near]] = y_near_by_target[[near, target]];
        }
    }
    let (near_by_target, z_extraction) = match symmetry {
        MatrixSymmetry::General => {
            let conjugate_z = provider.z_mode() == ZMode::Adjoint;
            let z_test = block.z_test.as_ref().ok_or(SampleError::MissingZ)?;
            let z_sketch = block.z_sketch.as_ref().ok_or(SampleError::MissingZ)?;
            let z_test_near = select_columns(z_test, near_indices, conjugate_z);
            let z_sketch_target = select_columns(z_sketch, target_indices, conjugate_z);
            solve_regularized_least_squares(
                &z_test_near,
                &z_sketch_target,
                regularization,
                least_squares,
            )
            .map_err(|error| SampleError::Provider(error.to_string()))?
        }
        MatrixSymmetry::Symmetric | MatrixSymmetry::Hermitian => {
            let conjugate = symmetry == MatrixSymmetry::Hermitian;
            let mut opposite =
                DynArray::<Item, 2>::from_shape([near_indices.len(), target_indices.len()]);
            for target in 0..target_indices.len() {
                for near in 0..near_indices.len() {
                    let value = target_by_near[[target, near]];
                    opposite[[near, target]] = if conjugate { value.conj() } else { value };
                }
            }
            (opposite, y_extraction)
        }
    };
    Ok(SampledNearBlocks {
        target_by_near,
        near_by_target,
        y_extraction,
        z_extraction,
    })
}

/// Read one prefix from a provider, remove the sampled near-field span, and
/// compute the shared row/column ID used by nonsymmetric RSRS.
#[allow(clippy::too_many_arguments)]
pub fn compute_sampled_id<Item, Provider>(
    provider: &Provider,
    sample_rows: usize,
    target_indices: &[usize],
    near_indices: &[usize],
    rank: IdRankSelection<Item::Real>,
    method: IdMethod<Item::Real>,
    regularization: Regularization<Item::Real>,
    least_squares: RegularizedLeastSquaresMethod,
) -> Result<SampledIdResult<Item>, SampleError>
where
    Item: MatrixId + Gemm,
    Item::Real: Into<Item>,
    Provider: SampleProvider<Item>,
{
    compute_sampled_id_with_combination(
        provider,
        sample_rows,
        target_indices,
        near_indices,
        rank,
        method,
        regularization,
        least_squares,
        IdCombination::Concat,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn compute_sampled_id_with_combination<Item, Provider>(
    provider: &Provider,
    sample_rows: usize,
    target_indices: &[usize],
    near_indices: &[usize],
    rank: IdRankSelection<Item::Real>,
    method: IdMethod<Item::Real>,
    regularization: Regularization<Item::Real>,
    least_squares: RegularizedLeastSquaresMethod,
    combination: IdCombination,
) -> Result<SampledIdResult<Item>, SampleError>
where
    Item: MatrixId + Gemm,
    Item::Real: Into<Item>,
    Provider: SampleProvider<Item>,
{
    compute_sampled_id_with_combination_and_nullification(
        provider,
        sample_rows,
        target_indices,
        near_indices,
        rank,
        method,
        NullificationMethod::Projection {
            solver: least_squares,
            regularization,
        },
        combination,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn compute_sampled_id_with_combination_and_nullification<Item, Provider>(
    provider: &Provider,
    sample_rows: usize,
    target_indices: &[usize],
    near_indices: &[usize],
    rank: IdRankSelection<Item::Real>,
    method: IdMethod<Item::Real>,
    nullification: NullificationMethod<Item::Real>,
    combination: IdCombination,
) -> Result<SampledIdResult<Item>, SampleError>
where
    Item: MatrixId + Gemm,
    Item::Real: Into<Item>,
    Provider: SampleProvider<Item>,
{
    if provider.available_rows() < sample_rows {
        return Err(SampleError::InsufficientRows {
            requested: sample_rows,
            available: provider.available_rows(),
        });
    }
    let resident = provider
        .resident_block()
        .filter(|block| block.rows() == sample_rows);
    let owned = if resident.is_none() {
        Some(
            provider
                .read_block(0, sample_rows)
                .map_err(|error| SampleError::Provider(error.to_string()))?,
        )
    } else {
        None
    };
    let block = resident.unwrap_or_else(|| owned.as_ref().expect("owned sample block is present"));
    block.validate()?;
    if block.dimension() != provider.dimension() {
        return Err(SampleError::InvalidShape(format!(
            "sample block dimension {} differs from provider dimension {}",
            block.dimension(),
            provider.dimension()
        )));
    }
    let (y_panel, y_report) = nullified_panel(
        &block.y_test,
        &block.y_sketch,
        target_indices,
        near_indices,
        false,
        nullification,
    )?;
    // An adjoint-generated Z family is converted to the same plain-transpose
    // algebra as historical nonsymmetric RSRS by conjugating tests and
    // sketches together.
    let z = if combination == IdCombination::YOnly {
        None
    } else {
        let z_test = block.z_test.as_ref().ok_or(SampleError::MissingZ)?;
        let z_sketch = block.z_sketch.as_ref().ok_or(SampleError::MissingZ)?;
        let conjugate_z = provider.z_mode() == ZMode::Adjoint;
        Some(nullified_panel(
            z_test,
            z_sketch,
            target_indices,
            near_indices,
            conjugate_z,
            nullification,
        )?)
    };
    let (y_rows, z_rows) = match (combination, z.as_ref()) {
        (IdCombination::YOnly, _) => (y_panel.shape()[0], 0),
        (IdCombination::ZOnly, Some((panel, _))) => (0, panel.shape()[0]),
        (IdCombination::Concat, Some((panel, _))) => (y_panel.shape()[0], panel.shape()[0]),
        _ => unreachable!("Z-only and concatenated IDs require Z samples"),
    };
    let mut combined = DynArray::<Item, 2>::from_shape([y_rows + z_rows, target_indices.len()]);
    if y_rows > 0 {
        for row in 0..y_rows {
            for column in 0..target_indices.len() {
                combined[[row, column]] = y_panel[[row, column]];
            }
        }
    }
    if z_rows > 0 {
        let z_panel = &z.as_ref().expect("Z panel is present").0;
        for row in 0..z_rows {
            for column in 0..target_indices.len() {
                combined[[y_rows + row, column]] = z_panel[[row, column]];
            }
        }
    }
    if std::env::var("RSRS_TRACE_CANDIDATES").as_deref() == Ok("1") {
        let mut target_hash = 0xcbf2_9ce4_8422_2325u64;
        for &index in target_indices {
            target_hash = (target_hash ^ index as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        let [rows, columns] = combined.shape();
        let mut sum_re = 0.0f64;
        let mut sum_im = 0.0f64;
        let mut norm_squared = 0.0f64;
        let mut weighted_re = 0.0f64;
        for row in 0..rows {
            for column in 0..columns {
                let value = combined[[row, column]];
                let re = num::ToPrimitive::to_f64(&value.re()).unwrap();
                let im = num::ToPrimitive::to_f64(&value.im()).unwrap();
                let weight = (1 + row * columns + column) as f64;
                sum_re += re;
                sum_im += im;
                norm_squared += re * re + im * im;
                weighted_re += weight * re;
            }
        }
        println!(
            "trace_candidate target_hash={target_hash:016x} rows={rows} columns={columns} sum_re={sum_re:.16e} sum_im={sum_im:.16e} norm_squared={norm_squared:.16e} weighted_re={weighted_re:.16e}"
        );
    }
    let id = compute_candidate_id(&combined, rank, method, false)
        .map_err(|error| SampleError::Provider(error.to_string()))?;
    Ok(SampledIdResult {
        id,
        nullified_panel: combined,
        y_projection: y_report,
        z_projection: z.and_then(|(_, report)| report),
    })
}

type NullifiedPanelResult<Item> = Result<
    (
        DynArray<Item, 2>,
        Option<RegularizedLeastSquaresReport<<Item as RlstScalar>::Real>>,
    ),
    SampleError,
>;

fn nullified_panel<Item>(
    test: &DynArray<Item, 2>,
    sketch: &DynArray<Item, 2>,
    target_indices: &[usize],
    near_indices: &[usize],
    conjugate: bool,
    nullification: NullificationMethod<Item::Real>,
) -> NullifiedPanelResult<Item>
where
    Item: MatrixId + Gemm,
    Item::Real: Into<Item>,
{
    let near_test = select_columns(test, near_indices, conjugate);
    let mut target_sketch = select_columns(sketch, target_indices, conjugate);
    if near_indices.is_empty() {
        // Produce a report with a harmless zero-column solve only when needed;
        // callers normally have a nonempty near field.
        return Ok((target_sketch, None));
    }
    match nullification {
        NullificationMethod::Projection {
            solver,
            regularization,
        } => {
            let (coefficients, report) =
                solve_regularized_least_squares(&near_test, &target_sketch, regularization, solver)
                    .map_err(|error| SampleError::Provider(error.to_string()))?;
            let projected = dot_blas!(near_test.r(), coefficients.r());
            for row in 0..target_sketch.shape()[0] {
                for column in 0..target_sketch.shape()[1] {
                    target_sketch[[row, column]] -= projected[[row, column]];
                }
            }
            Ok((target_sketch, Some(report)))
        }
        NullificationMethod::Qr { tolerance } => {
            let null = near_test
                .null_space(tolerance, NullSpaceMethod::LeftQr)
                .map_err(|error| SampleError::Provider(error.to_string()))?
                .null_space_arr;
            let null_adjoint = null.r().conj().transpose().eval();
            Ok((dot_blas!(null_adjoint.r(), target_sketch.r()), None))
        }
        NullificationMethod::Svd { tolerance } => {
            // The left null space of A is the right null space of A^H.
            let near_adjoint = near_test.r().conj().transpose().eval();
            let null = near_adjoint
                .null_space(tolerance, NullSpaceMethod::RightSvd)
                .map_err(|error| SampleError::Provider(error.to_string()))?
                .null_space_arr;
            let null_adjoint = null.r().conj().transpose().eval();
            Ok((dot_blas!(null_adjoint.r(), target_sketch.r()), None))
        }
    }
}

fn select_columns<Item: RlstScalar>(
    source: &DynArray<Item, 2>,
    indices: &[usize],
    conjugate: bool,
) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape([source.shape()[0], indices.len()]);
    for row in 0..source.shape()[0] {
        for (local, &global) in indices.iter().enumerate() {
            let value = source[[row, global]];
            result[[row, local]] = if conjugate { value.conj() } else { value };
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlst::base_types::c64;

    fn identity_y_provider(
        matrix: &DynArray<c64, 2>,
        corrupt_z_scale: Option<f64>,
    ) -> InMemorySampleProvider<c64> {
        let dimension = matrix.shape()[0];
        assert_eq!(matrix.shape()[1], dimension);
        let mut y_test = DynArray::<c64, 2>::from_shape([dimension, dimension]);
        let mut y_sketch = DynArray::<c64, 2>::from_shape([dimension, dimension]);
        let mut z_test = DynArray::<c64, 2>::from_shape([dimension, dimension]);
        let mut z_sketch = DynArray::<c64, 2>::from_shape([dimension, dimension]);
        for sample in 0..dimension {
            y_test[[sample, sample]] = c64::new(1.0, 0.0);
            for target in 0..dimension {
                // Row-wise convention: Y = Omega A^T.
                y_sketch[[sample, target]] = matrix[[target, sample]];
                // Deliberately invalid as samples of either A^T or A^*.
                if let Some(scale) = corrupt_z_scale {
                    z_test[[sample, target]] = c64::new(
                        scale * (sample + target + 1) as f64,
                        -scale * (2 * sample + target + 1) as f64,
                    );
                    z_sketch[[sample, target]] = c64::new(
                        -scale * (sample + 3 * target + 1) as f64,
                        scale * (sample + target + 2) as f64,
                    );
                }
            }
        }
        InMemorySampleProvider::new(
            SampleBlock {
                y_test,
                y_sketch,
                z_test: corrupt_z_scale.map(|_| z_test),
                z_sketch: corrupt_z_scale.map(|_| z_sketch),
            },
            ZMode::Transpose,
        )
        .unwrap()
    }

    fn check_symmetry_recovery(matrix: &DynArray<c64, 2>, symmetry: MatrixSymmetry) {
        let dimension = matrix.shape()[0];
        let provider = identity_y_provider(matrix, None);
        let indices = (0..dimension).collect::<Vec<_>>();
        let blocks = extract_sampled_near_blocks_with_symmetry(
            &provider,
            dimension,
            &indices,
            &indices,
            Regularization {
                mode: crate::regularized_least_squares::RegularizationMode::Scaled,
                relative: 0.0,
                absolute_floor: 0.0,
            },
            RegularizedLeastSquaresMethod::Cholesky,
            symmetry,
        )
        .unwrap();
        for row in 0..dimension {
            for column in 0..dimension {
                assert!(
                    (blocks.target_by_near[[row, column]] - matrix[[row, column]]).norm() < 1.0e-12
                );
                assert!(
                    (blocks.near_by_target[[row, column]] - matrix[[row, column]]).norm() < 1.0e-12
                );
            }
        }
    }

    fn complex_problem(z_mode: ZMode) -> (InMemorySampleProvider<c64>, DynArray<c64, 2>) {
        let n = 12;
        let samples = 16;
        let mut matrix = DynArray::<c64, 2>::from_shape([n, n]);
        for row in 0..n {
            for column in 0..n {
                matrix[[row, column]] = c64::new(
                    if row == column { 4.0 } else { 0.0 }
                        + ((3 * row + 5 * column + 1) % 17) as f64 / 31.0,
                    ((7 * row + 2 * column + 3) % 13) as f64 / 37.0,
                );
            }
        }
        // Targets 0..6 have shared rank-two interactions with the far set
        // 8..12 in both directions. Near entries remain generic.
        let coefficients = (0..6)
            .map(|candidate| {
                [
                    c64::new((candidate + 1) as f64 / 7.0, (candidate % 3) as f64 / 11.0),
                    c64::new((2 * candidate + 1) as f64 / 9.0, -(candidate as f64) / 13.0),
                ]
            })
            .collect::<Vec<_>>();
        for far in 8..12 {
            let left = [
                c64::new((far - 6) as f64 / 5.0, (far - 8) as f64 / 17.0),
                c64::new((far - 5) as f64 / 8.0, -(far as f64) / 29.0),
            ];
            let right = [
                c64::new((far - 7) as f64 / 6.0, -((far - 8) as f64) / 19.0),
                c64::new((far - 4) as f64 / 10.0, (far as f64) / 31.0),
            ];
            for target in 0..6 {
                matrix[[far, target]] =
                    left[0] * coefficients[target][0] + left[1] * coefficients[target][1];
                matrix[[target, far]] =
                    right[0] * coefficients[target][0] + right[1] * coefficients[target][1];
            }
        }
        let mut y_test = DynArray::<c64, 2>::from_shape([samples, n]);
        let mut z_test = DynArray::<c64, 2>::from_shape([samples, n]);
        for sample in 0..samples {
            for column in 0..n {
                y_test[[sample, column]] = c64::new(
                    ((sample * 7 + column * 3 + 1) % 19) as f64 / 9.0 - 1.0,
                    ((sample * 5 + column * 11 + 2) % 23) as f64 / 11.0 - 1.0,
                );
                z_test[[sample, column]] = c64::new(
                    ((sample * 13 + column * 2 + 4) % 29) as f64 / 14.0 - 1.0,
                    ((sample * 3 + column * 7 + 5) % 17) as f64 / 8.0 - 1.0,
                );
            }
        }
        let mut y_sketch = DynArray::<c64, 2>::from_shape([samples, n]);
        let mut z_sketch = DynArray::<c64, 2>::from_shape([samples, n]);
        for sample in 0..samples {
            for target in 0..n {
                for source in 0..n {
                    y_sketch[[sample, target]] +=
                        y_test[[sample, source]] * matrix[[target, source]];
                    let entry = match z_mode {
                        ZMode::Transpose => matrix[[source, target]],
                        ZMode::Adjoint => matrix[[source, target]].conj(),
                    };
                    z_sketch[[sample, target]] += z_test[[sample, source]] * entry;
                }
            }
        }
        let provider = InMemorySampleProvider::new(
            SampleBlock {
                y_test,
                y_sketch,
                z_test: Some(z_test),
                z_sketch: Some(z_sketch),
            },
            z_mode,
        )
        .unwrap();
        (provider, matrix)
    }

    fn check(z_mode: ZMode) {
        let (provider, matrix) = complex_problem(z_mode);
        let target = [0, 1, 2, 3, 4, 5];
        let near = [0, 1, 2, 3, 4, 5, 6, 7];
        let result = compute_sampled_id(
            &provider,
            11,
            &target,
            &near,
            IdRankSelection::Fixed(2),
            IdMethod::Rrqr,
            Regularization {
                mode: crate::regularized_least_squares::RegularizationMode::Scaled,
                relative: 1.0e-14,
                absolute_floor: 0.0,
            },
            RegularizedLeastSquaresMethod::Cholesky,
        )
        .unwrap();
        assert_eq!(result.id.rank, 2);
        // Test the selected ID against the actual two-sided far panels, not
        // merely against the random sketch used to select it.
        let far = [8, 9, 10, 11];
        let ordered = result
            .id
            .perm
            .iter()
            .map(|&local| target[local])
            .collect::<Vec<_>>();
        for redundant in 2..target.len() {
            for &far_index in &far {
                let mut column = c64::new(0.0, 0.0);
                let mut row = c64::new(0.0, 0.0);
                for skeleton in 0..2 {
                    let coefficient = result.id.id_mat[[redundant - 2, skeleton]].conj();
                    column += matrix[[far_index, ordered[skeleton]]] * coefficient;
                    row += matrix[[ordered[skeleton], far_index]] * coefficient;
                }
                assert!((column - matrix[[far_index, ordered[redundant]]]).norm() < 2.0e-9);
                assert!((row - matrix[[ordered[redundant], far_index]]).norm() < 2.0e-9);
            }
        }
    }

    #[test]
    fn qr_and_svd_nullification_select_the_expected_rank() {
        let (provider, _) = complex_problem(ZMode::Transpose);
        let target = [0, 1, 2, 3, 4, 5];
        let near = [0, 1, 2, 3, 4, 5, 6, 7];
        for nullification in [
            NullificationMethod::Qr { tolerance: 1.0e-12 },
            NullificationMethod::Svd { tolerance: 1.0e-12 },
        ] {
            let result = compute_sampled_id_with_combination_and_nullification(
                &provider,
                11,
                &target,
                &near,
                IdRankSelection::Fixed(2),
                IdMethod::Rrqr,
                nullification,
                IdCombination::Concat,
            )
            .unwrap();
            assert_eq!(result.id.rank, 2);
            assert!(result.y_projection.is_none());
            assert!(result.z_projection.is_none());
        }
    }

    #[test]
    fn complex_qr_and_svd_use_the_orthogonal_left_null_space() {
        let mut test = DynArray::<c64, 2>::from_shape([7, 3]);
        for row in 0..7 {
            let x = (row + 1) as f64;
            test[[row, 0]] = c64::new(1.0, 0.1 * x);
            test[[row, 1]] = c64::new(0.2 * x, 1.0 - 0.03 * x * x);
            test[[row, 2]] = c64::new(0.05 * x * x, -0.07 * x);
        }
        let indices = [0, 1, 2];
        for nullification in [
            NullificationMethod::Qr { tolerance: 1.0e-12 },
            NullificationMethod::Svd { tolerance: 1.0e-12 },
        ] {
            // Using the near-test matrix itself as the target sketch must
            // vanish after applying an orthonormal basis of null(A^H).
            let (nullified, report) =
                nullified_panel(&test, &test, &indices, &indices, false, nullification).unwrap();
            assert_eq!(nullified.shape(), [4, 3]);
            assert!(report.is_none());
            let maximum = nullified
                .iter_value()
                .map(|value| value.norm())
                .fold(0.0_f64, f64::max);
            assert!(maximum < 1.0e-12, "left-null residual was {maximum:e}");
        }
    }

    #[test]
    fn projection_retains_all_rows_from_each_sketch_family() {
        let (provider, _) = complex_problem(ZMode::Transpose);
        let target = [0, 1, 2, 3, 4, 5];
        let near = [0, 1, 2, 3, 4, 5, 6, 7];
        let result = compute_sampled_id_with_combination_and_nullification(
            &provider,
            11,
            &target,
            &near,
            IdRankSelection::Fixed(2),
            IdMethod::Rrqr,
            NullificationMethod::Projection {
                solver: RegularizedLeastSquaresMethod::Lu,
                regularization: Regularization {
                    mode: crate::regularized_least_squares::RegularizationMode::Historical,
                    relative: 1.0e-12,
                    absolute_floor: 0.0,
                },
            },
            IdCombination::Concat,
        )
        .unwrap();
        assert_eq!(result.nullified_panel.shape(), [22, target.len()]);
    }

    fn check_near_extraction(z_mode: ZMode) {
        let (provider, matrix) = complex_problem(z_mode);
        let target = [0, 1, 2, 3, 4, 5];
        let near = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let blocks = extract_sampled_near_blocks(
            &provider,
            16,
            &target,
            &near,
            Regularization {
                mode: crate::regularized_least_squares::RegularizationMode::Scaled,
                relative: 1.0e-14,
                absolute_floor: 0.0,
            },
            RegularizedLeastSquaresMethod::Cholesky,
        )
        .unwrap();
        let mut max_y = 0.0_f64;
        let mut max_z = 0.0_f64;
        for (local_target, &global_target) in target.iter().enumerate() {
            for (local_near, &global_near) in near.iter().enumerate() {
                max_y = max_y.max(
                    (blocks.target_by_near[[local_target, local_near]]
                        - matrix[[global_target, global_near]])
                    .norm(),
                );
                max_z = max_z.max(
                    (blocks.near_by_target[[local_near, local_target]]
                        - matrix[[global_near, global_target]])
                    .norm(),
                );
            }
        }
        eprintln!("near extraction mode={z_mode:?} max_y={max_y:e} max_z={max_z:e}");
        assert!(max_y < 2.0e-9);
        assert!(max_z < 2.0e-9);
    }

    #[test]
    fn complex_transpose_samples_recover_two_sided_id() {
        check(ZMode::Transpose);
    }

    #[test]
    fn complex_adjoint_samples_are_converted_before_id() {
        check(ZMode::Adjoint);
    }

    #[test]
    fn complex_transpose_samples_recover_near_blocks() {
        check_near_extraction(ZMode::Transpose);
    }

    #[test]
    fn complex_adjoint_samples_recover_near_blocks() {
        check_near_extraction(ZMode::Adjoint);
    }

    #[test]
    fn complex_symmetric_near_extraction_uses_y_and_plain_transpose() {
        let mut matrix = DynArray::<c64, 2>::from_shape([3, 3]);
        matrix[[0, 0]] = c64::new(3.0, 1.0);
        matrix[[1, 1]] = c64::new(4.0, -0.5);
        matrix[[2, 2]] = c64::new(5.0, 0.25);
        matrix[[0, 1]] = c64::new(1.0, 2.0);
        matrix[[1, 0]] = matrix[[0, 1]];
        matrix[[0, 2]] = c64::new(-0.5, 0.75);
        matrix[[2, 0]] = matrix[[0, 2]];
        matrix[[1, 2]] = c64::new(0.25, -1.5);
        matrix[[2, 1]] = matrix[[1, 2]];
        check_symmetry_recovery(&matrix, MatrixSymmetry::Symmetric);
    }

    #[test]
    fn hermitian_near_extraction_uses_y_and_conjugate_transpose() {
        let mut matrix = DynArray::<c64, 2>::from_shape([3, 3]);
        matrix[[0, 0]] = c64::new(3.0, 0.0);
        matrix[[1, 1]] = c64::new(4.0, 0.0);
        matrix[[2, 2]] = c64::new(5.0, 0.0);
        matrix[[0, 1]] = c64::new(1.0, 2.0);
        matrix[[1, 0]] = matrix[[0, 1]].conj();
        matrix[[0, 2]] = c64::new(-0.5, 0.75);
        matrix[[2, 0]] = matrix[[0, 2]].conj();
        matrix[[1, 2]] = c64::new(0.25, -1.5);
        matrix[[2, 1]] = matrix[[1, 2]].conj();
        check_symmetry_recovery(&matrix, MatrixSymmetry::Hermitian);

        let provider = identity_y_provider(&matrix, None);
        let indices = [0, 1, 2];
        let wrongly_symmetric = extract_sampled_near_blocks_with_symmetry(
            &provider,
            3,
            &indices,
            &indices,
            Regularization {
                mode: crate::regularized_least_squares::RegularizationMode::Scaled,
                relative: 0.0,
                absolute_floor: 0.0,
            },
            RegularizedLeastSquaresMethod::Cholesky,
            MatrixSymmetry::Symmetric,
        )
        .unwrap();
        assert!((wrongly_symmetric.near_by_target[[0, 1]] - matrix[[0, 1]]).norm() > 1.0);
    }

    #[test]
    fn provider_reports_insufficient_rows_without_sampling() {
        let (provider, _) = complex_problem(ZMode::Transpose);
        let result = compute_sampled_id(
            &provider,
            17,
            &[0, 1, 2],
            &[0, 1],
            IdRankSelection::Fixed(1),
            IdMethod::Rrqr,
            Regularization {
                mode: crate::regularized_least_squares::RegularizationMode::Scaled,
                relative: 1.0e-14,
                absolute_floor: 0.0,
            },
            RegularizedLeastSquaresMethod::Cholesky,
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("an insufficient provider unexpectedly succeeded"),
        };
        assert!(matches!(error, SampleError::InsufficientRows { .. }));
    }
}
