//! One nonsymmetric RSRS cycle driven entirely by matching test/sketch data.

use crate::{
    composed_factor::ComposedElementaryFactor,
    configuration::{MatrixSymmetry, PivotMethod},
    diagonal_block::SquareBlock,
    elementary_factor::{ElementaryFactor, FactorMode, IndexedLeftUpdate, IndexedRightUpdate},
    interpolative_decomposition::{IdMethod, IdRankSelection},
    regularized_least_squares::{Regularization, RegularizedLeastSquaresMethod},
    sampled_id::{
        compute_sampled_id_with_combination_and_nullification,
        extract_sampled_near_blocks_with_symmetry, IdCombination, SampleError, SampleProvider,
    },
    sampled_options::{
        ExtractionOptions, NullificationMethod, PivotOptions, SampledFactorizationOptions,
    },
};
use num::Zero;
use rlst::{
    dense::{array::DynArray, linalg::lapack::interpolative_decomposition::MatrixId},
    Gemm, Lapack,
};
use std::{collections::HashMap, error::Error, fmt};

#[derive(Debug)]
pub enum SampledCycleError {
    Samples(SampleError),
    LinearAlgebra(String),
    InvalidPlan(String),
    NotCompressible { rank: usize, target_size: usize },
}

impl fmt::Display for SampledCycleError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Samples(error) => write!(out, "{error}"),
            Self::LinearAlgebra(message) | Self::InvalidPlan(message) => out.write_str(message),
            Self::NotCompressible { rank, target_size } => write!(
                out,
                "sampled ID selected rank {rank} for {target_size} targets; no DOFs can be eliminated"
            ),
        }
    }
}

impl Error for SampledCycleError {}
impl From<SampleError> for SampledCycleError {
    fn from(value: SampleError) -> Self {
        Self::Samples(value)
    }
}

/// Sampled RSRS counterpart of `DenseSkeletonizationCycle`.
///
/// It stores exactly the same mathematical factors, but obtains the ID and
/// local elimination blocks from a [`SampleProvider`] rather than matrix
/// entries. Dense SRS and sampled RSRS can therefore coexist and share the
/// downstream factor-application machinery.
pub struct SampledSkeletonizationCycle<Item: MatrixId + Lapack> {
    dimension: usize,
    skeleton_indices: Vec<usize>,
    redundant_indices: Vec<usize>,
    remaining_indices: Vec<usize>,
    diagonal_block: DynArray<Item, 2>,
    id_factor: ElementaryFactor<Item>,
    /// General matrices retain an independent Z-derived lower factor.
    /// Symmetric/Hermitian matrices derive it from `upper_factor` on apply.
    lower_factor: Option<ElementaryFactor<Item>>,
    upper_factor: ComposedElementaryFactor<Item>,
    symmetry: MatrixSymmetry,
}

/// ID stage of one sampled cycle. Historical RSRS commits every ID in a
/// conflict-free batch to the live samples before extracting any near block.
/// Keeping this intermediate object makes that ordering explicit.
pub struct PreparedSampledCycle<Item: MatrixId + Lapack> {
    dimension: usize,
    skeleton_indices: Vec<usize>,
    redundant_indices: Vec<usize>,
    id_factor: ElementaryFactor<Item>,
    symmetry: MatrixSymmetry,
}

fn elimination_factor<Item: Lapack>(
    dimension: usize,
    source_indices: Vec<usize>,
    target_indices: Vec<usize>,
    pivot_matrix: &DynArray<Item, 2>,
    coupling: DynArray<Item, 2>,
    options: PivotOptions<Item::Real>,
) -> Result<ComposedElementaryFactor<Item>, SampledCycleError> {
    match options.method {
        PivotMethod::Lu => ComposedElementaryFactor::new(
            dimension,
            source_indices,
            target_indices,
            SquareBlock::lu_shifted(pivot_matrix, options.shift)
                .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?,
            coupling,
        )
        .map_err(SampledCycleError::InvalidPlan),
        PivotMethod::LuHybrid | PivotMethod::DirectInverse => {
            let pivot = match options.method {
                PivotMethod::LuHybrid => SquareBlock::lu_shifted(pivot_matrix, options.shift),
                PivotMethod::DirectInverse => {
                    SquareBlock::explicit_inverse_shifted(pivot_matrix, options.shift)
                }
                PivotMethod::Lu => unreachable!(),
            }
            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?;
            let mut update = coupling;
            pivot
                .apply_left_in_place(&mut update, FactorMode::NoTrans, true)
                .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?;
            ComposedElementaryFactor::from_materialized_update(
                dimension,
                source_indices,
                target_indices,
                update,
            )
            .map_err(SampledCycleError::InvalidPlan)
        }
    }
}

impl<Item: MatrixId + Lapack> PreparedSampledCycle<Item> {
    pub fn skeleton_indices(&self) -> &[usize] {
        &self.skeleton_indices
    }

    pub fn redundant_indices(&self) -> &[usize] {
        &self.redundant_indices
    }

    pub fn update_samples_id(
        &self,
        samples: &mut crate::sampled_id::SampleBlock<Item>,
    ) -> Result<(), SampledCycleError> {
        samples.validate()?;
        if samples.dimension() != self.dimension {
            return Err(SampledCycleError::InvalidPlan(
                "sample dimension differs from prepared cycle dimension".into(),
            ));
        }
        self.id_factor
            .apply_right_in_place(&mut samples.y_test, FactorMode::NoTrans, false);
        self.id_factor
            .apply_right_in_place(&mut samples.y_sketch, FactorMode::Transpose, true);
        if let Some(z_test) = samples.z_test.as_mut() {
            self.id_factor
                .apply_right_in_place(z_test, FactorMode::NoTrans, false);
        }
        if let Some(z_sketch) = samples.z_sketch.as_mut() {
            self.id_factor
                .apply_right_in_place(z_sketch, FactorMode::Transpose, true);
        }
        Ok(())
    }
}

impl<Item: MatrixId + Lapack + Gemm> SampledSkeletonizationCycle<Item>
where
    Item::Real: Into<Item>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_ranked_with_options<Provider: SampleProvider<Item>>(
        provider: &Provider,
        sample_rows: usize,
        target_indices: &[usize],
        near_indices: &[usize],
        rank_selection: IdRankSelection<Item::Real>,
        id_method: IdMethod<Item::Real>,
        options: SampledFactorizationOptions<Item>,
        symmetry: MatrixSymmetry,
    ) -> Result<PreparedSampledCycle<Item>, SampledCycleError> {
        if provider.dimension() == 0 || target_indices.len() < 2 {
            return Err(SampledCycleError::InvalidPlan(
                "sampled cycle requires at least two target indices".into(),
            ));
        }
        if target_indices
            .iter()
            .any(|&index| !near_indices.contains(&index))
        {
            return Err(SampledCycleError::InvalidPlan(
                "the near set must contain every target index".into(),
            ));
        }
        let combination = match symmetry {
            MatrixSymmetry::General => IdCombination::Concat,
            MatrixSymmetry::Symmetric | MatrixSymmetry::Hermitian => IdCombination::YOnly,
        };
        let sampled_id = compute_sampled_id_with_combination_and_nullification(
            provider,
            sample_rows,
            target_indices,
            near_indices,
            rank_selection,
            id_method,
            options.nullification,
            combination,
        )?;
        let rank = sampled_id.id.rank;
        if rank == 0 || rank >= target_indices.len() {
            return Err(SampledCycleError::NotCompressible {
                rank,
                target_size: target_indices.len(),
            });
        }
        let ordered = sampled_id
            .id
            .perm
            .iter()
            .map(|&local| target_indices[local])
            .collect::<Vec<_>>();
        let skeleton_indices = ordered[..rank].to_vec();
        let redundant_indices = ordered[rank..].to_vec();
        let mut interpolation =
            DynArray::<Item, 2>::from_shape([redundant_indices.len(), skeleton_indices.len()]);
        for redundant in 0..redundant_indices.len() {
            for skeleton in 0..skeleton_indices.len() {
                interpolation[[redundant, skeleton]] =
                    sampled_id.id.id_mat[[redundant, skeleton]].conj();
            }
        }
        let id_factor = ElementaryFactor::new(
            provider.dimension(),
            skeleton_indices.clone(),
            redundant_indices.clone(),
            interpolation,
        )
        .map_err(SampledCycleError::InvalidPlan)?;
        Ok(PreparedSampledCycle {
            dimension: provider.dimension(),
            skeleton_indices,
            redundant_indices,
            id_factor,
            symmetry,
        })
    }

    pub fn finish_after_id<Provider: SampleProvider<Item>>(
        provider: &Provider,
        sample_rows: usize,
        prepared: PreparedSampledCycle<Item>,
        near_indices: &[usize],
        options: SampledFactorizationOptions<Item>,
    ) -> Result<Self, SampledCycleError> {
        let redundant_lookup = prepared
            .redundant_indices
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        if prepared
            .redundant_indices
            .iter()
            .any(|index| !near_indices.contains(index))
        {
            return Err(SampledCycleError::InvalidPlan(
                "the post-ID near set must contain every redundant index".into(),
            ));
        }
        let mut remaining_indices = near_indices
            .iter()
            .copied()
            .filter(|index| !redundant_lookup.contains(index))
            .collect::<Vec<_>>();
        remaining_indices.sort_unstable();
        remaining_indices.dedup();
        let blocks = extract_sampled_near_blocks_with_symmetry(
            provider,
            sample_rows,
            &prepared.redundant_indices,
            near_indices,
            options.near_extraction.regularization,
            options.near_extraction.solver,
            prepared.symmetry,
        )?;
        let near_position = positions(near_indices);
        let upper_pivot = select_columns_by_global(
            &blocks.target_by_near,
            &near_position,
            &prepared.redundant_indices,
        );
        let right_coupling =
            select_columns_by_global(&blocks.target_by_near, &near_position, &remaining_indices);
        let upper_factor = elimination_factor(
            provider.dimension(),
            remaining_indices.clone(),
            prepared.redundant_indices.clone(),
            &upper_pivot,
            right_coupling,
            options.near_pivot,
        )?;
        let lower_factor = if prepared.symmetry == MatrixSymmetry::General {
            let lower_as_rows = transpose_copy(&blocks.near_by_target);
            let lower_pivot = select_columns_by_global(
                &lower_as_rows,
                &near_position,
                &prepared.redundant_indices,
            );
            let mut lower_transpose =
                select_columns_by_global(&lower_as_rows, &near_position, &remaining_indices);
            SquareBlock::from_pivot_method(
                &lower_pivot,
                options.near_pivot.method,
                options.near_pivot.shift,
            )
            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?
            .apply_left_in_place(&mut lower_transpose, FactorMode::NoTrans, true)
            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?;
            Some(
                ElementaryFactor::new(
                    provider.dimension(),
                    prepared.redundant_indices.clone(),
                    remaining_indices.clone(),
                    transpose_copy(&lower_transpose),
                )
                .map_err(SampledCycleError::InvalidPlan)?,
            )
        } else {
            None
        };
        Ok(Self {
            dimension: prepared.dimension,
            skeleton_indices: prepared.skeleton_indices,
            redundant_indices: prepared.redundant_indices,
            remaining_indices,
            diagonal_block: upper_pivot,
            id_factor: prepared.id_factor,
            lower_factor,
            upper_factor,
            symmetry: prepared.symmetry,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn factorize<Provider: SampleProvider<Item>>(
        provider: &Provider,
        sample_rows: usize,
        target_indices: &[usize],
        near_indices: &[usize],
        rank: usize,
        id_method: IdMethod<Item::Real>,
        regularization: Regularization<Item::Real>,
        least_squares: RegularizedLeastSquaresMethod,
    ) -> Result<Self, SampledCycleError> {
        Self::factorize_ranked(
            provider,
            sample_rows,
            target_indices,
            near_indices,
            IdRankSelection::Fixed(rank),
            id_method,
            regularization,
            least_squares,
        )
    }

    /// Factorize one cycle with either tolerance-selected or prescribed rank.
    #[allow(clippy::too_many_arguments)]
    pub fn factorize_ranked<Provider: SampleProvider<Item>>(
        provider: &Provider,
        sample_rows: usize,
        target_indices: &[usize],
        near_indices: &[usize],
        rank_selection: IdRankSelection<Item::Real>,
        id_method: IdMethod<Item::Real>,
        regularization: Regularization<Item::Real>,
        least_squares: RegularizedLeastSquaresMethod,
    ) -> Result<Self, SampledCycleError> {
        Self::factorize_ranked_with_symmetry(
            provider,
            sample_rows,
            target_indices,
            near_indices,
            rank_selection,
            id_method,
            regularization,
            least_squares,
            MatrixSymmetry::General,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn factorize_ranked_with_symmetry<Provider: SampleProvider<Item>>(
        provider: &Provider,
        sample_rows: usize,
        target_indices: &[usize],
        near_indices: &[usize],
        rank_selection: IdRankSelection<Item::Real>,
        id_method: IdMethod<Item::Real>,
        regularization: Regularization<Item::Real>,
        least_squares: RegularizedLeastSquaresMethod,
        symmetry: MatrixSymmetry,
    ) -> Result<Self, SampledCycleError> {
        let extraction = ExtractionOptions {
            solver: least_squares,
            regularization,
        };
        Self::factorize_ranked_with_options(
            provider,
            sample_rows,
            target_indices,
            near_indices,
            rank_selection,
            id_method,
            SampledFactorizationOptions {
                nullification: NullificationMethod::Projection {
                    solver: least_squares,
                    regularization,
                },
                near_extraction: extraction,
                diagonal_extraction: extraction,
                near_pivot: PivotOptions {
                    method: PivotMethod::Lu,
                    shift: Item::Real::zero(),
                },
                diagonal_pivot: PivotOptions {
                    method: PivotMethod::Lu,
                    shift: Item::Real::zero(),
                },
            },
            symmetry,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn factorize_ranked_with_options<Provider: SampleProvider<Item>>(
        provider: &Provider,
        sample_rows: usize,
        target_indices: &[usize],
        near_indices: &[usize],
        rank_selection: IdRankSelection<Item::Real>,
        id_method: IdMethod<Item::Real>,
        options: SampledFactorizationOptions<Item>,
        symmetry: MatrixSymmetry,
    ) -> Result<Self, SampledCycleError> {
        if provider.dimension() == 0 || target_indices.len() < 2 {
            return Err(SampledCycleError::InvalidPlan(
                "sampled cycle requires at least two target indices".into(),
            ));
        }
        if target_indices
            .iter()
            .any(|&index| !near_indices.contains(&index))
        {
            return Err(SampledCycleError::InvalidPlan(
                "the near set must contain every target index".into(),
            ));
        }
        let combination = match symmetry {
            MatrixSymmetry::General => IdCombination::Concat,
            MatrixSymmetry::Symmetric | MatrixSymmetry::Hermitian => IdCombination::YOnly,
        };
        let sampled_id = compute_sampled_id_with_combination_and_nullification(
            provider,
            sample_rows,
            target_indices,
            near_indices,
            rank_selection,
            id_method,
            options.nullification,
            combination,
        )?;
        let rank = sampled_id.id.rank;
        if rank == 0 || rank >= target_indices.len() {
            return Err(SampledCycleError::NotCompressible {
                rank,
                target_size: target_indices.len(),
            });
        }
        let blocks = extract_sampled_near_blocks_with_symmetry(
            provider,
            sample_rows,
            target_indices,
            near_indices,
            options.near_extraction.regularization,
            options.near_extraction.solver,
            symmetry,
        )?;
        let ordered = sampled_id
            .id
            .perm
            .iter()
            .map(|&local| target_indices[local])
            .collect::<Vec<_>>();
        let skeleton_indices = ordered[..rank].to_vec();
        let redundant_indices = ordered[rank..].to_vec();
        let mut interpolation =
            DynArray::<Item, 2>::from_shape([redundant_indices.len(), skeleton_indices.len()]);
        for redundant in 0..redundant_indices.len() {
            for skeleton in 0..skeleton_indices.len() {
                interpolation[[redundant, skeleton]] =
                    sampled_id.id.id_mat[[redundant, skeleton]].conj();
            }
        }
        let id_factor = ElementaryFactor::new(
            provider.dimension(),
            skeleton_indices.clone(),
            redundant_indices.clone(),
            copy_matrix(&interpolation),
        )
        .map_err(SampledCycleError::InvalidPlan)?;

        let redundant_lookup = redundant_indices
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let mut remaining_indices = near_indices
            .iter()
            .copied()
            .filter(|index| !redundant_lookup.contains(index))
            .collect::<Vec<_>>();
        remaining_indices.sort_unstable();
        remaining_indices.dedup();
        let target_position = positions(target_indices);
        let near_position = positions(near_indices);
        let block_positions = BlockPositions {
            target: &target_position,
            near: &near_position,
        };

        // Y recovers A[target,near] and supplies the upper elimination.
        let upper_pivot = transformed_block(
            &blocks.target_by_near,
            &block_positions,
            &skeleton_indices,
            &redundant_indices,
            &interpolation,
            &redundant_indices,
            &redundant_indices,
        );
        let right_coupling = transformed_block(
            &blocks.target_by_near,
            &block_positions,
            &skeleton_indices,
            &redundant_indices,
            &interpolation,
            &redundant_indices,
            &remaining_indices,
        );
        let upper_factor = elimination_factor(
            provider.dimension(),
            remaining_indices.clone(),
            redundant_indices.clone(),
            &upper_pivot,
            right_coupling,
            options.near_pivot,
        )?;

        let lower_factor = if symmetry == MatrixSymmetry::General {
            // Z recovers A[near,target]. Transpose it into the same row-oriented
            // helper, then use its own recovered pivot for the lower solve, as
            // in historical nonsymmetric RSRS.
            let z_target_by_near = transpose_copy(&blocks.near_by_target);
            let lower_pivot = transformed_block(
                &z_target_by_near,
                &block_positions,
                &skeleton_indices,
                &redundant_indices,
                &interpolation,
                &redundant_indices,
                &redundant_indices,
            );
            let left_coupling_as_transpose = transformed_block(
                &z_target_by_near,
                &block_positions,
                &skeleton_indices,
                &redundant_indices,
                &interpolation,
                &redundant_indices,
                &remaining_indices,
            );
            let mut lower_transpose = left_coupling_as_transpose;
            SquareBlock::from_pivot_method(
                &lower_pivot,
                options.near_pivot.method,
                options.near_pivot.shift,
            )
            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?
            .apply_left_in_place(&mut lower_transpose, FactorMode::NoTrans, true)
            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?;
            Some(
                ElementaryFactor::new(
                    provider.dimension(),
                    redundant_indices.clone(),
                    remaining_indices.clone(),
                    transpose_copy(&lower_transpose),
                )
                .map_err(SampledCycleError::InvalidPlan)?,
            )
        } else {
            None
        };

        Ok(Self {
            dimension: provider.dimension(),
            skeleton_indices,
            redundant_indices,
            remaining_indices,
            diagonal_block: upper_pivot,
            id_factor,
            lower_factor,
            upper_factor,
            symmetry,
        })
    }

    pub fn skeleton_indices(&self) -> &[usize] {
        &self.skeleton_indices
    }
    pub fn redundant_indices(&self) -> &[usize] {
        &self.redundant_indices
    }
    pub fn remaining_indices(&self) -> &[usize] {
        &self.remaining_indices
    }
    pub fn diagonal_block(&self) -> &DynArray<Item, 2> {
        &self.diagonal_block
    }
    /// Interpolation update retained by the ID factor. Exposed for numerical
    /// diagnostics and controlled historical/refactored comparisons.
    pub fn id_update(&self) -> &DynArray<Item, 2> {
        self.id_factor.update()
    }

    pub(crate) fn storage_bytes(&self) -> (u64, u64) {
        let id_bytes = self.id_factor.storage_bytes();
        let elimination_bytes = self.upper_factor.storage_bytes()
            + self
                .lower_factor
                .as_ref()
                .map_or(0, ElementaryFactor::storage_bytes);
        (id_bytes, elimination_bytes)
    }

    /// Apply this sampled cycle's congruence/elimination transforms to a dense
    /// matrix. This is a validation/debug operation; production RSRS updates
    /// sketches and retained blocks instead of materializing the operator.
    pub fn transform_dense(
        &self,
        matrix: &DynArray<Item, 2>,
    ) -> Result<DynArray<Item, 2>, SampledCycleError> {
        let mut result = copy_matrix(matrix);
        self.id_factor
            .apply_left_in_place(&mut result, FactorMode::NoTrans, true);
        self.id_factor
            .apply_right_in_place(&mut result, FactorMode::Transpose, true);
        self.lower_left(&mut result, FactorMode::NoTrans, true)?;
        self.upper_factor
            .apply_right_in_place(&mut result, FactorMode::NoTrans, true)
            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?;
        Ok(result)
    }

    pub fn reconstruct(
        &self,
        transformed: &DynArray<Item, 2>,
    ) -> Result<DynArray<Item, 2>, SampledCycleError> {
        assert_eq!(transformed.shape(), [self.dimension, self.dimension]);
        let mut result = copy_matrix(transformed);
        self.lower_left(&mut result, FactorMode::NoTrans, false)?;
        self.upper_factor
            .apply_right_in_place(&mut result, FactorMode::NoTrans, false)
            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?;
        self.id_factor
            .apply_left_in_place(&mut result, FactorMode::NoTrans, false);
        self.id_factor
            .apply_right_in_place(&mut result, FactorMode::Transpose, false);
        Ok(result)
    }

    /// Advance canonical row-oriented samples from A to the transformed
    /// operator represented after this cycle.
    pub fn update_samples(
        &self,
        samples: &mut crate::sampled_id::SampleBlock<Item>,
    ) -> Result<(), SampledCycleError> {
        samples.validate()?;
        if samples.dimension() != self.dimension {
            return Err(SampledCycleError::InvalidPlan(
                "sample dimension differs from cycle dimension".into(),
            ));
        }
        self.update_samples_id(samples)?;
        self.update_samples_lu(samples)
    }

    pub fn update_samples_id(
        &self,
        samples: &mut crate::sampled_id::SampleBlock<Item>,
    ) -> Result<(), SampledCycleError> {
        samples.validate()?;
        if samples.dimension() != self.dimension {
            return Err(SampledCycleError::InvalidPlan(
                "sample dimension differs from cycle dimension".into(),
            ));
        }
        // A1 = E^-1 A E^-T.
        self.id_factor
            .apply_right_in_place(&mut samples.y_test, FactorMode::NoTrans, false);
        self.id_factor
            .apply_right_in_place(&mut samples.y_sketch, FactorMode::Transpose, true);
        if let Some(z_test) = samples.z_test.as_mut() {
            self.id_factor
                .apply_right_in_place(z_test, FactorMode::NoTrans, false);
        }
        if let Some(z_sketch) = samples.z_sketch.as_mut() {
            self.id_factor
                .apply_right_in_place(z_sketch, FactorMode::Transpose, true);
        }
        Ok(())
    }

    pub fn update_samples_lu(
        &self,
        samples: &mut crate::sampled_id::SampleBlock<Item>,
    ) -> Result<(), SampledCycleError> {
        samples.validate()?;
        if samples.dimension() != self.dimension {
            return Err(SampledCycleError::InvalidPlan(
                "sample dimension differs from cycle dimension".into(),
            ));
        }
        // A2 = L^-1 A1 U^-1. For Y^T=T A^T and Z^T=T A the corresponding
        // test/sketch transformations differ exactly as below.
        self.upper_factor
            .apply_right_in_place(&mut samples.y_test, FactorMode::Transpose, false)
            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?;
        self.lower_right(&mut samples.y_sketch, FactorMode::Transpose, true)?;
        if let Some(z_test) = samples.z_test.as_mut() {
            self.lower_right(z_test, FactorMode::NoTrans, false)?;
        }
        if let Some(z_sketch) = samples.z_sketch.as_mut() {
            self.upper_factor
                .apply_right_in_place(z_sketch, FactorMode::NoTrans, true)
                .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?;
        }
        Ok(())
    }

    pub(crate) fn id_left_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> IndexedLeftUpdate<Item> {
        self.id_factor.left_update(target, mode, inverse)
    }
    pub(crate) fn id_right_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> IndexedRightUpdate<Item> {
        self.id_factor.right_update(target, mode, inverse)
    }
    pub(crate) fn lower_left(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> Result<(), SampledCycleError> {
        if let Some(lower) = &self.lower_factor {
            lower.apply_left_in_place(target, mode, inverse);
            Ok(())
        } else {
            self.upper_factor
                .apply_left_in_place(target, partner_mode(self.symmetry, mode), inverse)
                .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))
        }
    }
    pub(crate) fn lower_left_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> Result<IndexedLeftUpdate<Item>, SampledCycleError> {
        if let Some(lower) = &self.lower_factor {
            Ok(lower.left_update(target, mode, inverse))
        } else {
            self.upper_factor
                .indexed_left_update(target, partner_mode(self.symmetry, mode), inverse)
                .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))
        }
    }
    pub(crate) fn lower_right(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> Result<(), SampledCycleError> {
        if let Some(lower) = &self.lower_factor {
            lower.apply_right_in_place(target, mode, inverse);
            Ok(())
        } else {
            self.upper_factor
                .apply_right_in_place(target, partner_mode(self.symmetry, mode), inverse)
                .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))
        }
    }
    pub(crate) fn lower_right_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> Result<IndexedRightUpdate<Item>, SampledCycleError> {
        if let Some(lower) = &self.lower_factor {
            Ok(lower.right_update(target, mode, inverse))
        } else {
            self.upper_factor
                .indexed_right_update(target, partner_mode(self.symmetry, mode), inverse)
                .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))
        }
    }
    pub(crate) fn upper_left_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> Result<IndexedLeftUpdate<Item>, SampledCycleError> {
        self.upper_factor
            .indexed_left_update(target, mode, inverse)
            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))
    }
    pub(crate) fn upper_right_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> Result<IndexedRightUpdate<Item>, SampledCycleError> {
        self.upper_factor
            .indexed_right_update(target, mode, inverse)
            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))
    }
}

fn partner_mode(symmetry: MatrixSymmetry, mode: FactorMode) -> FactorMode {
    match (symmetry, mode) {
        (MatrixSymmetry::Symmetric, FactorMode::NoTrans) => FactorMode::Transpose,
        (MatrixSymmetry::Symmetric, FactorMode::Transpose) => FactorMode::NoTrans,
        (MatrixSymmetry::Symmetric, FactorMode::Adjoint) => FactorMode::Conjugate,
        (MatrixSymmetry::Symmetric, FactorMode::Conjugate) => FactorMode::Adjoint,
        (MatrixSymmetry::Hermitian, FactorMode::NoTrans) => FactorMode::Adjoint,
        (MatrixSymmetry::Hermitian, FactorMode::Adjoint) => FactorMode::NoTrans,
        (MatrixSymmetry::Hermitian, FactorMode::Transpose) => FactorMode::Conjugate,
        (MatrixSymmetry::Hermitian, FactorMode::Conjugate) => FactorMode::Transpose,
        (MatrixSymmetry::General, _) => unreachable!("general matrices store a lower factor"),
    }
}

fn positions(indices: &[usize]) -> HashMap<usize, usize> {
    indices
        .iter()
        .copied()
        .enumerate()
        .map(|(local, global)| (global, local))
        .collect()
}

struct BlockPositions<'a> {
    target: &'a HashMap<usize, usize>,
    near: &'a HashMap<usize, usize>,
}

fn select_columns_by_global<Item: Lapack>(
    source: &DynArray<Item, 2>,
    source_positions: &HashMap<usize, usize>,
    columns: &[usize],
) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape([source.shape()[0], columns.len()]);
    for row in 0..source.shape()[0] {
        for (column, global) in columns.iter().enumerate() {
            result[[row, column]] = source[[row, source_positions[global]]];
        }
    }
    result
}

/// Extract a block after `E_ID^-1 A E_ID^-T` from A[target,near].
fn transformed_block<Item: Lapack>(
    target_by_near: &DynArray<Item, 2>,
    block_positions: &BlockPositions<'_>,
    skeletons: &[usize],
    redundant: &[usize],
    interpolation: &DynArray<Item, 2>,
    rows: &[usize],
    columns: &[usize],
) -> DynArray<Item, 2> {
    let redundant_position = positions(redundant);
    let mut result = DynArray::<Item, 2>::from_shape([rows.len(), columns.len()]);
    for (local_row, &row) in rows.iter().enumerate() {
        let row_target = block_positions.target[&row];
        let row_redundant = redundant_position.get(&row).copied();
        for (local_column, &column) in columns.iter().enumerate() {
            let column_near = block_positions.near[&column];
            let mut value = target_by_near[[row_target, column_near]];
            if let Some(r) = row_redundant {
                for (s, &skeleton) in skeletons.iter().enumerate() {
                    value -= interpolation[[r, s]]
                        * target_by_near[[block_positions.target[&skeleton], column_near]];
                }
            }
            if let Some(&column_redundant) = redundant_position.get(&column) {
                for (s, &skeleton) in skeletons.iter().enumerate() {
                    let skeleton_column = block_positions.near[&skeleton];
                    let mut transformed_row_skeleton =
                        target_by_near[[row_target, skeleton_column]];
                    if let Some(r) = row_redundant {
                        for (q, &row_skeleton) in skeletons.iter().enumerate() {
                            transformed_row_skeleton -= interpolation[[r, q]]
                                * target_by_near
                                    [[block_positions.target[&row_skeleton], skeleton_column]];
                        }
                    }
                    value -= transformed_row_skeleton * interpolation[[column_redundant, s]];
                }
            }
            result[[local_row, local_column]] = value;
        }
    }
    result
}

fn copy_matrix<Item: Lapack>(source: &DynArray<Item, 2>) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape(source.shape());
    result.fill_from(source);
    result
}

fn transpose_copy<Item: Lapack>(source: &DynArray<Item, 2>) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape([source.shape()[1], source.shape()[0]]);
    for row in 0..source.shape()[0] {
        for column in 0..source.shape()[1] {
            result[[column, row]] = source[[row, column]];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dense_cycle::DenseSkeletonizationCycle,
        sampled_id::{InMemorySampleProvider, SampleBlock, ZMode},
    };
    use rlst::{base_types::c64, RlstScalar};

    #[test]
    fn sampled_cycle_matches_dense_srs_cycle() {
        let n = 10;
        let target = vec![0, 1, 2, 3];
        let near = vec![0, 1, 2, 3, 4, 5];
        let far = vec![6, 7, 8, 9];
        let mut matrix = DynArray::<c64, 2>::from_shape([n, n]);
        for row in 0..n {
            for column in 0..n {
                matrix[[row, column]] = c64::new(
                    if row == column { 5.0 } else { 0.0 }
                        + ((3 * row + 2 * column + 1) % 11) as f64 / 19.0,
                    ((5 * row + column + 2) % 7) as f64 / 23.0,
                );
            }
        }
        let coefficients = [
            [c64::new(1.0, 0.0), c64::new(0.0, 0.0)],
            [c64::new(0.0, 0.0), c64::new(1.0, 0.0)],
            [c64::new(0.4, 0.2), c64::new(-0.3, 0.1)],
            [c64::new(-0.2, 0.3), c64::new(0.7, -0.4)],
        ];
        for (local_far, &global_far) in far.iter().enumerate() {
            let column_basis = [
                c64::new((local_far + 1) as f64 / 5.0, (local_far + 2) as f64 / 17.0),
                c64::new((2 * local_far + 3) as f64 / 7.0, -(local_far as f64) / 13.0),
            ];
            let row_basis = [
                c64::new((local_far + 2) as f64 / 6.0, -(local_far as f64) / 11.0),
                c64::new(
                    (3 * local_far + 1) as f64 / 8.0,
                    (local_far + 1) as f64 / 14.0,
                ),
            ];
            for candidate in 0..target.len() {
                matrix[[global_far, target[candidate]]] = column_basis[0]
                    * coefficients[candidate][0]
                    + column_basis[1] * coefficients[candidate][1];
                matrix[[target[candidate], global_far]] = row_basis[0] * coefficients[candidate][0]
                    + row_basis[1] * coefficients[candidate][1];
            }
        }
        // Identity tests make near and far sample subspaces orthogonal. They
        // provide an exact, deterministic equivalence test for the two paths.
        let mut test = DynArray::<c64, 2>::from_shape([n, n]);
        let sample_coordinates = far.iter().chain(near.iter()).copied().collect::<Vec<_>>();
        for (sample, &coordinate) in sample_coordinates.iter().enumerate() {
            test[[sample, coordinate]] = c64::new(1.0, 0.0);
        }
        let mut y_sketch = DynArray::<c64, 2>::from_shape([n, n]);
        let mut z_sketch = DynArray::<c64, 2>::from_shape([n, n]);
        for (sample, &source) in sample_coordinates.iter().enumerate() {
            for coordinate in 0..n {
                y_sketch[[sample, coordinate]] = matrix[[coordinate, source]];
                z_sketch[[sample, coordinate]] = matrix[[source, coordinate]];
            }
        }
        let provider = InMemorySampleProvider::new(
            SampleBlock {
                y_test: copy_matrix(&test),
                y_sketch,
                z_test: Some(test),
                z_sketch: Some(z_sketch),
            },
            ZMode::Transpose,
        )
        .unwrap();
        let (_, dense_transformed) =
            DenseSkeletonizationCycle::factorize(&matrix, &target, &near, &far, 2, IdMethod::Rrqr)
                .unwrap();
        let regularization = Regularization {
            mode: crate::regularized_least_squares::RegularizationMode::Scaled,
            relative: 1.0e-15,
            absolute_floor: 0.0,
        };
        for pivot_method in [
            PivotMethod::DirectInverse,
            PivotMethod::Lu,
            PivotMethod::LuHybrid,
        ] {
            let extraction = ExtractionOptions {
                solver: RegularizedLeastSquaresMethod::Cholesky,
                regularization,
            };
            let sampled = SampledSkeletonizationCycle::factorize_ranked_with_options(
                &provider,
                n,
                &target,
                &near,
                IdRankSelection::Fixed(2),
                IdMethod::Rrqr,
                SampledFactorizationOptions {
                    nullification: NullificationMethod::Projection {
                        solver: RegularizedLeastSquaresMethod::Cholesky,
                        regularization,
                    },
                    near_extraction: extraction,
                    diagonal_extraction: extraction,
                    near_pivot: PivotOptions {
                        method: pivot_method,
                        shift: 0.0,
                    },
                    diagonal_pivot: PivotOptions {
                        method: pivot_method,
                        shift: 0.0,
                    },
                },
                MatrixSymmetry::General,
            )
            .unwrap();
            let sampled_transformed = sampled.transform_dense(&matrix).unwrap();
            let mut updated_samples = provider.read_block(0, n).unwrap();
            sampled.update_samples(&mut updated_samples).unwrap();
            for sample in 0..n {
                for coordinate in 0..n {
                    let mut expected_y = c64::new(0.0, 0.0);
                    let mut expected_z = c64::new(0.0, 0.0);
                    for source in 0..n {
                        expected_y += updated_samples.y_test[[sample, source]]
                            * sampled_transformed[[coordinate, source]];
                        expected_z += updated_samples.z_test.as_ref().unwrap()[[sample, source]]
                            * sampled_transformed[[source, coordinate]];
                    }
                    assert!(
                        (updated_samples.y_sketch[[sample, coordinate]] - expected_y).norm()
                            < 3.0e-10,
                        "Y sample mismatch for {pivot_method:?}"
                    );
                    assert!(
                        (updated_samples.z_sketch.as_ref().unwrap()[[sample, coordinate]]
                            - expected_z)
                            .norm()
                            < 3.0e-10,
                        "Z sample mismatch for {pivot_method:?}"
                    );
                }
            }
            let mut max_error = 0.0_f64;
            for row in 0..n {
                for column in 0..n {
                    max_error = max_error.max(RlstScalar::abs(
                        sampled_transformed[[row, column]] - dense_transformed[[row, column]],
                    ));
                }
            }
            assert!(
                max_error < 2.0e-10,
                "sampled/dense cycle error for {pivot_method:?}: {max_error:e}"
            );
            let reconstructed = sampled.reconstruct(&sampled_transformed).unwrap();
            for row in 0..n {
                for column in 0..n {
                    assert!(
                        RlstScalar::abs(reconstructed[[row, column]] - matrix[[row, column]])
                            < 2.0e-10,
                        "reconstruction mismatch for {pivot_method:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn symmetric_cycle_stores_only_one_elimination_factor() {
        let n = 8;
        let target = vec![0, 1, 2, 3];
        let near = vec![0, 1, 2, 3, 4, 5];
        let mut matrix = DynArray::<c64, 2>::from_shape([n, n]);
        for row in 0..n {
            for column in row..n {
                let value = c64::new(
                    if row == column { 7.0 } else { 0.0 }
                        + ((3 * row + 5 * column + 1) % 13) as f64 / 17.0,
                    ((row + 2 * column + 3) % 11) as f64 / 29.0,
                );
                matrix[[row, column]] = value;
                matrix[[column, row]] = value;
            }
        }

        let mut y_test = DynArray::<c64, 2>::from_shape([n, n]);
        let mut y_sketch = DynArray::<c64, 2>::from_shape([n, n]);
        for sample in 0..n {
            y_test[[sample, sample]] = c64::new(1.0, 0.0);
            for coordinate in 0..n {
                y_sketch[[sample, coordinate]] = matrix[[coordinate, sample]];
            }
        }
        let provider = InMemorySampleProvider::new(
            SampleBlock {
                y_test,
                y_sketch,
                z_test: None,
                z_sketch: None,
            },
            ZMode::Transpose,
        )
        .unwrap();
        let cycle = SampledSkeletonizationCycle::factorize_ranked_with_symmetry(
            &provider,
            n,
            &target,
            &near,
            IdRankSelection::Fixed(2),
            IdMethod::Rrqr,
            Regularization::none(),
            RegularizedLeastSquaresMethod::Lu,
            MatrixSymmetry::Symmetric,
        )
        .unwrap();

        assert!(cycle.lower_factor.is_none());
        let transformed = cycle.transform_dense(&matrix).unwrap();
        let reconstructed = cycle.reconstruct(&transformed).unwrap();
        for row in 0..n {
            for column in 0..n {
                assert!(
                    RlstScalar::abs(reconstructed[[row, column]] - matrix[[row, column]]) < 2.0e-10
                );
            }
        }
    }
}
