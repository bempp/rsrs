//! Serial multilevel RSRS driven by a mutable sample state.

use crate::{
    configuration::{MatrixSymmetry, ParentRankPolicy, PivotMethod, RankSelection},
    diagonal_block::{IndexedDiagonalBlock, SquareBlock},
    elementary_factor::{commit_left_update, commit_right_update, FactorMode},
    interpolative_decomposition::IdMethod,
    regularized_least_squares::{Regularization, RegularizedLeastSquaresMethod},
    reporting::{BatchReport, CycleReport, FactorMemoryReport, FactorizationReport, LevelReport},
    sampled_cycle::{SampledCycleError, SampledSkeletonizationCycle},
    sampled_id::{
        extract_sampled_near_blocks, extract_sampled_near_blocks_with_symmetry, SampleBlock,
        SampleError, SampleProvider, ZMode,
    },
    sampled_options::{
        ExtractionOptions, NullificationMethod, PivotOptions, SampledFactorizationOptions,
    },
    scheduling::FactorizationPlan,
};
use num::Zero;
use rlst::{
    dense::{array::DynArray, linalg::lapack::interpolative_decomposition::MatrixId},
    Gemm, Lapack,
};
use std::collections::HashMap;
use std::time::Instant;

// RLST/LAPACK kernels used by the historical implementation require a larger
// worker stack for the largest local extraction problems. Keep the same stack
// budget in the joint path rather than relying on Rayon's small default.
const RSRS_THREAD_STACK_BYTES: usize = 128 * 1024 * 1024;

fn configured_rank<Item: MatrixId>(
    selection: RankSelection,
) -> crate::interpolative_decomposition::IdRankSelection<Item::Real> {
    use crate::interpolative_decomposition::IdRankSelection;
    match selection {
        RankSelection::Fixed { rank } => IdRankSelection::Fixed(rank),
        RankSelection::Tolerance { tolerance } => IdRankSelection::Tolerance(Item::real(tolerance)),
        RankSelection::ToleranceWithMinimumRank {
            tolerance,
            minimum_rank,
        } => IdRankSelection::ToleranceWithMinimumRank {
            tolerance: Item::real(tolerance),
            minimum_rank,
        },
        RankSelection::ToleranceWithMaximumRank {
            tolerance,
            maximum_rank,
        } => IdRankSelection::ToleranceWithMaximumRank {
            tolerance: Item::real(tolerance),
            maximum_rank,
        },
    }
}

fn inherited_rank(policy: ParentRankPolicy, ranks: &[usize]) -> Option<usize> {
    if ranks.is_empty() || policy == ParentRankPolicy::Tolerance {
        return None;
    }
    let minimum = *ranks.iter().min()?;
    let maximum = *ranks.iter().max()?;
    match policy {
        ParentRankPolicy::Minimum => Some(minimum),
        ParentRankPolicy::DoubleMinimum => Some(minimum.saturating_mul(2)),
        ParentRankPolicy::Maximum => Some(maximum),
        ParentRankPolicy::Average => Some(ranks.iter().sum::<usize>() / ranks.len()),
        ParentRankPolicy::Midpoint => Some(minimum + (maximum - minimum) / 2),
        ParentRankPolicy::Tolerance => None,
    }
}

#[derive(Debug, Clone)]
pub struct SampledCyclePlan {
    pub box_id: usize,
    pub parent_id: Option<usize>,
    pub target_indices: Vec<usize>,
    pub near_indices: Vec<usize>,
    pub rank: usize,
    /// Prefix used by this box in per-level fixed-rank mode. `None` selects
    /// the full active prefix (the historical constant-sample mode).
    pub sample_rows: Option<usize>,
}

pub struct ExecutedSampledCycle<Item: MatrixId + Lapack> {
    pub level: usize,
    pub batch: usize,
    pub cycle: SampledSkeletonizationCycle<Item>,
}

/// Multilevel sampled RSRS. Only O(s n) test/sketch storage is retained; the
/// dense operator is never materialized.
pub struct SampledMultilevelFactorization<Item: MatrixId + Lapack> {
    dimension: usize,
    cycles: Vec<ExecutedSampledCycle<Item>>,
    active_indices: Vec<usize>,
    diagonal: Vec<IndexedDiagonalBlock<Item>>,
    apply_pool: rayon::ThreadPool,
}

#[derive(Clone, Copy)]
enum CycleFactor {
    Id,
    Lower,
    Upper,
}

fn sample_block_bytes<Item: MatrixId + Lapack>(samples: &SampleBlock<Item>) -> u64 {
    let matrix_bytes = |matrix: &DynArray<Item, 2>| {
        (matrix.shape()[0] * matrix.shape()[1] * std::mem::size_of::<Item>()) as u64
    };
    matrix_bytes(&samples.y_test)
        + matrix_bytes(&samples.y_sketch)
        + samples.z_test.as_ref().map_or(0, matrix_bytes)
        + samples.z_sketch.as_ref().map_or(0, matrix_bytes)
}

fn factor_memory<Item: MatrixId + Lapack + Gemm>(
    cycles: &[ExecutedSampledCycle<Item>],
    diagonal: &[IndexedDiagonalBlock<Item>],
) -> FactorMemoryReport
where
    Item::Real: Into<Item>,
{
    let (id_bytes, elimination_bytes) = cycles.iter().fold((0_u64, 0_u64), |sum, executed| {
        let current = executed.cycle.storage_bytes();
        (sum.0 + current.0, sum.1 + current.1)
    });
    let diagonal_bytes = diagonal
        .iter()
        .map(IndexedDiagonalBlock::storage_bytes)
        .sum::<u64>();
    FactorMemoryReport {
        total_bytes: id_bytes + elimination_bytes + diagonal_bytes,
        id_bytes,
        elimination_bytes,
        diagonal_bytes,
        id_count: cycles.len(),
        elimination_count: cycles.len(),
        diagonal_count: diagonal.len(),
    }
}

impl<Item: MatrixId + Lapack + Gemm> SampledMultilevelFactorization<Item>
where
    Item::Real: Into<Item>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn factorize<Provider: SampleProvider<Item>>(
        provider: &Provider,
        sample_rows: usize,
        levels: &[Vec<SampledCyclePlan>],
        id_method: IdMethod<Item::Real>,
        regularization: Regularization<Item::Real>,
        least_squares: RegularizedLeastSquaresMethod,
    ) -> Result<Self, SampledCycleError> {
        let required = levels
            .iter()
            .flatten()
            .map(|plan| plan.near_indices.len() + plan.rank)
            .max()
            .unwrap_or(0);
        Self::factorize_with_report(
            provider,
            sample_rows,
            required,
            levels,
            id_method,
            regularization,
            least_squares,
        )
        .map(|(factorization, _)| factorization)
    }

    /// Factorize and return a stable report suitable for JSON serialization.
    #[allow(clippy::too_many_arguments)]
    pub fn factorize_with_report<Provider: SampleProvider<Item>>(
        provider: &Provider,
        sample_rows: usize,
        required_sample_rows: usize,
        levels: &[Vec<SampledCyclePlan>],
        id_method: IdMethod<Item::Real>,
        regularization: Regularization<Item::Real>,
        least_squares: RegularizedLeastSquaresMethod,
    ) -> Result<(Self, FactorizationReport), SampledCycleError> {
        let total_start = Instant::now();
        if provider.available_rows() < sample_rows {
            return Err(SampleError::InsufficientRows {
                requested: sample_rows,
                available: provider.available_rows(),
            }
            .into());
        }
        if sample_rows < required_sample_rows {
            return Err(SampleError::InsufficientRows {
                requested: required_sample_rows,
                available: sample_rows,
            }
            .into());
        }
        let loading_start = Instant::now();
        let mut samples = provider
            .read_block(0, sample_rows)
            .map_err(|error| SampleError::Provider(error.to_string()))?;
        samples.validate()?;
        // Internally Z always uses the canonical plain-transpose convention.
        if provider.z_mode() == ZMode::Adjoint {
            if let Some(z_test) = samples.z_test.as_mut() {
                conjugate_in_place(z_test);
            }
            if let Some(z_sketch) = samples.z_sketch.as_mut() {
                conjugate_in_place(z_sketch);
            }
        }
        let loading_seconds = loading_start.elapsed().as_secs_f64();
        let dimension = provider.dimension();
        let mut active = vec![true; dimension];
        let mut cycles = Vec::new();
        let mut diagonal = Vec::new();
        let mut cycle_reports = Vec::new();
        let batch_reports = Vec::new();
        let mut level_reports = Vec::new();

        for (level, plans) in levels.iter().enumerate() {
            let level_start = Instant::now();
            let active_before = active.iter().filter(|&&value| value).count();
            let cycles_before = cycles.len();
            for (batch, plan) in plans.iter().enumerate() {
                let target = filter_active(&plan.target_indices, &active);
                if target.len() <= plan.rank || plan.rank == 0 {
                    continue;
                }
                let mut near = filter_active(&plan.near_indices, &active);
                for &index in &target {
                    if !near.contains(&index) {
                        near.push(index);
                    }
                }
                let cycle_start = Instant::now();
                let state = WorkingSampleProvider { block: &samples };
                let cycle = SampledSkeletonizationCycle::factorize(
                    &state,
                    sample_rows,
                    &target,
                    &near,
                    plan.rank.min(target.len() - 1),
                    id_method,
                    regularization,
                    least_squares,
                )?;
                let skeleton_size = cycle.skeleton_indices().len();
                let redundant = cycle.redundant_indices().to_vec();
                let redundant_size = redundant.len();
                diagonal.push(
                    IndexedDiagonalBlock::new(
                        dimension,
                        redundant.clone(),
                        SquareBlock::lu(cycle.diagonal_block())
                            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?,
                    )
                    .map_err(SampledCycleError::InvalidPlan)?,
                );
                cycle.update_samples(&mut samples)?;
                for &index in &redundant {
                    active[index] = false;
                }
                cycles.push(ExecutedSampledCycle {
                    level,
                    batch,
                    cycle,
                });
                cycle_reports.push(CycleReport {
                    level,
                    batch,
                    target_size: target.len(),
                    near_size: near.len(),
                    skeleton_size,
                    redundant_size,
                    elapsed_seconds: cycle_start.elapsed().as_secs_f64(),
                });
            }
            let level_cycles = &cycle_reports[cycles_before..];
            let ranks = level_cycles
                .iter()
                .map(|report| report.skeleton_size)
                .collect::<Vec<_>>();
            let memory = factor_memory(&cycles, &diagonal);
            level_reports.push(LevelReport {
                level,
                boxes: cycles.len() - cycles_before,
                batches: plans.len(),
                active_dofs_before: active_before,
                active_dofs_after: active.iter().filter(|&&value| value).count(),
                elapsed_seconds: level_start.elapsed().as_secs_f64(),
                maximum_sample_rows_used: sample_rows,
                minimum_rank: ranks.iter().copied().min().unwrap_or(0),
                maximum_rank: ranks.iter().copied().max().unwrap_or(0),
                mean_rank: if ranks.is_empty() {
                    0.0
                } else {
                    ranks.iter().sum::<usize>() as f64 / ranks.len() as f64
                },
                retained_factor_bytes: memory.total_bytes,
                retained_id_bytes: memory.id_bytes,
                retained_elimination_bytes: memory.elimination_bytes,
                retained_diagonal_bytes: memory.diagonal_bytes,
            });
        }
        let active_indices = active
            .iter()
            .enumerate()
            .filter_map(|(index, &is_active)| is_active.then_some(index))
            .collect::<Vec<_>>();
        if !active_indices.is_empty() {
            let state = WorkingSampleProvider { block: &samples };
            let blocks = extract_sampled_near_blocks(
                &state,
                sample_rows,
                &active_indices,
                &active_indices,
                regularization,
                least_squares,
            )?;
            let mut final_block = blocks.target_by_near;
            // Average the independent Y and Z estimates of the final block.
            let half = Item::real(0.5).into();
            for row in 0..active_indices.len() {
                for column in 0..active_indices.len() {
                    final_block[[row, column]] =
                        half * (final_block[[row, column]] + blocks.near_by_target[[row, column]]);
                }
            }
            diagonal.push(
                IndexedDiagonalBlock::new(
                    dimension,
                    active_indices.clone(),
                    SquareBlock::lu(&final_block)
                        .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?,
                )
                .map_err(SampledCycleError::InvalidPlan)?,
            );
        }
        let apply_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .stack_size(RSRS_THREAD_STACK_BYTES)
            .build()
            .map_err(|error| SampledCycleError::InvalidPlan(error.to_string()))?;
        let sample_buffer_bytes = sample_block_bytes(&samples);
        let factor_memory = factor_memory(&cycles, &diagonal);
        let factorization = Self {
            dimension,
            cycles,
            active_indices,
            diagonal,
            apply_pool,
        };
        let report = FactorizationReport {
            dimension,
            available_sample_rows: provider.available_rows(),
            used_sample_rows: sample_rows,
            maximum_sample_rows_used: sample_rows,
            required_sample_rows,
            loading_seconds,
            factorization_seconds: total_start.elapsed().as_secs_f64(),
            final_active_dofs: factorization.active_indices.len(),
            sample_buffer_bytes,
            factor_memory,
            cycles: cycle_reports,
            batches: batch_reports,
            levels: level_reports,
        };
        Ok((factorization, report))
    }

    /// Production fixed-rank path. Conflict-free cycles in one planned batch
    /// are computed concurrently from the same immutable sample state; their
    /// disjoint sample updates are then committed in deterministic order.
    #[allow(clippy::too_many_arguments)]
    pub fn factorize_planned_with_report<Provider: SampleProvider<Item> + Sync>(
        provider: &Provider,
        sample_rows: usize,
        plan: &FactorizationPlan,
        id_method: IdMethod<Item::Real>,
        regularization: Regularization<Item::Real>,
        least_squares: RegularizedLeastSquaresMethod,
        rayon_threads: usize,
    ) -> Result<(Self, FactorizationReport), SampledCycleError>
    where
        Item: Send + Sync,
        Item::Real: Send + Sync,
    {
        Self::factorize_planned_with_report_and_symmetry(
            provider,
            sample_rows,
            plan,
            id_method,
            regularization,
            least_squares,
            rayon_threads,
            MatrixSymmetry::General,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn factorize_planned_with_report_and_symmetry<Provider: SampleProvider<Item> + Sync>(
        provider: &Provider,
        sample_rows: usize,
        plan: &FactorizationPlan,
        id_method: IdMethod<Item::Real>,
        regularization: Regularization<Item::Real>,
        least_squares: RegularizedLeastSquaresMethod,
        rayon_threads: usize,
        symmetry: MatrixSymmetry,
    ) -> Result<(Self, FactorizationReport), SampledCycleError>
    where
        Item: Send + Sync,
        Item::Real: Send + Sync,
    {
        let extraction = ExtractionOptions {
            solver: least_squares,
            regularization,
        };
        Self::factorize_planned_with_report_and_options(
            provider,
            sample_rows,
            plan,
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
            rayon_threads,
            symmetry,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn factorize_planned_with_report_and_options<Provider: SampleProvider<Item> + Sync>(
        provider: &Provider,
        sample_rows: usize,
        plan: &FactorizationPlan,
        id_method: IdMethod<Item::Real>,
        options: SampledFactorizationOptions<Item>,
        rayon_threads: usize,
        symmetry: MatrixSymmetry,
    ) -> Result<(Self, FactorizationReport), SampledCycleError>
    where
        Item: Send + Sync,
        Item::Real: Send + Sync,
    {
        use rayon::prelude::*;

        if provider.dimension() != plan.dimension {
            return Err(SampledCycleError::InvalidPlan(format!(
                "factorization plan dimension {} differs from provider dimension {}",
                plan.dimension,
                provider.dimension()
            )));
        }
        if provider.available_rows() < sample_rows {
            return Err(SampleError::InsufficientRows {
                requested: sample_rows,
                available: provider.available_rows(),
            }
            .into());
        }
        if sample_rows < plan.required_sample_rows {
            return Err(SampleError::InsufficientRows {
                requested: plan.required_sample_rows,
                available: sample_rows,
            }
            .into());
        }
        let total_start = Instant::now();
        let loading_start = Instant::now();
        let mut samples = provider
            .read_block(0, sample_rows)
            .map_err(|error| SampleError::Provider(error.to_string()))?;
        samples.validate()?;
        if provider.z_mode() == ZMode::Adjoint {
            if let Some(z_test) = samples.z_test.as_mut() {
                conjugate_in_place(z_test);
            }
            if let Some(z_sketch) = samples.z_sketch.as_mut() {
                conjugate_in_place(z_sketch);
            }
        }
        let loading_seconds = loading_start.elapsed().as_secs_f64();
        let dimension = provider.dimension();
        let mut active = vec![true; dimension];
        let mut cycles = Vec::new();
        let mut diagonal = Vec::new();
        let mut cycle_reports = Vec::new();
        let mut batch_reports = Vec::new();
        let mut level_reports = Vec::new();
        let mut inherited_ranks = HashMap::<usize, Vec<usize>>::new();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(rayon_threads.max(1))
            .stack_size(RSRS_THREAD_STACK_BYTES)
            .build()
            .map_err(|error| SampledCycleError::InvalidPlan(error.to_string()))?;

        for planned_level in &plan.levels {
            let level_start = Instant::now();
            let active_before = active.iter().filter(|&&value| value).count();
            let cycles_before = cycles.len();
            let mut next_inherited_ranks = HashMap::<usize, Vec<usize>>::new();
            for (batch_index, batch) in planned_level.batches.iter().enumerate() {
                let batch_started = Instant::now();
                let active_batch_before = active.iter().filter(|&&value| value).count();
                let tasks = batch
                    .cycles
                    .iter()
                    .filter_map(|cycle_plan| {
                        let target = filter_active(&cycle_plan.target_indices, &active);
                        if target.len() < 2 {
                            return None;
                        }
                        let selected_rank = inherited_ranks
                            .get(&cycle_plan.box_id)
                            .and_then(|ranks| inherited_rank(plan.parent_rank_policy, ranks));
                        let rank_selection = selected_rank
                            .map(crate::interpolative_decomposition::IdRankSelection::Fixed)
                            .unwrap_or_else(|| configured_rank::<Item>(plan.rank_selection));
                        if let crate::interpolative_decomposition::IdRankSelection::Fixed(rank) =
                            rank_selection
                        {
                            if rank == 0 || target.len() <= rank {
                                return None;
                            }
                        }
                        let mut near = filter_active(&cycle_plan.near_indices, &active);
                        near.extend(target.iter().copied());
                        near.sort_unstable();
                        near.dedup();
                        Some((
                            target,
                            near,
                            rank_selection,
                            cycle_plan
                                .sample_rows
                                .unwrap_or(sample_rows)
                                .min(sample_rows),
                            cycle_plan.parent_id,
                        ))
                    })
                    .collect::<Vec<_>>();
                let max_sample_rows = tasks
                    .iter()
                    .map(|(_, _, _, local_sample_rows, _)| *local_sample_rows)
                    .max()
                    .unwrap_or(0);
                let state = WorkingSampleProvider { block: &samples };
                let id_started = Instant::now();
                let prepared = pool.install(|| {
                    tasks
                        .par_iter()
                        .map(
                            |(target, near, rank_selection, local_sample_rows, parent_id)| {
                                let started = Instant::now();
                                let prepared =
                                    match SampledSkeletonizationCycle::prepare_ranked_with_options(
                                        &state,
                                        *local_sample_rows,
                                        target,
                                        near,
                                        *rank_selection,
                                        id_method,
                                        options,
                                        symmetry,
                                    ) {
                                        Ok(prepared) => prepared,
                                        Err(SampledCycleError::NotCompressible { .. }) => {
                                            return Ok(None);
                                        }
                                        Err(error) => return Err(error),
                                    };
                                Ok::<_, SampledCycleError>(Some((
                                    prepared,
                                    near.clone(),
                                    *local_sample_rows,
                                    target.len(),
                                    near.len(),
                                    started.elapsed().as_secs_f64(),
                                    *parent_id,
                                )))
                            },
                        )
                        .collect::<Result<Vec<_>, _>>()
                })?;
                let prepared = prepared.into_iter().flatten().collect::<Vec<_>>();
                let id_seconds = id_started.elapsed().as_secs_f64();
                // Historical joint factorization commits the whole ID batch
                // before it extracts any near block for the LU stage.
                let id_update_started = Instant::now();
                for (prepared_cycle, _, _, _, _, _, _) in &prepared {
                    prepared_cycle.update_samples_id(&mut samples)?;
                }
                let id_update_seconds = id_update_started.elapsed().as_secs_f64();
                let state = WorkingSampleProvider { block: &samples };
                let near_extraction_started = Instant::now();
                let computed = pool.install(|| {
                    prepared
                        .into_par_iter()
                        .map(
                            |(
                                prepared_cycle,
                                near,
                                local_sample_rows,
                                target_size,
                                near_size,
                                id_seconds,
                                parent_id,
                            )| {
                                let started = Instant::now();
                                let cycle = SampledSkeletonizationCycle::finish_after_id(
                                    &state,
                                    local_sample_rows,
                                    prepared_cycle,
                                    &near,
                                    options,
                                )?;
                                Ok::<_, SampledCycleError>((
                                    cycle,
                                    target_size,
                                    near_size,
                                    id_seconds + started.elapsed().as_secs_f64(),
                                    parent_id,
                                ))
                            },
                        )
                        .collect::<Result<Vec<_>, _>>()
                })?;
                let near_extraction_seconds = near_extraction_started.elapsed().as_secs_f64();
                // Keep the batch intact through the LU stage, matching the
                // historical joint cycle. Every LU factor below observes the
                // same post-ID sample state; no LU update or active-set change
                // is visible while another factor in this batch is extracted.
                let mut completed_batch = Vec::with_capacity(computed.len());
                for (cycle, target_size, near_size, elapsed_seconds, parent_id) in computed {
                    let skeleton_size = cycle.skeleton_indices().len();
                    let redundant = cycle.redundant_indices().to_vec();
                    let redundant_size = redundant.len();
                    diagonal.push(
                        IndexedDiagonalBlock::new(
                            dimension,
                            redundant.clone(),
                            SquareBlock::from_pivot_method(
                                cycle.diagonal_block(),
                                options.diagonal_pivot.method,
                                options.diagonal_pivot.shift,
                            )
                            .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?,
                        )
                        .map_err(SampledCycleError::InvalidPlan)?,
                    );
                    cycle_reports.push(CycleReport {
                        level: planned_level.level,
                        batch: batch_index,
                        target_size,
                        near_size,
                        skeleton_size,
                        redundant_size,
                        elapsed_seconds,
                    });
                    completed_batch.push((cycle, redundant, parent_id));
                }

                // Historical `joint_id_and_lu` forms the complete commutative
                // LU batch, applies that batch to the sketches, and only then
                // removes its redundant DOFs from all subsequent near fields.
                // Applying the independent indexed factors in deterministic
                // order is algebraically the same batch application.
                let lu_update_started = Instant::now();
                for (cycle, _, _) in &completed_batch {
                    cycle.update_samples_lu(&mut samples)?;
                }
                let lu_update_seconds = lu_update_started.elapsed().as_secs_f64();
                let completed_cycles = completed_batch.len();
                for (_, redundant, _) in &completed_batch {
                    for &index in redundant {
                        debug_assert!(active[index], "a batch eliminated one DOF twice");
                        active[index] = false;
                    }
                }
                let active_batch_after = active.iter().filter(|&&value| value).count();
                for (cycle, _, parent_id) in completed_batch {
                    if let Some(parent_id) = parent_id {
                        next_inherited_ranks
                            .entry(parent_id)
                            .or_default()
                            .push(cycle.skeleton_indices().len());
                    }
                    cycles.push(ExecutedSampledCycle {
                        level: planned_level.level,
                        batch: batch_index,
                        cycle,
                    });
                }
                batch_reports.push(BatchReport {
                    level: planned_level.level,
                    batch: batch_index,
                    cycles: completed_cycles,
                    active_dofs_before: active_batch_before,
                    active_dofs_after: active_batch_after,
                    eliminated_dofs: active_batch_before - active_batch_after,
                    max_sample_rows,
                    id_seconds,
                    id_update_seconds,
                    near_extraction_seconds,
                    lu_update_seconds,
                    elapsed_seconds: batch_started.elapsed().as_secs_f64(),
                });
            }
            inherited_ranks = next_inherited_ranks;
            let level_cycles = &cycle_reports[cycles_before..];
            let ranks = level_cycles
                .iter()
                .map(|report| report.skeleton_size)
                .collect::<Vec<_>>();
            let maximum_sample_rows_used = batch_reports
                .iter()
                .filter(|report| report.level == planned_level.level)
                .map(|report| report.max_sample_rows)
                .max()
                .unwrap_or(0);
            let memory = factor_memory(&cycles, &diagonal);
            level_reports.push(LevelReport {
                level: planned_level.level,
                boxes: cycles.len() - cycles_before,
                batches: planned_level.batches.len(),
                active_dofs_before: active_before,
                active_dofs_after: active.iter().filter(|&&value| value).count(),
                elapsed_seconds: level_start.elapsed().as_secs_f64(),
                maximum_sample_rows_used,
                minimum_rank: ranks.iter().copied().min().unwrap_or(0),
                maximum_rank: ranks.iter().copied().max().unwrap_or(0),
                mean_rank: if ranks.is_empty() {
                    0.0
                } else {
                    ranks.iter().sum::<usize>() as f64 / ranks.len() as f64
                },
                retained_factor_bytes: memory.total_bytes,
                retained_id_bytes: memory.id_bytes,
                retained_elimination_bytes: memory.elimination_bytes,
                retained_diagonal_bytes: memory.diagonal_bytes,
            });
        }

        let active_indices = active
            .iter()
            .enumerate()
            .filter_map(|(index, &is_active)| is_active.then_some(index))
            .collect::<Vec<_>>();
        if !active_indices.is_empty() {
            let state = WorkingSampleProvider { block: &samples };
            let blocks = extract_sampled_near_blocks_with_symmetry(
                &state,
                sample_rows,
                &active_indices,
                &active_indices,
                options.diagonal_extraction.regularization,
                options.diagonal_extraction.solver,
                symmetry,
            )?;
            let mut final_block = blocks.target_by_near;
            let half = Item::real(0.5).into();
            for row in 0..active_indices.len() {
                for column in 0..active_indices.len() {
                    final_block[[row, column]] =
                        half * (final_block[[row, column]] + blocks.near_by_target[[row, column]]);
                }
            }
            diagonal.push(
                IndexedDiagonalBlock::new(
                    dimension,
                    active_indices.clone(),
                    SquareBlock::from_pivot_method(
                        &final_block,
                        options.diagonal_pivot.method,
                        options.diagonal_pivot.shift,
                    )
                    .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))?,
                )
                .map_err(SampledCycleError::InvalidPlan)?,
            );
        }
        let maximum_sample_rows_used = batch_reports
            .iter()
            .map(|report| report.max_sample_rows)
            .max()
            .unwrap_or(0);
        let sample_buffer_bytes = sample_block_bytes(&samples);
        let factor_memory = factor_memory(&cycles, &diagonal);
        let factorization = Self {
            dimension,
            cycles,
            active_indices,
            diagonal,
            apply_pool: pool,
        };
        let report = FactorizationReport {
            dimension,
            available_sample_rows: provider.available_rows(),
            used_sample_rows: sample_rows,
            maximum_sample_rows_used,
            required_sample_rows: plan.required_sample_rows,
            loading_seconds,
            factorization_seconds: total_start.elapsed().as_secs_f64(),
            final_active_dofs: factorization.active_indices.len(),
            sample_buffer_bytes,
            factor_memory,
            cycles: cycle_reports,
            batches: batch_reports,
            levels: level_reports,
        };
        Ok((factorization, report))
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn cycles(&self) -> &[ExecutedSampledCycle<Item>] {
        &self.cycles
    }
    pub fn active_indices(&self) -> &[usize] {
        &self.active_indices
    }

    /// Ranges of consecutive cycles belonging to one historical joint batch.
    /// Batch boundaries are part of the factorization: all IDs in a batch are
    /// applied before any LU factor from that batch.
    fn joint_batch_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::new();
        let mut first = 0;
        while first < self.cycles.len() {
            let level = self.cycles[first].level;
            let batch = self.cycles[first].batch;
            let mut end = first + 1;
            while end < self.cycles.len()
                && self.cycles[end].level == level
                && self.cycles[end].batch == batch
            {
                end += 1;
            }
            ranges.push(first..end);
            first = end;
        }
        ranges
    }

    /// Apply the factorized approximation or its inverse to one or more RHSs.
    pub fn apply(
        &self,
        input: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> Result<DynArray<Item, 2>, SampledCycleError>
    where
        Item: Send + Sync,
    {
        assert_eq!(input.shape()[0], self.dimension);
        match mode {
            FactorMode::NoTrans => self.apply_left(input, inverse),
            FactorMode::Transpose | FactorMode::Adjoint => {
                let conjugate = mode == FactorMode::Adjoint;
                let mut rows = transpose_copy(input, conjugate);
                self.apply_right(&mut rows, inverse)?;
                Ok(transpose_copy(&rows, conjugate))
            }
            FactorMode::Conjugate => {
                let mut conjugated = copy_matrix(input);
                conjugate_in_place(&mut conjugated);
                let mut result = self.apply_left(&conjugated, inverse)?;
                conjugate_in_place(&mut result);
                Ok(result)
            }
        }
    }

    fn apply_left(
        &self,
        input: &DynArray<Item, 2>,
        inverse: bool,
    ) -> Result<DynArray<Item, 2>, SampledCycleError>
    where
        Item: Send + Sync,
    {
        let mut result = copy_matrix(input);
        let batches = self.joint_batch_ranges();
        if !inverse {
            for batch in &batches {
                self.apply_left_batch(
                    &mut result,
                    batch.clone(),
                    CycleFactor::Id,
                    FactorMode::Transpose,
                    false,
                )?;
                self.apply_left_batch(
                    &mut result,
                    batch.clone(),
                    CycleFactor::Upper,
                    FactorMode::NoTrans,
                    false,
                )?;
            }
            self.apply_diagonal_left(&mut result, FactorMode::NoTrans, false)?;
            for batch in batches.iter().rev() {
                self.apply_left_batch(
                    &mut result,
                    batch.clone(),
                    CycleFactor::Lower,
                    FactorMode::NoTrans,
                    false,
                )?;
                self.apply_left_batch(
                    &mut result,
                    batch.clone(),
                    CycleFactor::Id,
                    FactorMode::NoTrans,
                    false,
                )?;
            }
        } else {
            for batch in &batches {
                self.apply_left_batch(
                    &mut result,
                    batch.clone(),
                    CycleFactor::Id,
                    FactorMode::NoTrans,
                    true,
                )?;
                self.apply_left_batch(
                    &mut result,
                    batch.clone(),
                    CycleFactor::Lower,
                    FactorMode::NoTrans,
                    true,
                )?;
            }
            self.apply_diagonal_left(&mut result, FactorMode::NoTrans, true)?;
            for batch in batches.iter().rev() {
                self.apply_left_batch(
                    &mut result,
                    batch.clone(),
                    CycleFactor::Upper,
                    FactorMode::NoTrans,
                    true,
                )?;
                self.apply_left_batch(
                    &mut result,
                    batch.clone(),
                    CycleFactor::Id,
                    FactorMode::Transpose,
                    true,
                )?;
            }
        }
        Ok(result)
    }

    fn apply_left_batch(
        &self,
        target: &mut DynArray<Item, 2>,
        batch: std::ops::Range<usize>,
        factor: CycleFactor,
        mode: FactorMode,
        inverse: bool,
    ) -> Result<(), SampledCycleError>
    where
        Item: Send + Sync,
    {
        use rayon::prelude::*;
        let input: &DynArray<Item, 2> = target;
        let updates = self.apply_pool.install(|| {
            self.cycles[batch]
                .par_iter()
                .map(|executed| match factor {
                    CycleFactor::Id => Ok(executed.cycle.id_left_update(input, mode, inverse)),
                    CycleFactor::Lower => executed.cycle.lower_left_update(input, mode, inverse),
                    CycleFactor::Upper => executed.cycle.upper_left_update(input, mode, inverse),
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        for update in updates {
            commit_left_update(target, update);
        }
        Ok(())
    }

    fn apply_diagonal_left(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> Result<(), SampledCycleError>
    where
        Item: Send + Sync,
    {
        use rayon::prelude::*;
        let input: &DynArray<Item, 2> = target;
        let updates = self.apply_pool.install(|| {
            self.diagonal
                .par_iter()
                .map(|block| {
                    block
                        .left_update(input, mode, inverse)
                        .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        for update in updates {
            commit_left_update(target, update);
        }
        Ok(())
    }

    fn apply_right(
        &self,
        result: &mut DynArray<Item, 2>,
        inverse: bool,
    ) -> Result<(), SampledCycleError>
    where
        Item: Send + Sync,
    {
        let batches = self.joint_batch_ranges();
        if !inverse {
            for batch in &batches {
                self.apply_right_batch(
                    result,
                    batch.clone(),
                    CycleFactor::Id,
                    FactorMode::NoTrans,
                    false,
                )?;
                self.apply_right_batch(
                    result,
                    batch.clone(),
                    CycleFactor::Lower,
                    FactorMode::NoTrans,
                    false,
                )?;
            }
            self.apply_diagonal_right(result, FactorMode::NoTrans, false)?;
            for batch in batches.iter().rev() {
                self.apply_right_batch(
                    result,
                    batch.clone(),
                    CycleFactor::Upper,
                    FactorMode::NoTrans,
                    false,
                )?;
                self.apply_right_batch(
                    result,
                    batch.clone(),
                    CycleFactor::Id,
                    FactorMode::Transpose,
                    false,
                )?;
            }
        } else {
            for batch in &batches {
                self.apply_right_batch(
                    result,
                    batch.clone(),
                    CycleFactor::Id,
                    FactorMode::Transpose,
                    true,
                )?;
                self.apply_right_batch(
                    result,
                    batch.clone(),
                    CycleFactor::Upper,
                    FactorMode::NoTrans,
                    true,
                )?;
            }
            self.apply_diagonal_right(result, FactorMode::NoTrans, true)?;
            for batch in batches.iter().rev() {
                self.apply_right_batch(
                    result,
                    batch.clone(),
                    CycleFactor::Lower,
                    FactorMode::NoTrans,
                    true,
                )?;
                self.apply_right_batch(
                    result,
                    batch.clone(),
                    CycleFactor::Id,
                    FactorMode::NoTrans,
                    true,
                )?;
            }
        }
        Ok(())
    }

    fn apply_right_batch(
        &self,
        target: &mut DynArray<Item, 2>,
        batch: std::ops::Range<usize>,
        factor: CycleFactor,
        mode: FactorMode,
        inverse: bool,
    ) -> Result<(), SampledCycleError>
    where
        Item: Send + Sync,
    {
        use rayon::prelude::*;
        let input: &DynArray<Item, 2> = target;
        let updates = self.apply_pool.install(|| {
            self.cycles[batch]
                .par_iter()
                .map(|executed| match factor {
                    CycleFactor::Id => Ok(executed.cycle.id_right_update(input, mode, inverse)),
                    CycleFactor::Lower => executed.cycle.lower_right_update(input, mode, inverse),
                    CycleFactor::Upper => executed.cycle.upper_right_update(input, mode, inverse),
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        for update in updates {
            commit_right_update(target, update);
        }
        Ok(())
    }

    fn apply_diagonal_right(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> Result<(), SampledCycleError>
    where
        Item: Send + Sync,
    {
        use rayon::prelude::*;
        let input: &DynArray<Item, 2> = target;
        let updates = self.apply_pool.install(|| {
            self.diagonal
                .par_iter()
                .map(|block| {
                    block
                        .right_update(input, mode, inverse)
                        .map_err(|error| SampledCycleError::LinearAlgebra(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        for update in updates {
            commit_right_update(target, update);
        }
        Ok(())
    }
}

struct WorkingSampleProvider<'a, Item: MatrixId + Lapack> {
    block: &'a SampleBlock<Item>,
}

impl<Item: MatrixId + Lapack> SampleProvider<Item> for WorkingSampleProvider<'_, Item> {
    type Error = SampleError;
    fn available_rows(&self) -> usize {
        self.block.rows()
    }
    fn dimension(&self) -> usize {
        self.block.dimension()
    }
    fn z_mode(&self) -> ZMode {
        ZMode::Transpose
    }
    fn read_block(&self, first: usize, rows: usize) -> Result<SampleBlock<Item>, Self::Error> {
        if first.saturating_add(rows) > self.block.rows() {
            return Err(SampleError::InsufficientRows {
                requested: first.saturating_add(rows),
                available: self.block.rows(),
            });
        }
        let copy = |source: &DynArray<Item, 2>| {
            let mut result = DynArray::<Item, 2>::from_shape([rows, self.dimension()]);
            for row in 0..rows {
                for column in 0..self.dimension() {
                    result[[row, column]] = source[[first + row, column]];
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
        Some(self.block)
    }
}

fn filter_active(indices: &[usize], active: &[bool]) -> Vec<usize> {
    indices
        .iter()
        .copied()
        .filter(|&index| index < active.len() && active[index])
        .collect()
}
fn conjugate_in_place<Item: Lapack>(matrix: &mut DynArray<Item, 2>) {
    for value in matrix
        .data_mut()
        .expect("owned dense arrays are contiguous")
    {
        *value = value.conj();
    }
}
fn copy_matrix<Item: Lapack>(source: &DynArray<Item, 2>) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape(source.shape());
    result.fill_from(source);
    result
}
fn transpose_copy<Item: Lapack>(source: &DynArray<Item, 2>, conjugate: bool) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape([source.shape()[1], source.shape()[0]]);
    for row in 0..source.shape()[0] {
        for column in 0..source.shape()[1] {
            let value = source[[row, column]];
            result[[column, row]] = if conjugate { value.conj() } else { value };
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        sampled_id::InMemorySampleProvider,
        scheduling::{FactorizationPlan, PlannedBatch, PlannedLevel},
    };
    use rlst::{base_types::c64, RlstScalar};

    #[test]
    fn two_level_sample_state_remains_consistent_and_operator_inverts() {
        let n = 12;
        let mut matrix = DynArray::<c64, 2>::from_shape([n, n]);
        for row in 0..n {
            for column in 0..n {
                matrix[[row, column]] = c64::new(
                    if row == column { 10.0 } else { 0.0 }
                        + ((3 * row + 5 * column + 1) % 17) as f64 / 31.0,
                    ((7 * row + 2 * column + 3) % 13) as f64 / 37.0,
                );
            }
        }
        let mut test = DynArray::<c64, 2>::from_shape([n, n]);
        for index in 0..n {
            test[[index, index]] = c64::new(1.0, 0.0);
        }
        let mut y = DynArray::<c64, 2>::from_shape([n, n]);
        let mut z = DynArray::<c64, 2>::from_shape([n, n]);
        for sample in 0..n {
            for coordinate in 0..n {
                y[[sample, coordinate]] = matrix[[coordinate, sample]];
                z[[sample, coordinate]] = matrix[[sample, coordinate]];
            }
        }
        let provider = InMemorySampleProvider::new(
            SampleBlock {
                y_test: copy_matrix(&test),
                y_sketch: y,
                z_test: Some(test),
                z_sketch: Some(z),
            },
            ZMode::Transpose,
        )
        .unwrap();
        let levels = vec![
            vec![SampledCyclePlan {
                box_id: 0,
                parent_id: None,
                target_indices: vec![0, 1, 2, 3],
                near_indices: vec![0, 1, 2, 3, 4, 5, 6, 7],
                rank: 2,
                sample_rows: None,
            }],
            vec![SampledCyclePlan {
                box_id: 0,
                parent_id: None,
                target_indices: vec![4, 5, 6, 7],
                near_indices: (0..10).collect(),
                rank: 2,
                sample_rows: None,
            }],
        ];
        let factors = SampledMultilevelFactorization::factorize(
            &provider,
            n,
            &levels,
            IdMethod::Rrqr,
            Regularization {
                mode: crate::regularized_least_squares::RegularizationMode::Scaled,
                relative: 1.0e-14,
                absolute_floor: 0.0,
            },
            RegularizedLeastSquaresMethod::Cholesky,
        )
        .unwrap();
        assert_eq!(factors.cycles().len(), 2);
        let mut rhs = DynArray::<c64, 2>::from_shape([n, 2]);
        for row in 0..n {
            for column in 0..2 {
                rhs[[row, column]] = c64::new(
                    (row + column + 1) as f64 / 9.0,
                    (2 * row + column) as f64 / 17.0,
                );
            }
        }
        for mode in [
            FactorMode::NoTrans,
            FactorMode::Transpose,
            FactorMode::Adjoint,
            FactorMode::Conjugate,
        ] {
            let applied = factors.apply(&rhs, mode, false).unwrap();
            let recovered = factors.apply(&applied, mode, true).unwrap();
            for row in 0..n {
                for column in 0..2 {
                    assert!(
                        RlstScalar::abs(recovered[[row, column]] - rhs[[row, column]]) < 2.0e-9
                    );
                }
            }
        }

        // The planned joint path must retire redundant DOFs only after the
        // complete LU batch and exclude them from the next batch's near field.
        let plan = FactorizationPlan {
            dimension: n,
            required_sample_rows: n,
            rank_selection: crate::configuration::RankSelection::Fixed { rank: 2 },
            parent_rank_policy: crate::configuration::ParentRankPolicy::Minimum,
            levels: levels
                .iter()
                .enumerate()
                .map(|(level, cycles)| PlannedLevel {
                    level,
                    batches: vec![PlannedBatch {
                        cycles: cycles.clone(),
                    }],
                })
                .collect(),
        };
        let regularization = Regularization {
            mode: crate::regularized_least_squares::RegularizationMode::Scaled,
            relative: 1.0e-14,
            absolute_floor: 0.0,
        };
        let extraction = ExtractionOptions {
            solver: RegularizedLeastSquaresMethod::Cholesky,
            regularization,
        };
        let options = SampledFactorizationOptions {
            nullification: NullificationMethod::Projection {
                solver: RegularizedLeastSquaresMethod::Cholesky,
                regularization,
            },
            near_extraction: extraction,
            diagonal_extraction: extraction,
            near_pivot: PivotOptions {
                method: PivotMethod::Lu,
                shift: 0.0,
            },
            diagonal_pivot: PivotOptions {
                method: PivotMethod::LuHybrid,
                shift: 0.0,
            },
        };
        let (serial_factors, report) =
            SampledMultilevelFactorization::factorize_planned_with_report_and_options(
                &provider,
                n,
                &plan,
                IdMethod::Rrqr,
                options,
                1,
                MatrixSymmetry::General,
            )
            .unwrap();
        let (parallel_factors, _) =
            SampledMultilevelFactorization::factorize_planned_with_report_and_options(
                &provider,
                n,
                &plan,
                IdMethod::Rrqr,
                options,
                4,
                MatrixSymmetry::General,
            )
            .unwrap();
        assert_eq!(report.batches.len(), 2);
        assert_eq!(report.batches[0].active_dofs_before, 12);
        assert_eq!(report.batches[0].active_dofs_after, 10);
        assert_eq!(report.batches[1].active_dofs_before, 10);
        assert_eq!(report.batches[1].active_dofs_after, 8);
        assert_eq!(report.cycles[1].near_size, 8);
        assert_eq!(report.maximum_sample_rows_used, n);
        assert_eq!(
            report.sample_buffer_bytes,
            (4 * n * n * std::mem::size_of::<c64>()) as u64
        );
        assert_eq!(
            report.factor_memory.total_bytes,
            report.factor_memory.id_bytes
                + report.factor_memory.elimination_bytes
                + report.factor_memory.diagonal_bytes
        );
        assert!(report.factor_memory.total_bytes > 0);
        assert_eq!(report.factor_memory.id_count, 2);
        assert_eq!(report.factor_memory.elimination_count, 2);
        assert!(report
            .levels
            .windows(2)
            .all(|levels| levels[0].retained_factor_bytes <= levels[1].retained_factor_bytes));

        for mode in [
            FactorMode::NoTrans,
            FactorMode::Transpose,
            FactorMode::Adjoint,
            FactorMode::Conjugate,
        ] {
            for inverse in [false, true] {
                let serial = serial_factors.apply(&rhs, mode, inverse).unwrap();
                let parallel = parallel_factors.apply(&rhs, mode, inverse).unwrap();
                for row in 0..n {
                    for column in 0..2 {
                        assert!(
                            RlstScalar::abs(serial[[row, column]] - parallel[[row, column]])
                                < 2.0e-12
                        );
                    }
                }
            }
        }

        // A tolerance-selected full-rank ID is a valid "do not eliminate"
        // decision, not a factorization error. The remaining block is handled
        // by the final diagonal factor.
        let mut tolerance_plan = plan.clone();
        tolerance_plan.required_sample_rows = 0;
        tolerance_plan.rank_selection =
            crate::configuration::RankSelection::Tolerance { tolerance: 1.0e-14 };
        for level in &mut tolerance_plan.levels {
            for batch in &mut level.batches {
                for cycle in &mut batch.cycles {
                    cycle.rank = 0;
                }
            }
        }
        let (tolerance_factors, tolerance_report) =
            SampledMultilevelFactorization::factorize_planned_with_report_and_options(
                &provider,
                n,
                &tolerance_plan,
                IdMethod::Rrqr,
                SampledFactorizationOptions {
                    nullification: NullificationMethod::Projection {
                        solver: RegularizedLeastSquaresMethod::Cholesky,
                        regularization,
                    },
                    near_extraction: extraction,
                    diagonal_extraction: extraction,
                    near_pivot: PivotOptions {
                        method: PivotMethod::Lu,
                        shift: 0.0,
                    },
                    diagonal_pivot: PivotOptions {
                        method: PivotMethod::Lu,
                        shift: 0.0,
                    },
                },
                1,
                MatrixSymmetry::General,
            )
            .unwrap();
        assert_eq!(tolerance_report.required_sample_rows, 0);
        let applied = tolerance_factors
            .apply(&rhs, FactorMode::NoTrans, false)
            .unwrap();
        let recovered = tolerance_factors
            .apply(&applied, FactorMode::NoTrans, true)
            .unwrap();
        for row in 0..n {
            for column in 0..2 {
                assert!(RlstScalar::abs(recovered[[row, column]] - rhs[[row, column]]) < 2.0e-9);
            }
        }
    }

    #[test]
    fn inherited_rank_policies_match_historical_definitions() {
        let ranks = [3, 7, 5];
        assert_eq!(inherited_rank(ParentRankPolicy::Minimum, &ranks), Some(3));
        assert_eq!(
            inherited_rank(ParentRankPolicy::DoubleMinimum, &ranks),
            Some(6)
        );
        assert_eq!(inherited_rank(ParentRankPolicy::Maximum, &ranks), Some(7));
        assert_eq!(inherited_rank(ParentRankPolicy::Average, &ranks), Some(5));
        assert_eq!(inherited_rank(ParentRankPolicy::Midpoint, &ranks), Some(5));
        assert_eq!(inherited_rank(ParentRankPolicy::Tolerance, &ranks), None);
        assert_eq!(inherited_rank(ParentRankPolicy::Minimum, &[]), None);
    }
}
