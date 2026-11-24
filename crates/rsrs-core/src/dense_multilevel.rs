use crate::{
    configuration::RankSelection, dense_cycle::DenseSkeletonizationCycle,
    interpolative_decomposition::IdMethod, scheduling::FactorizationPlan,
};
use rlst::{
    base_types::RlstResult,
    dense::{array::DynArray, linalg::lapack::interpolative_decomposition::MatrixId},
    Lapack,
};

#[derive(Debug, Clone)]
pub struct DenseCyclePlan {
    pub target_indices: Vec<usize>,
    pub near_indices: Vec<usize>,
    pub far_indices: Vec<usize>,
    pub rank: usize,
}

/// Convert the shared octree schedule into dense-SRS cycles. Dense SRS uses
/// exact far-field matrix entries, so the far set is simply the complement of
/// the planned near set. Tolerance-selected dense IDs will be added through
/// the same rank-selection abstraction; for now the explicit dense reference
/// deliberately accepts only fixed-rank plans.
pub fn dense_levels_from_plan(
    plan: &FactorizationPlan,
) -> Result<Vec<Vec<DenseCyclePlan>>, String> {
    if !matches!(plan.rank_selection, RankSelection::Fixed { .. }) {
        return Err("dense SRS currently requires a fixed-rank plan".into());
    }
    Ok(plan
        .levels
        .iter()
        .map(|level| {
            level
                .batches
                .iter()
                .flat_map(|batch| &batch.cycles)
                .map(|cycle| {
                    let near = cycle
                        .near_indices
                        .iter()
                        .copied()
                        .collect::<std::collections::HashSet<_>>();
                    DenseCyclePlan {
                        target_indices: cycle.target_indices.clone(),
                        near_indices: cycle.near_indices.clone(),
                        far_indices: (0..plan.dimension)
                            .filter(|index| !near.contains(index))
                            .collect(),
                        rank: cycle.rank,
                    }
                })
                .collect()
        })
        .collect())
}

pub struct ExecutedDenseCycle<Item: MatrixId + Lapack> {
    pub level: usize,
    pub batch: usize,
    pub cycle: DenseSkeletonizationCycle<Item>,
}

/// Serial multilevel reference factorization.
///
/// Levels and batches are retained explicitly even though execution is serial;
/// this is the correctness reference that the Rayon scheduler will later use.
pub struct DenseMultilevelFactorization<Item: MatrixId + Lapack> {
    dimension: usize,
    cycles: Vec<ExecutedDenseCycle<Item>>,
    active_indices: Vec<usize>,
}

impl<Item: MatrixId + Lapack> DenseMultilevelFactorization<Item> {
    pub fn factorize(
        matrix: &DynArray<Item, 2>,
        levels: &[Vec<DenseCyclePlan>],
        method: IdMethod<Item::Real>,
    ) -> RlstResult<(Self, DynArray<Item, 2>)> {
        let [rows, columns] = matrix.shape();
        assert_eq!(rows, columns);
        let dimension = rows;
        let mut active = vec![true; dimension];
        let mut transformed = copy_matrix(matrix);
        let mut cycles = Vec::new();

        for (level, plans) in levels.iter().enumerate() {
            for (batch, plan) in plans.iter().enumerate() {
                let target = filter_active(&plan.target_indices, &active);
                if target.len() <= plan.rank {
                    continue;
                }
                let mut near = filter_active(&plan.near_indices, &active);
                for &index in &target {
                    if !near.contains(&index) {
                        near.push(index);
                    }
                }
                let far = filter_active(&plan.far_indices, &active);
                if far.is_empty() {
                    continue;
                }
                let (cycle, next) = DenseSkeletonizationCycle::factorize(
                    &transformed,
                    &target,
                    &near,
                    &far,
                    plan.rank.min(target.len() - 1),
                    method,
                )?;
                for &index in cycle.redundant_indices() {
                    active[index] = false;
                }
                transformed = next;
                cycles.push(ExecutedDenseCycle {
                    level,
                    batch,
                    cycle,
                });
            }
        }
        let active_indices = active
            .iter()
            .enumerate()
            .filter_map(|(index, &is_active)| is_active.then_some(index))
            .collect();
        Ok((
            Self {
                dimension,
                cycles,
                active_indices,
            },
            transformed,
        ))
    }

    pub fn cycles(&self) -> &[ExecutedDenseCycle<Item>] {
        &self.cycles
    }
    pub fn active_indices(&self) -> &[usize] {
        &self.active_indices
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn reconstruct(&self, transformed: &DynArray<Item, 2>) -> RlstResult<DynArray<Item, 2>> {
        assert_eq!(transformed.shape(), [self.dimension, self.dimension]);
        let mut result = copy_matrix(transformed);
        for executed in self.cycles.iter().rev() {
            result = executed.cycle.reconstruct(&result)?;
        }
        Ok(result)
    }

    /// Drop the residual couplings between eliminated blocks and the rest of
    /// the transformed matrix. Reconstructing this matrix gives the actual
    /// dense reference approximation represented by the skeletonization.
    pub fn block_diagonal_approximation(
        &self,
        transformed: &DynArray<Item, 2>,
    ) -> DynArray<Item, 2> {
        assert_eq!(transformed.shape(), [self.dimension, self.dimension]);
        let mut result = copy_matrix(transformed);
        for executed in &self.cycles {
            let redundant = executed.cycle.redundant_indices();
            let mut in_block = vec![false; self.dimension];
            for &index in redundant {
                in_block[index] = true;
            }
            for &row in redundant {
                for column in 0..self.dimension {
                    if !in_block[column] {
                        result[[row, column]] = Item::zero();
                        result[[column, row]] = Item::zero();
                    }
                }
            }
        }
        result
    }

    pub fn approximate_matrix(
        &self,
        transformed: &DynArray<Item, 2>,
    ) -> RlstResult<DynArray<Item, 2>> {
        self.reconstruct(&self.block_diagonal_approximation(transformed))
    }
}

fn filter_active(indices: &[usize], active: &[bool]) -> Vec<usize> {
    indices
        .iter()
        .copied()
        .filter(|&index| index < active.len() && active[index])
        .collect()
}

fn copy_matrix<Item: Lapack>(source: &DynArray<Item, 2>) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape(source.shape());
    result.fill_from(source);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        configuration::ParentRankPolicy,
        sampled_multilevel::SampledCyclePlan,
        scheduling::{PlannedBatch, PlannedLevel},
    };
    use rlst::{base_types::c64, RlstScalar};

    fn check<Item, Make>(make: Make, tolerance: Item::Real)
    where
        Item: MatrixId + Lapack,
        Make: Fn(f64, f64) -> Item,
    {
        let n = 10;
        let mut matrix = DynArray::<Item, 2>::from_shape([n, n]);
        for row in 0..n {
            for column in 0..n {
                matrix[[row, column]] = make(
                    if row == column { 6.0 } else { 0.0 }
                        + ((row * 3 + column * 5 + 1) % 13) as f64 / 21.0,
                    ((row * 7 + column * 2 + 3) % 11) as f64 / 29.0,
                );
            }
        }
        let all = (0..n).collect::<Vec<_>>();
        let levels = vec![
            vec![DenseCyclePlan {
                target_indices: vec![0, 1, 2, 3],
                near_indices: all.clone(),
                far_indices: vec![6, 7, 8, 9],
                rank: 2,
            }],
            vec![DenseCyclePlan {
                target_indices: vec![4, 5, 6, 7],
                near_indices: all.clone(),
                far_indices: vec![0, 1, 8, 9],
                rank: 2,
            }],
        ];
        let original = copy_matrix(&matrix);
        let (factors, transformed) =
            DenseMultilevelFactorization::factorize(&matrix, &levels, IdMethod::Rrqr).unwrap();
        assert_eq!(factors.cycles().len(), 2);
        assert_eq!(factors.active_indices().len(), 6);
        assert_eq!(factors.cycles()[0].level, 0);
        assert_eq!(factors.cycles()[1].level, 1);
        let reconstructed = factors.reconstruct(&transformed).unwrap();
        for row in 0..n {
            for column in 0..n {
                assert!(
                    RlstScalar::abs(reconstructed[[row, column]] - original[[row, column]])
                        <= tolerance
                );
            }
        }
    }

    #[test]
    fn real_multilevel_reconstructs() {
        check::<f64, _>(|real, _| real, 3.0e-10);
    }

    #[test]
    fn complex_multilevel_reconstructs() {
        check::<c64, _>(c64::new, 5.0e-10);
    }

    #[test]
    fn shared_fixed_rank_plan_builds_dense_near_and_far_sets() {
        let plan = FactorizationPlan {
            dimension: 8,
            required_sample_rows: 0,
            rank_selection: RankSelection::Fixed { rank: 2 },
            parent_rank_policy: ParentRankPolicy::Minimum,
            levels: vec![PlannedLevel {
                level: 3,
                batches: vec![PlannedBatch {
                    cycles: vec![SampledCyclePlan {
                        box_id: 0,
                        parent_id: None,
                        target_indices: vec![0, 1, 2, 3],
                        near_indices: vec![0, 1, 2, 3, 4],
                        rank: 2,
                        sample_rows: None,
                    }],
                }],
            }],
        };
        let levels = dense_levels_from_plan(&plan).unwrap();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].len(), 1);
        assert_eq!(levels[0][0].near_indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(levels[0][0].far_indices, vec![5, 6, 7]);

        let mut tolerance = plan;
        tolerance.rank_selection = RankSelection::Tolerance { tolerance: 1.0e-6 };
        assert!(dense_levels_from_plan(&tolerance).is_err());
    }
}
