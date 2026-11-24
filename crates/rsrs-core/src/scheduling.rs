//! Provider-neutral hierarchical scheduling and sample-budget calculation.
//!
//! Geometry packages build [`HierarchyLevel`] values from their octree.  The
//! numerical core only needs the point indices in each box and the box-level
//! near-field graph. This keeps MPI and a particular octree implementation out
//! of the factorization kernel while preserving historical RSRS scheduling.

use crate::{
    configuration::{ParentRankPolicy, RankSelection},
    sampled_multilevel::SampledCyclePlan,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedRankSamplingMode {
    /// Historical mode: each box consumes the smallest safe prefix.
    PerLevel,
    /// Every box consumes the complete active sample prefix.
    Constant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchingStrategy {
    /// Historical greedy coloring on the graph of launched boxes only.
    HistoricalActiveNearFields,
    /// More conservative coloring requiring complete point near sets to be disjoint.
    ConservativeDisjointNearFields,
    /// Deterministic reference/debug mode with one cycle in each batch.
    Serial,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HierarchyBox {
    pub id: usize,
    pub indices: Vec<usize>,
    /// Box ids at the same level whose active points form the near field.
    pub near_box_ids: Vec<usize>,
    /// Box id at the next coarser planned level. This is explicit because an
    /// adaptive tree may carry a coarse leaf unchanged instead of mapping it
    /// to `indices[0]`'s geometric parent.
    #[serde(default)]
    pub parent_id: Option<usize>,
    /// Whether this box launches a skeletonization cycle at this level.
    /// Carried coarse leaves are present in near fields but do not launch at
    /// a deeper level than their own.
    #[serde(default = "default_launch")]
    pub launch: bool,
}

fn default_launch() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HierarchyLevel {
    pub level: usize,
    pub boxes: Vec<HierarchyBox>,
}

#[derive(Debug, Clone)]
pub struct PlannedBatch {
    pub cycles: Vec<SampledCyclePlan>,
}

#[derive(Debug, Clone)]
pub struct PlannedLevel {
    pub level: usize,
    pub batches: Vec<PlannedBatch>,
}

#[derive(Debug, Clone)]
pub struct FactorizationPlan {
    pub dimension: usize,
    pub levels: Vec<PlannedLevel>,
    pub required_sample_rows: usize,
    pub rank_selection: RankSelection,
    pub parent_rank_policy: ParentRankPolicy,
}

impl FactorizationPlan {
    pub fn flatten(&self) -> Vec<Vec<SampledCyclePlan>> {
        self.levels
            .iter()
            .map(|level| {
                level
                    .batches
                    .iter()
                    .flat_map(|batch| batch.cycles.iter().cloned())
                    .collect()
            })
            .collect()
    }

    pub fn from_cycle_levels(
        dimension: usize,
        levels: Vec<Vec<SampledCyclePlan>>,
        required_sample_rows: usize,
    ) -> Self {
        let maximum_rank = levels
            .iter()
            .flatten()
            .map(|cycle| cycle.rank)
            .max()
            .unwrap_or(0);
        let levels = levels
            .into_iter()
            .enumerate()
            .map(|(level, cycles)| PlannedLevel {
                level,
                batches: color_independent_cycles(cycles)
                    .into_iter()
                    .map(|cycles| PlannedBatch { cycles })
                    .collect(),
            })
            .collect();
        Self {
            dimension,
            levels,
            required_sample_rows,
            rank_selection: RankSelection::Fixed { rank: maximum_rank },
            parent_rank_policy: ParentRankPolicy::Minimum,
        }
    }
}

/// Build fixed-rank plans and greedily color boxes whose near fields overlap.
/// Cycles in one batch are independent and may be computed concurrently.
pub fn build_fixed_rank_plan(
    dimension: usize,
    levels: &[HierarchyLevel],
    rank: usize,
    oversampling: usize,
) -> FactorizationPlan {
    build_fixed_rank_plan_with_options(
        dimension,
        levels,
        rank,
        oversampling,
        FixedRankSamplingMode::PerLevel,
        BatchingStrategy::HistoricalActiveNearFields,
    )
}

pub fn build_fixed_rank_plan_with_options(
    dimension: usize,
    levels: &[HierarchyLevel],
    rank: usize,
    oversampling: usize,
    sampling_mode: FixedRankSamplingMode,
    batching_strategy: BatchingStrategy,
) -> FactorizationPlan {
    build_fixed_rank_plan_with_budget_options(
        dimension,
        levels,
        rank,
        oversampling,
        0,
        sampling_mode,
        batching_strategy,
    )
}

pub fn build_fixed_rank_plan_with_budget_options(
    dimension: usize,
    levels: &[HierarchyLevel],
    rank: usize,
    oversampling: usize,
    minimum_rows: usize,
    sampling_mode: FixedRankSamplingMode,
    batching_strategy: BatchingStrategy,
) -> FactorizationPlan {
    // Historical fixed-rank RSRS rounds the effective additive oversampling to
    // a multiple of five. Keeping that rule makes cache-size predictions and
    // loaded prefixes agree with the original implementation.
    let effective_oversampling = round_up_to_multiple_of_five(oversampling);
    let mut required_sample_rows = 0;
    let mut planned_levels = Vec::with_capacity(levels.len());
    let mut previous_post_counts: Vec<(Option<usize>, usize)> = Vec::new();
    for (level_index, level) in levels.iter().enumerate() {
        let active_counts = if level_index == 0 {
            level
                .boxes
                .iter()
                .map(|box_data| box_data.indices.len())
                .collect::<Vec<_>>()
        } else {
            let mut counts = vec![0usize; level.boxes.len()];
            for (parent_id, count) in &previous_post_counts {
                if let Some(parent_id) = parent_id {
                    if let Some(parent) = counts.get_mut(*parent_id) {
                        *parent = parent.saturating_add(*count);
                    }
                }
            }
            counts
        };
        let mut weighted_cycles = Vec::new();
        for (target_position, target_box) in level.boxes.iter().enumerate() {
            let target_active = active_counts[target_position];
            if !target_box.launch || target_active <= rank {
                continue;
            }
            let near_ids = target_box
                .near_box_ids
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            let mut near = level
                .boxes
                .iter()
                .filter(|candidate| {
                    candidate.id == target_box.id || near_ids.contains(&candidate.id)
                })
                .flat_map(|candidate| candidate.indices.iter().copied())
                .collect::<Vec<_>>();
            near.extend_from_slice(&target_box.indices);
            near.sort_unstable();
            near.dedup();
            let near_active = level
                .boxes
                .iter()
                .enumerate()
                .filter(|(_, candidate)| {
                    candidate.id == target_box.id || near_ids.contains(&candidate.id)
                })
                .map(|(position, _)| active_counts[position])
                .sum::<usize>();
            let box_sample_rows = near_active + rank.min(target_active) + effective_oversampling;
            required_sample_rows = required_sample_rows.max(box_sample_rows);
            weighted_cycles.push((
                target_active + near_active,
                target_box.id,
                target_box.near_box_ids.clone(),
                SampledCyclePlan {
                    box_id: target_box.id,
                    parent_id: target_box.parent_id,
                    target_indices: target_box.indices.clone(),
                    near_indices: near,
                    rank,
                    sample_rows: (sampling_mode == FixedRankSamplingMode::PerLevel)
                        .then_some(box_sample_rows),
                },
            ));
        }
        // Historical RSRS schedules the cheapest active boxes first. The
        // stable tie order is the octree's Morton order. Greedy coloring is
        // order-dependent, so preserving this ordering is required both for
        // matching its batch count and for matching which factors observe the
        // same pre-update samples.
        weighted_cycles.sort_by_key(|(work, _, _, _)| *work);
        let batches = match batching_strategy {
            BatchingStrategy::HistoricalActiveNearFields => {
                color_historical_octree_cycles(weighted_cycles)
            }
            BatchingStrategy::ConservativeDisjointNearFields => color_independent_cycles(
                weighted_cycles
                    .into_iter()
                    .map(|(_, _, _, cycle)| cycle)
                    .collect(),
            ),
            BatchingStrategy::Serial => weighted_cycles
                .into_iter()
                .map(|(_, _, _, cycle)| vec![cycle])
                .collect(),
        }
        .into_iter()
        .map(|cycles| PlannedBatch { cycles })
        .collect();
        planned_levels.push(PlannedLevel {
            level: level.level,
            batches,
        });
        previous_post_counts = level
            .boxes
            .iter()
            .zip(active_counts)
            .map(|(box_data, active)| {
                let post = if box_data.launch {
                    active.min(rank)
                } else {
                    active
                };
                (box_data.parent_id, post)
            })
            .collect();
    }
    let final_active = previous_post_counts
        .iter()
        .map(|(_, count)| *count)
        .sum::<usize>();
    required_sample_rows =
        required_sample_rows.max(final_active.saturating_add(effective_oversampling));
    if minimum_rows > required_sample_rows {
        required_sample_rows = minimum_rows;
        for level in &mut planned_levels {
            for batch in &mut level.batches {
                for cycle in &mut batch.cycles {
                    cycle.sample_rows = None;
                }
            }
        }
    }
    FactorizationPlan {
        dimension,
        levels: planned_levels,
        required_sample_rows,
        rank_selection: RankSelection::Fixed { rank },
        parent_rank_policy: ParentRankPolicy::Minimum,
    }
}

/// Build the geometry and batching part of a tolerance-selected plan.
///
/// Unlike fixed-rank planning, the exact sample requirement is data
/// dependent. The factorization therefore consumes the user-supplied prefix
/// and returns an insufficient-samples error if a local nullification or
/// extraction needs more rows. It never samples or retries internally.
pub fn build_tolerance_plan_with_options(
    dimension: usize,
    levels: &[HierarchyLevel],
    rank_selection: RankSelection,
    parent_rank_policy: ParentRankPolicy,
    batching_strategy: BatchingStrategy,
) -> FactorizationPlan {
    assert!(
        !matches!(rank_selection, RankSelection::Fixed { .. }),
        "tolerance planning requires a tolerance-selected rank"
    );
    let mut planned_levels = Vec::with_capacity(levels.len());
    for level in levels {
        let mut weighted_cycles = level
            .boxes
            .iter()
            .filter(|box_data| box_data.launch && box_data.indices.len() >= 2)
            .map(|target_box| {
                let near_ids = target_box
                    .near_box_ids
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>();
                let mut near = level
                    .boxes
                    .iter()
                    .filter(|candidate| {
                        candidate.id == target_box.id || near_ids.contains(&candidate.id)
                    })
                    .flat_map(|candidate| candidate.indices.iter().copied())
                    .collect::<Vec<_>>();
                near.extend_from_slice(&target_box.indices);
                near.sort_unstable();
                near.dedup();
                (
                    target_box.indices.len() + near.len(),
                    target_box.id,
                    target_box.near_box_ids.clone(),
                    SampledCyclePlan {
                        box_id: target_box.id,
                        parent_id: target_box.parent_id,
                        target_indices: target_box.indices.clone(),
                        near_indices: near,
                        // The planned factorization resolves the tolerance or
                        // inherited parent rank at execution time.
                        rank: 0,
                        sample_rows: None,
                    },
                )
            })
            .collect::<Vec<_>>();
        weighted_cycles.sort_by_key(|(work, _, _, _)| *work);
        let batches = match batching_strategy {
            BatchingStrategy::HistoricalActiveNearFields => {
                color_historical_octree_cycles(weighted_cycles)
            }
            BatchingStrategy::ConservativeDisjointNearFields => color_independent_cycles(
                weighted_cycles
                    .into_iter()
                    .map(|(_, _, _, cycle)| cycle)
                    .collect(),
            ),
            BatchingStrategy::Serial => weighted_cycles
                .into_iter()
                .map(|(_, _, _, cycle)| vec![cycle])
                .collect(),
        }
        .into_iter()
        .map(|cycles| PlannedBatch { cycles })
        .collect();
        planned_levels.push(PlannedLevel {
            level: level.level,
            batches,
        });
    }
    FactorizationPlan {
        dimension,
        levels: planned_levels,
        required_sample_rows: 0,
        rank_selection,
        parent_rank_policy,
    }
}

/// Reproduce historical RSRS batching on the active box graph.
///
/// Only boxes that actually launch a cycle participate in conflicts. A small
/// leaf that is skipped at fixed rank therefore cannot create an artificial
/// conflict between two launched boxes merely because their geometric near
/// fields both include it.
fn color_historical_octree_cycles(
    cycles: Vec<(usize, usize, Vec<usize>, SampledCyclePlan)>,
) -> Vec<Vec<SampledCyclePlan>> {
    let launched = cycles
        .iter()
        .map(|(_, id, _, _)| *id)
        .collect::<HashSet<_>>();
    let mut group_contents: Vec<HashSet<usize>> = Vec::new();
    let mut batches: Vec<Vec<SampledCyclePlan>> = Vec::new();
    'cycle: for (_, id, neighbours, cycle) in cycles {
        let mut closed_near = neighbours
            .into_iter()
            .filter(|candidate| launched.contains(candidate))
            .collect::<HashSet<_>>();
        closed_near.insert(id);
        for (group, batch) in group_contents.iter_mut().zip(batches.iter_mut()) {
            if closed_near.is_disjoint(group) {
                group.extend(closed_near);
                batch.push(cycle);
                continue 'cycle;
            }
        }
        group_contents.push(closed_near);
        batches.push(vec![cycle]);
    }
    batches
}

fn round_up_to_multiple_of_five(value: usize) -> usize {
    value.saturating_add(4) / 5 * 5
}

/// Deterministic greedy coloring of the near-field conflict graph.
///
/// Two cycles may share a batch only when their complete near fields are
/// disjoint. Their sample-column updates then commute, so the batch can be
/// evaluated in parallel from one immutable sample state and committed in a
/// deterministic order afterwards.
fn color_independent_cycles(cycles: Vec<SampledCyclePlan>) -> Vec<Vec<SampledCyclePlan>> {
    let mut batches: Vec<Vec<SampledCyclePlan>> = Vec::new();
    'cycle: for cycle in cycles {
        for batch in &mut batches {
            if batch.iter().all(|other| independent(&cycle, other)) {
                batch.push(cycle);
                continue 'cycle;
            }
        }
        batches.push(vec![cycle]);
    }
    batches
}

fn independent(left: &SampledCyclePlan, right: &SampledCyclePlan) -> bool {
    let left_near = left.near_indices.iter().copied().collect::<HashSet<_>>();
    !right
        .near_indices
        .iter()
        .any(|index| left_near.contains(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_budget_and_conflict_coloring_are_deterministic() {
        let level = HierarchyLevel {
            level: 2,
            boxes: vec![
                HierarchyBox {
                    id: 0,
                    indices: vec![0, 1, 2],
                    near_box_ids: vec![1],
                    parent_id: None,
                    launch: true,
                },
                HierarchyBox {
                    id: 1,
                    indices: vec![3, 4, 5],
                    near_box_ids: vec![0, 2],
                    parent_id: None,
                    launch: true,
                },
                HierarchyBox {
                    id: 2,
                    indices: vec![6, 7, 8],
                    near_box_ids: vec![1],
                    parent_id: None,
                    launch: true,
                },
            ],
        };
        let plan = build_fixed_rank_plan(9, &[level], 1, 2);
        assert_eq!(plan.required_sample_rows, 15);
        assert_eq!(plan.levels[0].batches.len(), 3);
        assert_eq!(plan.flatten()[0].len(), 3);
    }

    #[test]
    fn adaptive_parent_counts_preserve_carried_coarse_leaves() {
        let fine = HierarchyLevel {
            level: 3,
            boxes: vec![
                HierarchyBox {
                    id: 0,
                    indices: vec![0, 1, 2, 3],
                    near_box_ids: vec![0, 1],
                    parent_id: Some(0),
                    launch: true,
                },
                HierarchyBox {
                    id: 1,
                    indices: vec![4, 5, 6],
                    near_box_ids: vec![0, 1],
                    parent_id: Some(0),
                    launch: false,
                },
            ],
        };
        let coarse = HierarchyLevel {
            level: 2,
            boxes: vec![HierarchyBox {
                id: 0,
                indices: (0..7).collect(),
                near_box_ids: vec![0],
                parent_id: None,
                launch: true,
            }],
        };
        let plan = build_fixed_rank_plan(7, &[fine, coarse], 2, 3);
        assert_eq!(plan.required_sample_rows, 14);
        assert_eq!(plan.flatten()[0].len(), 1);
        assert_eq!(plan.flatten()[1].len(), 1);

        let floored = build_fixed_rank_plan_with_budget_options(
            7,
            &[HierarchyLevel {
                level: 3,
                boxes: vec![HierarchyBox {
                    id: 0,
                    indices: vec![0, 1, 2, 3],
                    near_box_ids: vec![0],
                    parent_id: None,
                    launch: true,
                }],
            }],
            2,
            3,
            30,
            FixedRankSamplingMode::PerLevel,
            BatchingStrategy::HistoricalActiveNearFields,
        );
        assert_eq!(floored.required_sample_rows, 30);
        assert!(floored
            .levels
            .iter()
            .flat_map(|level| &level.batches)
            .flat_map(|batch| &batch.cycles)
            .all(|cycle| cycle.sample_rows.is_none()));
    }
}
