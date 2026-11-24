//! Geometry-facing octree scheduling for RSRS.

use bempp_octree::{morton::MortonKey, Octree};
use mpi::traits::CommunicatorCollectives;
use rsrs_core::configuration::{ParentRankPolicy, RankSelection};
use rsrs_core::scheduling::{
    build_fixed_rank_plan_with_budget_options, build_fixed_rank_plan_with_options,
    build_tolerance_plan_with_options, BatchingStrategy, FactorizationPlan, FixedRankSamplingMode,
    HierarchyBox, HierarchyLevel,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Build the exact geometry hierarchy used by bempp-octree, while keeping the
/// numerical core independent from MPI and the octree implementation.
pub fn fixed_rank_plan_from_octree<C: CommunicatorCollectives>(
    tree: &Octree<'_, C>,
    minimum_level: usize,
    rank: usize,
    oversampling: usize,
) -> FactorizationPlan {
    fixed_rank_plan_from_octree_with_options(
        tree,
        minimum_level,
        rank,
        oversampling,
        FixedRankSamplingMode::PerLevel,
        BatchingStrategy::HistoricalActiveNearFields,
    )
}

pub fn fixed_rank_plan_from_octree_with_options<C: CommunicatorCollectives>(
    tree: &Octree<'_, C>,
    minimum_level: usize,
    rank: usize,
    oversampling: usize,
    sampling_mode: FixedRankSamplingMode,
    batching_strategy: BatchingStrategy,
) -> FactorizationPlan {
    let levels = hierarchy_from_octree(tree, minimum_level);
    build_fixed_rank_plan_with_options(
        tree.local_number_of_points(),
        &levels,
        rank,
        oversampling,
        sampling_mode,
        batching_strategy,
    )
}

pub fn fixed_rank_plan_from_octree_with_budget_options<C: CommunicatorCollectives>(
    tree: &Octree<'_, C>,
    minimum_level: usize,
    rank: usize,
    oversampling: usize,
    minimum_rows: usize,
    sampling_mode: FixedRankSamplingMode,
    batching_strategy: BatchingStrategy,
) -> FactorizationPlan {
    let levels = hierarchy_from_octree(tree, minimum_level);
    build_fixed_rank_plan_with_budget_options(
        tree.local_number_of_points(),
        &levels,
        rank,
        oversampling,
        minimum_rows,
        sampling_mode,
        batching_strategy,
    )
}

pub fn tolerance_plan_from_octree_with_options<C: CommunicatorCollectives>(
    tree: &Octree<'_, C>,
    minimum_level: usize,
    rank_selection: RankSelection,
    parent_rank_policy: ParentRankPolicy,
    batching_strategy: BatchingStrategy,
) -> FactorizationPlan {
    let levels = hierarchy_from_octree(tree, minimum_level);
    build_tolerance_plan_with_options(
        tree.local_number_of_points(),
        &levels,
        rank_selection,
        parent_rank_policy,
        batching_strategy,
    )
}

pub fn hierarchy_from_octree<C: CommunicatorCollectives>(
    tree: &Octree<'_, C>,
    minimum_level: usize,
) -> Vec<HierarchyLevel> {
    let maximum_level = tree.global_max_level();
    let mut box_points = HashMap::new();
    for (&leaf, local_indices) in tree.leaf_keys_to_local_point_indices() {
        // RSRS factors act on the operator's local row/column ordering.  A
        // Point::global_id is metadata and is not required to be unique (the
        // historical BIEGrid constructors set every global id to zero).
        // Converting octree-local point positions to global ids here therefore
        // collapses coordinates and creates the wrong hierarchy.
        box_points.insert(leaf, local_indices.clone());
    }
    let neighbour_map = tree.neighbour_map();
    let mut level_keys = tree.leaf_keys().iter().copied().collect::<BTreeSet<_>>();
    let mut current_level = maximum_level;
    let mut levels = Vec::new();
    while current_level > minimum_level {
        let keys = level_keys.iter().copied().collect::<Vec<_>>();
        let key_ids = keys
            .iter()
            .enumerate()
            .map(|(id, &key)| (key, id))
            .collect::<HashMap<_, _>>();
        let next_keys = next_level_keys(&level_keys, current_level, maximum_level);
        let next_key_ids = next_keys
            .iter()
            .copied()
            .enumerate()
            .map(|(id, key)| (key, id))
            .collect::<HashMap<_, _>>();
        let boxes = keys
            .iter()
            .enumerate()
            .map(|(id, &key)| {
                let mut near_box_ids = neighbour_map
                    .get(&key)
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|candidate| {
                        current_level == maximum_level || candidate.level() == current_level
                    })
                    .filter_map(|candidate| key_ids.get(&candidate).copied())
                    .collect::<Vec<_>>();
                near_box_ids.push(id);
                near_box_ids.sort_unstable();
                near_box_ids.dedup();
                let mut indices = box_points.get(&key).cloned().unwrap_or_default();
                indices.sort_unstable();
                indices.dedup();
                let parent_id = next_key_ids
                    .get(&key.parent())
                    .or_else(|| next_key_ids.get(&key))
                    .copied();
                HierarchyBox {
                    id,
                    indices,
                    near_box_ids,
                    parent_id,
                    launch: current_level < maximum_level || key.level() == maximum_level,
                }
            })
            .collect();
        levels.push(HierarchyLevel {
            level: current_level,
            boxes,
        });

        let mut next_box_points = BTreeMap::<MortonKey, Vec<usize>>::new();
        for key in &keys {
            let destination = if next_key_ids.contains_key(&key.parent()) {
                key.parent()
            } else if next_key_ids.contains_key(key) {
                *key
            } else {
                continue;
            };
            next_box_points
                .entry(destination)
                .or_default()
                .extend(box_points.get(key).into_iter().flatten().copied());
        }
        box_points = next_box_points.into_iter().collect();
        level_keys = next_keys;
        current_level -= 1;
    }
    levels
}

/// Reproduce the historical mixed-depth key transition exactly. Coarse leaves
/// are carried unchanged for one level while maximum-depth leaves move to
/// their parent; keys that do not belong to the next level are then removed.
fn next_level_keys(
    level_keys: &BTreeSet<MortonKey>,
    current_level: usize,
    maximum_level: usize,
) -> BTreeSet<MortonKey> {
    level_keys
        .iter()
        .map(|&key| {
            if current_level == maximum_level {
                if key.level() == maximum_level {
                    key.parent()
                } else {
                    key
                }
            } else if key.level() == current_level - 1 {
                key
            } else {
                key.parent()
            }
        })
        .filter(|key| current_level == maximum_level || key.level() == current_level - 1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpi::topology::SimpleCommunicator;

    #[test]
    fn regular_plan_has_leaf_to_root_order_and_covers_points() {
        let points = (0..9)
            .flat_map(|y| {
                (0..9).map(move |x| {
                    bempp_octree::Point::new([x as f64 / 8.0, y as f64 / 8.0, 0.0], 0)
                })
            })
            .collect::<Vec<_>>();
        let universe = mpi::initialize().expect("MPI is not already initialized");
        let communicator = SimpleCommunicator::self_comm();
        let tree = Octree::new(&points, 5, 8, &communicator);
        let plan = fixed_rank_plan_from_octree(&tree, 1, 2, 4);
        assert_eq!(plan.dimension, 81);
        assert!(plan
            .levels
            .windows(2)
            .all(|pair| pair[0].level > pair[1].level));
        assert!(plan
            .levels
            .iter()
            .flat_map(|level| &level.batches)
            .flat_map(|batch| &batch.cycles)
            .all(|cycle| !cycle.target_indices.is_empty()));
        let mut covered = plan
            .levels
            .first()
            .unwrap()
            .batches
            .iter()
            .flat_map(|batch| &batch.cycles)
            .flat_map(|cycle| cycle.target_indices.iter().copied())
            .collect::<Vec<_>>();
        covered.sort_unstable();
        covered.dedup();
        assert!(covered.iter().all(|&index| index < points.len()));
        assert!(
            covered.len() > 1,
            "duplicate global ids must not collapse points"
        );

        let mut adaptive_points = (0..128)
            .map(|index| {
                let x = (index % 8) as f64 * 1.0e-4;
                let y = ((index / 8) % 8) as f64 * 1.0e-4;
                let z = (index / 64) as f64 * 1.0e-4;
                bempp_octree::Point::new([x, y, z], index)
            })
            .collect::<Vec<_>>();
        for (index, coordinate) in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        .into_iter()
        .enumerate()
        {
            adaptive_points.push(bempp_octree::Point::new(coordinate, 128 + index));
        }
        let adaptive_tree = Octree::new(&adaptive_points, 6, 8, &communicator);
        let maximum_level = adaptive_tree.global_max_level();
        assert!(adaptive_tree
            .leaf_keys()
            .iter()
            .any(|key| key.level() < maximum_level));
        let adaptive_plan = fixed_rank_plan_from_octree(&adaptive_tree, 1, 2, 8);
        let deepest_cycles = adaptive_plan.levels[0]
            .batches
            .iter()
            .flat_map(|batch| &batch.cycles)
            .count();
        let deep_occupied_leaves = adaptive_tree
            .leaf_keys_to_local_point_indices()
            .iter()
            .filter(|(key, indices)| key.level() == maximum_level && indices.len() > 2)
            .count();
        assert_eq!(deepest_cycles, deep_occupied_leaves);
        drop(universe);
    }
}
