use bempp_octree::{morton::MortonKey, octree::Octree};
use mpi::traits::CommunicatorCollectives;
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
pub struct TreeData {
    pub level_keys: HashSet<MortonKey>,
    pub boxes_map: HashMap<MortonKey, Vec<usize>>,
    pub max_level: usize,
    pub current_level: usize,
    neighbour_map: HashMap<MortonKey, Vec<MortonKey>>,
}

pub trait TreeIndexing: Sized {
    fn new<C: CommunicatorCollectives>(octree_data: &Octree<'_, C>) -> Self;
    //Returns indices of points in neighboring boxes

    fn update_level_keys(&mut self);

    fn next_level_keys(&mut self) -> HashSet<MortonKey>;

    fn get_box_near_field_keys(&self, box_key: &MortonKey, level: usize) -> HashSet<MortonKey>;
}

impl TreeIndexing for TreeData {
    fn new<C: CommunicatorCollectives>(octree_data: &Octree<'_, C>) -> Self {
        let neighbour_map: HashMap<MortonKey, Vec<MortonKey>> = octree_data.neighbour_map().clone();
        let boxes_map: HashMap<MortonKey, Vec<usize>> =
            octree_data.leaf_keys_to_local_point_indices().clone();
        let leaf_tree_keys = octree_data.leaf_keys().iter().cloned();
        let max_level = octree_data.global_max_level();
        let level_keys = leaf_tree_keys.collect::<HashSet<_>>();
        let current_level = max_level;
        Self {
            level_keys,
            boxes_map,
            neighbour_map,
            max_level: octree_data.global_max_level(),
            current_level,
        }
    }

    fn update_level_keys(&mut self) {
        self.level_keys = self.next_level_keys();
        self.current_level -= 1;
    }

    fn next_level_keys(&mut self) -> HashSet<MortonKey> {
        let next_level_keys = self
            .level_keys
            .iter()
            .map(|&key| {
                if self.current_level == self.max_level {
                    if key.level() == self.max_level {
                        key.parent()
                    } else {
                        key
                    }
                } else if key.level() == self.current_level - 1 {
                    key
                } else {
                    key.parent()
                }
            })
            .filter(|key| {
                self.current_level == self.max_level || key.level() == self.current_level - 1
            })
            .collect::<HashSet<_>>();

        next_level_keys
    }

    fn get_box_near_field_keys(&self, box_key: &MortonKey, level: usize) -> HashSet<MortonKey> {
        if level == self.max_level {
            self.neighbour_map
                .get(box_key)
                .unwrap()
                .iter()
                .cloned()
                .collect()
        } else {
            self.neighbour_map
                .get(box_key)
                .unwrap()
                .iter()
                .cloned()
                .filter(|&key| key.level() == level)
                .collect()
        }
    }
}
