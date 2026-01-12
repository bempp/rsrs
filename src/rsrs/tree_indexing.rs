use bempp_octree::{morton::MortonKey, octree::Octree};
use mpi::traits::CommunicatorCollectives;
use std::collections::{HashMap, HashSet};

pub struct TreeData {
    pub level_keys: HashSet<MortonKey>,
    pub boxes_map: HashMap<MortonKey, Vec<usize>>,
    pub max_level: usize,
    pub current_level: usize,
    neighbour_map: HashMap<MortonKey, Vec<MortonKey>>,
}

pub trait TreeIndexing: Sized {
    fn new<C: CommunicatorCollectives>(octree_data: &Octree<'_, C>) -> Self;

    fn update_level_keys(&mut self);
    fn next_level_keys(&mut self) -> HashSet<MortonKey>;

    fn get_box_near_field_keys(&self, box_key: &MortonKey, level: usize) -> HashSet<MortonKey>;
    fn get_box_indices(&self, box_key: &MortonKey) -> Option<&Vec<usize>>;
    fn get_box_far_field_keys(&self, box_key: &MortonKey) -> HashSet<MortonKey>;
    fn get_neighbouring_indices(&self, box_key: &MortonKey) -> Option<Vec<usize>>;
}

impl TreeIndexing for TreeData {
    fn new<C: CommunicatorCollectives>(octree_data: &Octree<'_, C>) -> Self {
        let neighbour_map: HashMap<MortonKey, Vec<MortonKey>> = octree_data.neighbour_map().clone();

        // Leaf-only initially (this is fine at max level)
        let boxes_map: HashMap<MortonKey, Vec<usize>> =
            octree_data.leaf_keys_to_local_point_indices().clone();

        let level_keys = octree_data
            .leaf_keys()
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        let max_level = octree_data.global_max_level();
        let current_level = max_level;

        Self {
            level_keys,
            boxes_map,
            neighbour_map,
            max_level,
            current_level,
        }
    }

    fn update_level_keys(&mut self) {
        // Already at root
        if self.current_level == 0 {
            return;
        }

        let next_level = self.current_level - 1;

        // Compute the set of keys at the next level (coarser)
        let next_keys = self.next_level_keys();

        // Aggregate indices from current boxes_map into next level boxes_map
        let mut next_boxes_map: HashMap<MortonKey, Vec<usize>> = HashMap::new();

        for (&key, inds) in self.boxes_map.iter() {
            // Map this key up to next_level (possibly multiple parents if key is deeper)
            let mut k = key;
            while k.level() > next_level {
                k = k.parent();
            }

            // Only keep keys that are part of the next level set
            if next_keys.contains(&k) {
                next_boxes_map
                    .entry(k)
                    .or_insert_with(Vec::new)
                    .extend_from_slice(inds);
            }
        }

        // Remove any accidental empties (shouldn't happen, but keeps invariant strict)
        next_boxes_map.retain(|_, v| !v.is_empty());

        // IMPORTANT: also drop keys that ended up empty after aggregation (paranoia)
        let next_keys: HashSet<MortonKey> = next_keys
            .into_iter()
            .filter(|k| next_boxes_map.get(k).map_or(false, |v| !v.is_empty()))
            .collect();

        // Update state
        self.boxes_map = next_boxes_map;
        self.level_keys = next_keys;
        self.current_level = next_level;
    }

    fn next_level_keys(&mut self) -> HashSet<MortonKey> {
        // Move one level up
        if self.current_level == 0 {
            return self.level_keys.clone();
        }
        let next_level = self.current_level - 1;

        self.level_keys
            .iter()
            .map(|&key| {
                let mut k = key;
                while k.level() > next_level {
                    k = k.parent();
                }
                k
            })
            .filter(|k| k.level() == next_level)
            .collect()
    }

    fn get_box_near_field_keys(&self, box_key: &MortonKey, level: usize) -> HashSet<MortonKey> {
        let neigh = match self.neighbour_map.get(box_key) {
            Some(v) => v,
            None => return HashSet::new(),
        };

        if level == self.max_level {
            neigh.iter().cloned().collect()
        } else {
            neigh
                .iter()
                .cloned()
                .filter(|k| k.level() == level)
                .collect()
        }
    }

    fn get_box_indices(&self, box_key: &MortonKey) -> Option<&Vec<usize>> {
        self.boxes_map.get(box_key)
    }

    fn get_box_far_field_keys(&self, box_key: &MortonKey) -> HashSet<MortonKey> {
        let level_keys: &HashSet<MortonKey> = &self.level_keys;
        let near_keys: HashSet<MortonKey> =
            self.get_box_near_field_keys(box_key, self.current_level);
        level_keys.difference(&near_keys).cloned().collect()
    }

    fn get_neighbouring_indices(&self, box_key: &MortonKey) -> Option<Vec<usize>> {
        match self.boxes_map.get(box_key) {
            Some(indices) => {
                let mut neighbour_indices: Vec<usize> = Vec::new();
                neighbour_indices.extend_from_slice(indices);

                let neighbour_keys = self
                    .get_box_near_field_keys(box_key, self.current_level)
                    .into_iter();

                for neighbour_key in neighbour_keys {
                    if let Some(inds) = self.boxes_map.get(&neighbour_key) {
                        neighbour_indices.extend_from_slice(inds);
                    }
                }
                Some(neighbour_indices)
            }
            None => None,
        }
    }
}
