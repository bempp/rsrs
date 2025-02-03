use bempp_octree::{morton::MortonKey, octree::Octree};
use std::{collections::{HashMap, HashSet}, usize};
use mpi::traits::CommunicatorCollectives;

pub struct TreeData//<'o, C: CommunicatorCollectives> 
{
    //octree_data: Octree<'o, C>,
    pub level_keys: HashSet<MortonKey>,
    pub boxes_map: HashMap<MortonKey, Vec<usize>>,
    pub max_level: usize,
    current_level: usize,
    neighbour_map: HashMap<MortonKey, Vec<MortonKey>>
}

pub trait TreeIndexing: Sized{

    //fn new(points: &[Point], max_level: usize, max_leaf_points: usize, comm: &'o C) -> Self;

    fn new<C: CommunicatorCollectives>(octree_data: &Octree<'_, C>)-> Self;
    //Returns indices of points in neighboring boxes

    fn update_level_keys(&mut self);

    fn get_box_near_field_keys(&self, box_key: &MortonKey)-> HashSet<MortonKey>;

    //Function to get indices of points in a box
    fn get_box_indices(&self, box_key: &MortonKey)-> Option<&Vec<usize>>;

    fn get_box_far_field_keys(&self, box_key: &MortonKey)-> HashSet<MortonKey>;

    fn get_neighbouring_indices(&self, box_key: &MortonKey)-> Option<Vec<usize>>;

    //fn permute_box(&mut self, box_key: &MortonKey, perm: &[usize]);

    //fn update_box(&mut self, box_key: &MortonKey, new_inds: Vec<usize>);

    //Returns the parent box of the given box
    //fn get_box_parent(self, box_key: MortonKey);

    //Returns the level of the box
    //fn get_box_level(self, box_key: MortonKey);

    //Returns whether a box is a leaf
    //fn is_leaf(self, box_key: MortonKey);

}


impl TreeIndexing for  TreeData{

    //fn new(points: &[Point], max_level: usize, max_leaf_points: usize, comm: &'o C)-> Self{
    fn new<C: CommunicatorCollectives>(octree_data: &Octree<'_, C>)-> Self{
        let neighbour_map: HashMap<MortonKey, Vec<MortonKey>> = octree_data.neighbour_map().clone();
        let boxes_map: HashMap<MortonKey, Vec<usize>> = octree_data.leaf_keys_to_local_point_indices().clone();
        let leaf_tree_keys = octree_data.leaf_keys().iter().cloned();
        let max_level = octree_data.global_max_level();
        let level_keys = leaf_tree_keys.collect::<HashSet<_>>();//leaf_tree_keys.filter(|&key| key.level()==max_level).collect::<HashSet<_>>();
        let current_level = max_level;
        Self{level_keys, boxes_map, neighbour_map, max_level: octree_data.global_max_level(), current_level}
    }

    fn update_level_keys(&mut self){
        if self.current_level == self.max_level{
            self.level_keys = self.level_keys.iter().map(|&key| if key.level() == self.max_level {key.parent()} else{key}).collect::<HashSet<_>>();
        }
        else{
            self.level_keys = self.level_keys.iter().filter(|key| key.level()>0 && key.level()==self.current_level).map(|key| key.parent()).collect::<HashSet<_>>();
        }
        self.current_level-=1;
    }

    fn get_box_near_field_keys(&self, box_key: &MortonKey)-> HashSet<MortonKey>{
        if self.current_level == self.max_level{
            self.neighbour_map.get(box_key).unwrap().iter().cloned().collect()
        }
        else{
            self.neighbour_map.get(box_key).unwrap().iter().cloned().filter(|&key| key.level()==self.current_level).collect()
        }
    }

    fn get_box_indices(&self, box_key: &MortonKey)-> Option<&Vec<usize>>{
        self.boxes_map.get(box_key)
    }

    fn get_box_far_field_keys(&self, box_key: &MortonKey)-> HashSet<MortonKey>{
        let level_keys: &HashSet<MortonKey> = &self.level_keys;
        let near_keys: HashSet<MortonKey> = self.get_box_near_field_keys(box_key);
        let far_keys: HashSet<MortonKey> = level_keys.difference(&near_keys).cloned().collect::<HashSet<_>>();
        far_keys
    }

    fn get_neighbouring_indices(&self, box_key: &MortonKey)-> Option<Vec<usize>>{
        match  self.boxes_map.get(box_key){
            Some(indices) => {
                let mut neighbour_indices: Vec<usize> = Vec::new();
                neighbour_indices.extend_from_slice(indices);
                let neighbour_keys: std::collections::hash_set::IntoIter<MortonKey> = self.get_box_near_field_keys(box_key).into_iter();
                for neighbour_key in neighbour_keys{
                    if let Some(indices) = self.boxes_map.get(&neighbour_key) {
                        neighbour_indices.extend_from_slice(indices);
                    }
                }
                Some(neighbour_indices)

            },
            None => {
                None
            },
        }
        
    }

    /*fn permute_box(&mut self, box_key: &MortonKey, perm: &[usize]){
        
        if let Some(indices) =self.boxes_map.get_mut(box_key) {
            let mut aux_indices = indices.clone();
            for (id, &elem) in perm.iter().enumerate(){
                *indices.get_mut(id).unwrap() = *aux_indices.get_mut(elem).unwrap();
            }
        }
    }*/

    //fn update_box(&mut self, box_key: &MortonKey, new_inds: Vec<usize>){
    //    *self.boxes_map.get_mut(box_key).unwrap() = new_inds;
    //}

    //fn get_box_parent(self, box_key: MortonKey){}

    //fn get_box_level(self, box_key: MortonKey){}

    //fn is_leaf(self, box_key: MortonKey){}

}
