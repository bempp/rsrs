use bempp_octree::{morton::MortonKey, octree::Octree};
use std::collections::{HashMap, HashSet};
use mpi::traits::CommunicatorCollectives;

pub struct TreeData//<'o, C: CommunicatorCollectives> 
{
    //octree_data: Octree<'o, C>,
    pub level_keys: HashSet<MortonKey>,
    boxes_map: HashMap<MortonKey, Vec<usize>>,
    neighbour_map: HashMap<MortonKey, Vec<MortonKey>>
}



pub trait TreeIndexing<'o, C>: Sized{

    //fn new(points: &[Point], max_level: usize, max_leaf_points: usize, comm: &'o C) -> Self;

    fn new(octree_data: Octree<'o, C>)-> Self;
    //Returns indices of points in neighboring boxes
    fn get_box_near_field_keys(&self, box_key: &MortonKey)->  HashSet<MortonKey>;

    fn get_box_far_field_keys(&self, box_key: &MortonKey)-> HashSet<MortonKey>;

    //Function to get indices of points in a box
    fn get_box_indices(&self, box_key: &MortonKey)-> Vec<usize>;

    fn get_neighbouring_indices(&mut self, box_key: &MortonKey)-> Vec<usize>;

    //Returns the parent box of the given box
    //fn get_box_parent(self, box_key: MortonKey);

    //Returns the level of the box
    //fn get_box_level(self, box_key: MortonKey);

    //Returns whether a box is a leaf
    //fn is_leaf(self, box_key: MortonKey);

    fn permute_box(&mut self, box_key: &MortonKey, perm: &[usize]);

    fn update_box(&mut self, box_key: &MortonKey, new_inds: Vec<usize>);
        
}


impl <'o, C: CommunicatorCollectives>TreeIndexing<'o, C> for  TreeData{

    //fn new(points: &[Point], max_level: usize, max_leaf_points: usize, comm: &'o C)-> Self{
    fn new(octree_data: Octree<'o, C>)-> Self{
       // let octree_data: Octree<'_, C> = Octree::new(points, max_level, max_leaf_points, comm);
        
        let neighbour_map: HashMap<MortonKey, Vec<MortonKey>> = octree_data.neighbour_map().clone();
        let boxes_map: HashMap<MortonKey, Vec<usize>> = octree_data.leaf_keys_to_local_point_indices().clone();
        let level_keys : HashSet<MortonKey> = octree_data.all_keys().keys().cloned().collect();
        Self{level_keys, boxes_map, neighbour_map}
    }

    fn get_box_near_field_keys(&self, box_key: &MortonKey)-> HashSet<MortonKey>{
        self.neighbour_map.get(box_key).unwrap().iter().cloned().collect()
    }

    fn get_box_far_field_keys(&self, box_key: &MortonKey)-> HashSet<MortonKey>{
        let level_keys: &HashSet<MortonKey> = &self.level_keys;
        let near_keys: HashSet<MortonKey> = <TreeData as TreeIndexing<'_, C>>::get_box_near_field_keys(&self, box_key);
        let far_keys: HashSet<MortonKey> = level_keys.difference(&near_keys).cloned().collect::<HashSet<_>>();
        far_keys
    }

    fn get_box_indices(&self, box_key: &MortonKey)-> Vec<usize>{
        self.boxes_map.get(box_key).unwrap().to_vec()
    }

    fn get_neighbouring_indices(&mut self, box_key: &MortonKey)-> Vec<usize>{
        let mut neighbour_indices: Vec<usize> = Vec::new();
        if let Some(indices) = self.boxes_map.get_mut(box_key) {
            neighbour_indices.append(indices);
        }
        
        let neighbour_keys: std::collections::hash_set::IntoIter<MortonKey> = <TreeData as TreeIndexing<'_, C>>::get_box_near_field_keys(self, box_key).into_iter();
        
        for neighbour_key in neighbour_keys{
            if let Some(indices) = self.boxes_map.get_mut(&neighbour_key) {
                neighbour_indices.append(indices);
            }
        }
        neighbour_indices
    }

    fn permute_box(&mut self, box_key: &MortonKey, perm: &[usize]){
        
        if let Some(indices) =self.boxes_map.get_mut(box_key) {
            let mut aux_indices = indices.clone();
            for (id, &elem) in perm.iter().enumerate(){
                *indices.get_mut(id).unwrap() = *aux_indices.get_mut(elem).unwrap();
            }
        }
    }

    fn update_box(&mut self, box_key: &MortonKey, new_inds: Vec<usize>){
        *self.boxes_map.get_mut(box_key).unwrap() = new_inds;
    }

    //fn get_box_parent(self, box_key: MortonKey){}

    //fn get_box_level(self, box_key: MortonKey){}

    //fn is_leaf(self, box_key: MortonKey){}

}
