use bempp_octree::{morton::MortonKey, octree::Octree};
use std::collections::HashMap;
use mpi::{
    collective::SystemOperation,
    traits::{CommunicatorCollectives, Root},
};


pub trait TreeData: Sized{

    fn new() -> Self;

    //Returns indices of points in neighboring boxes
    fn get_box_neighbours_inds(&self, box_ind: MortonKey)-> &HashMap<MortonKey, Vec<MortonKey>>;

    //Function to get indices of points in a box
    fn get_box_inds(self, box_ind: MortonKey);

    //Returns the parent box of the given box
    fn get_box_parent(self, box_ind: MortonKey);

    //Returns the level of the box
    fn get_box_level(self, box_ind: MortonKey);

    //Returns whether a box is a leaf
    fn is_leaf(self, box_ind: MortonKey);
        
}


impl <'o, C: CommunicatorCollectives>TreeData for  Octree<'o, C>{

    fn new()-> Self{

    }

    fn get_box_neighbours_inds(&self, box_id: MortonKey)-> &HashMap<MortonKey, Vec<MortonKey>>{
        return &self.neighbour_map().get(box_id)
    }

    fn get_box_inds(self, box_ind: MortonKey){}

    fn get_box_parent(self, box_ind: MortonKey){}

    fn get_box_level(self, box_ind: MortonKey){}

    fn is_leaf(self, box_ind: MortonKey){}

}
