pub use rlst::prelude::*;
use mpi::traits::CommunicatorCollectives;
use bempp_octree::{Point, MortonKey, octree::Octree};
pub use rlst::dense::array::empty_array;
use crate::sketch::BoxesData;

type ArrayImpl<Item> = BaseArray<Item, VectorContainer<Item>, 2>;

pub struct RsrsData<'o, Item: RlstScalar, C: CommunicatorCollectives> 
{
    octree_data: Octree<'o, C>,
    arr: Array<Item, ArrayImpl<Item>, 2>,
    null_tol: Item,
    y_data: BoxesData<Item>,
    z_data: BoxesData<Item>
}

pub trait Rsrs<'o, C: CommunicatorCollectives> {
    type Item: RlstScalar;
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;
    fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, points: &[Point], max_level: usize, max_leaf_points: usize, comm: &'o C)->Self;

    //Octree index handling

    //Returns indices of points in neighboring boxes
    fn get_box_neighbours_inds(&self, box_ind: MortonKey)-> Option<&Vec<MortonKey>>;

    //Function to get indices of points in a box
    fn get_box_inds(self, box_ind: MortonKey);

    //Returns the parent box of the given box
    fn get_box_parent(self, box_ind: MortonKey);

    //Returns the level of the box
    fn get_box_level(self, box_ind: MortonKey);

    //Returns whether a box is a leaf
    fn is_leaf(self, box_ind: MortonKey);

    fn get_level_indices(self, level: usize);

    //RSRS functionality

    fn get_far_fields(self, box_ind: MortonKey);


    
}

macro_rules! impl_rsrs{
    ($scalar:ty) => {
        impl <'o, C: CommunicatorCollectives>Rsrs<'o, C> for RsrsData<'o, $scalar, C> {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl<$scalar>;

            fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, points: &[Point], max_level: usize, max_leaf_points: usize, comm: &'o C)->Self{
                let octree_data = Octree::new(&points, max_level, max_leaf_points, comm);
                
                let num_samples = 100;//TODO: change

            

                Self{arr, octree_data, y_sketch, z_sketch, y_test, z_test, null_tol: 1e-15}

            }

            fn get_box_neighbours_inds(&self, box_id: MortonKey)-> Option<&Vec<MortonKey>>{
                return self.octree_data.neighbour_map().get(&box_id)
            }
        
            fn get_box_inds(self, box_ind: MortonKey){}
        
            fn get_box_parent(self, box_ind: MortonKey){}
        
            fn get_box_level(self, box_ind: MortonKey){}
        
            fn is_leaf(self, box_ind: MortonKey){}

            fn get_level_indices(self, level: usize){}

            fn get_far_fields(self, box_ind: MortonKey){
            
                
            }

        }
    }
}

impl_rsrs!(f64);