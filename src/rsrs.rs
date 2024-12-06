pub use rlst::prelude::*;
use mpi::traits::CommunicatorCollectives;
use bempp_octree::{Point, MortonKey, octree::Octree};
pub use rlst::dense::array::empty_array;

type ArrayImpl<Item> = BaseArray<Item, VectorContainer<Item>, 2>;

pub struct RsrsData<'o, Item: RlstScalar, C: CommunicatorCollectives> 
{
    octree_data: Octree<'o, C>,
    arr: Array<Item, ArrayImpl<Item>, 2>,
    null_tol: Item,
    y_sketch: Array<Item, ArrayImpl<Item>, 2>,
    z_sketch: Array<Item, ArrayImpl<Item>, 2>, 
    y_test: Array<Item, ArrayImpl<Item>, 2>,
    z_test: Array<Item, ArrayImpl<Item>, 2>,
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

    //Useful permutations

    //Permutation of the complete matrix after a rsrs iteration
    fn it_permutation(self);

    //
    fn permute_indices(self);
    /* , ind_list, pbind=None, bind=None):
        if pbind is None and bind is None:
            bind, pbind = self.get_perm()
        return pbind[np.in1d(bind, ind_list).nonzero()[0]]*/

    fn permute_near_field(self);
    /* , box, pbind=None, bind=None):
        self.near_field_inds[box] = self.permute_indices(
            self.near_field_inds[box], pbind, bind)*/

    fn permute_sketches(self);
    /* , inds, perm_inds):
        self.y[inds] = self.y[perm_inds]
        self.z[inds] = self.z[perm_inds]
        self.omega[inds] = self.omega[perm_inds]
        self.psi[inds] = self.psi[perm_inds]*/
    
}

macro_rules! impl_rsrs{
    ($scalar:ty) => {
        impl <'o, C: CommunicatorCollectives>Rsrs<'o, C> for RsrsData<'o, $scalar, C> {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl<$scalar>;

            fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, points: &[Point], max_level: usize, max_leaf_points: usize, comm: &'o C)->Self{
                let octree_data = Octree::new(&points, max_level, max_leaf_points, comm);
                
                let num_samples = 100;//TODO: change

                //TODO: change this implementation when the mat-vec product is available
                let y_test: DynamicArray<Self::Item, 2> = rlst_dynamic_array2!(Self::Item, [num_samples, arr.shape()[1]]);
                let z_test: DynamicArray<Self::Item, 2> = rlst_dynamic_array2!(Self::Item, [num_samples, arr.shape()[1]]);
                let y_sketch: DynamicArray<Self::Item, 2> = empty_array().simple_mult_into_resize(arr.view(), y_test.view());
                let z_sketch: DynamicArray<Self::Item, 2> = empty_array().mult_into_resize(TransMode::Trans,
                    TransMode::NoTrans,
                    <Self::Item as num::One>::one(),
                    arr.view(),
                    z_test.view(),
                    <Self::Item as num::Zero>::zero());

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

            fn it_permutation(self){}

            fn permute_indices(self){}

            fn permute_near_field(self){}
        
            fn permute_sketches(self){}

        }
    }
}

impl_rsrs!(f64);