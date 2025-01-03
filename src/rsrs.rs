use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::dense::tools::RandScalar;
pub use rlst::prelude::*;
use mpi::traits::CommunicatorCollectives;
use crate::{box_skeletonisation::Tols, tree_indexing::{self, TreeIndexing}};
use bempp_octree::{octree, MortonKey, Octree, Point};
pub use rlst::dense::array::empty_array;
use crate::{sketch::{BoxesData, SketchOps}, tree_indexing::TreeData};
use crate::box_skeletonisation::{BoxFeatures, SkelBox, Rank};

type ArrayImpl<Item> = BaseArray<Item, VectorContainer<Item>, 2>;


pub struct RsrsData<Item: RlstScalar> 
{
    pub level_indexing: TreeData,
    arr: Array<Item, ArrayImpl<Item>, 2>,
    pub tols: Tols<Item>,
    y_data: BoxesData<Item>,
    z_data: BoxesData<Item>,
    num_samples: usize
}

pub trait Rsrs<'o, C: CommunicatorCollectives> {
    type Item: RlstScalar;
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;
    fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, tols: Tols<Self::Item>, min_num_samples: usize, octree: Octree<'o, C>, comm: &C)->Self;
    fn tree_cycle();
    fn level_iteration(&mut self);
    
}


impl <'o, T:RlstScalar  +
MatrixId + MatrixNull + 
MatrixInverse + MatrixPseudoInverse + 
RandScalar + mpi::datatype::Equivalence, C: CommunicatorCollectives>Rsrs<'o, C> for RsrsData<T> 
where StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>{
    type Item = T;
    type ArrayImpl = ArrayImpl<T>;

    fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, tols: Tols<Self::Item>, min_num_samples: usize, octree: Octree<'o, C>, comm: &C)->Self{
        let level_indexing: TreeData = <TreeData as TreeIndexing<'o, C>>::new(octree);
        let y_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(&arr,  min_num_samples, false, comm);
        let z_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(&arr, min_num_samples, true, comm);
        Self{arr, level_indexing, y_data, z_data, tols, num_samples: min_num_samples}
    }


    fn tree_cycle(){

    }

    fn level_iteration(&mut self){
        for (key_ind, box_key) in self.level_indexing.level_keys.clone().into_iter().enumerate(){
            let target_inds: Vec<usize> = <tree_indexing::TreeData as tree_indexing::TreeIndexing<'_, C>>::get_box_indices(&self.level_indexing, &box_key);
            let near_field_inds: Vec<usize> = <tree_indexing::TreeData as tree_indexing::TreeIndexing<'_, C>>::get_neighbouring_indices(&mut self.level_indexing, &box_key); 

            let min_num_samples = near_field_inds.len();

            println!("num samples {}", min_num_samples);

            let extra_num_samples = min_num_samples - self.y_data.num_samples;

            if extra_num_samples > 0{
                self.y_data.add_samples(extra_num_samples, &self.arr);
                self.z_data.add_samples(extra_num_samples, &self.arr);
            }

            let mut points_box: BoxFeatures = <BoxFeatures as SkelBox<Self::Item>>::new(target_inds, near_field_inds);
            let _: Rank<Self::Item> = points_box.decouple(self.arr.shape()[0], &mut self.y_data, &mut self.z_data, &self.tols);
            <tree_indexing::TreeData as tree_indexing::TreeIndexing<'_, C>>::update_box(&mut self.level_indexing, &box_key, points_box.target_inds);
        }
    }

}