use super::{box_skeletonisation::{DecoupledBox, Tols, Rank, BoxFeatures, SkelBox}, sketch::{BoxesData, SketchOps}, tree_indexing::{TreeData, TreeIndexing}};
use rand_distr::{Distribution, Standard, StandardNormal};
use mpi::traits::CommunicatorCollectives;
use bempp_octree::{MortonKey, Octree};
use rlst::dense::tools::RandScalar;
use std::time::{Duration, Instant};
pub use rlst::prelude::*;

type Inds<T> = Vec<Vec<T>>;

pub struct DecoupledBoxData<Item:RlstScalar>
{
    pub operators: DecoupledBox<Item>,
    pub box_features: BoxFeatures
}

pub struct RsrsData<Item: RlstScalar> 
{
    pub level_indexing: TreeData,
    arr: DynamicArray<Item, 2>,
    pub tols: Tols<Item>,
    y_data: BoxesData<Item>,
    z_data: BoxesData<Item>,
    pub level_boxes: Vec<DecoupledBoxData<Item>>,
    dim: usize,
    pub acc_ind_r: Vec<usize>,
    ind_s: Inds<usize>,
    target_inds: Inds<usize>,
    near_inds: Inds<usize>,
}

pub trait Rsrs {
    type Item: RlstScalar;
    fn new<C: CommunicatorCollectives>(arr: DynamicArray<Self::Item, 2>, tols: Tols<Self::Item>, octree: Octree<'_, C>)->Self;
    fn tree_cycle(&mut self);
    fn level_iteration(&mut self);
    fn get_level_indices(&mut self, level: usize);
    
}


impl <T:RlstScalar  +
MatrixId + MatrixNull + 
MatrixInverse + MatrixPseudoInverse + 
RandScalar + mpi::datatype::Equivalence>Rsrs for RsrsData<T> 
where StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>{
    type Item = T;

    fn new<C: CommunicatorCollectives>(arr: DynamicArray<Self::Item, 2>, tols: Tols<Self::Item>, octree: Octree<'_, C>)->Self{
        let dim: usize = arr.shape()[0];
        let level_indexing: TreeData = <TreeData as TreeIndexing>::new(octree);
        let target_inds: Inds<usize> = Vec::new();
        let near_inds: Inds<usize> = Vec::new();
        let ind_s: Inds<usize> = Vec::new();
        let y_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(&arr,  1, false);
        let z_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(&arr, 1, true);
        let level_boxes: Vec<DecoupledBoxData<Self::Item>> = Vec::new();
        let acc_ind_r: Vec<usize> = Vec::new();
        
        Self{arr, level_indexing, y_data, z_data, tols, level_boxes, dim, acc_ind_r, ind_s, target_inds, near_inds}
    }

    fn tree_cycle(&mut self){
        let start: Instant = Instant::now();
        let mut level: usize = self.level_indexing.max_level;
        let min_level: usize = 1;
        
        while level > min_level{
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            self.get_level_indices(level);
            println!("Current Level: {}\n\n", level);
            self.level_iteration();
            println!("End level cycle\n");
            level -=1;
            let duration: Duration = start.elapsed();

            println!("Elapsed time: {} s", duration.as_secs());
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            let len_s: usize = self.ind_s.iter().map(|sketch_inds| sketch_inds.len()).sum();
            println!("Sketch Points: {}", len_s);
            let len_r: usize = self.acc_ind_r.len();
            println!("Residual Points: {}", len_r);

            if level <= min_level || (self.y_data.num_samples >= len_s && (len_r + len_s)== self.dim){
                break;
            }
        }

        let duration = start.elapsed();
        println!("Elapsed time: {} s", duration.as_secs());
    }

    fn level_iteration(&mut self){
        self.ind_s.resize(self.level_indexing.level_keys.len(), [].to_vec());
        let mut box_indices: Vec<usize> = (0..self.target_inds.len()).collect::<Vec<_>>();
        box_indices.sort_by_key(|&i| self.target_inds[i].len() + self.near_inds[i].len());
        for (box_num, &box_ind) in box_indices.iter().enumerate(){
            let target_inds: Vec<usize> = self.target_inds[box_ind].clone();
            let near_field_inds: Vec<usize> = self.near_inds[box_ind].clone();

            if !target_inds.is_empty(){
                println!("--------------------------------------------------\n");
                println!("Box {} of {}, with index {}\n", box_num + 1, self.target_inds.len(), box_ind);
                
                let min_num_samples: usize = near_field_inds.len() + target_inds.len();
                println!("Current number of samples: {}. Minimum number of samples: {}\n", self.y_data.num_samples, min_num_samples);
                
                if min_num_samples > self.y_data.num_samples {
                    let extra_num_samples: usize = min_num_samples - self.y_data.num_samples;

                    println!("***************");
                    println!("Extra samples: {}", extra_num_samples);

                    self.y_data.add_samples(extra_num_samples, &self.arr);
                    self.z_data.add_samples(extra_num_samples, &self.arr);

                    println!("***************\n");
                }

                let mut box_features: BoxFeatures = <BoxFeatures as SkelBox<Self::Item>>::new(target_inds.clone(), near_field_inds);
                let rank: Rank<Self::Item> = box_features.decouple(self.arr.shape()[0], &mut self.y_data, &mut self.z_data, &self.tols);
                match rank{
                    Rank::Low(decoupled_box) => {
                        self.ind_s[box_ind].clone_from(&box_features.ind_s);
                        self.acc_ind_r.extend_from_slice(&box_features.ind_r.clone());
                        self.level_boxes.push(DecoupledBoxData{operators: decoupled_box, box_features});
                    },
                    Rank::Full => {
                        self.ind_s[box_ind] = target_inds;
                    },
                }

                let len_s: usize = self.ind_s.iter().map(|sketch_inds| sketch_inds.len()).sum();
                println!("Sketch Points: {}", len_s);
                let len_r: usize = self.acc_ind_r.len();
                println!("Residual Points: {}", len_r);
                
                if  self.y_data.num_samples >= len_s && (len_r + len_s)== self.dim
                {
                    break;
                }

            }
        }
    }

    fn get_level_indices(&mut self, level: usize){//target_inds: &mut Inds<usize>, near_inds: &mut Inds<usize>){
            println!("Computing Indices...\n");
            if level < self.level_indexing.max_level{
                let binding: std::collections::HashSet<MortonKey> = self.level_indexing.level_keys.clone();
                let upper_level_keys: Vec<&MortonKey> = binding.iter().collect::<Vec<_>>();
                self.level_indexing.update_level_keys();
                let binding: std::collections::HashSet<MortonKey> = self.level_indexing.level_keys.clone();
                let current_level_keys: Vec<&MortonKey> = binding.iter().collect::<Vec<_>>();
                self.target_inds.clear();
                self.near_inds.clear();
                self.target_inds.resize(current_level_keys.len(), Vec::new());
                self.near_inds.resize(current_level_keys.len(), Vec::new());

                for (box_ind, box_key) in upper_level_keys.iter().enumerate(){
                    if let Some(parent_index) = current_level_keys.iter().position(|&r| *r == box_key.parent()){
                        self.target_inds[parent_index].extend_from_slice(&self.ind_s[box_ind]);
                    }
                }

                for (box_ind, box_key) in current_level_keys.iter().enumerate(){
                    if !upper_level_keys.iter().any(|&&k| k == box_key.parent()){
                        if let Some(parent_box) = self.level_indexing.boxes_map.get(box_key){
                            self.target_inds[box_ind].extend_from_slice(parent_box);
                        }
                    }
                }

                for (box_ind, box_key) in current_level_keys.iter().enumerate(){
                    self.near_inds[box_ind].extend_from_slice(&self.target_inds[box_ind]);
                    let near_keys: std::collections::HashSet<MortonKey> = self.level_indexing.get_box_near_field_keys(box_key);
                    for near_box_key in near_keys.iter(){
                        if let Some(near_box_ind) = current_level_keys.iter().position(|&r| *r == near_box_key.parent()){
                            self.near_inds[box_ind].extend_from_slice(&self.target_inds[near_box_ind]);
                        }
                    }
                }
                self.ind_s.clear();
                let boxes_lengths: Vec<usize> = self.target_inds.iter().map(|ind| ind.len()).collect::<Vec<_>>();
                let near_boxes_lengths: Vec<usize> = self.near_inds.iter().map(|ind| ind.len()).collect::<Vec<_>>();
                println!("New {} boxes of lengths {:?} ", self.target_inds.len(), boxes_lengths);
                println!("with respective neighbouring lengths {:?} ", near_boxes_lengths);

            }
            else {
                self.target_inds.resize(self.level_indexing.level_keys.len(), Vec::new());
                self.near_inds.resize(self.level_indexing.level_keys.len(), Vec::new());
                for (box_ind, box_key) in self.level_indexing.level_keys.clone().iter().enumerate(){
                    if let Some(box_indices) = self.level_indexing.boxes_map.get(box_key){
                        self.target_inds[box_ind] = box_indices.to_vec();
                        if let Some(neighbouring_indices) = self.level_indexing.get_neighbouring_indices(box_key){
                            self.near_inds[box_ind] = neighbouring_indices;
                        }
                    }
                }
            }
        
    }

}
