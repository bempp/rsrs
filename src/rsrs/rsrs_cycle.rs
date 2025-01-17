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
    pub box_features: BoxFeatures,
    pub level_it: usize,
}

pub struct RsrsData<Item: RlstScalar> 
{
    pub level_indexing: TreeData,
    pub tols: Tols<Item>,
    y_data: BoxesData<Item>,
    z_data: BoxesData<Item>,
    pub dec_boxes: Vec<DecoupledBoxData<Item>>,
    dim: usize,
    pub acc_ind_r: Vec<usize>,
    ind_s: Inds<usize>,
    target_inds: Inds<usize>,
    near_inds: Inds<usize>,
}

pub trait Rsrs {
    type Item: RlstScalar;
    fn new<C: CommunicatorCollectives>(arr: &DynamicArray<Self::Item, 2>, tols: Tols<Self::Item>, octree: Octree<'_, C>)->Self;
    fn tree_cycle(&mut self, arr: &DynamicArray<Self::Item, 2>);
    fn level_iteration(&mut self, level_it: usize, arr: &DynamicArray<Self::Item, 2>)->State;
    fn get_level_indices(&mut self, level: usize);
    fn get_near_indices(&mut self, box_ind: usize)->Vec<usize>;
    fn tree_cycle_and_diag_block_extraction(&mut self, arr: &DynamicArray<Self::Item, 2>);
    
}

pub enum State {
    PartialSketching,
    FullSketching,
}


impl <T:RlstScalar  +
MatrixId + MatrixNull + 
MatrixInverse + MatrixPseudoInverse + 
RandScalar + mpi::datatype::Equivalence>Rsrs for RsrsData<T> 
where StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>{
    type Item = T;

    fn new<C: CommunicatorCollectives>(arr: &DynamicArray<Self::Item, 2>, tols: Tols<Self::Item>, octree: Octree<'_, C>)->Self{
        let dim: usize = arr.shape()[0];
        let level_indexing: TreeData = <TreeData as TreeIndexing>::new(octree);
        let target_inds: Inds<usize> = Vec::new();
        let near_inds: Inds<usize> = Vec::new();
        let ind_s: Inds<usize> = Vec::new();
        let y_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr,  1, false);
        let z_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr, 1, true);
        let dec_boxes: Vec<DecoupledBoxData<Self::Item>> = Vec::new();
        let acc_ind_r: Vec<usize> = Vec::new();
        
        Self{level_indexing, y_data, z_data, tols, dec_boxes, dim, acc_ind_r, ind_s, target_inds, near_inds}
    }

    fn tree_cycle_and_diag_block_extraction(&mut self, arr: &DynamicArray<Self::Item, 2>){
        let start: Instant = Instant::now();
        self.tree_cycle(arr);
        let duration = start.elapsed();
        println!("Tree cycle elapsed time: {} s", duration.as_secs());

    }

    fn tree_cycle(&mut self, arr: &DynamicArray<Self::Item, 2>){
        
        let mut level: usize = self.level_indexing.max_level;
        let min_level: usize = 1;

        while level > min_level{
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            let start: Instant = Instant::now();
            self.get_level_indices(level);
            println!("Current Level: {}\n\n", level);
            self.level_iteration(level, arr);
            println!("End level cycle\n");
            let duration: Duration = start.elapsed();
            println!("Elapsed time: {} s", duration.as_secs());
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

            let len_s: usize = self.ind_s.iter().map(|sketch_inds| sketch_inds.len()).sum();
            println!("Sketch Points: {}", len_s);
            let len_r: usize = self.acc_ind_r.len();
            println!("Residual Points: {}", len_r);

            level -=1;

            if level <= min_level{
                println!("\nReached lower level");
                break;
            }
        }
    }

    fn level_iteration(&mut self, level_it: usize, arr: &DynamicArray<Self::Item, 2>)->State{
        self.ind_s.clone_from(&self.target_inds);//.resize(self.level_indexing.level_keys.len(), [].to_vec());
        let mut box_indices: Vec<usize> = (0..self.target_inds.len()).collect::<Vec<_>>();
        box_indices.sort_by_key(|&i| self.target_inds[i].len() + self.near_inds[i].len());
        let mut len_sketch: usize = 0;
        let mut state = State::FullSketching;
        let mut len_residual = 0;

        for (box_num, &box_ind) in box_indices.iter().enumerate(){
            let near_field_inds: Vec<usize> = self.get_near_indices(box_ind);

            if !self.target_inds[box_ind].is_empty(){
                println!("--------------------------------------------------\n");
                println!("Box {} of {}, with index {}\n", box_num + 1, self.target_inds[box_ind].len(), box_ind);
                
                let min_box_samples: usize = near_field_inds.len() + self.target_inds[box_ind].len();
                let min_sketch_samples = self.dim-len_residual;

                if min_sketch_samples > min_box_samples
                {
                    println!("Current number of samples: {}. Minimum number of samples: {}\n", self.y_data.num_samples, min_box_samples);
                    
                    if min_box_samples > self.y_data.num_samples {
                        let extra_num_samples: usize = min_box_samples - self.y_data.num_samples;

                        println!("***************");
                        println!("Extra samples: {}", extra_num_samples);

                        self.y_data.add_samples(extra_num_samples, arr, &self.dec_boxes);
                        self.z_data.add_samples(extra_num_samples, arr, &self.dec_boxes);

                        println!("***************\n");
                    }

                    let mut box_features: BoxFeatures = <BoxFeatures as SkelBox<Self::Item>>::new(near_field_inds);

                    let rank: Rank<Self::Item> = box_features.decouple(&mut self.target_inds[box_ind], &mut self.y_data, &mut self.z_data, &self.tols);
                    match rank{
                        Rank::Low(decoupled_box) => {
                            self.ind_s[box_ind].clone_from(&box_features.ind_s);
                            self.acc_ind_r.extend_from_slice(&box_features.ind_r.clone());
                            self.dec_boxes.push(DecoupledBoxData{operators: decoupled_box, box_features, level_it});
                        },
                        Rank::Full => {},
                    }

                    len_sketch += self.ind_s[box_ind].len();
                    println!("Level sketch points: {}", len_sketch);
                    len_residual = self.acc_ind_r.len();
                    println!("Residual points: {}", len_residual);
                    println!("Remaining points in sketch: {}", self.dim-len_residual);

                }
                else if min_sketch_samples > self.y_data.num_samples {
                    let extra_num_samples: usize = min_sketch_samples - self.y_data.num_samples;

                    println!("***************");
                    println!("Extra samples: {}", extra_num_samples);

                    self.y_data.add_samples(extra_num_samples, arr, &self.dec_boxes);
                    self.z_data.add_samples(extra_num_samples, arr, &self.dec_boxes);

                    println!("***************\n");
                }

                if self.y_data.num_samples >= self.dim-len_residual{
                    println!("Enough Samples");
                    state = State::PartialSketching;
                    break;
                }
            }
        }
        state
    }

    fn get_near_indices(&mut self, box_ind: usize)-> Vec<usize>{
        let mut near_indices = Vec::new();
        for ind in self.near_inds[box_ind].iter(){
            near_indices.extend_from_slice(&self.target_inds[*ind]);
        } 
        near_indices
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
                    if box_key.level() > 0{
                        let box_parent = box_key.parent();
                        if !upper_level_keys.iter().any(|&&k| k == box_parent){
                            if let Some(parent_box) = self.level_indexing.boxes_map.get(box_key){
                                self.target_inds[box_ind].extend_from_slice(parent_box);
                            }
                        }
                    }
                }
                for (box_ind, box_key) in current_level_keys.iter().enumerate(){
                    self.near_inds[box_ind].push(box_ind);
                    let near_keys: std::collections::HashSet<MortonKey> = self.level_indexing.get_box_near_field_keys(box_key);
                    for near_box_key in near_keys.iter(){
                        if let Some(near_box_ind) = current_level_keys.iter().position(|&r| *r == near_box_key.parent()){
                            self.near_inds[box_ind].push(near_box_ind);
                        }
                    }
                }
                self.ind_s.clear();
                let boxes_lengths: Vec<usize> = self.target_inds.iter().map(|ind| ind.len()).collect::<Vec<_>>();
                println!("New {} boxes of lengths {:?} ", self.target_inds.len(), boxes_lengths);

            }
            else {
                self.target_inds.resize(self.level_indexing.level_keys.len(), Vec::new());
                self.near_inds.resize(self.level_indexing.level_keys.len(), Vec::new());
                for (box_ind, box_key) in self.level_indexing.level_keys.clone().iter().enumerate(){
                    if let Some(box_indices) = self.level_indexing.boxes_map.get(box_key){
                        self.target_inds[box_ind] = box_indices.to_vec();
                    }
                }

                for (box_ind, box_key) in self.level_indexing.level_keys.iter().enumerate(){
                    self.near_inds[box_ind].push(box_ind);
                    let near_keys: std::collections::HashSet<MortonKey> = self.level_indexing.get_box_near_field_keys(box_key);
                    for near_box_key in near_keys.iter(){
                        if let Some(near_box_ind) = self.level_indexing.level_keys.iter().position(|&r| r == *near_box_key){
                            self.near_inds[box_ind].push(near_box_ind);
                        }
                    }
                }
            }
        
    }

}


/* 
impl <T:RlstScalar  +
MatrixId + MatrixNull + 
MatrixInverse + MatrixPseudoInverse + 
RandScalar + mpi::datatype::Equivalence>Rsrs for RsrsData<T> 
where StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>{
    type Item = T;

    fn new<C: CommunicatorCollectives>(arr: &DynamicArray<Self::Item, 2>, tols: Tols<Self::Item>, octree: Octree<'_, C>)->Self{
        let dim: usize = arr.shape()[0];
        let level_indexing: TreeData = <TreeData as TreeIndexing>::new(octree);
        let target_inds: Inds<usize> = Vec::new();
        let near_inds: Inds<usize> = Vec::new();
        let ind_s: Inds<usize> = Vec::new();
        let y_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr,  1, false);
        let z_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr, 1, true);
        let dec_boxes: Vec<DecoupledBoxData<Self::Item>> = Vec::new();
        let acc_ind_r: Vec<usize> = Vec::new();
        
        Self{level_indexing, y_data, z_data, tols, dec_boxes, dim, acc_ind_r, ind_s, target_inds, near_inds}
    }

    fn tree_cycle_and_diag_block_extraction(&mut self, arr: &DynamicArray<Self::Item, 2>){
        let start: Instant = Instant::now();
        self.tree_cycle(arr);
        let duration = start.elapsed();
        println!("Tree cycle elapsed time: {} s", duration.as_secs());

    }

    fn tree_cycle(&mut self, arr: &DynamicArray<Self::Item, 2>){
        
        let mut level: usize = self.level_indexing.max_level;
        let min_level: usize = 1;

        while level > min_level{
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            let start: Instant = Instant::now();
            self.get_level_indices(level);
            println!("Current Level: {}\n\n", level);
            self.level_iteration(level, arr);
            println!("End level cycle\n");
            let duration: Duration = start.elapsed();
            println!("Elapsed time: {} s", duration.as_secs());
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

            let len_s: usize = self.ind_s.iter().map(|sketch_inds| sketch_inds.len()).sum();
            println!("Sketch Points: {}", len_s);
            let len_r: usize = self.acc_ind_r.len();
            println!("Residual Points: {}", len_r);

            level -=1;
            /*if level <= min_level || (self.y_data.num_samples >= len_s && (len_r + len_s)== self.dim){
                break;
            }*/
            if level <= min_level{
                println!("\nReached lower level");
                break;
            }
        }
    }

    fn level_iteration(&mut self, level_it: usize, arr: &DynamicArray<Self::Item, 2>)->State{
        self.ind_s.clone_from(&self.target_inds);//.resize(self.level_indexing.level_keys.len(), [].to_vec());
        let mut box_indices: Vec<usize> = (0..self.target_inds.len()).collect::<Vec<_>>();
        box_indices.sort_by_key(|&i| self.target_inds[i].len() + self.near_inds[i].len());
        let mut len_sketch: usize = 0;
        let mut state = State::FullSketching;
        let mut len_residual = 0;
        //let mut dec_boxes: Vec<DecoupledBoxData<T>>;
        //dec_boxes.clone_from(&self.dec_boxes);

        for (box_num, &box_ind) in box_indices.iter().enumerate(){
            let target_inds: Vec<usize> = self.target_inds[box_ind].clone();
            let near_field_inds: Vec<usize> = self.near_inds[box_ind].clone();
            

            if !target_inds.is_empty(){
                println!("--------------------------------------------------\n");
                println!("Box {} of {}, with index {}\n", box_num + 1, self.target_inds.len(), box_ind);
                
                let min_box_samples: usize = near_field_inds.len() + target_inds.len();
                let min_sketch_samples = self.dim-len_residual;

                if min_sketch_samples > min_box_samples
                {
                    println!("Current number of samples: {}. Minimum number of samples: {}\n", self.y_data.num_samples, min_box_samples);
                    
                    if min_box_samples > self.y_data.num_samples {
                        let extra_num_samples: usize = min_box_samples - self.y_data.num_samples;

                        println!("***************");
                        println!("Extra samples: {}", extra_num_samples);

                        self.y_data.add_samples(extra_num_samples, arr, &self.dec_boxes);
                        self.z_data.add_samples(extra_num_samples, arr, &self.dec_boxes);

                        println!("***************\n");
                    }

                    let mut box_features: BoxFeatures = <BoxFeatures as SkelBox<Self::Item>>::new(target_inds.clone(), near_field_inds);
                    let rank: Rank<Self::Item> = box_features.decouple(arr.shape()[0], &mut self.y_data, &mut self.z_data, &self.tols);
                    match rank{
                        Rank::Low(decoupled_box) => {
                            self.ind_s[box_ind].clone_from(&box_features.ind_s);
                            self.acc_ind_r.extend_from_slice(&box_features.ind_r.clone());
                            self.dec_boxes.push(DecoupledBoxData{operators: decoupled_box, box_features, level_it});
                        },
                        Rank::Full => {},
                    }

                    len_sketch += self.ind_s[box_ind].len();
                    println!("Level sketch points: {}", len_sketch);
                    len_residual = self.acc_ind_r.len();
                    println!("Residual points: {}", len_residual);
                    println!("Remaining points in sketch: {}", self.dim-len_residual);

                }
                else if min_sketch_samples > self.y_data.num_samples {
                    let extra_num_samples: usize = min_sketch_samples - self.y_data.num_samples;

                    println!("***************");
                    println!("Extra samples: {}", extra_num_samples);

                    self.y_data.add_samples(extra_num_samples, arr, &self.dec_boxes);
                    self.z_data.add_samples(extra_num_samples, arr, &self.dec_boxes);

                    println!("***************\n");
                }

                if self.y_data.num_samples >= self.dim-len_residual{
                    println!("Enough Samples");
                    state = State::PartialSketching;
                    break;
                }
            }
        }
        state
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
                    if box_key.level() > 0{
                        let box_parent = box_key.parent();
                        if !upper_level_keys.iter().any(|&&k| k == box_parent){
                            if let Some(parent_box) = self.level_indexing.boxes_map.get(box_key){
                                self.target_inds[box_ind].extend_from_slice(parent_box);
                            }
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
*/