use super::{box_skeletonisation::{BoxNearField, Rank, SkelBox, Tols}, rsrs_factors::{RsrsFactors, RsrsFactorsOps}, sketch::{BoxesData, SketchOps}, tree_indexing::{TreeData, TreeIndexing}};
use rand_distr::{Distribution, Standard, StandardNormal};
use mpi::traits::CommunicatorCollectives;
use bempp_octree::{MortonKey, Octree};
use rlst::dense::tools::RandScalar;
use std::time::{Duration, Instant};
pub use rlst::prelude::*;

type Inds<T> = Vec<Vec<T>>;

pub struct RsrsData<Item: RlstScalar> 
{
    level_indexing: TreeData,
    tols: Tols<Item>,
    pub y_data: BoxesData<Item>,
    pub z_data: BoxesData<Item>,
    dim: usize,
    ind_s: Inds<usize>,
    ind_r: Inds<usize>,
    target_inds: Inds<usize>,
    near_inds: Inds<usize>,
}

pub struct RsrsOptions{
    pub hermitian: bool,
    pub silent: bool
}

pub trait Rsrs{
    type Item: RlstScalar;
    fn new<C: CommunicatorCollectives>(arr: &DynamicArray<Self::Item, 2>, tols: Tols<Self::Item>, octree: &Octree<'_, C>)->Self;
    fn tree_cycle_and_diag_block_extraction(&mut self, arr: &DynamicArray<Self::Item, 2>, options: RsrsOptions)->RsrsFactors<Self::Item>;
    fn tree_cycle(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions);
    fn level_iteration(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions)->State;
    fn get_level_indices(&mut self, level: usize, options: &RsrsOptions);
    fn get_near_indices(&mut self, box_ind: usize)->Vec<usize>;
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

    fn new<C: CommunicatorCollectives>(arr: &DynamicArray<Self::Item, 2>, tols: Tols<Self::Item>, octree: &Octree<'_, C>)->Self{
        let dim: usize = arr.shape()[0];
        let level_indexing: TreeData = <TreeData as TreeIndexing>::new(octree);
        let target_inds: Inds<usize> = Vec::new();
        let near_inds: Inds<usize> = Vec::new();
        let ind_s: Inds<usize> = Vec::new();
        let ind_r: Inds<usize> = Vec::new();
        let y_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr, false);
        let z_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr,true);
        Self{level_indexing, y_data, z_data, tols, dim, ind_s, ind_r, target_inds, near_inds}
    }

    fn tree_cycle_and_diag_block_extraction(&mut self, arr: &DynamicArray<Self::Item, 2>, options: RsrsOptions)-> RsrsFactors<Self::Item>
    {
        let algo_start: Instant = Instant::now();
        let mut rsrs_factors = <RsrsFactors<Self::Item> as RsrsFactorsOps>::new();
        let start: Instant = Instant::now();
        self.tree_cycle(arr, &mut rsrs_factors, &options);
        let duration = start.elapsed();
        println!("Tree cycle elapsed time: {} s", duration.as_secs());
        println!("Extracting diagonal blocks");
        let start: Instant = Instant::now();
        self.y_data.extract_diag_boxes(self.ind_r.clone(), self.ind_s.clone(), self.tols.lstq, &mut rsrs_factors);
        let duration = start.elapsed();
        println!("Extraction time: {} s\n", duration.as_secs());
        let duration = algo_start.elapsed();
        println!("Total elapsed time: {} s, with {} samples\n", duration.as_secs(), self.y_data.num_samples);

        rsrs_factors
    }

    fn tree_cycle(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions){
        
        let mut level: usize = self.level_indexing.max_level;
        let min_level: usize = 1;

        while level > min_level{
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            let start: Instant = Instant::now();
            self.get_level_indices(level, options);
            println!("Current Level: {}\n\n", level);
            self.level_iteration(arr, rsrs_factors, options);
            println!("End level cycle\n");
            let duration: Duration = start.elapsed();
            println!("Elapsed time: {} s", duration.as_secs());
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

            let len_s: usize = self.ind_s.iter().map(|sketch_inds| sketch_inds.len()).sum();
            let len_r: usize = self.ind_r.iter().map(|residual_inds| residual_inds.len()).sum();

            println!("Sketch Points: {}", len_s);
            println!("Residual Points: {}", len_r);

            level -=1;

            if level <= min_level{
                println!("\nReached lower level: {}", level);

                let len_residual: usize = self.ind_r.iter().map(|residual_inds| residual_inds.len()).sum();
                let min_sketch_samples = self.dim-len_residual;

                if min_sketch_samples > self.y_data.num_samples{
                    let extra_num_samples = min_sketch_samples - self.y_data.num_samples;

                    if !options.silent{
                        println!("Extra {} samples", extra_num_samples);
                    }

                    self.y_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent, 0_u64);
                    if !options.hermitian{
                        self.z_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent, 0_u64);
                    }
                }

                break;
            }
        }
    }

    fn level_iteration(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions)->State{
        let mut box_indices: Vec<usize> = (0..self.target_inds.len()).collect::<Vec<_>>();
        box_indices = box_indices.into_iter().filter(|&box_ind| !self.ind_s[box_ind].is_empty()).collect::<Vec<_>>();
        box_indices.sort_by_key(|&box_ind| self.ind_s[box_ind].len() + self.get_near_indices(box_ind).len());
        let mut len_sketch: usize = 0;
        let mut state = State::FullSketching;
        let mut len_residual = 0;

        for (box_num, &box_ind) in box_indices.iter().enumerate(){
            let near_field_inds: Vec<usize> = self.get_near_indices(box_ind);
            if !options.silent{
                println!("--------------------------------------------------\n");
                println!("Box {} of {} with {} targets and {} near indices\n", box_num + 1, self.ind_s.len(), self.ind_s[box_ind].len(), near_field_inds.len());
            }
            
            let min_box_samples = near_field_inds.len() + self.ind_s[box_ind].len();
            let min_sketch_samples = self.dim-len_residual;

            if min_sketch_samples > min_box_samples
            {
                if !options.silent{
                    println!("Current number of samples: {}. Minimum number of samples: {}", self.y_data.num_samples, min_box_samples);
                    println!("Minimum samples to finish {}\n", min_sketch_samples);
                }
                
                if min_box_samples > self.y_data.num_samples{
                    let extra_num_samples = min_box_samples - self.y_data.num_samples;

                    if !options.silent{
                        println!("***************");
                        println!("Extra samples: {}", extra_num_samples);
                    }

                    self.y_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent, box_num as u64);
                    if !options.hermitian{
                        self.z_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent, box_num as u64);
                    } 
                    
                    if !options.silent{
                        println!("***************\n");
                    }
                }

                let mut box_features: BoxNearField= <BoxNearField as SkelBox<Self::Item>>::new(near_field_inds);

                let rank: Rank = box_features.decouple(&mut self.ind_s[box_ind], &mut self.y_data, &mut self.z_data, rsrs_factors, &self.tols, options);
                
                match rank{
                    Rank::Low(ind_r, ind_s) => {
                        self.ind_s[box_ind] = ind_s;
                        self.ind_r.push(ind_r);
                    },
                    Rank::Full => {},
                }

                len_sketch += self.ind_s[box_ind].len();
                
                len_residual = self.ind_r.iter().map(|residual_inds| residual_inds.len()).sum();

                if !options.silent{
                    println!("Level sketch points: {}", len_sketch);
                    println!("Residual points: {}", len_residual);
                    println!("Remaining points to be decomposed: {}", self.dim-len_residual);
                }

            }
            else if min_sketch_samples > self.y_data.num_samples{
                let extra_num_samples  = min_sketch_samples - self.y_data.num_samples;

                if !options.silent{
                    println!("***************");
                    println!("Extra samples: {}", extra_num_samples);
                }

                self.y_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent,(box_num + 1) as u64);
                if !options.hermitian{
                    self.z_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent,(box_num + 1) as u64);
                }

                if !options.silent{
                    println!("***************\n");
                }
            }

            if self.y_data.num_samples >= self.dim-len_residual{
                if !options.silent{
                    println!("Enough Samples");
                }
                state = State::PartialSketching;
                break;
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

    fn get_level_indices(&mut self, level: usize, options: &RsrsOptions){
        if !options.silent{
            println!("Computing Indices...\n");
        }
        if level < self.level_indexing.max_level{
            let binding: std::collections::HashSet<MortonKey> = self.level_indexing.level_keys.clone();
            let previous_level_keys: Vec<&MortonKey> = binding.iter().collect::<Vec<_>>();
            self.level_indexing.update_level_keys();
            let binding: std::collections::HashSet<MortonKey> = self.level_indexing.level_keys.clone();
            let current_level_keys: Vec<&MortonKey> = binding.iter().collect::<Vec<_>>();
            let mut target_inds: Inds<usize> = Vec::new();
            self.near_inds.clear();
            self.near_inds.resize(current_level_keys.len(), Vec::new());
            target_inds.resize(current_level_keys.len(), Vec::new());

            for (box_ind, box_key) in previous_level_keys.iter().enumerate(){
                if box_key.level() > 0{
                    if let Some(parent_index) = current_level_keys.iter().position(|&r| *r == box_key.parent()){
                        target_inds[parent_index].extend_from_slice(&self.ind_s[box_ind]);
                    }
                    else if let Some(parent_index) = current_level_keys.iter().position(|&r| r == *box_key){
                        target_inds[parent_index].extend_from_slice(&self.target_inds[box_ind]);
                    }
                }
            }

            self.target_inds.clear();
            self.target_inds = target_inds;
            self.ind_s.clear();
            self.ind_s.clone_from(&self.target_inds);
            
            for (box_ind, box_key) in current_level_keys.iter().enumerate(){
                self.near_inds[box_ind].push(box_ind);
                let near_keys: std::collections::HashSet<MortonKey> = self.level_indexing.get_box_near_field_keys(box_key);
                for near_box_key in near_keys.iter(){
                    if let Some(near_box_ind) = current_level_keys.iter().position(|&r| r == near_box_key){
                        self.near_inds[box_ind].push(near_box_ind);
                    }
                }
            }
    
            let boxes_lengths: Vec<usize> = self.ind_s.iter().map(|ind| ind.len()).collect::<Vec<_>>();

            if !options.silent{
                println!("New {} boxes of lengths {:?}, and active indices: {}", self.ind_s.len(), boxes_lengths, boxes_lengths.iter().sum::<usize>());
            }

        }
        else {
            self.target_inds.resize(self.level_indexing.level_keys.len(), Vec::new());
            self.ind_s.resize(self.level_indexing.level_keys.len(), Vec::new());
            self.near_inds.resize(self.level_indexing.level_keys.len(), Vec::new());
            for (box_ind, box_key) in self.level_indexing.level_keys.clone().iter().enumerate(){
                if let Some(box_indices) = self.level_indexing.boxes_map.get(box_key){
                    self.target_inds[box_ind] = box_indices.to_vec();
                    if box_key.level() == self.level_indexing.max_level{
                        self.ind_s[box_ind] = box_indices.to_vec();
                    }
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

            let boxes_lengths: Vec<usize> = self.target_inds.iter().map(|ind| ind.len()).collect::<Vec<_>>();
            if !options.silent{
                println!("New {} boxes of lengths {:?}, and active indices: {}", self.target_inds.len(), boxes_lengths, boxes_lengths.iter().sum::<usize>());
            }
        }
    }
}

