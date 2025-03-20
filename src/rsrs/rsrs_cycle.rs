use super::{box_skeletonisation::{BoxStats, IdTimes, Rank, Skel, Tols, UpdateTimes}, rsrs_factors::{LuTimes, RsrsFactors, RsrsFactorsOps}, sketch::{BoxesData, SketchOps}, tree_indexing::{TreeData, TreeIndexing}};
use rand_distr::{Distribution, Standard, StandardNormal};
use mpi::traits::CommunicatorCollectives;
use bempp_octree::{MortonKey, Octree};
use rlst::dense::tools::RandScalar;
use std::time::{Duration, Instant};
pub use rlst::prelude::*;
use num::FromPrimitive;

type Inds<T> = Vec<Vec<T>>;
pub struct Stats{
    pub sampling_time: Vec<u128>,
    pub id_times: Vec<IdTimes>,
    pub lu_times: Vec<LuTimes>,
    pub update_times: Vec<UpdateTimes>,
    pub total_elapsed_time: u64,
    pub extraction_time: u128,
    pub residual_size: usize,
    pub ranks: Vec<usize>,
    pub box_sizes: Vec<usize>,
    pub near_field_sizes: Vec<usize>,
    pub dec_boxes_per_level: Vec<usize>
}

pub struct RsrsData<Item: RlstScalar> 
{
    level_indexing: TreeData,
    tols: Tols<Item>,
    pub y_data: BoxesData<Item>,
    pub z_data: BoxesData<Item>,
    dim: usize,
    ind_s: Inds<usize>,
    ind_r: Inds<usize>,
    box_types: Vec<BoxType>,
    target_inds: Inds<usize>,
    near_inds: Inds<usize>,
    pub stats: Stats
}

#[derive(Clone)]
pub enum BoxType{
    Merged,
    New
}

pub enum Termination{
    EnoughSamples,
    ReachRoot
}

type Real<T> = <T as rlst::RlstScalar>::Real;

pub struct RsrsOptions{
    pub split: bool,
    pub termination: Termination,
    pub hermitian: bool,
    pub silent: bool,
    pub oversampling: usize,
    pub adaptive_tol: bool
}

pub trait Rsrs{
    type Item: RlstScalar;
    fn new<C: CommunicatorCollectives>(arr: &DynamicArray<Self::Item, 2>, tols: Tols<Self::Item>, octree: &Octree<'_, C>)->Self;
    fn tree_cycle_and_diag_block_extraction(&mut self, arr: &DynamicArray<Self::Item, 2>, options: RsrsOptions)->RsrsFactors<Self::Item>;
    fn tree_cycle(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions);
    fn level_iteration(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions);
    fn split_level_iteration(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions);
    fn id_level_iteration(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions)->usize;
    fn lu_level_iteration(&mut self, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions, level_num_dec_boxes: usize);
    fn get_level_indices(&mut self, level: usize, options: &RsrsOptions);
    fn get_near_indices(&mut self, box_ind: usize)->Vec<usize>;
}

fn oversample(samples: usize, oversampling:usize)->usize{
    samples + (samples/100)*oversampling
}


impl <T:RlstScalar  +
MatrixId + MatrixNull + 
MatrixInverse + MatrixPseudoInverse + 
RandScalar>Rsrs for RsrsData<T> 
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
        let box_types: Vec<BoxType> = Vec::new();
        let y_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr, false);
        let z_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr,true);
        let stats = Stats{ 
            sampling_time: Vec::new(),  
            id_times: Vec::new(), 
            lu_times: Vec::new(), 
            update_times: Vec::new(), 
            total_elapsed_time: 0_u64, 
            extraction_time: 0_u128, 
            residual_size: 0,
            ranks: Vec::new(),
            box_sizes: Vec::new(),
            near_field_sizes: Vec::new(),
            dec_boxes_per_level: Vec::new()};

        Self{level_indexing, y_data, z_data, tols, dim, ind_s, ind_r, box_types, target_inds, near_inds, stats}
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
        let extraction_time = start.elapsed();
        println!("Extraction time: {} s, {} ms \n", extraction_time.as_secs(), extraction_time.as_millis());
        self.stats.extraction_time = extraction_time.as_millis();
        let duration = algo_start.elapsed();
        println!("Total elapsed time: {} s, with {} samples\n", duration.as_secs(), self.y_data.num_samples);
        self.stats.total_elapsed_time = duration.as_secs();

        rsrs_factors
    }

    fn tree_cycle(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions){
        
        let mut level: usize = self.level_indexing.max_level;
        let max_level: usize = self.level_indexing.max_level;
        let min_level: usize = 1;

        while level > min_level{
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

            /*let last_len_s: usize = self.ind_s.iter().map(|sketch_inds| sketch_inds.len()).sum();
            let last_len_r: usize = self.ind_r.iter().map(|residual_inds| residual_inds.len()).sum();
            let last_len_active: usize = last_len_s + last_len_r;*/

            let start: Instant = Instant::now();
            self.get_level_indices(level, options);
            println!("Current Level: {}\n\n", level);

            
            //println!("Active Points last iteration: {}, dimension: {}", last_len_active, self.y_data.dim);

            if level<max_level-1 && options.adaptive_tol{
                self.tols.id_2 = self.tols.id_2 * Real::<Self::Item>::from_f64(5.0).unwrap();
                if self.tols.id_2 > Real::<Self::Item>::from_f64(1e-1).unwrap(){
                    self.tols.id_2 = Real::<Self::Item>::from_f64(1e-1).unwrap();
                }
            }

            if options.split{
                self.split_level_iteration(arr, rsrs_factors, options);
            }
            else{
                self.level_iteration(arr, rsrs_factors, options);
            }
            println!("End level cycle\n");
            let duration: Duration = start.elapsed();
            println!("Elapsed time: {} s", duration.as_secs());
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

            let len_s: usize = self.ind_s.iter().map(|sketch_inds| sketch_inds.len()).sum();
            let len_r: usize = self.ind_r.iter().map(|residual_inds| residual_inds.len()).sum();

            println!("Sketch Points: {}", len_s);
            println!("Residual Points: {}", len_r);
            println!("Current Number of Samples: {}", self.y_data.num_samples);

            level -=1;

            if level <= min_level{
                println!("\nReached lower level: {}", level);
                let len_residual: usize = self.ind_r.iter().map(|residual_inds| residual_inds.len()).sum();
                self.stats.residual_size = len_residual;
                let min_sketch_samples = oversample(self.dim-len_residual, options.oversampling);//(self.dim-len_residual) + ((self.dim-len_residual)/100)*options.oversampling;

                if min_sketch_samples > self.y_data.num_samples{
                    let extra_num_samples = min_sketch_samples - self.y_data.num_samples;

                    if !options.silent{
                        println!("Extra {} samples", extra_num_samples);
                    }

                    let mut tot_sampling_time = self.y_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent, true, 0_u64);
                    if !options.hermitian{
                        let sampling_z_time = self.z_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent, true, 0_u64);
                        tot_sampling_time += sampling_z_time;
                    }
                    println!("Sampling Time: {:?} s", tot_sampling_time.as_secs());
                    self.stats.sampling_time.push(tot_sampling_time.as_millis());
                }

                break;
            }
        }
    }

    fn id_level_iteration(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions)->usize{
        let mut box_indices: Vec<usize> = (0..self.target_inds.len()).collect::<Vec<_>>();
        box_indices = box_indices.into_iter().filter(|&box_ind| !self.ind_s[box_ind].is_empty()).collect::<Vec<_>>();
        box_indices.sort_by_key(|&box_ind| self.ind_s[box_ind].len() + self.get_near_indices(box_ind).len());
        let last_box_index = *box_indices.last().unwrap();
        let min_num_samples = oversample(self.ind_s[last_box_index].len() + self.get_near_indices(last_box_index).len(), options.oversampling);
        
        let mut len_sketch: usize = 0;
        let mut len_residual = 0;
        let mut len_full_rank = 0;
        let mut num_dec_boxes = 0;

        let extra_num_samples = min_num_samples.saturating_sub(self.y_data.num_samples);

        if !options.silent{
            println!("***************");
            println!("Extra samples: {}", extra_num_samples);
        }

        if extra_num_samples > 0{
            let mut tot_sampling_time = self.y_data.add_samples(
                extra_num_samples, 
                arr, 
                rsrs_factors, 
                options.silent, 
                true, 
                0);

            if !options.hermitian{
                let sampling_z_time = self.z_data.add_samples(
                    extra_num_samples, 
                    arr, 
                    rsrs_factors, 
                    options.silent, 
                    true, 
                    0);

                tot_sampling_time += sampling_z_time;
            } 
            self.stats.sampling_time.push(tot_sampling_time.as_millis());
        }

        if !options.silent{
            println!("***************\n");
        }
        
        for (box_num, &box_ind) in box_indices.iter().enumerate(){
            let mut near_field_inds: Vec<usize> = self.get_near_indices(box_ind);
            if !options.silent{
                println!("--------------------------------------------------\n");
                println!("Box {} of {} with {} targets and {} near indices\n", box_num + 1, self.ind_s.len(), self.ind_s[box_ind].len(), near_field_inds.len());
            }
            let min_box_samples = oversample(near_field_inds.len() + self.ind_s[box_ind].len(), options.oversampling);
            let min_sketch_samples = self.dim-len_residual;

            if !options.silent{
                println!("Current number of samples: {}. Minimum number of samples: {}", self.y_data.num_samples, min_box_samples);
                println!("Minimum samples to finish {}\n", min_sketch_samples);
            }
            let mut skel_box = <Self::Item as Default>::default();

            let box_size = self.ind_s[box_ind].len();

            let rank = skel_box.id_step(
                &self.box_types[box_ind],
                &mut self.ind_s[box_ind], 
                &mut near_field_inds, 
                &mut self.y_data, 
                &mut self.z_data, 
                rsrs_factors, 
                min_box_samples, 
                &self.tols, 
                options);
            
            match rank{
                Rank::Low(id_times) => {
                    self.ind_s[box_ind] = rsrs_factors.dec_factors.last().unwrap().id_factor.ind_s.clone();
                    self.ind_r.push(rsrs_factors.dec_factors.last().unwrap().id_factor.ind_r.clone());
                    self.stats.id_times.push(id_times);
                    self.stats.ranks.push(self.ind_s[box_ind].len());
                    self.stats.box_sizes.push(box_size);
                    self.stats.near_field_sizes.push(near_field_inds.len());
                    len_sketch += self.ind_s[box_ind].len();
                    num_dec_boxes +=1;

                },
                Rank::Full(id_times) => {
                    self.stats.id_times.push(id_times);
                    len_full_rank += self.ind_s[box_ind].len();
                },
            }
            len_residual = self.ind_r.iter().map(|residual_inds| residual_inds.len()).sum();

            if !options.silent{
                println!("Level sketch points: {}", len_sketch);
                println!("Full rank points: {}", len_full_rank);
                println!("Residual points: {}", len_residual);
                println!("Remaining points to be decomposed: {}", self.dim-len_residual);
            }

            
        }
        self.stats.dec_boxes_per_level.push(num_dec_boxes);
        num_dec_boxes
    }

    fn lu_level_iteration(&mut self, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions, level_num_dec_boxes: usize){
        let tot_num_dec_factors = rsrs_factors.dec_factors.len();
        let dec_factor_indices: Vec<usize> = rsrs_factors.dec_factors.iter().enumerate().filter(|(box_ind, _)| *box_ind + 1 > tot_num_dec_factors-level_num_dec_boxes).map(|(box_ind, _)| box_ind).collect();
        
        for box_ind in dec_factor_indices {
            let dec_factor = &rsrs_factors.dec_factors[box_ind];
            let min_num_samples = oversample(dec_factor.id_factor.ind_r.len() + dec_factor.id_factor.ind_s.len() + dec_factor.near_field_inds.len(), options.oversampling);
            let skel_box = <Self::Item as Default>::default();
            let (lu_times, update_times) = skel_box.lu_step(&mut self.y_data, &mut self.z_data, rsrs_factors, min_num_samples, &self.tols, options, Some(box_ind));
            self.stats.lu_times.push(lu_times);
            self.stats.update_times.push(update_times);
        }
    }

    fn split_level_iteration(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions){
        let level_num_dec_boxes = self.id_level_iteration(arr, rsrs_factors, options);
        self.lu_level_iteration(rsrs_factors, options, level_num_dec_boxes);
        
    }

    fn level_iteration(&mut self, arr: &DynamicArray<Self::Item, 2>, rsrs_factors: &mut RsrsFactors<Self::Item>, options: &RsrsOptions){
        let mut box_indices: Vec<usize> = (0..self.target_inds.len()).collect::<Vec<_>>();
        box_indices = box_indices.into_iter().filter(|&box_ind| !self.ind_s[box_ind].is_empty()).collect::<Vec<_>>();
        box_indices.sort_by_key(|&box_ind| self.ind_s[box_ind].len() + self.get_near_indices(box_ind).len());
        let mut len_sketch: usize = 0;
        let mut len_residual = 0;
        let mut len_full_rank = 0;

        for (box_num, &box_ind) in box_indices.iter().enumerate(){
            let mut near_field_inds: Vec<usize> = self.get_near_indices(box_ind);
            if !options.silent{
                println!("--------------------------------------------------\n");
                println!("Box {} of {} with {} targets and {} near indices\n", box_num + 1, self.ind_s.len(), self.ind_s[box_ind].len(), near_field_inds.len());
            }
            
            let min_box_samples = oversample(near_field_inds.len() + self.ind_s[box_ind].len(), options.oversampling);
            let min_sketch_samples = oversample(self.dim-len_residual ,options.oversampling);

            match options.termination {
                Termination::EnoughSamples => {
                    if min_sketch_samples <= min_box_samples{
                        if min_sketch_samples > self.y_data.num_samples{
                            let extra_num_samples  = min_sketch_samples - self.y_data.num_samples;
            
                            if !options.silent{
                                println!("***************");
                                println!("Extra samples: {}", extra_num_samples);
                            }
            
                            self.y_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent, true, (box_num + 1) as u64);
                            if !options.hermitian{
                                self.z_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent, true, (box_num + 1) as u64);
                            }
            
                            if !options.silent{
                                println!("***************\n");
                            }
                        }

                        if !options.silent{
                            println!("Enough Samples");
                        }
                        break;
                    }
                },
                Termination::ReachRoot => {},
            }
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

                let mut tot_sampling_time = self.y_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent, true, box_num as u64);
                if !options.hermitian{
                    let sampling_z_time = self.z_data.add_samples(extra_num_samples, arr, rsrs_factors, options.silent, true, box_num as u64);
                    tot_sampling_time += sampling_z_time;
                } 
                self.stats.sampling_time.push(tot_sampling_time.as_millis());
                
                if !options.silent{
                    println!("***************\n");
                }
            }

            let mut skel_box = <Self::Item as Default>::default();

            let rank  = skel_box.id_and_lu_steps(&self.box_types[box_ind], &mut self.ind_s[box_ind], &mut near_field_inds, &mut self.y_data, &mut self.z_data, rsrs_factors, min_box_samples, &self.tols, options);
            
            match rank{
                BoxStats::Low(dec_times) => {
                    self.ind_s[box_ind] = rsrs_factors.dec_factors.last().unwrap().id_factor.ind_s.clone();
                    self.ind_r.push(rsrs_factors.dec_factors.last().unwrap().id_factor.ind_r.clone());
                    self.stats.id_times.push(dec_times.id_times);
                    self.stats.lu_times.push(dec_times.lu_times);
                    self.stats.update_times.push(dec_times.update_times);
                    len_sketch += self.ind_s[box_ind].len();
                },
                BoxStats::Full(id_times) => {
                    self.stats.id_times.push(id_times);
                    len_full_rank += self.ind_s[box_ind].len();
                },
            }
            
            len_residual = self.ind_r.iter().map(|residual_inds| residual_inds.len()).sum();

            if !options.silent{
                println!("Level sketch points: {}", len_sketch);
                println!("Full rank points: {}", len_full_rank);
                println!("Residual points: {}", len_residual);
                println!("Remaining points to be decomposed: {}", self.dim-len_residual);
            }

            if self.y_data.num_samples >= self.dim-len_residual{
                if !options.silent{
                    println!("Enough Samples");
                }
                break;
            }
        }
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
            let mut num_sons: Vec<usize> = Vec::new();
            self.near_inds.clear();
            self.near_inds.resize(current_level_keys.len(), Vec::new());
            self.box_types.resize(current_level_keys.len(), BoxType::New);
            target_inds.resize(current_level_keys.len(), Vec::new());
            num_sons.resize(current_level_keys.len(), 0);


            for (box_ind, box_key) in previous_level_keys.iter().enumerate(){
                if let Some(parent_index) = current_level_keys.iter().position(|&r| *r == box_key.parent()){
                    target_inds[parent_index].extend_from_slice(&self.ind_s[box_ind]);
                    num_sons[parent_index] +=1;
                    self.ind_s[box_ind].clear();
                    self.box_types[parent_index] = BoxType::Merged;
                }
            }
            
            for (box_ind, box_key) in previous_level_keys.iter().enumerate(){
                if let Some(parent_index) = current_level_keys.iter().position(|&r| r == *box_key){
                    if num_sons[parent_index] == 0{
                        if self.level_indexing.max_level-level==1
                        {
                            target_inds[parent_index].extend_from_slice(&self.target_inds[box_ind]);
                            num_sons[parent_index] +=1;
                            self.target_inds[box_ind].clear();
                        }
                        else{
                            target_inds[parent_index].extend_from_slice(&self.ind_s[box_ind]);
                            num_sons[parent_index] +=1;
                            self.ind_s[box_ind].clear();
                        }
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
            println!("New {} boxes and number with {} active indices", self.target_inds.len(), boxes_lengths.iter().sum::<usize>());

            if !options.silent{
                println!("New {} boxes of lengths {:?}, and active indices: {}", self.ind_s.len(), boxes_lengths, boxes_lengths.iter().sum::<usize>());
            }

        }
        else {
            self.target_inds.resize(self.level_indexing.level_keys.len(), Vec::new());
            self.ind_s.resize(self.level_indexing.level_keys.len(), Vec::new());
            self.near_inds.resize(self.level_indexing.level_keys.len(), Vec::new());
            self.box_types.resize(self.level_indexing.level_keys.len(), BoxType::New);

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
            println!("New {} boxes and number of active indices: {}", self.target_inds.len(), boxes_lengths.iter().sum::<usize>());
            if !options.silent{
                println!("New {} boxes of lengths {:?}, and active indices: {}", self.target_inds.len(), boxes_lengths, boxes_lengths.iter().sum::<usize>());
            }
        }
    }
}

