use crate::{
    rsrs::{
        rsrs_factors::FactorType,
        sketch::{update_sketch_id, update_sketch_lu},
    },
    //with_openblas_threads,
};

use super::{
    box_skeletonisation::{
        IdTimes, IdTimesOperations, LuTimesOperations, Rank, Skel, Tols, UpdateTimes,
        UpdateTimesOperations,
    },
    rsrs_factors::{IdFactor, LuFactor, LuTimes, RsrsFactors, RsrsFactorsOps},
    sketch::{BoxesData, SketchOps},
    tree_indexing::{TreeData, TreeIndexing},
};
use bempp_octree::{MortonKey, Octree};
use mpi::traits::CommunicatorCollectives;
use num::FromPrimitive;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rlst::dense::tools::RandScalar;
pub use rlst::prelude::*;
use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

type Inds<T> = Vec<Vec<T>>;

#[derive(Debug)]
pub struct Stats {
    pub sampling_time: Vec<u128>,
    pub sampling_extraction_time: u128,
    pub id_times: Vec<IdTimes>,
    pub tot_id_time: u128,
    pub lu_times: Vec<LuTimes>,
    pub tot_lu_time: u128,
    pub update_times: Vec<UpdateTimes>,
    pub total_elapsed_time: u64,
    pub extraction_time: u128,
    pub residual_size: usize,
    pub ranks: Vec<usize>,
    pub box_sizes: Vec<usize>,
    pub near_field_sizes: Vec<usize>,
    pub dec_boxes_per_level: Vec<usize>,
}

pub struct RsrsData<Item: RlstScalar> {
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
    current_box_indices: Vec<usize>,
    pub stats: Stats,
}

#[derive(Debug, Clone)]
pub enum BoxType {
    Merged,
    New,
}

pub enum Termination {
    EnoughSamples,
    ReachRoot,
}

pub struct RsrsOptions {
    pub termination: Termination,
    pub hermitian: bool,
    pub silent: bool,
    pub oversampling: usize,
    pub adaptive_tol: bool,
}

type Real<T> = <T as rlst::RlstScalar>::Real;

pub trait Rsrs {
    type Item: RlstScalar;
    fn new<C: CommunicatorCollectives>(
        arr: &DynamicArray<Self::Item, 2>,
        tols: Tols<Self::Item>,
        octree: &Octree<'_, C>,
    ) -> Self;
    fn tree_cycle_and_diag_block_extraction(
        &mut self,
        arr: &DynamicArray<Self::Item, 2>,
        options: &RsrsOptions,
    ) -> RsrsFactors<Self::Item>;
    fn tree_cycle(
        &mut self,
        arr: &DynamicArray<Self::Item, 2>,
        rsrs_factors: &mut RsrsFactors<Self::Item>,
        options: &RsrsOptions,
    );
    fn split_level_iteration(
        &mut self,
        arr: &DynamicArray<Self::Item, 2>,
        rsrs_factors: &mut RsrsFactors<Self::Item>,
        options: &RsrsOptions,
        level_it: usize,
    );
    fn sampling_step(
        &mut self,
        arr: &DynamicArray<Self::Item, 2>,
        rsrs_factors: &RsrsFactors<Self::Item>,
        options: &RsrsOptions,
    );
    fn id_level_iteration(
        &mut self,
        options: &RsrsOptions,
    ) -> (Vec<IdFactor<Self::Item>>, Vec<Vec<usize>>, Vec<Vec<usize>>);
    fn lu_level_iteration(
        &mut self,
        level_near_field_inds: &Vec<Vec<usize>>,
        level_ind_r: &Vec<Vec<usize>>,
        options: &RsrsOptions,
    ) -> Vec<Vec<LuFactor<Self::Item>>>;
    fn get_level_indices(&mut self, level: usize, options: &RsrsOptions);
    fn get_near_indices(&mut self, box_ind: usize) -> Vec<usize>;
}

fn oversample(samples: usize, oversampling: usize) -> usize {
    samples + (samples / 100) * oversampling
}

impl<T: RlstScalar + MatrixId + MatrixNull + MatrixInverse + MatrixPseudoInverse + RandScalar> Rsrs
    for RsrsData<T>
where
    StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>,
{
    type Item = T;

    fn new<C: CommunicatorCollectives>(
        arr: &DynamicArray<Self::Item, 2>,
        tols: Tols<Self::Item>,
        octree: &Octree<'_, C>,
    ) -> Self {
        let dim: usize = arr.shape()[0];
        let level_indexing: TreeData = <TreeData as TreeIndexing>::new(octree);
        let target_inds: Inds<usize> = Vec::new();
        let near_inds: Inds<usize> = Vec::new();
        let ind_s: Inds<usize> = Vec::new();
        let ind_r: Inds<usize> = Vec::new();
        let box_types: Vec<BoxType> = Vec::new();
        let y_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr, false);
        let z_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr, true);
        let current_box_indices = Vec::new();
        let id_times = Vec::new();
        let lu_times = Vec::new();
        let update_times = Vec::new();

        let stats = Stats {
            sampling_time: Vec::new(),
            sampling_extraction_time: 0_u128,
            id_times,
            tot_id_time: 0_u128,
            lu_times,
            tot_lu_time: 0_u128,
            update_times,
            total_elapsed_time: 0_u64,
            extraction_time: 0_u128,
            residual_size: 0,
            ranks: Vec::new(),
            box_sizes: Vec::new(),
            near_field_sizes: Vec::new(),
            dec_boxes_per_level: Vec::new(),
        };

        Self {
            level_indexing,
            y_data,
            z_data,
            tols,
            dim,
            ind_s,
            ind_r,
            box_types,
            target_inds,
            near_inds,
            current_box_indices,
            stats,
        }
    }

    fn tree_cycle_and_diag_block_extraction(
        &mut self,
        arr: &DynamicArray<Self::Item, 2>,
        options: &RsrsOptions,
    ) -> RsrsFactors<Self::Item> {
        let num_levels: usize = self.level_indexing.max_level;
        let algo_start: Instant = Instant::now();
        let mut rsrs_factors = <RsrsFactors<Self::Item> as RsrsFactorsOps>::new(num_levels);
        let start: Instant = Instant::now();
        self.tree_cycle(arr, &mut rsrs_factors, &options);
        let duration = start.elapsed();
        println!("Tree cycle elapsed time: {} s", duration.as_secs());
        println!("Extracting diagonal blocks");
        let start: Instant = Instant::now();
        self.y_data.extract_diag_boxes(
            self.ind_r.clone(),
            self.ind_s.clone(),
            self.tols.lstq,
            &mut rsrs_factors,
        );
        let extraction_time = start.elapsed();
        println!(
            "Extraction time: {} s, {} ms \n",
            extraction_time.as_secs(),
            extraction_time.as_millis()
        );
        self.stats.extraction_time = extraction_time.as_millis();
        let duration = algo_start.elapsed();
        println!(
            "Total elapsed time: {} s, with {} samples\n",
            duration.as_secs(),
            self.y_data.num_samples
        );
        self.stats.total_elapsed_time = duration.as_secs();

        rsrs_factors
    }

    fn tree_cycle(
        &mut self,
        arr: &DynamicArray<Self::Item, 2>,
        rsrs_factors: &mut RsrsFactors<Self::Item>,
        options: &RsrsOptions,
    ) {
        let mut level: usize = self.level_indexing.max_level;
        let mut level_it = 0;
        let min_level: usize = 1;

        while level > min_level {
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            let start: Instant = Instant::now();
            self.get_level_indices(level, options);
            println!("Current Level: {}\n\n", level);
            self.split_level_iteration(arr, rsrs_factors, options, level_it);
            println!("End level cycle\n");
            let duration: Duration = start.elapsed();
            println!("Elapsed time: {} s", duration.as_secs());
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

            let len_s: usize = self.ind_s.iter().map(|sketch_inds| sketch_inds.len()).sum();
            let len_r: usize = self
                .ind_r
                .iter()
                .map(|residual_inds| residual_inds.len())
                .sum();

            println!("Sketch Points: {}", len_s);
            println!("Residual Points: {}", len_r);
            println!("Current Number of Samples: {}", self.y_data.num_samples);

            level -= 1;
            level_it += 1;

            if level <= min_level {
                println!("\nReached lower level: {}", level);
                let len_residual: usize = self
                    .ind_r
                    .iter()
                    .map(|residual_inds| residual_inds.len())
                    .sum();
                self.stats.residual_size = len_residual;
                let min_sketch_samples = oversample(self.dim - len_residual, options.oversampling); //(self.dim-len_residual) + ((self.dim-len_residual)/100)*options.oversampling;

                if min_sketch_samples > self.y_data.num_samples {
                    let extra_num_samples = min_sketch_samples - self.y_data.num_samples;

                    if !options.silent {
                        println!("Extra {} samples", extra_num_samples);
                    }

                    let (mut tot_sampling_time, mut tot_id_update, mut tot_lu_update) = self.y_data.add_samples(
                        extra_num_samples,
                        arr,
                        rsrs_factors,
                        options.silent,
                        0_u64,
                    );
                    if !options.hermitian {
                        let (tot_z_sampling_time, tot_z_id_update, tot_z_lu_update) = self.z_data.add_samples(
                            extra_num_samples,
                            arr,
                            rsrs_factors,
                            options.silent,
                            0_u64,
                        );
                        tot_sampling_time += tot_z_sampling_time;
                        tot_id_update += tot_z_id_update;
                        tot_lu_update += tot_z_lu_update;
                    }
                    println!("Sampling Time: {:?} ms", tot_sampling_time);
                    println!("Update times: {}, {} ms", tot_id_update, tot_lu_update);

                    self.stats.sampling_extraction_time = tot_sampling_time;
                    let mut update_times = UpdateTimes::new();
                    update_times.sum(tot_id_update, tot_lu_update);
                    self.stats.update_times.push(update_times);
                }

                break;
            }
        }
    }

    fn sampling_step(
        &mut self,
        arr: &DynamicArray<Self::Item, 2>,
        rsrs_factors: &RsrsFactors<Self::Item>,
        options: &RsrsOptions,
    ) {
        let mut box_indices: Vec<usize> = (0..self.target_inds.len()).collect::<Vec<_>>();

        box_indices = box_indices
            .into_iter()
            .filter(|&box_ind| !self.ind_s[box_ind].is_empty())
            .collect::<Vec<_>>();

        box_indices.sort_by_key(|&box_ind| {
            self.ind_s[box_ind].len() + self.get_near_indices(box_ind).len()
        });

        self.current_box_indices = box_indices;

        let last_box_index = *self.current_box_indices.last().unwrap();
        let min_num_samples = oversample(
            self.ind_s[last_box_index].len() + self.get_near_indices(last_box_index).len(),
            options.oversampling,
        );

        let extra_num_samples = min_num_samples.saturating_sub(self.y_data.num_samples);

        if !options.silent {
            println!("***************");
            println!("Extra samples: {}", extra_num_samples);
        }

        if extra_num_samples > 0 {
            let (mut tot_sampling_time, mut tot_id_update, mut tot_lu_update) = self.y_data.add_samples(
                extra_num_samples,
                arr,
                rsrs_factors,
                options.silent,
                1,
            );

            if !options.hermitian {
                let (tot_z_sampling_time, tot_z_id_update, tot_z_lu_update) = self.z_data.add_samples(
                    extra_num_samples,
                    arr,
                    rsrs_factors,
                    options.silent,
                    1,
                );
                tot_sampling_time += tot_z_sampling_time;
                tot_id_update += tot_z_id_update;
                tot_lu_update += tot_z_lu_update;
            }
            println!("Sampling Time: {:?} ms", tot_sampling_time);
            println!("Update times: {}, {} ms", tot_id_update, tot_lu_update);

            self.stats.sampling_time.push(tot_sampling_time);
            let mut update_times = UpdateTimes::new();
            update_times.sum(tot_id_update, tot_lu_update);
            self.stats.update_times.push(update_times);
        }

        if !options.silent {
            println!("***************\n");
        }
    }

    fn id_level_iteration(
        &mut self,
        options: &RsrsOptions,
    ) -> (Vec<IdFactor<T>>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let current_box_indices = self.current_box_indices.clone();
        let mut current_near_field_indices = Vec::new();
        current_box_indices
            .iter()
            .for_each(|box_ind| current_near_field_indices.push(self.get_near_indices(*box_ind)));

        let mut box_id_level_iteration_res: Vec<_> = current_box_indices
            .par_iter()
            .map(|&box_ind| {
                let box_num = current_box_indices
                    .iter()
                    .position(|cbi| *cbi == box_ind)
                    .unwrap();
                let mut near_field_inds = &current_near_field_indices[box_num];
                if !options.silent {
                    println!("--------------------------------------------------\n");
                    println!(
                        "Box {} of {} with {} targets and {} near indices\n",
                        box_ind,
                        self.ind_s.len(),
                        self.ind_s[box_ind].len(),
                        near_field_inds.len()
                    );
                }
                let min_box_samples = oversample(
                    near_field_inds.len() + self.ind_s[box_ind].len(),
                    options.oversampling,
                );

                let mut skel_box = <Self::Item as Default>::default();

                let rank = skel_box.id_step(
                    &self.box_types[box_ind],
                    &self.ind_s[box_ind],
                    &mut near_field_inds,
                    &self.y_data,
                    &self.z_data,
                    min_box_samples,
                    &self.tols,
                    options,
                );

                (box_ind, rank)
            })
            .collect();

        

        let mut len_sketch = 0;
        let mut len_full_rank = 0;
        let mut num_dec_boxes = 0;

        box_id_level_iteration_res.sort_by_key(|&(i, _)| i);
        self.current_box_indices.clear();

        let mut level_near_field_inds = Vec::new();
        let mut level_ind_r = Vec::new();
        let mut id_times = IdTimes::new();

        let low_rank_res: Vec<_> = box_id_level_iteration_res
            .into_iter()
            .filter_map(|res| {
                let (box_ind, result) = res;

                match result {
                    Rank::Low(low_rank_result) => {
                        self.current_box_indices.push(box_ind);
                        level_near_field_inds.push(low_rank_result.near_field_inds.clone());
                        level_ind_r.push(low_rank_result.id_factor.ind_r.clone());

                        self.ind_s[box_ind] = low_rank_result.id_factor.ind_s.clone();
                        self.ind_r.push(low_rank_result.id_factor.ind_r.clone());
                        let box_size = self.target_inds[box_ind].len();

                        id_times.sum(
                            low_rank_result.id_times.nullification,
                            low_rank_result.id_times.id,
                        );
                        self.stats.ranks.push(self.ind_s[box_ind].len());
                        self.stats.box_sizes.push(box_size);
                        self.stats
                            .near_field_sizes
                            .push(low_rank_result.near_field_inds.len());

                        len_sketch += self.ind_s[box_ind].len();
                        num_dec_boxes += 1;

                        Some(low_rank_result.id_factor)
                    }
                    Rank::Full(it_id_times) => {
                        len_full_rank += self.ind_s[box_ind].len();
                        id_times.sum(it_id_times.nullification, it_id_times.id);
                        None
                    }
                }
            })
            .collect();

        self.stats.dec_boxes_per_level.push(num_dec_boxes);
        self.stats.id_times.push(id_times);

        let len_residual: usize = self
            .ind_r
            .iter()
            .map(|residual_inds| residual_inds.len())
            .sum();

        if !options.silent {
            println!("Level sketch points: {}", len_sketch);
            println!("Full rank points: {}", len_full_rank);
            println!("Residual points: {}", len_residual);
            println!(
                "Remaining points to be decomposed: {}",
                self.dim - len_residual
            );
        }

        (low_rank_res, level_near_field_inds, level_ind_r)
    }

    fn lu_level_iteration(
        &mut self,
        level_near_field_inds: &Vec<Vec<usize>>,
        level_ind_r: &Vec<Vec<usize>>,
        options: &RsrsOptions,
    ) -> Vec<Vec<LuFactor<T>>> {
        let independent_near_fields = group_near_fields(level_near_field_inds);

        let batches_res: Vec<_> = independent_near_fields
            .into_iter()
            .map(|batch| {
                let batch_res: Vec<_> = batch
                    .par_iter()
                    .map(|box_num| {
                        let skel_box = <Self::Item as Default>::default();
                        let box_ind = self.current_box_indices[*box_num];
                        let min_num_samples = oversample(
                            self.target_inds[box_ind].len() + level_near_field_inds[*box_num].len(),
                            options.oversampling,
                        );
                        let (lu_factor, lu_times) = skel_box.lu_step(
                            &self.y_data,
                            &self.z_data,
                            &level_ind_r[*box_num],
                            &level_near_field_inds[*box_num],
                            min_num_samples,
                            &self.tols,
                            options,
                        );
                        (*box_num, lu_factor, lu_times)
                    })
                    .collect();

                let batch_reduced_res: Vec<_> = batch_res
                    .into_iter()
                    .map(|(box_ind, lu_factor, lu_times)| {
                        let start: Instant = Instant::now();
                        update_sketch_lu(
                            &mut self.y_data.sketch,
                            &mut self.y_data.test,
                            &lu_factor,
                            FactorType::F,
                            FactorType::S,
                            false,
                        );
                        if !options.hermitian {
                            update_sketch_lu(
                                &mut self.z_data.sketch,
                                &mut self.z_data.test,
                                &lu_factor,
                                FactorType::S,
                                FactorType::F,
                                true,
                            );
                        }
                        let update_lu_time: Duration = start.elapsed();
                        if !options.silent {
                            println!("Update from LU in {} ms", update_lu_time.as_millis());
                        }

                        (box_ind, lu_factor, lu_times, update_lu_time)
                    })
                    .collect();
                batch_reduced_res
            })
            .collect();

        self.current_box_indices.clear();

        let mut lu_times = LuTimes::new();
        let mut update_times = UpdateTimes::new();
        let batches_res: Vec<_> = batches_res
            .into_iter()
            .map(|batch_res| {
                let batch_res: Vec<_> = batch_res
                    .into_iter()
                    .map(|(_box_ind, lu_factor, it_lu_times, update_lu_time)| {
                        lu_times.sum(it_lu_times.extraction, it_lu_times.lu);
                        update_times.sum(0_u128, update_lu_time.as_millis());
                        lu_factor
                    })
                    .collect();
                batch_res
            })
            .collect();

        self.stats.lu_times.push(lu_times);
        self.stats.update_times.push(update_times);

        batches_res
    }

    fn split_level_iteration(
        &mut self,
        arr: &DynamicArray<Self::Item, 2>,
        rsrs_factors: &mut RsrsFactors<Self::Item>,
        options: &RsrsOptions,
        level_it: usize,
    ) {
        let merged_count = self.box_types.iter().filter(|box_type| matches!(box_type, BoxType::Merged)).count();
        println!("Number of merged boxes: {}", merged_count);

        if level_it > 1 && options.adaptive_tol {
            self.tols.id_2 = self.tols.id_2 * Real::<Self::Item>::from_f64(10.0).unwrap();
            if self.tols.id_2 >= Real::<Self::Item>::from_f64(1e4).unwrap()*self.tols.id {
                self.tols.id_2 = Real::<Self::Item>::from_f64(1e4).unwrap()*self.tols.id;
            }
            if self.tols.id_2 > Real::<Self::Item>::from_f64(1e-1).unwrap() {
                self.tols.id_2 = Real::<Self::Item>::from_f64(1e-1).unwrap();
            }
            
        }

        println!("Current tolerances: {}, {}", self.tols.id, self.tols.id_2);

        self.sampling_step(arr, rsrs_factors, options);
        let id_step_start: Instant = Instant::now();
        let (id_factors_res, level_near_field_inds, level_ind_r) = self.id_level_iteration(options);
        rsrs_factors.id_factors[level_it] = id_factors_res;
        let id_step_duration = id_step_start.elapsed();
        self.stats.tot_id_time += id_step_duration.as_millis();

        let start_tot_update: Instant = Instant::now();

        let id_update_times: Vec<_> = rsrs_factors.id_factors[level_it]
            .iter()
            .map(|id_factor| {
                let start: Instant = Instant::now();
                update_sketch_id(
                    &mut self.y_data.sketch,
                    &mut self.y_data.test,
                    &id_factor,
                    FactorType::F,
                    FactorType::S,
                    false,
                );
                if !options.hermitian {
                    update_sketch_id(
                        &mut self.z_data.sketch,
                        &mut self.z_data.test,
                        &id_factor,
                        FactorType::S,
                        FactorType::F,
                        true,
                    );
                }
                let update_id_time: Duration = start.elapsed();

                update_id_time
            })
            .collect();

        let update_id_time: Duration = start_tot_update.elapsed();
        let mut update_times = UpdateTimes::new();

        id_update_times.iter().for_each(|update_id_time| {
            update_times.sum(update_id_time.as_millis(), 0_u128);
        });
        self.stats.update_times.push(update_times);

        if !options.silent {
            println!(
                "Update from ID factors in {} ms",
                update_id_time.as_millis()
            );
        }

        let lu_step_start: Instant = Instant::now();
        rsrs_factors.lu_factors[level_it] =
            self.lu_level_iteration(&level_near_field_inds, &level_ind_r, options);
        let lu_step_duration =
            lu_step_start.elapsed().as_millis() - self.stats.update_times.last().unwrap().lu;
        self.stats.tot_lu_time += lu_step_duration;
    }

    fn get_near_indices(&mut self, box_ind: usize) -> Vec<usize> {
        let mut near_indices = Vec::new();
        for ind in self.near_inds[box_ind].iter() {
            near_indices.extend_from_slice(&self.target_inds[*ind]);
        }
        near_indices
    }

    fn get_level_indices(&mut self, level: usize, options: &RsrsOptions) {
        if !options.silent {
            println!("Computing Indices...\n");
        }
        if level < self.level_indexing.max_level {
            let binding: std::collections::HashSet<MortonKey> =
                self.level_indexing.level_keys.clone();
            let previous_level_keys: Vec<&MortonKey> = binding.iter().collect::<Vec<_>>();
            self.level_indexing.update_level_keys();
            let binding: std::collections::HashSet<MortonKey> =
                self.level_indexing.level_keys.clone();
            let current_level_keys: Vec<&MortonKey> = binding.iter().collect::<Vec<_>>();
            let mut target_inds: Inds<usize> = Vec::new();
            let mut num_sons: Vec<usize> = Vec::new();
            self.near_inds.clear();
            self.near_inds.resize(current_level_keys.len(), Vec::new());
            let mut box_types = Vec::new();
            box_types
                .resize(current_level_keys.len(), BoxType::New);
            target_inds.resize(current_level_keys.len(), Vec::new());
            num_sons.resize(current_level_keys.len(), 0);

            for (box_ind, box_key) in previous_level_keys.iter().enumerate() {
                if let Some(parent_index) = current_level_keys
                    .iter()
                    .position(|&r| *r == box_key.parent())
                {
                    /*if self.ind_s[box_ind].len() < self.target_inds[box_ind].len() || matches!(self.box_types[box_ind], BoxType::Merged){
                        box_types[parent_index] = BoxType::Merged;
                    }*/
                    box_types[parent_index] = BoxType::Merged;
                    target_inds[parent_index].extend_from_slice(&self.ind_s[box_ind]);
                    num_sons[parent_index] += 1;
                    self.ind_s[box_ind].clear();
                }
            }

            for (box_ind, box_key) in previous_level_keys.iter().enumerate() {
                if let Some(parent_index) = current_level_keys.iter().position(|&r| r == *box_key) {
                    if num_sons[parent_index] == 0 {
                        if self.level_indexing.max_level - level == 1 {
                            target_inds[parent_index].extend_from_slice(&self.target_inds[box_ind]);
                            num_sons[parent_index] += 1;
                            self.target_inds[box_ind].clear();
                        } else {
                            target_inds[parent_index].extend_from_slice(&self.ind_s[box_ind]);
                            num_sons[parent_index] += 1;
                            self.ind_s[box_ind].clear();
                        }
                        
                    }
                }
            }

            self.box_types.clear();
            self.box_types = box_types;
            self.target_inds.clear();
            self.target_inds = target_inds;
            self.ind_s.clear();
            self.ind_s.clone_from(&self.target_inds);

            for (box_ind, box_key) in current_level_keys.iter().enumerate() {
                self.near_inds[box_ind].push(box_ind);
                let near_keys: std::collections::HashSet<MortonKey> =
                    self.level_indexing.get_box_near_field_keys(box_key);
                for near_box_key in near_keys.iter() {
                    if let Some(near_box_ind) =
                        current_level_keys.iter().position(|&r| r == near_box_key)
                    {
                        self.near_inds[box_ind].push(near_box_ind);
                    }
                }
            }

            let boxes_lengths: Vec<usize> =
                self.ind_s.iter().map(|ind| ind.len()).collect::<Vec<_>>();
            println!(
                "New {} boxes and number with {} active indices",
                self.target_inds.len(),
                boxes_lengths.iter().sum::<usize>()
            );

            if !options.silent {
                println!(
                    "New {} boxes of lengths {:?}, and active indices: {}",
                    self.ind_s.len(),
                    boxes_lengths,
                    boxes_lengths.iter().sum::<usize>()
                );
            }
        } else {
            self.target_inds
                .resize(self.level_indexing.level_keys.len(), Vec::new());
            self.ind_s
                .resize(self.level_indexing.level_keys.len(), Vec::new());
            self.near_inds
                .resize(self.level_indexing.level_keys.len(), Vec::new());
            self.box_types
                .resize(self.level_indexing.level_keys.len(), BoxType::New);

            for (box_ind, box_key) in self.level_indexing.level_keys.clone().iter().enumerate() {
                if let Some(box_indices) = self.level_indexing.boxes_map.get(box_key) {
                    self.target_inds[box_ind] = box_indices.to_vec();
                    if box_key.level() == self.level_indexing.max_level {
                        self.ind_s[box_ind] = box_indices.to_vec();
                    }
                }
            }

            for (box_ind, box_key) in self.level_indexing.level_keys.iter().enumerate() {
                self.near_inds[box_ind].push(box_ind);
                let near_keys: std::collections::HashSet<MortonKey> =
                    self.level_indexing.get_box_near_field_keys(box_key);
                for near_box_key in near_keys.iter() {
                    if let Some(near_box_ind) = self
                        .level_indexing
                        .level_keys
                        .iter()
                        .position(|&r| r == *near_box_key)
                    {
                        self.near_inds[box_ind].push(near_box_ind);
                    }
                }
            }

            let boxes_lengths: Vec<usize> = self
                .target_inds
                .iter()
                .map(|ind| ind.len())
                .collect::<Vec<_>>();
            println!(
                "New {} boxes and number of active indices: {}",
                self.target_inds.len(),
                boxes_lengths.iter().sum::<usize>()
            );
            if !options.silent {
                println!(
                    "New {} boxes of lengths {:?}, and active indices: {}",
                    self.target_inds.len(),
                    boxes_lengths,
                    boxes_lengths.iter().sum::<usize>()
                );
            }
        }
    }
}

fn group_near_fields(near_fields: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut acc_near_field_inds_groups: Vec<Vec<usize>> = Vec::new();
    let mut near_field_group_inds: Vec<Vec<usize>> = Vec::new();
    let mut used_ind: Vec<bool> = Vec::new();
    used_ind.resize(near_fields.len(), false);

    acc_near_field_inds_groups.push(Vec::new());
    near_field_group_inds.push(Vec::new());

    near_fields
        .iter()
        .enumerate()
        .for_each(|(box_ind, box_near_fields)| {
            let mut has_common = false;
            acc_near_field_inds_groups.iter_mut().enumerate().for_each(
                |(acc_ind, acc_near_field_inds)| {
                    let acc_near_field_inds_set: HashSet<_> =
                        acc_near_field_inds.iter().copied().collect();
                    let current_near_field_inds_set: HashSet<_> =
                        box_near_fields.iter().copied().collect();
                    if acc_near_field_inds_set.is_disjoint(&current_near_field_inds_set)
                        && !used_ind[box_ind]
                    {
                        acc_near_field_inds.extend_from_slice(box_near_fields);
                        near_field_group_inds[acc_ind].push(box_ind);
                        used_ind[box_ind] = true;
                    } else {
                        has_common = true;
                    }
                },
            );

            if has_common && !used_ind[box_ind] {
                acc_near_field_inds_groups.push(Vec::new());
                near_field_group_inds.push(Vec::new());
                let last_index = acc_near_field_inds_groups.len() - 1;
                acc_near_field_inds_groups[last_index].extend_from_slice(box_near_fields);
                near_field_group_inds[acc_near_field_inds_groups.len() - 1].push(box_ind);
                used_ind[box_ind] = true;
            }
        });

    near_field_group_inds
}
