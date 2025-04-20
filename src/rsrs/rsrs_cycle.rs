use super::{
    box_skeletonisation::{
        IdTimes, IdTimesOperations, LuTimesOperations, Rank, Skel, Tols, UpdateTimes,
        UpdateTimesOperations,
    },
    rsrs_factors::{IdFactor, LuFactor, LuTimes, RsrsFactors, RsrsFactorsOps},
    sketch::{update_sketch_lu_no_subs, BoxesData, SketchOps},
    tree_indexing::{TreeData, TreeIndexing},
};
use crate::rsrs::{
    rsrs_factors::{FactorBatch, FactorBatchOperations, FactorType},
    sketch::{update_sketch_id, update_sketch_lu, update_sketch_lu_subs, UpdateType},
};
use bempp_octree::{MortonKey, Octree};
use mpi::traits::CommunicatorCollectives;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rlst::dense::{linalg::lu::MatrixLu, tools::RandScalar};
pub use rlst::prelude::*;
use std::{
    collections::{HashMap, HashSet},
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
    pub index_calculation: u128,
    pub sorting_near_field: u128,
    pub residual_calculation: u128,
}

pub struct RsrsData<Item: RlstScalar> {
    level_indexing: TreeData,
    tols: Tols<Item>,
    pub y_data: BoxesData<Item>,
    pub z_data: BoxesData<Item>,
    dim: usize,
    ind_s: Inds<usize>,
    ind_r: Inds<usize>,
    box_types: Vec<BoxType<Real<Item>>>,
    target_inds: Inds<usize>,
    near_inds: Inds<usize>,
    pub stats: Stats,
}

#[derive(Debug, Clone)]
pub enum BoxType<Item: RlstScalar> {
    Merged(usize),
    Full(Real<Item>),
}

pub enum Termination {
    EnoughSamples,
    ReachRoot,
}

pub struct RsrsOptions {
    pub termination: Termination,
    pub hermitian: bool,
    pub oversampling: usize,
    pub adaptive_tol: bool,
    pub initial_num_samples: usize,
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
        start_sample: bool,
        level_it: usize,
        options: &RsrsOptions,
    ) -> Vec<usize>;
    fn id_level_iteration(
        &mut self,
        current_box_indices: &Vec<usize>,
        options: &RsrsOptions,
    ) -> (Vec<IdFactor<Self::Item>>, Vec<usize>, Vec<Vec<usize>>);
    fn lu_level_iteration(
        &mut self,
        current_box_indices: &Vec<usize>,
        level_ind_r: &Vec<Vec<usize>>,
        options: &RsrsOptions,
    ) -> Vec<Vec<LuFactor<Self::Item>>>;
    fn get_level_indices(&mut self, level: usize);
    fn get_near_indices(&mut self, box_ind: usize) -> Vec<usize>;
}

fn oversample(samples: usize, oversampling: usize) -> usize {
    samples + (samples / 100) * oversampling
}

impl<
        T: RlstScalar
            + MatrixId
            + MatrixNull
            + MatrixInverse
            + MatrixPseudoInverse
            + RandScalar
            + MatrixLu
            + MatrixQr,
    > Rsrs for RsrsData<T>
where
    StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>,
    LuDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixLuDecomposition<Item = T>,
    QrDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixQrDecomposition<Item = T>,
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
        let box_types: Vec<BoxType<Real<Self::Item>>> = Vec::new();
        let y_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr, false);
        let z_data: BoxesData<T> = <BoxesData<Self::Item> as SketchOps>::new(arr, true);
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
            index_calculation: 0_u128,
            sorting_near_field: 0_u128,
            residual_calculation: 0_u128,
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
            self.y_data.test.shape()[1],
            self.tols.lstq,
            &mut rsrs_factors,
        );
        let extraction_time = start.elapsed();
        println!("Extraction time: {:?}s\n", extraction_time.as_secs());
        self.stats.extraction_time = extraction_time.as_millis();
        let duration = algo_start.elapsed();
        println!(
            "Total elapsed time: {} s, with {} samples\n",
            duration.as_secs(),
            self.y_data.num_samples
        );
        println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
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
            self.get_level_indices(level);
            let duration: Duration = start.elapsed();
            println!(
                "Current Level: {}. Indices computed in {:?}\n\n",
                level, duration
            );
            self.stats.index_calculation += duration.as_millis();

            let start: Instant = Instant::now();
            self.split_level_iteration(arr, rsrs_factors, options, level_it);
            println!("End level cycle. Summary:");
            println!("-------------------------");
            let duration: Duration = start.elapsed();
            println!("Elapsed time: {} s", duration.as_secs());

            let start: Instant = Instant::now();
            let len_r: usize = self
                .ind_r
                .iter()
                .map(|residual_inds| residual_inds.len())
                .sum();
            let len_s = self.dim - len_r;
            let duration: Duration = start.elapsed();
            self.stats.residual_calculation += duration.as_millis();

            println!("Sketch Points: {}", len_s);
            println!("Residual Points: {}", len_r);
            println!("Current Number of Samples: {}\n", self.y_data.num_samples);

            level -= 1;
            level_it += 1;

            if level <= min_level {
                println!("-------------------------");
                println!("\nReached lower level: {}", level);
                self.stats.residual_size = len_r;
                let min_sketch_samples = oversample(len_s, options.oversampling);

                if min_sketch_samples > self.y_data.num_samples {
                    let extra_num_samples = min_sketch_samples - self.y_data.num_samples;

                    println!("Extra {} samples", extra_num_samples);

                    let (mut tot_sampling_time, mut tot_id_update, mut tot_lu_update) = self
                        .y_data
                        .add_samples(extra_num_samples, arr, rsrs_factors, level_it,0_u64);
                    if !options.hermitian {
                        let (tot_z_sampling_time, tot_z_id_update, tot_z_lu_update) = self
                            .z_data
                            .add_samples(extra_num_samples, arr, rsrs_factors, level_it, 0_u64);
                        tot_sampling_time += tot_z_sampling_time;
                        tot_id_update += tot_z_id_update;
                        tot_lu_update += tot_z_lu_update;
                    }

                    println!("Sampling Time: {}ms", tot_sampling_time);
                    println!("Update times: {}ms, {}ms", tot_id_update, tot_lu_update);

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
        start_sample: bool,
        level_it: usize,
        options: &RsrsOptions,
    ) -> Vec<usize> {
        let mut extra_num_samples = options.initial_num_samples;

        let mut box_indices: Vec<usize> = (0..self.target_inds.len()).collect::<Vec<_>>();

        box_indices = box_indices
            .into_iter()
            .filter(|&box_ind| !self.ind_s[box_ind].is_empty())
            .collect::<Vec<_>>();

        box_indices.sort_by_key(|&box_ind| {
            self.ind_s[box_ind].len() + self.get_near_indices(box_ind).len()
        });

        let current_box_indices = box_indices;

        if !start_sample {
            let last_box_index = *current_box_indices.last().unwrap();
            let min_num_samples = oversample(
                self.ind_s[last_box_index].len() + self.get_near_indices(last_box_index).len(),
                options.oversampling,
            );
            extra_num_samples = min_num_samples.saturating_sub(self.y_data.num_samples);
        }

        println!("Sampling step. New {} samples", extra_num_samples);

        if extra_num_samples > 0 {
            let (mut tot_sampling_time, mut tot_id_update, mut tot_lu_update) = self
                .y_data
                .add_samples(extra_num_samples, arr, rsrs_factors, level_it, 1);

            if !options.hermitian {
                let (tot_z_sampling_time, tot_z_id_update, tot_z_lu_update) = self
                    .z_data
                    .add_samples(extra_num_samples, arr, rsrs_factors, level_it, 1);
                tot_sampling_time += tot_z_sampling_time;
                tot_id_update += tot_z_id_update;
                tot_lu_update += tot_z_lu_update;
            }
            println!("Sampling Time: {}ms", tot_sampling_time);
            println!("Update times: {}ms, {}ms\n", tot_id_update, tot_lu_update);

            self.stats.sampling_time.push(tot_sampling_time);
            let mut update_times = UpdateTimes::new();
            update_times.sum(tot_id_update, tot_lu_update);
            self.stats.update_times.push(update_times);
        }

        current_box_indices
    }

    fn id_level_iteration(
        &mut self,
        current_box_indices: &Vec<usize>,
        options: &RsrsOptions,
    ) -> (Vec<IdFactor<T>>, Vec<usize>, Vec<Vec<usize>>) {
        println!("Starting ID step");
        let mut current_near_field_indices = Vec::new();
        current_box_indices
            .iter()
            .for_each(|box_ind| current_near_field_indices.push(self.get_near_indices(*box_ind)));
        let current_near_field_ind_to_num: HashMap<_, _> = current_box_indices
            .iter()
            .enumerate()
            .map(|(box_num, box_ind)| (*box_ind, box_num))
            .collect();

        let mut box_id_level_iteration_res: Vec<_> = current_box_indices
            .par_iter()
            .map(|&box_ind| {
                let box_num = *current_near_field_ind_to_num.get(&box_ind).unwrap();
                let mut near_field_inds = &current_near_field_indices[box_num];
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
        let mut current_box_indices = Vec::new();

        let mut level_ind_r = Vec::new();
        let mut id_times = IdTimes::new();

        let low_rank_res: Vec<_> = box_id_level_iteration_res
            .into_iter()
            .filter_map(|res| {
                let (box_ind, result) = res;

                match result {
                    Rank::Low(low_rank_result) => {
                        current_box_indices.push(box_ind);
                        level_ind_r.push(low_rank_result.id_factor.ind_r.clone());
                        self.target_inds[box_ind] = low_rank_result.target_inds.clone();
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

        (low_rank_res, current_box_indices, level_ind_r)
    }

    fn lu_level_iteration(
        &mut self,
        current_box_indices: &Vec<usize>,
        level_ind_r: &Vec<Vec<usize>>,
        options: &RsrsOptions,
    ) -> Vec<Vec<LuFactor<T>>> {
        println!("Start LU step");

        let start: Instant = Instant::now();
        let level_near_field_reduced_inds: Vec<_> = current_box_indices
            .iter()
            .map(|&box_ind| self.near_inds[box_ind].clone())
            .collect();
        let independent_near_fields = group_near_fields(&level_near_field_reduced_inds);
        let level_near_field_inds: Vec<_> = current_box_indices
            .iter()
            .map(|&box_ind| self.get_near_indices(box_ind))
            .collect();
        let time_independent_nf = start.elapsed();

        self.stats.sorting_near_field += time_independent_nf.as_millis();

        println!("Batches computed in {:?}", time_independent_nf);

        let mut update_parallel_batch_time = 0;
        let lu_step_start: Instant = Instant::now();
        let batches_res: Vec<_> = independent_near_fields
            .into_iter()
            .map(|batch| {
                let mut lu_batch: FactorBatch<Self::Item> = FactorBatchOperations::new();
                let mut lu_batch_time = LuTimes::new();
                let lu_times_and_factor: Vec<_> = batch
                    .par_iter()
                    .map(|box_num| {
                        let skel_box = <Self::Item as Default>::default();
                        let box_ind = current_box_indices[*box_num];
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

                        (lu_times, lu_factor)
                    })
                    .collect();

                lu_times_and_factor
                    .into_iter()
                    .for_each(|(lu_time, lu_factor)| {
                        lu_batch.add_factor(lu_factor);
                        lu_batch_time.sum(lu_time.lu, lu_time.extraction);
                    });

                let parallel_batch_start: Instant = Instant::now();

                let update_type = UpdateType::Lu(
                    &lu_batch
                );

                self.update_samples(0, self.active_samples, level_it, &update_type);
                let parallel_batch_duration = parallel_batch_start.elapsed().as_millis();
                update_parallel_batch_time += parallel_batch_duration;
                (lu_batch_time, lu_batch)
            })
            .collect();

        let batches_res: Vec<_> = batches_res
            .into_iter()
            .map(|(batch_lu_times, lu_batch)| {
                lu_times.sum(batch_lu_times.extraction, batch_lu_times.lu);
                lu_batch
            })
            .collect();

        update_times.sum(0_u128, update_parallel_batch_time);

        self.stats.lu_times.push(lu_times);
        self.stats.update_times.push(update_times);
        let lu_step_duration = lu_step_start.elapsed().as_millis() - update_parallel_batch_time;
        self.stats.tot_lu_time += lu_step_duration;
        println!(
            "LU step in {}ms, with updates in {}ms\n",
            lu_step_duration, update_parallel_batch_time
        );

        batches_res
    }

    fn split_level_iteration(
        &mut self,
        arr: &DynamicArray<Self::Item, 2>,
        rsrs_factors: &mut RsrsFactors<Self::Item>,
        options: &RsrsOptions,
        level_it: usize,
    ) {
        let merged_count = self
            .box_types
            .iter()
            .filter(|box_type| matches!(box_type, BoxType::Merged(_rank)))
            .count();
        println!("Number of merged boxes: {}\n", merged_count);

        let current_box_indices = self.sampling_step(arr, rsrs_factors, level_it == 0, level_it, options);

        let id_step_start: Instant = Instant::now();
        let (id_factors_res, current_box_indices, level_ind_r) =
            self.id_level_iteration(&current_box_indices, options);
        rsrs_factors.id_factors[level_it] = id_factors_res;
        let id_step_duration = id_step_start.elapsed();
        self.stats.tot_id_time += id_step_duration.as_millis();

        println!("ID step in {:?}", id_step_duration);

        let start_id_update: Instant = Instant::now();
        let _id_update_times: Vec<_> = rsrs_factors.id_factors[level_it]
            .iter()
            .map(|id_factor| {
                let start: Instant = Instant::now();
                update_sketch_id(
                    &mut self.y_data.sketch,
                    &mut self.y_data.test,
                    &id_factor,
                    &FactorType::F,
                    &FactorType::S,
                    false,
                );
                if !options.hermitian {
                    update_sketch_id(
                        &mut self.z_data.sketch,
                        &mut self.z_data.test,
                        &id_factor,
                        &FactorType::S,
                        &FactorType::F,
                        true,
                    );
                }
                let update_id_time: Duration = start.elapsed();

                update_id_time
            })
            .collect();

        let update_id_time: Duration = start_id_update.elapsed();
        let mut update_times = UpdateTimes::new();
        update_times.sum(update_id_time.as_millis(), 0_u128);
        self.stats.update_times.push(update_times);
        println!("ID updated in {:?}\n", update_id_time);

        rsrs_factors.lu_factors[level_it] =
            self.lu_level_iteration(&current_box_indices, &level_ind_r, options);
    }

    fn get_near_indices(&mut self, box_ind: usize) -> Vec<usize> {
        let mut near_indices = Vec::new();
        for ind in self.near_inds[box_ind].iter() {
            near_indices.extend_from_slice(&self.target_inds[*ind]);
        }
        near_indices
    }

    fn get_level_indices(&mut self, level: usize) {
        println!("Computing Indices...\n");
        if level < self.level_indexing.max_level {
            let binding: std::collections::HashSet<MortonKey> =
                self.level_indexing.level_keys.clone();
            let previous_level_keys: Vec<&MortonKey> = binding.iter().collect::<Vec<_>>();
            self.level_indexing.update_level_keys();
            let binding: std::collections::HashSet<MortonKey> =
                self.level_indexing.level_keys.clone();
            let current_level_keys: Vec<&MortonKey> = binding.iter().collect::<Vec<_>>();
            let current_level_key_to_index: HashMap<_, _> = current_level_keys
                .iter()
                .enumerate()
                .map(|(i, key)| (*key, i))
                .collect();
            let mut target_inds: Inds<usize> = Vec::new();
            let mut num_sons: Vec<usize> = Vec::new();

            self.near_inds.clear();
            self.near_inds.resize(current_level_keys.len(), Vec::new());
            let mut local_box_ranks = Vec::new();
            local_box_ranks.resize(current_level_keys.len(), Vec::new());
            let mut box_types = Vec::new();
            box_types.resize(current_level_keys.len(), BoxType::Full(self.tols.id));
            target_inds.resize(current_level_keys.len(), Vec::new());
            num_sons.resize(current_level_keys.len(), 0);

            for (box_ind, box_key) in previous_level_keys.iter().enumerate() {
                if let Some(&parent_index) = current_level_key_to_index.get(&box_key.parent()) {
                    if self.ind_s[box_ind].len() < self.target_inds[box_ind].len() {
                        local_box_ranks[parent_index].push(BoxType::Merged::<Real<Self::Item>>(
                            self.ind_s[box_ind].len(),
                        ));
                    }
                    target_inds[parent_index].extend_from_slice(&self.ind_s[box_ind]);
                    num_sons[parent_index] += 1;
                    self.ind_s[box_ind].clear();
                }
            }

            for (box_ind, box_key) in previous_level_keys.iter().enumerate() {
                if let Some(&parent_index) = current_level_key_to_index.get(box_key) {
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

            current_level_key_to_index
                .iter()
                .for_each(|(_box_key, &parent_index)| {
                    let merged_ranks: Vec<_> = local_box_ranks[parent_index]
                        .iter()
                        .filter_map(|box_type| match box_type {
                            BoxType::Merged(rank) => Some(rank),
                            BoxType::Full(_) => None,
                        })
                        .collect();

                    if merged_ranks.len() > 0 {
                        let rank = merged_ranks.iter().min().unwrap();
                        box_types[parent_index] = BoxType::Merged(**rank);
                    }
                });

            self.box_types.clear();
            self.box_types = box_types;
            self.target_inds.clear();
            self.target_inds = target_inds;
            self.ind_s.clear();
            self.ind_s.clone_from(&self.target_inds);

            for (box_key, &box_ind) in current_level_key_to_index.iter() {
                self.near_inds[box_ind].push(box_ind);

                let near_keys: HashSet<MortonKey> =
                    self.level_indexing.get_box_near_field_keys(box_key);

                for near_box_key in near_keys.iter() {
                    if let Some(&near_box_ind) = current_level_key_to_index.get(near_box_key) {
                        self.near_inds[box_ind].push(near_box_ind);
                    }
                }
            }

            let boxes_lengths: Vec<usize> =
                self.ind_s.iter().map(|ind| ind.len()).collect::<Vec<_>>();

            println!(
                "New {} boxes, and active indices: {}",
                self.ind_s.len(),
                boxes_lengths.iter().sum::<usize>()
            );
        } else {
            self.target_inds
                .resize(self.level_indexing.level_keys.len(), Vec::new());
            self.ind_s
                .resize(self.level_indexing.level_keys.len(), Vec::new());
            self.near_inds
                .resize(self.level_indexing.level_keys.len(), Vec::new());
            self.box_types.resize(
                self.level_indexing.level_keys.len(),
                BoxType::Full(self.tols.id),
            );

            for (box_ind, box_key) in self.level_indexing.level_keys.clone().iter().enumerate() {
                if let Some(box_indices) = self.level_indexing.boxes_map.get(box_key) {
                    self.target_inds[box_ind] = box_indices.to_vec();
                    if box_key.level() == self.level_indexing.max_level {
                        self.ind_s[box_ind] = box_indices.to_vec();
                    }
                }
            }

            let key_to_index: HashMap<_, _> = self
                .level_indexing
                .level_keys
                .iter()
                .enumerate()
                .map(|(i, key)| (*key, i))
                .collect();

            for (box_key, &box_ind) in key_to_index.iter() {
                self.near_inds[box_ind].push(box_ind);

                let near_keys: HashSet<MortonKey> =
                    self.level_indexing.get_box_near_field_keys(box_key);

                for near_key in near_keys.iter() {
                    if let Some(&near_ind) = key_to_index.get(near_key) {
                        self.near_inds[box_ind].push(near_ind);
                    }
                }
            }

            let boxes_lengths: Vec<usize> = self
                .target_inds
                .iter()
                .map(|ind| ind.len())
                .collect::<Vec<_>>();

            println!(
                "New {} boxes, and active indices: {}",
                self.target_inds.len(),
                boxes_lengths.iter().sum::<usize>()
            );
        }
    }
}

fn group_near_fields(near_fields: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut near_field_groups: Vec<Vec<Vec<usize>>> = Vec::new();
    let mut near_field_group_inds: Vec<Vec<usize>> = Vec::new();

    'outer: for (near_field_ind, near_field) in near_fields.iter().enumerate() {
        let near_field_set: HashSet<_> = near_field.iter().copied().collect();

        for (near_field_group_ind, near_field_group) in
            &mut near_field_groups.iter_mut().enumerate()
        {
            let mut has_common = false;

            for existing_near_field in near_field_group.iter() {
                let existing_near_field_set: HashSet<_> =
                    existing_near_field.iter().copied().collect();
                if !existing_near_field_set.is_disjoint(&near_field_set) {
                    has_common = true;
                    break;
                }
            }

            if !has_common {
                near_field_group.push(near_field.to_vec());
                near_field_group_inds[near_field_group_ind].push(near_field_ind);
                continue 'outer;
            }
        }

        near_field_groups.push(vec![near_field.to_vec()]);
        near_field_group_inds.push(Vec::new());
        near_field_group_inds[near_field_groups.len() - 1].push(near_field_ind);
    }
    near_field_group_inds
}

fn par_batch_update_map<
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + MatrixLu,
>(
    batch_res: Vec<(usize, LuFactor<Item>, LuTimes)>,
    rsrs_data: &mut RsrsData<Item>,
    hermitian: bool,
) -> Vec<(
    usize,
    LuFactor<Item>,
    LuTimes,
    Duration,
    Vec<(DynamicArray<Item, 2>, DynamicArray<Item, 2>)>,
)>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let batch_reduced_res: Vec<_> = batch_res
        .into_par_iter()
        .map(|(box_ind, lu_factor, lu_times)| {
            let mut sketches = Vec::new();
            let start: Instant = Instant::now();
            {
                let y_sketch_data = update_sketch_lu_no_subs(
                    &rsrs_data.y_data.sketch,
                    &rsrs_data.y_data.test,
                    &lu_factor,
                    &FactorType::F,
                    &FactorType::S,
                    false,
                );
                sketches.push(y_sketch_data);
                if !hermitian {
                    let z_sketch_data = update_sketch_lu_no_subs(
                        &rsrs_data.z_data.sketch,
                        &rsrs_data.z_data.test,
                        &lu_factor,
                        &FactorType::S,
                        &FactorType::F,
                        true,
                    );
                    sketches.push(z_sketch_data);
                }
            }
            let update_lu_time: Duration = start.elapsed();

            (box_ind, lu_factor, lu_times, update_lu_time, sketches)
        })
        .collect();

    batch_reduced_res
}

fn _single_node_batch_update_map<
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + MatrixLu,
>(
    batch_res: Vec<(usize, LuFactor<Item>, LuTimes)>,
    rsrs_data: &mut RsrsData<Item>,
    hermitian: bool,
) -> Vec<(usize, LuFactor<Item>, LuTimes, Duration)>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let batch_reduced_res: Vec<_> = batch_res
        .into_iter()
        .map(|(box_ind, lu_factor, lu_times)| {
            let start: Instant = Instant::now();
            {
                update_sketch_lu(
                    &mut rsrs_data.y_data.sketch,
                    &mut rsrs_data.y_data.test,
                    &lu_factor,
                    &FactorType::F,
                    &FactorType::S,
                    false,
                );
                if !hermitian {
                    update_sketch_lu(
                        &mut rsrs_data.z_data.sketch,
                        &mut rsrs_data.z_data.test,
                        &lu_factor,
                        &FactorType::S,
                        &FactorType::F,
                        true,
                    );
                }
            }
            let update_lu_time: Duration = start.elapsed();

            (box_ind, lu_factor, lu_times, update_lu_time)
        })
        .collect();

    batch_reduced_res
}
