use super::{
    box_skeletonisation::{Rank, Skel},
    sketch::SketchData,
    tree_indexing::{TreeData, TreeIndexing},
};
use crate::{
    rsrs::{
        args::{RankPicking, RsrsOptions},
        rsrs_factors::{
            commutative_factors::{
                BoxType, CommutativeFactors, CommutativeFactorsOperations, DiagBoxFactor, Factor,
                MultiLevelIdFactors, RsrsFactors,
            },
            rsrs_operator::{FactType, LocalFromSpaces, RsrsFactorsImpl, RsrsOperator},
        },
        sketch::{SamplingSpace, UpdateType},
        statistics::{
            IdTimes, IdTimesOperations, LevelEffort, LimitingFactors, LimitingLevel, LuTimes,
            LuTimesOperations, Stats, Times, UpdateTimes, UpdateTimesOperations,
        },
    },
    utils::io::IOData,
};
use bempp_octree::{MortonKey, Octree};
use mpi::traits::CommunicatorCollectives;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    ThreadPoolBuilder,
};
use rlst::dense::{linalg::lu::MatrixLu, tools::RandScalar};
pub use rlst::prelude::*;
use rustc_hash::FxHashSet;
use std::{
    collections::HashMap,
    path::Path,
    time::{Duration, Instant},
};

type Inds<T> = Vec<Vec<T>>;

type Real<T> = <T as rlst::RlstScalar>::Real;

pub struct Rsrs<Item: RlstScalar> {
    level_indexing: TreeData,
    pub y_data: SketchData<Item>,
    pub z_data: SketchData<Item>,
    dim: usize,
    ind_s: Inds<usize>,
    ind_r: Inds<usize>,
    box_types: Vec<BoxType<Real<Item>>>,
    target_inds: Inds<usize>,
    near_inds: Inds<usize>,
    pub active_samples: usize,
    pub stats: Stats,
    options: RsrsOptions<Item>,
}

fn oversample<Item: RlstScalar>(
    samples: usize,
    oversampling: usize,
    id_tol: <Item as RlstScalar>::Real,
    ms: usize,
) -> usize {
    if id_tol < num::One::one() {
        (samples + (samples / 100) * oversampling).max(ms)
    } else {
        (samples + num::ToPrimitive::to_usize(&id_tol).unwrap()).max(ms)
    }
}

fn local_oversample(_min_samples: usize, active_samples: usize) -> usize {
    active_samples
    //min_samples + (active_samples - min_samples) / 2
}

fn auto_min_len(batch_len: usize, num_threads: usize) -> usize {
    if batch_len <= num_threads {
        1
    } else {
        let raw = batch_len / (2 * num_threads);
        raw.max(1).min(32)
    }
}

impl<
        Item: RlstScalar
            + MatrixId
            + MatrixNull
            + MatrixInverse
            + MatrixPseudoInverse
            + RandScalar
            + MatrixLu
            + MatrixQr,
    > Rsrs<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    QrDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixQrDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
    Item: IOData<Item>,
    Item: std::convert::From<<Item as IOData<Item>>::Item>,
{
    pub fn new<C: CommunicatorCollectives>(
        octree: &Octree<'_, C>,
        options: RsrsOptions<Item>,
        dim: usize,
    ) -> Self {
        let level_indexing: TreeData = <TreeData as TreeIndexing>::new(octree);
        let target_inds: Inds<usize> = Vec::new();
        let near_inds: Inds<usize> = Vec::new();
        let ind_s: Inds<usize> = Vec::new();
        let ind_r: Inds<usize> = Vec::new();
        let box_types: Vec<BoxType<Real<Item>>> = Vec::new();
        let y_data: SketchData<Item> = SketchData::new(dim, TransMode::NoTrans);
        let z_data: SketchData<Item> = SketchData::new(dim, TransMode::Trans);
        let id_times = Vec::new();
        let lu_times = Vec::new();
        let update_times = Vec::new();
        let limiting_level = LimitingLevel {
            level: 0,
            num_boxes: 0,
            active_points: 0,
            elapsed_time: 0,
        };
        let limiting_factors = LimitingFactors {
            min_samples: 0,
            max_level: 0,
            limiting_level,
            leaf_count: 0,
        };

        println!(
            "Current number of threads = {}",
            rayon::current_num_threads()
        );
        let stats = Stats {
            sampling_time: Vec::new(),
            sampling_extraction_time: 0_u128,
            id_times,
            tot_id_time: 0_u128,
            lu_times,
            tot_lu_time: 0_u128,
            update_times,
            total_elapsed_time: 0_u128,
            total_elapsed_time_wo_sampling: 0_u128,
            extraction_time: 0_u128,
            residual_size: 0,
            ranks: Vec::new(),
            box_sizes: Vec::new(),
            near_field_sizes: Vec::new(),
            dec_boxes_per_level: Vec::new(),
            index_calculation: 0_u128,
            sorting_near_field: 0_u128,
            residual_calculation: 0_u128,
            limiting_factors,
            dim,
            level_effort: Vec::new(),
            mv_avg_time: Vec::new(),
        };

        Self {
            level_indexing,
            y_data,
            z_data,
            dim,
            ind_s,
            ind_r,
            box_types,
            target_inds,
            near_inds,
            stats,
            active_samples: 0,
            options,
        }
    }

    pub fn get_rsrs_operator<'a, Space, OpImpl>(
        &mut self,
        operator: Operator<OpImpl>,
    ) -> RsrsOperator<'a, Item, Space, RsrsFactors<Item>>
    where
        Space: SamplingSpace<F = Item> + 'a,
        OpImpl: AsApply<Domain = Space, Range = Space>,
        RsrsOperator<'a, Item, Space, RsrsFactors<Item>>:
            LocalFromSpaces<'a, Item, Space, RsrsFactors<Item>>,
    {
        let domain = std::rc::Rc::clone(&operator.domain());
        let range = std::rc::Rc::clone(&operator.range());
        let rsrs_factors = self.run(operator.r());
        // Move rsrs_factors into a Box to extend its lifetime
        let boxed_factors = Box::new(rsrs_factors);
        // Create a static reference by leaking the Box (caller must ensure cleanup if needed)
        let static_factors: &'a mut RsrsFactors<Item> = Box::leak(boxed_factors);

        RsrsOperator::from_local_spaces(static_factors, domain, range)
    }
    pub fn run<Space: SamplingSpace<F = Item>, OpImpl: AsApply<Domain = Space, Range = Space>>(
        &mut self,
        operator: Operator<OpImpl>,
    ) -> RsrsFactors<Item> {
        let num_levels: usize = self.level_indexing.max_level;
        let algo_start: Instant = Instant::now();
        let mut rsrs_factors = <RsrsFactors<Item> as RsrsFactorsImpl<Item>>::new(
            num_levels,
            self.dim,
            &self.options.fact_type,
            self.options.num_threads,
        );
        let start: Instant = Instant::now();
        self.tree_cycle(operator.r(), &mut rsrs_factors);
        let duration = start.elapsed();
        println!("Tree cycle elapsed time: {} s", duration.as_secs());
        println!(
            "Extracting diagonal blocks with {} active samples",
            self.active_samples
        );
        let start: Instant = Instant::now();

        let (mut diag_box_factors, rows, cols) = self.extract_step();
        if self.options.flush_factors {
            diag_box_factors.flush();
        }
        rsrs_factors.diag_box_factors = diag_box_factors;
        rsrs_factors.perm_factor.orig_indices = cols;
        rsrs_factors.perm_factor.perm_indices = rows;
        let extraction_time = start.elapsed();
        println!("Extraction time: {:?}s\n", extraction_time.as_secs());
        self.stats.extraction_time = extraction_time.as_millis();
        let duration = algo_start.elapsed();
        self.stats.total_elapsed_time = duration.as_millis();
        let sampling_time =
            self.stats.sampling_extraction_time + self.stats.sampling_time.iter().sum::<u128>();
        self.stats.total_elapsed_time_wo_sampling =
            self.stats.total_elapsed_time.saturating_sub(sampling_time);
        println!(
            "Total elapsed time: {:?} ({}ms for sampling, {}ms for RSRS), with {} active samples\n",
            duration, sampling_time, self.stats.total_elapsed_time_wo_sampling, self.active_samples
        );
        println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

        rsrs_factors
    }

    fn tree_cycle<
        Space: SamplingSpace<F = Item>,
        OpImpl: AsApply<Domain = Space, Range = Space>,
    >(
        &mut self,
        operator: Operator<OpImpl>,
        rsrs_factors: &mut RsrsFactors<Item>,
    ) {
        let mut level: usize = self.level_indexing.max_level;
        let mut level_it = 0;
        let min_level: usize = self.options.min_level;

        while level > min_level {
            println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            let start: Instant = Instant::now();
            self.get_level_indices(level);
            let duration: Duration = start.elapsed();
            println!("Current Level: {level}. Indices computed in {duration:?}\n\n");
            let active_boxes = self.ind_s.iter().filter(|v| !v.is_empty()).count();
            self.stats.index_calculation += duration.as_millis();

            let len_r_s: usize = self
                .ind_r
                .iter()
                .map(|residual_inds| residual_inds.len())
                .sum();
            let start: Instant = Instant::now();
            let (level_duration, num_batches) =
                self.level_cycle(operator.r(), rsrs_factors, level_it);
            println!("End level cycle. Summary:");
            println!("-------------------------");
            let duration: Duration = start.elapsed();
            println!("Elapsed time: {} s", duration.as_secs());
            println!("Leaf count: {}", self.stats.limiting_factors.leaf_count);
            if self.stats.limiting_factors.limiting_level.level == self.level_indexing.current_level
            {
                self.stats.limiting_factors.limiting_level.elapsed_time = duration.as_millis()
            }

            let start: Instant = Instant::now();
            let len_r: usize = self
                .ind_r
                .iter()
                .map(|residual_inds| residual_inds.len())
                .sum();
            let len_s = self.dim - len_r;
            let duration: Duration = start.elapsed();
            self.stats.residual_calculation += duration.as_millis();
            let active_indices: usize = self.ind_s.iter().map(Vec::len).sum();
            println!("Sketch Points: {len_s} ({active_indices})");
            println!("Residual Points: {len_r}");
            println!(
                "Current Number of Samples: {} of which {} are active\n",
                self.y_data.test.shape()[0],
                self.active_samples
            );
            let level_effort = LevelEffort {
                time: level_duration,
                num_boxes: active_boxes,
                num_batches,
                effective_dofs: len_r - len_r_s,
                residual_len: len_r,
                sketch_len: active_indices,
            };
            self.stats.level_effort.push(level_effort);

            level -= 1;
            level_it += 1;

            if level <= min_level {
                println!("-------------------------");
                println!("\nReached lower level: {level}");
                self.stats.residual_size = len_r;
                let min_oversamples = oversample::<Item>(
                    len_s,
                    self.options.sketching.oversampling_diag_blocks,
                    num::One::one(),
                    self.options.sketching.min_num_samples,
                );
                println!("Minimum samples: {min_oversamples}");

                let (tot_sampling_time, tot_id_update, tot_lu_update) = self.add_samples(
                    min_oversamples,
                    operator.r(),
                    rsrs_factors,
                    level_it,
                    false,
                    false,
                    0_u64,
                );
                self.active_samples = min_oversamples.max(self.active_samples);

                self.stats.sampling_extraction_time = tot_sampling_time;
                let mut update_times = UpdateTimes::new();
                update_times.sum(tot_id_update, tot_lu_update);
                self.stats.update_times.push(update_times);

                break;
            }
        }
    }

    fn level_cycle<
        Space: SamplingSpace<F = Item>,
        OpImpl: AsApply<Domain = Space, Range = Space>,
    >(
        &mut self,
        operator: Operator<OpImpl>,
        rsrs_factors: &mut RsrsFactors<Item>,
        level_it: usize,
    ) -> (u128, usize) {
        let merged_count = self
            .box_types
            .iter()
            .filter(|box_type| matches!(box_type, BoxType::Merged(_rank)))
            .count();
        println!("Number of merged boxes: {merged_count}\n");

        let current_box_indices =
            self.sampling_step(operator.r(), rsrs_factors, level_it == 0, level_it);

        let level_start: Instant = Instant::now();
        let num_batches = match self.options.fact_type {
            FactType::Joint => {
                self.joint_id_and_lu::<Space>(rsrs_factors, &current_box_indices, level_it)
            }
            FactType::Split => {
                self.split_id_and_lu::<Space>(rsrs_factors, current_box_indices, level_it);
                0
            }
        };
        let level_duration_wo_sampling = level_start.elapsed();
        (level_duration_wo_sampling.as_millis(), num_batches)
    }

    fn sampling_step<
        Space: SamplingSpace<F = Item>,
        OpImpl: AsApply<Domain = Space, Range = Space>,
    >(
        &mut self,
        operator: Operator<OpImpl>,
        rsrs_factors: &RsrsFactors<Item>,
        start: bool,
        level_it: usize,
    ) -> Vec<usize> {
        let mut box_indices: Vec<usize> = (0..self.target_inds.len()).collect::<Vec<_>>();

        box_indices = box_indices
            .into_iter()
            .filter(|&box_ind| !self.ind_s[box_ind].is_empty())
            .collect::<Vec<_>>();

        box_indices.sort_by_key(|&box_ind| {
            self.ind_s[box_ind].len() + self.get_near_indices(box_ind).len()
        });

        let current_box_indices = box_indices;
        let last_box_index = *current_box_indices.last().unwrap();

        let min_oversamples = oversample::<Item>(
            self.ind_s[last_box_index].len() + self.get_near_indices(last_box_index).len(),
            self.options.sketching.oversampling,
            self.options.id_options.tol_id,
            self.options.sketching.min_num_samples,
        );

        let min_samples = if start {
            self.options
                .sketching
                .initial_num_samples
                .max(min_oversamples)
        } else {
            min_oversamples
        };

        let load_samples = if start { true } else { false };

        let (tot_sampling_time, tot_id_update, tot_lu_update) = self.add_samples(
            min_samples,
            operator.r(),
            rsrs_factors,
            level_it,
            start,
            load_samples,
            1,
        );

        self.stats.limiting_factors.min_samples = self
            .stats
            .limiting_factors
            .min_samples
            .max(self.active_samples);
        self.active_samples = min_oversamples.max(self.active_samples);

        self.stats.sampling_time.push(tot_sampling_time);
        let mut update_times = UpdateTimes::new();
        update_times.sum(tot_id_update, tot_lu_update);
        self.stats.update_times.push(update_times);

        println!("Active samples: {}\n", self.active_samples);

        current_box_indices
    }

    fn add_samples<
        Space: SamplingSpace<F = Item>,
        OpImpl: AsApply<Domain = Space, Range = Space>,
    >(
        &mut self,
        min_samples: usize,
        operator: Operator<OpImpl>,
        rsrs_factors: &RsrsFactors<Item>,
        level_it: usize,
        start: bool,
        load_samples: bool,
        _seed: u64,
    ) -> (u128, u128, u128) {
        if load_samples {
            if Path::new("y_test_file.h5").exists() && Path::new("y_sketch_file.h5").exists() {
                let test = <Item as IOData<Item>>::load("y_test_file.h5").unwrap();
                let sketch = <Item as IOData<Item>>::load("y_sketch_file.h5").unwrap();
                let num_existing_samples = test.len() / self.dim;
                self.y_data
                    .test
                    .resize_in_place([num_existing_samples, self.dim]);
                self.y_data
                    .sketch
                    .resize_in_place([num_existing_samples, self.dim]);
                self.y_data
                    .test
                    .data_mut()
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, d)| {
                        *d = test[i].into();
                    });
                self.y_data
                    .sketch
                    .data_mut()
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, d)| {
                        *d = sketch[i].into();
                    });

                println!(
                    "{} samples loaded and {} min samples",
                    num_existing_samples, min_samples
                );
            }

            if !self.options.symmetric {
                if Path::new("z_test_file.h5").exists() && Path::new("z_sketch_file.h5").exists() {
                    let test = <Item as IOData<Item>>::load("z_test_file.h5").unwrap();
                    let sketch = <Item as IOData<Item>>::load("z_sketch_file.h5").unwrap();
                    let num_existing_samples = test.len() / self.dim;
                    self.z_data
                        .test
                        .resize_in_place([num_existing_samples, self.dim]);
                    self.z_data
                        .sketch
                        .resize_in_place([num_existing_samples, self.dim]);
                    self.z_data
                        .test
                        .data_mut()
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, d)| {
                            *d = test[i].into();
                        });
                    self.z_data
                        .sketch
                        .data_mut()
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, d)| {
                            *d = sketch[i].into();
                        });

                    println!(
                        "{} samples loaded and {} min samples",
                        num_existing_samples, min_samples
                    );
                }
            }
        }

        let mut tot_sampling_time = 0_u128;
        let test_shape = self.y_data.test.shape();

        if min_samples > test_shape[0] {
            let extra_samples = min_samples.saturating_sub(self.y_data.test.shape()[0]);
            println!("Sampling step. Sampling new {extra_samples} vectors\n");

            tot_sampling_time += self.y_data.add_samples(
                extra_samples,
                operator.r(),
                &self.options.sketching.shift,
                self.options.sketching.save_samples,
                0_u64,
            );

            if !self.options.symmetric {
                let tot_z_sampling_time = self.z_data.add_samples(
                    extra_samples,
                    operator.r(),
                    &self.options.sketching.shift,
                    self.options.sketching.save_samples,
                    0_u64,
                );
                tot_sampling_time += tot_z_sampling_time;
            }

            println!("Total samples: {}", self.y_data.test.shape()[0]);
            println!("Sampling time: {tot_sampling_time}ms\n");
        }
        if !start && min_samples > self.active_samples {
            let extra_active_samples = min_samples.saturating_sub(self.active_samples);
            println!("New {extra_active_samples} samples, with {min_samples} min samples.");
            let update_start = self.active_samples;
            let (tot_id_update, tot_lu_update, avg_mv_time) = self.update_samples(
                update_start,
                extra_active_samples,
                level_it,
                &UpdateType::Both(rsrs_factors),
            );

            match avg_mv_time {
                Some(avg_time) => {
                    println!("Average update time: {} ms", avg_time);
                    self.stats.mv_avg_time.push(avg_time)
                }
                None => todo!(),
            };

            println!("Update times: {tot_id_update}ms (ID), {tot_lu_update}ms (LU)");
            return (tot_sampling_time, tot_id_update, tot_lu_update);
        }
        (tot_sampling_time, 0_u128, 0_u128)
    }

    fn update_samples(
        &mut self,
        update_start: usize,
        samples_to_update: usize,
        level: usize,
        update_type: &UpdateType<Item>,
    ) -> (u128, u128, Option<u128>) {
        let (mut tot_id_update, mut tot_lu_update) = self.y_data.update_samples(
            update_start,
            samples_to_update,
            level,
            update_type,
            &self.options.fact_type,
            self.options.num_threads,
        );

        let avg_update_time = if samples_to_update == 0 {
            None
        } else {
            Some((tot_id_update + tot_lu_update) / (samples_to_update as u128))
        };

        if !self.options.symmetric {
            let (tot_z_id_update, tot_z_lu_update) = self.z_data.update_samples(
                update_start,
                samples_to_update,
                level,
                update_type,
                &self.options.fact_type,
                self.options.num_threads,
            );
            tot_id_update += tot_z_id_update;
            tot_lu_update += tot_z_lu_update;
        }

        (tot_id_update, tot_lu_update, avg_update_time)
    }

    fn split_id_and_lu<Space: SamplingSpace<F = Item>>(
        &mut self,
        rsrs_factors: &mut RsrsFactors<Item>,
        current_box_indices: Vec<usize>,
        level_it: usize,
    ) {
        let id_step_start: Instant = Instant::now();
        let (id_factors_res, current_box_indices, level_ind_r) =
            self.id_level_iteration::<Space>(&current_box_indices);
        let id_step_duration = id_step_start.elapsed();
        self.stats.tot_id_time += id_step_duration.as_millis();
        println!("ID step in {id_step_duration:?}");
        match &mut rsrs_factors.id_factors {
            MultiLevelIdFactors::Single(level_factors) => {
                level_factors[level_it] = id_factors_res;
                let start_id_update: Instant = Instant::now();
                let update_type = UpdateType::Id(&level_factors[level_it]);
                self.update_samples(0, self.active_samples, level_it, &update_type);
                let update_id_time: Duration = start_id_update.elapsed();

                let mut update_times = UpdateTimes::new();
                update_times.sum(update_id_time.as_millis(), 0_u128);
                self.stats.update_times.push(update_times);

                println!("ID updated in {update_id_time:?}\n");
            }
            MultiLevelIdFactors::Batched(_) => todo!(),
        }
        //rsrs_factors.id_factors[level_it] = id_factors_res;

        rsrs_factors.lu_factors[level_it] =
            self.lu_level_iteration::<Space>(&current_box_indices, &level_ind_r, level_it);
    }
    fn joint_id_and_lu<Space: SamplingSpace<F = Item>>(
        &mut self,
        rsrs_factors: &mut RsrsFactors<Item>,
        current_box_indices: &[usize],
        level_it: usize,
    ) -> usize {
        println!("ID and LU step");
        let start = Instant::now();

        // 1. Build independent batches (MIS-based)
        let independent_near_fields = self.group_near_fields(current_box_indices);

        // 2. Build level_near_field_inds once (per level)
        let mut level_near_field_inds: Vec<_> = current_box_indices
            .iter()
            .map(|&box_ind| self.get_near_indices(box_ind))
            .collect();

        let time_independent_nf = start.elapsed();
        self.stats.sorting_near_field += time_independent_nf.as_millis();

        let num_batches = independent_near_fields.len();
        println!(
            "Batches computed in {time_independent_nf:?}, number of batches: {}",
            independent_near_fields.len()
        );

        // --- timing / stats accumulators over all batches at this level ---
        let mut update_lu_batch_time: u128 = 0;
        let mut update_id_batch_time: u128 = 0;
        let mut lu_step_duration: u128 = 0;
        let mut id_step_duration: u128 = 0;

        let mut level_ind_r: Vec<Vec<usize>> = vec![Vec::new(); current_box_indices.len()];
        let inactive_inds: Vec<usize> = Vec::new();

        let mut lu_times = LuTimes::new();
        let mut id_times = IdTimes::new();
        let mut update_times = UpdateTimes::new();
        let mut num_dec_boxes = 0;
        let mut len_sketch = 0;
        let mut len_full_rank = 0;

        // Build pool once, use it for all batches
        let pool = ThreadPoolBuilder::new()
            .num_threads(self.options.num_threads)
            .build()
            .unwrap();

        // Process all batches *sequentially*, each batch using Rayon internally
        let batches_res: Vec<_> = pool.install(|| {
            independent_near_fields
                .into_iter()
                .map(|batch| {
                    // One batch = independent set of LOCAL indices into current_box_indices
                    let mut id_batch: CommutativeFactors<Item> =
                        CommutativeFactorsOperations::new();
                    let mut id_batch_time = IdTimes::new();
                    let mut lu_batch: CommutativeFactors<Item> =
                        CommutativeFactorsOperations::new();
                    let mut lu_batch_time = LuTimes::new();

                    // ---- ID STEP ----
                    let id_step_start = Instant::now();

                    let num_threads = rayon::current_num_threads();
                    let min_len_id = auto_min_len(batch.len(), num_threads);

                    let id_batch_res: Vec<_> = batch
                        .par_iter()
                        .with_min_len(min_len_id)
                        .map(|box_num| {
                            let box_ind = current_box_indices[*box_num];
                            let mut skel_box = <Item as Default>::default();
                            let os = oversample::<Item>(
                                self.target_inds[box_ind].len()
                                    + level_near_field_inds[*box_num].len(),
                                self.options.sketching.oversampling,
                                self.options.id_options.tol_id,
                                self.options.sketching.min_num_samples,
                            );
                            let min_num_samples = local_oversample(os, self.active_samples);

                            let rank = <Item as Skel<Item, Space>>::id_step(
                                &mut skel_box,
                                &self.box_types[box_ind],
                                &self.ind_s[box_ind],
                                &level_near_field_inds[*box_num],
                                &self.y_data,
                                &self.z_data,
                                min_num_samples,
                                &self.options,
                            );

                            let leaf_counter = match self.box_types[box_ind] {
                                BoxType::Merged(_) => 0,
                                BoxType::Full(_) => match rank {
                                    Rank::Low(_) => 1,
                                    Rank::Full(_) => 0,
                                },
                            };

                            (*box_num, box_ind, rank, leaf_counter)
                        })
                        .collect();

                    let mut active_batch: Vec<usize> = Vec::new();

                    id_batch_res.into_iter().for_each(
                        |(box_num, box_ind, result, leaf_counter)| match result {
                            Rank::Low(low_rank_result) => {
                                active_batch.push(box_num);

                                level_ind_r[box_num] = low_rank_result.id_factor.ind_r.clone();
                                self.target_inds[box_ind] = low_rank_result.target_inds.clone();
                                self.ind_s[box_ind] = low_rank_result.id_factor.ind_s.clone();
                                self.ind_r.push(low_rank_result.id_factor.ind_r.clone());

                                let box_size = self.target_inds[box_ind].len();

                                let res_id_times = match low_rank_result.id_times {
                                    Times::Lu(_) => IdTimes {
                                        nullification: 0,
                                        id: 0,
                                    },
                                    Times::Id(id_times) => id_times,
                                };

                                id_batch_time.sum(res_id_times.nullification, res_id_times.id);

                                self.stats.ranks.push(self.ind_s[box_ind].len());
                                self.stats.box_sizes.push(box_size);
                                self.stats
                                    .near_field_sizes
                                    .push(low_rank_result.near_field_inds.len());

                                len_sketch += self.ind_s[box_ind].len();
                                num_dec_boxes += 1;
                                self.stats.limiting_factors.leaf_count += leaf_counter;

                                id_batch.add_factor(Factor::Id(low_rank_result.id_factor));
                            }
                            Rank::Full(full_rank_result) => {
                                if let Times::Id(id_times) = full_rank_result.id_times {
                                    id_batch_time.sum(id_times.nullification, id_times.id);
                                }
                                self.stats.ranks.push(full_rank_result.len_target_inds);
                                self.stats.box_sizes.push(full_rank_result.len_target_inds);
                                self.stats
                                    .near_field_sizes
                                    .push(full_rank_result.len_near_field_inds);
                                self.stats.limiting_factors.leaf_count += leaf_counter;
                                len_full_rank += self.ind_s[box_ind].len();
                            }
                        },
                    );

                    id_step_duration += id_step_start.elapsed().as_millis();

                    // update samples after ID
                    let id_batch_start = Instant::now();
                    let update_type = UpdateType::Id(&id_batch);
                    self.update_samples(0, self.active_samples, level_it, &update_type);
                    update_id_batch_time += id_batch_start.elapsed().as_millis();

                    if self.options.flush_factors {
                        id_batch.flush();
                    }
                    // ---- LU STEP ----
                    if !active_batch.is_empty() {
                        let lu_step_start = Instant::now();

                        let min_len_lu = auto_min_len(active_batch.len(), num_threads);

                        let lu_batch_res: Vec<_> = active_batch
                            .par_iter()
                            .with_min_len(min_len_lu)
                            .filter_map(|box_num| {
                                let skel_box = <Item as Default>::default();
                                let box_ind = current_box_indices[*box_num];
                                let os = oversample::<Item>(
                                    self.target_inds[box_ind].len()
                                        + level_near_field_inds[*box_num].len(),
                                    self.options.sketching.oversampling,
                                    self.options.id_options.tol_id,
                                    self.options.sketching.min_num_samples,
                                );
                                let min_num_samples = local_oversample(os, self.active_samples);

                                <Item as Skel<Item, Space>>::lu_step(
                                    &skel_box,
                                    &self.y_data,
                                    &self.z_data,
                                    &level_ind_r[*box_num],
                                    &level_near_field_inds[*box_num],
                                    &inactive_inds,
                                    min_num_samples,
                                    &self.options,
                                )
                                .map(|(lu_factor, lu_times)| (lu_times, lu_factor))
                            })
                            .collect();

                        lu_batch_res
                            .into_iter()
                            .for_each(|(it_lu_times, lu_factor)| {
                                lu_batch.add_factor(Factor::Lu(lu_factor));
                                if let Times::Lu(lu_times) = it_lu_times {
                                    lu_batch_time.sum(lu_times.lu, lu_times.extraction)
                                }
                            });

                        lu_step_duration += lu_step_start.elapsed().as_millis();

                        // update samples after LU
                        let lu_batch_start = Instant::now();
                        let update_type = UpdateType::Lu(&lu_batch);
                        self.update_samples(0, self.active_samples, level_it, &update_type);
                        update_lu_batch_time += lu_batch_start.elapsed().as_millis();
                    }

                    // prune inactive inds (currently no-op if inactive_inds is empty)
                    level_near_field_inds = level_near_field_inds
                        .iter()
                        .map(|inds| {
                            inds.iter()
                                .filter(|el| !inactive_inds.contains(el))
                                .cloned()
                                .collect()
                        })
                        .collect();

                    if self.options.flush_factors {
                        lu_batch.flush();
                    }

                    (id_batch_time, id_batch, lu_batch_time, lu_batch)
                })
                .collect()
        });

        // accumulate over batches
        batches_res
            .into_iter()
            .for_each(|(id_batch_time, id_batch, lu_batch_time, lu_batch)| {
                id_times.sum(id_batch_time.nullification, id_batch_time.id);
                lu_times.sum(lu_batch_time.extraction, lu_batch_time.lu);

                rsrs_factors.lu_factors[level_it].push(lu_batch);
                match &mut rsrs_factors.id_factors {
                    MultiLevelIdFactors::Single(_) => todo!(),
                    MultiLevelIdFactors::Batched(level_factors) => {
                        level_factors[level_it].push(id_batch)
                    }
                }
            });

        update_times.sum(update_id_batch_time, update_lu_batch_time);

        self.stats.dec_boxes_per_level.push(num_dec_boxes);
        self.stats.id_times.push(id_times);
        self.stats.lu_times.push(lu_times);
        self.stats.update_times.push(update_times);

        self.stats.tot_id_time += id_step_duration;
        self.stats.tot_lu_time += lu_step_duration;

        println!("ID and LU steps completed");
        println!("ID step in {id_step_duration}ms, with updates in {update_id_batch_time}ms");
        println!("LU step in {lu_step_duration}ms, with updates in {update_lu_batch_time}ms\n");

        num_batches
    }

    fn id_level_iteration<Space: SamplingSpace<F = Item>>(
        &mut self,
        current_box_indices: &[usize],
    ) -> (CommutativeFactors<Item>, Vec<usize>, Vec<Vec<usize>>) {
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

        let mut current_box_indices = current_box_indices.to_vec();
        current_box_indices.sort_by_key(|&box_ind| {
            let box_num = *current_near_field_ind_to_num.get(&box_ind).unwrap();
            let near_field_len = current_near_field_indices[box_num].len();
            let source_len = self.ind_s[box_ind].len();
            oversample::<Item>(
                near_field_len + source_len,
                self.options.sketching.oversampling,
                self.options.id_options.tol_id,
                self.options.sketching.min_num_samples,
            )
        });
        let pool_threads = ThreadPoolBuilder::new()
            .num_threads(self.options.num_threads)
            .build()
            .unwrap();
        let start = Instant::now();
        let id_level_iteration_res: Vec<_> = pool_threads.install(|| {
            current_box_indices
                .par_iter()
                .map(|&box_ind| {
                    let box_num = *current_near_field_ind_to_num.get(&box_ind).unwrap();
                    let near_field_inds = &current_near_field_indices[box_num];
                    let os = oversample::<Item>(
                        near_field_inds.len() + self.ind_s[box_ind].len(),
                        self.options.sketching.oversampling,
                        self.options.id_options.tol_id,
                        self.options.sketching.min_num_samples,
                    );
                    let min_box_samples = local_oversample(os, self.active_samples);
                    let mut skel_box = <Item as Default>::default();

                    let rank = <Item as Skel<Item, Space>>::id_step(
                        &mut skel_box,
                        &self.box_types[box_ind],
                        &self.ind_s[box_ind],
                        near_field_inds,
                        &self.y_data,
                        &self.z_data,
                        min_box_samples,
                        &self.options,
                    );
                    (box_ind, rank)
                })
                .collect()
        });
        let id_level_duration = start.elapsed();
        println!("ID calculations in {id_level_duration:?}",);

        let start = Instant::now();
        let mut len_sketch = 0;
        let mut len_full_rank = 0;
        let mut num_dec_boxes = 0;
        let mut current_box_indices = Vec::new();
        let mut level_ind_r = Vec::new();
        let mut id_times = IdTimes::new();
        let mut id_level: CommutativeFactors<Item> = CommutativeFactorsOperations::new();

        id_level_iteration_res
            .into_iter()
            .for_each(|(box_ind, result)| match result {
                Rank::Low(low_rank_result) => {
                    current_box_indices.push(box_ind);
                    level_ind_r.push(low_rank_result.id_factor.ind_r.clone());
                    self.target_inds[box_ind] = low_rank_result.target_inds.clone();
                    self.ind_s[box_ind] = low_rank_result.id_factor.ind_s.clone();
                    self.ind_r.push(low_rank_result.id_factor.ind_r.clone());
                    let box_size = self.target_inds[box_ind].len();

                    let res_id_times = match low_rank_result.id_times {
                        Times::Lu(_lu_times) => IdTimes {
                            nullification: 0,
                            id: 0,
                        },
                        Times::Id(id_times) => id_times,
                    };

                    id_times.sum(res_id_times.nullification, res_id_times.id);

                    self.stats.ranks.push(self.ind_s[box_ind].len());
                    self.stats.box_sizes.push(box_size);
                    self.stats
                        .near_field_sizes
                        .push(low_rank_result.near_field_inds.len());

                    len_sketch += self.ind_s[box_ind].len();
                    num_dec_boxes += 1;

                    id_level.add_factor(Factor::Id(low_rank_result.id_factor));
                }
                Rank::Full(full_rank_result) => {
                    len_full_rank += self.ind_s[box_ind].len();
                    self.stats.ranks.push(full_rank_result.len_target_inds);
                    self.stats.box_sizes.push(full_rank_result.len_target_inds);
                    self.stats
                        .near_field_sizes
                        .push(full_rank_result.len_near_field_inds);
                    let res_id_times = match full_rank_result.id_times {
                        Times::Lu(_lu_times) => IdTimes {
                            nullification: 0,
                            id: 0,
                        },
                        Times::Id(id_times) => id_times,
                    };

                    id_times.sum(res_id_times.nullification, res_id_times.id);
                }
            });

        self.stats.dec_boxes_per_level.push(num_dec_boxes);
        self.stats.id_times.push(id_times);
        let id_level_duration = start.elapsed();
        println!("ID postprocessing in {id_level_duration:?}");

        (id_level, current_box_indices, level_ind_r)
    }

    fn lu_level_iteration<Space: SamplingSpace<F = Item>>(
        &mut self,
        current_box_indices: &[usize],
        level_ind_r: &[Vec<usize>],
        level_it: usize,
    ) -> Vec<CommutativeFactors<Item>> {
        println!("Start LU step");

        let start: Instant = Instant::now();
        let independent_near_fields = self.group_near_fields(current_box_indices);

        let level_near_field_inds: Vec<_> = current_box_indices
            .iter()
            .map(|&box_ind| self.get_near_indices(box_ind))
            .collect();

        let time_independent_nf = start.elapsed();

        self.stats.sorting_near_field += time_independent_nf.as_millis();

        println!("Batches computed in {time_independent_nf:?}");

        let mut update_parallel_batch_time = 0;
        let mut lu_times = LuTimes::new();
        let mut update_times = UpdateTimes::new();
        let lu_step_start: Instant = Instant::now();

        let mut inactive_inds = Vec::new();
        let pool_threads = ThreadPoolBuilder::new()
            .num_threads(self.options.num_threads)
            .build()
            .unwrap();
        let batches_res: Vec<_> = independent_near_fields
            .into_iter()
            .map(|batch| {
                let mut lu_batch: CommutativeFactors<Item> = CommutativeFactorsOperations::new();
                let mut lu_batch_time = LuTimes::new();
                let lu_times_and_factors: Vec<_> = pool_threads.install(|| {
                    batch
                        .par_iter()
                        .filter_map(|box_num| {
                            let skel_box = <Item as Default>::default();
                            let box_ind = current_box_indices[*box_num];
                            let os = oversample::<Item>(
                                self.target_inds[box_ind].len()
                                    + level_near_field_inds[*box_num].len(),
                                self.options.sketching.oversampling,
                                self.options.id_options.tol_id,
                                self.options.sketching.min_num_samples,
                            );
                            let min_num_samples = local_oversample(os, self.active_samples);
                            <Item as Skel<Item, Space>>::lu_step(
                                &skel_box,
                                &self.y_data,
                                &self.z_data,
                                &mut level_ind_r[*box_num].clone(),
                                &mut level_near_field_inds[*box_num].clone(),
                                &inactive_inds,
                                min_num_samples,
                                &self.options,
                            )
                            .map(|(lu_factor, lu_times)| {
                                (lu_times, lu_factor, level_ind_r[*box_num].clone())
                            })
                        })
                        .collect()
                });

                lu_times_and_factors
                    .into_iter()
                    .for_each(|(lu_time, lu_factor, r_inds)| {
                        lu_batch.add_factor(Factor::Lu(lu_factor));
                        inactive_inds.extend_from_slice(&r_inds);
                        match lu_time {
                            Times::Lu(lu_times) => {
                                lu_batch_time.sum(lu_times.lu, lu_times.extraction)
                            }
                            Times::Id(_id_times) => {}
                        }
                    });

                let parallel_batch_start: Instant = Instant::now();
                let update_type = UpdateType::Lu(&lu_batch);
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
            "LU step in {lu_step_duration}ms, with updates in {update_parallel_batch_time}ms\n"
        );

        batches_res
    }

    fn extract_step(&self) -> (CommutativeFactors<Item>, Vec<usize>, Vec<usize>) {
        let rows: Vec<usize> = (0..self.y_data.dim).collect();
        let mut acc_ind_s = Vec::new();
        let mut acc_ind_r = Vec::new();

        for inds in self.ind_s.iter() {
            acc_ind_s.extend_from_slice(inds);
        }

        for inds in self.ind_r.iter() {
            acc_ind_r.extend_from_slice(inds);
        }

        let mut cols = acc_ind_r;
        cols.extend_from_slice(&acc_ind_s);

        let remaining_indices = rows
            .clone()
            .into_iter()
            .filter(|&el| !cols.contains(&el))
            .collect::<Vec<_>>();
        cols.extend_from_slice(&remaining_indices);

        let pool_threads = ThreadPoolBuilder::new()
            .num_threads(self.options.num_threads)
            .build()
            .unwrap();

        let mut diag_box_factors: CommutativeFactors<Item> = CommutativeFactorsOperations::new();
        let mut diag_box_res: Vec<_> = pool_threads.install(|| {
            self.ind_r
                .par_iter()
                .map(|inds| {
                    DiagBoxFactor::new(
                        &mut inds.to_vec(),
                        &self.y_data,
                        self.active_samples,
                        &self.options.extract_db_options,
                    )
                })
                .collect()
        });

        if self.options.symmetric {
            diag_box_res.push(DiagBoxFactor::new(
                &mut acc_ind_s.to_vec(),
                &self.y_data,
                self.active_samples,
                &self.options.extract_db_options,
            ));
        } else {
            diag_box_res.push(DiagBoxFactor::new_no_symm(
                &mut acc_ind_s.to_vec(),
                &self.y_data,
                &self.z_data,
                self.active_samples,
                &self.options.extract_db_options,
            ));
        }

        diag_box_res.into_iter().for_each(|(dbres, _dbtime)| {
            diag_box_factors.add_factor(Factor::Diag(dbres.unwrap()));
        });

        (diag_box_factors, cols, rows)
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
            // Step 1: Extract and snapshot keys before and after update
            let previous_level_keys: Vec<MortonKey> =
                self.level_indexing.level_keys.iter().cloned().collect();
            self.level_indexing.update_level_keys();
            let current_level_keys: Vec<MortonKey> =
                self.level_indexing.level_keys.iter().cloned().collect();

            // Step 2: Build index map from MortonKey to index
            let current_level_key_to_index: HashMap<_, _> = current_level_keys
                .iter()
                .enumerate()
                .map(|(i, key)| (*key, i))
                .collect();

            // Step 3: Allocate and initialize structures
            let num_boxes = current_level_keys.len();
            self.near_inds = vec![Vec::new(); num_boxes];
            let mut local_box_ranks = vec![Vec::new(); num_boxes];
            let mut box_types = vec![BoxType::Full(self.options.id_options.tol_id); num_boxes];
            let mut target_inds: Inds<usize> = vec![Vec::new(); num_boxes];
            let mut num_sons = vec![0; num_boxes];

            // Step 4: Migrate children to parent boxes
            for (box_ind, &box_key) in previous_level_keys.iter().enumerate() {
                if let Some(&parent_index) = current_level_key_to_index.get(&box_key.parent()) {
                    if self.ind_s[box_ind].len() < self.target_inds[box_ind].len() {
                        local_box_ranks[parent_index]
                            .push(BoxType::Merged::<Real<Item>>(self.ind_s[box_ind].len()));
                    }
                    target_inds[parent_index].extend_from_slice(&self.ind_s[box_ind]);
                    num_sons[parent_index] += 1;
                    self.ind_s[box_ind].clear();
                }
            }

            // Step 5: Handle orphan boxes (no children moved up)
            for (box_ind, &box_key) in previous_level_keys.iter().enumerate() {
                if let Some(&parent_index) = current_level_key_to_index.get(&box_key) {
                    if num_sons[parent_index] == 0 {
                        if self.level_indexing.max_level - level == 1 {
                            target_inds[parent_index].extend_from_slice(&self.target_inds[box_ind]);
                            self.target_inds[box_ind].clear();
                        } else {
                            target_inds[parent_index].extend_from_slice(&self.ind_s[box_ind]);
                            self.ind_s[box_ind].clear();
                        }
                        num_sons[parent_index] += 1;
                    }
                }
            }

            // Step 6: Update box types based on merged rank
            for (&_box_key, &parent_index) in current_level_key_to_index.iter() {
                let rank = pick_ranks(&self.options.rank_picking, &local_box_ranks[parent_index]);
                if let Some(min_rank) = rank {
                    let min_rank = min_rank.min(target_inds[parent_index].len());
                    box_types[parent_index] = BoxType::Merged(min_rank);
                } else if matches!(self.options.rank_picking, RankPicking::Tol) {
                    box_types[parent_index] = BoxType::Full(self.options.id_options.tol_id);
                }
            }

            // Step 7: Update self with new structures
            self.box_types = box_types;
            self.target_inds = target_inds;
            self.ind_s.clone_from(&self.target_inds);

            // Step 8: Build near field interaction indices
            for (box_key, &box_ind) in current_level_key_to_index.iter() {
                self.near_inds[box_ind].push(box_ind);

                let near_keys = self
                    .level_indexing
                    .get_box_near_field_keys(box_key, self.level_indexing.current_level);
                for near_box_key in &near_keys {
                    if let Some(&near_box_ind) = current_level_key_to_index.get(near_box_key) {
                        self.near_inds[box_ind].push(near_box_ind);
                    }
                }
            }

            // Step 9: Debug / info output
            let boxes_lengths: Vec<_> = self.ind_s.iter().map(Vec::len).collect();
            let active_indices = boxes_lengths.iter().sum::<usize>();
            let active_boxes = self.ind_s.iter().filter(|v| !v.is_empty()).count();

            println!("New {active_boxes} active boxes of a total of {num_boxes}, and active indices: {active_indices}");

            if self.stats.limiting_factors.limiting_level.active_points < active_indices {
                self.stats.limiting_factors.limiting_level.level =
                    self.level_indexing.current_level;
                self.stats.limiting_factors.limiting_level.active_points = active_indices;
                self.stats.limiting_factors.limiting_level.num_boxes = self.ind_s.len();
            }
        } else {
            let level_keys: Vec<MortonKey> =
                self.level_indexing.level_keys.iter().cloned().collect();
            let num_boxes = level_keys.len();

            // Resize all necessary structures once
            self.target_inds.resize(num_boxes, Vec::new());
            self.ind_s.resize(num_boxes, Vec::new());
            self.near_inds.resize(num_boxes, Vec::new());
            self.box_types
                .resize(num_boxes, BoxType::Full(self.options.id_options.tol_id));

            // Fill in target_inds and ind_s if at max level
            for (box_ind, box_key) in level_keys.iter().enumerate() {
                if let Some(box_indices) = self.level_indexing.boxes_map.get(box_key) {
                    self.target_inds[box_ind] = box_indices.clone();
                    if box_key.level() == self.level_indexing.max_level {
                        self.ind_s[box_ind] = box_indices.clone();
                    }
                }
            }

            // Build key-to-index map
            let key_to_index: HashMap<_, _> = level_keys
                .iter()
                .enumerate()
                .map(|(i, key)| (*key, i))
                .collect();

            // Fill near_inds
            for (box_key, &box_ind) in &key_to_index {
                self.near_inds[box_ind].push(box_ind);

                let near_keys = self
                    .level_indexing
                    .get_box_near_field_keys(box_key, self.level_indexing.current_level);
                for near_key in &near_keys {
                    if let Some(&near_ind) = key_to_index.get(near_key) {
                        self.near_inds[box_ind].push(near_ind);
                    }
                }
            }

            // Print box info
            let total_active: usize = self.target_inds.iter().map(Vec::len).sum();
            let active_boxes = self.ind_s.iter().filter(|v| !v.is_empty()).count();
            println!("New {active_boxes} active boxes of a total of {num_boxes}, and active indices: {total_active}");

            self.stats.limiting_factors.max_level = self.level_indexing.current_level;
        }
    }

    fn group_near_fields(&mut self, current_box_indices: &[usize]) -> Vec<Vec<usize>> {
        let num_indices = current_box_indices.len();

        // Occupancy predicate (source of truth)
        let is_occupied = |b: usize| !self.ind_s[b].is_empty();
        // If you prefer: let is_occupied = |b: usize| !self.target_inds[b].is_empty();

        // Only consider conflicts among boxes that are actually part of current_box_indices
        let current_set: FxHashSet<usize> = current_box_indices.iter().copied().collect();

        let mut group_contents: Vec<FxHashSet<usize>> = Vec::with_capacity(num_indices);
        let mut group_indices: Vec<Vec<usize>> = Vec::with_capacity(num_indices);

        // Largest-first ordering (degree counts only OCCUPIED neighbors inside current_set)
        let mut nodes_with_degree: Vec<(usize, usize)> = (0..num_indices)
            .map(|local_idx| {
                let g = current_box_indices[local_idx];
                let degree = self.near_inds[g]
                    .iter()
                    .copied()
                    .filter(|&n| current_set.contains(&n) && is_occupied(n))
                    .count();
                (local_idx, degree)
            })
            .collect();

        nodes_with_degree.sort_by_key(|&(_, degree)| std::cmp::Reverse(degree));

        'outer: for (ind, _degree) in nodes_with_degree {
            let g = current_box_indices[ind];

            // Skip empty boxes defensively (shouldn't happen if caller filtered, but safe)
            if !is_occupied(g) {
                continue;
            }

            // Try to place g into an existing group
            for (group_set, group) in group_contents.iter_mut().zip(group_indices.iter_mut()) {
                // Conflict if ANY occupied neighbor (within current_set) is already reserved
                let conflict = self.near_inds[g]
                    .iter()
                    .copied()
                    .filter(|&n| current_set.contains(&n) && is_occupied(n))
                    .any(|n| group_set.contains(&n));

                if !conflict {
                    // Reserve: add g itself and its occupied neighbors (within current_set)
                    group_set.insert(g);
                    group_set.extend(
                        self.near_inds[g]
                            .iter()
                            .copied()
                            .filter(|&n| current_set.contains(&n) && is_occupied(n)),
                    );

                    // Store local index into current_box_indices (as in your original code)
                    group.push(ind);
                    continue 'outer;
                }
            }

            // No group found -> create a new group
            let mut new_set = FxHashSet::default();
            new_set.insert(g);
            new_set.extend(
                self.near_inds[g]
                    .iter()
                    .copied()
                    .filter(|&n| current_set.contains(&n) && is_occupied(n)),
            );
            group_contents.push(new_set);
            group_indices.push(vec![ind]);
        }

        group_indices
    }
}

fn pick_ranks<Item: RlstScalar>(
    rank_picking: &RankPicking,
    local_box_ranks: &[BoxType<Item>],
) -> std::option::Option<usize> {
    match rank_picking {
        RankPicking::Min => local_box_ranks
            .iter()
            .filter_map(|b| match b {
                BoxType::Merged(rank) => Some(*rank),
                _ => None,
            })
            .min(),
        RankPicking::DoubleMin => {
            let min = local_box_ranks
                .iter()
                .filter_map(|b| match b {
                    BoxType::Merged(rank) => Some(2 * (*rank)),
                    _ => None,
                })
                .min();
            min
        }
        RankPicking::Max => local_box_ranks
            .iter()
            .filter_map(|b| match b {
                BoxType::Merged(rank) => Some(*rank),
                _ => None,
            })
            .max(),
        RankPicking::Avg => local_box_ranks
            .iter()
            .filter_map(|b| match b {
                BoxType::Merged(rank) => Some(*rank), // Dereference to get the value of rank
                _ => None,
            })
            .collect::<Vec<_>>() // Collect into a Vec
            .into_iter()
            .fold(None, |acc, rank| {
                match acc {
                    Some((sum, count)) => Some((sum + rank, count + 1)),
                    None => Some((rank, 1)), // Start with the first element
                }
            })
            .map(|(sum, count)| sum / count),
        RankPicking::Mid => {
            let min = local_box_ranks
                .iter()
                .filter_map(|b| match b {
                    BoxType::Merged(rank) => Some(*rank),
                    _ => None,
                })
                .min();
            let max = local_box_ranks
                .iter()
                .filter_map(|b| match b {
                    BoxType::Merged(rank) => Some(*rank),
                    _ => None,
                })
                .max();

            match (min, max) {
                (Some(min_val), Some(max_val)) => Some((min_val + max_val) / 2),
                _ => None,
            }
        }
        RankPicking::Tol => None,
    }
}
