use super::{
    box_skeletonisation::{
        IdTimesOperations, LuTimesOperations, Rank, Skel, UpdateTimes, UpdateTimesOperations,
    },
    rsrs_factors::{DiagBoxFactor, FactorOperations, LuTimes, RsrsFactors, RsrsFactorsImpl},
    sketch::SketchData,
    tree_indexing::{TreeData, TreeIndexing},
};
use crate::rsrs::{
    rsrs_factors::{CommutativeFactors, CommutativeFactorsOperations, Factor},
    sketch::UpdateType,
};
use crate::{
    rsrs::rsrs_factors::{IdTimes, Times},
    utils::least_squares_and_null::NullMethod,
};
use bempp_octree::{MortonKey, Octree};
use mpi::traits::CommunicatorCollectives;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rlst::dense::{linalg::lu::MatrixLu, tools::RandScalar};
pub use rlst::prelude::*;
use rustc_hash::FxHashSet;
use std::{
    collections::HashMap,
    fmt::Write,
    time::{Duration, Instant},
}; // Ensure IndexableSpace is in scope

type Inds<T> = Vec<Vec<T>>;

#[derive(Debug)]
pub struct LimitingLevel {
    pub level: usize,
    pub num_boxes: usize,
    pub active_points: usize,
    pub elapsed_time: u128,
}

#[derive(Debug)]
pub struct LimitingFactors {
    pub min_samples: usize,
    pub max_level: usize,
    pub limiting_level: LimitingLevel,
}

#[derive(Debug)]
pub struct Stats {
    pub sampling_time: Vec<u128>,
    pub sampling_extraction_time: u128,
    pub id_times: Vec<IdTimes>,
    pub tot_id_time: u128,
    pub lu_times: Vec<LuTimes>,
    pub tot_lu_time: u128,
    pub update_times: Vec<UpdateTimes>,
    pub total_elapsed_time: u128,
    pub total_elapsed_time_wo_sampling: u128,
    pub dim: usize,
    pub extraction_time: u128,
    pub residual_size: usize,
    pub ranks: Vec<usize>,
    pub box_sizes: Vec<usize>,
    pub near_field_sizes: Vec<usize>,
    pub dec_boxes_per_level: Vec<usize>,
    pub index_calculation: u128,
    pub sorting_near_field: u128,
    pub residual_calculation: u128,
    pub limiting_factors: LimitingFactors,
}

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

#[derive(Debug, Clone)]
pub enum BoxType<Item: RlstScalar> {
    Merged(usize),
    Full(Real<Item>),
}

#[derive(Debug, Clone)]
pub enum RankPicking {
    Min,
    DoubleMin,
    Max,
    Avg,
    Mid,
    Tol,
    AdTol,
}

#[derive(Debug, Clone)]
pub struct IdOptions<Item: RlstScalar> {
    pub null_method: NullMethod,
    pub tol_null: Real<Item>,
    pub tol_id: Real<Item>,
}

#[derive(Debug, Clone)]
pub struct SketchingOptions {
    pub oversampling: usize,
    pub oversampling_diag_blocks: usize,
    pub initial_num_samples: usize,
}

#[derive(Debug, Clone)]
pub struct RsrsOptions<Item: RlstScalar> {
    pub sketching: SketchingOptions,
    pub id_options: IdOptions<Item>,
    pub min_rank: usize,
    pub hermitian: bool,
    pub rank_picking: RankPicking,
    pub tol_ext_near: Real<Item>,
    pub tol_lu: Real<Item>,
    pub tol_diag_ext: Real<Item>,
}

impl<Item: RlstScalar + std::fmt::Display> RsrsOptions<Item> {
    pub fn new(
        oversampling: usize,
        oversampling_diag_blocks: usize,
        initial_num_samples: usize,
        null_method: NullMethod,
        tol_null: Real<Item>,
        tol_id: Real<Item>,
        tol_ext_near: Real<Item>,
        tol_lu: Real<Item>,
        tol_diag_ext: Real<Item>,
        min_rank: usize,
        hermitian: bool,
        rank_picking: RankPicking,
    ) -> Self {
        Self {
            sketching: SketchingOptions {
                oversampling,
                oversampling_diag_blocks,
                initial_num_samples,
            },
            id_options: IdOptions {
                null_method,
                tol_null,
                tol_id,
            },
            min_rank,
            hermitian,
            rank_picking,
            tol_ext_near,
            tol_lu,
            tol_diag_ext,
        }
    }

    pub fn to_identifier(&self) -> String {
        let mut id = String::from("rsrs");

        write!(
            &mut id,
            "_null_{:?}_toln_{:e}",
            self.id_options.null_method, self.id_options.tol_null,
        )
        .unwrap();

        write!(
            &mut id,
            "_os_{os}_osdiag_{osdiag}_initsam_{init}",
            os = self.sketching.oversampling,
            osdiag = self.sketching.oversampling_diag_blocks,
            init = self.sketching.initial_num_samples
        )
        .unwrap();

        write!(
            &mut id,
            "_mrnk_{}_herm_{}_rpick_{:?}_tolextn_{:e}_tolu_{:e}_toldext_{:e}",
            self.min_rank,
            self.hermitian,
            self.rank_picking,
            self.tol_ext_near,
            self.tol_lu,
            self.tol_diag_ext
        )
        .unwrap();

        id
    }
}

type Real<T> = <T as rlst::RlstScalar>::Real;

fn oversample(samples: usize, oversampling: usize) -> usize {
    samples + (samples / 100) * oversampling
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
{
    pub fn new<C: CommunicatorCollectives>(
        dim: usize,
        octree: &Octree<'_, C>,
        options: RsrsOptions<Item>,
    ) -> Self {
        let level_indexing: TreeData = <TreeData as TreeIndexing>::new(octree);
        let target_inds: Inds<usize> = Vec::new();
        let near_inds: Inds<usize> = Vec::new();
        let ind_s: Inds<usize> = Vec::new();
        let ind_r: Inds<usize> = Vec::new();
        let box_types: Vec<BoxType<Real<Item>>> = Vec::new();
        let y_data: SketchData<Item> = SketchData::new(dim, false);
        let z_data: SketchData<Item> = SketchData::new(dim, true);
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
            limiting_level: limiting_level,
        };

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

    pub fn run<OpImpl: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>>(
        &mut self,
        operator: &OpImpl,
    ) -> RsrsFactors<Item> {
        let num_levels: usize = self.level_indexing.max_level;
        let algo_start: Instant = Instant::now();
        let mut rsrs_factors =
            <RsrsFactors<Item> as RsrsFactorsImpl<Item>>::new(num_levels, self.dim);
        let start: Instant = Instant::now();
        self.tree_cycle(operator, &mut rsrs_factors);
        let duration = start.elapsed();
        println!("Tree cycle elapsed time: {} s", duration.as_secs());
        println!(
            "Extracting diagonal blocks with {} active samples",
            self.active_samples
        );
        let start: Instant = Instant::now();

        let (diag_box_factors, rows, cols) = self.extract_step();
        rsrs_factors.diag_box_factors = diag_box_factors;
        rsrs_factors.perm_factor.orig_indices = cols;
        rsrs_factors.perm_factor.perm_indices = rows;
        let extraction_time = start.elapsed();
        println!("Extraction time: {:?}s\n", extraction_time.as_secs());
        self.stats.extraction_time = extraction_time.as_millis();
        let duration = algo_start.elapsed();
        println!(
            "Total elapsed time: {:?}, with {} active samples\n",
            duration, self.active_samples
        );
        println!("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
        self.stats.total_elapsed_time = duration.as_millis();
        self.stats.total_elapsed_time_wo_sampling = self.stats.total_elapsed_time.saturating_sub(
            self.stats.sampling_extraction_time + self.stats.sampling_time.iter().sum::<u128>(),
        );

        rsrs_factors
    }

    fn tree_cycle<
        OpImpl: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
    >(
        &mut self,
        operator: &OpImpl,
        rsrs_factors: &mut RsrsFactors<Item>,
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
            self.level_cycle(operator, rsrs_factors, level_it);
            println!("End level cycle. Summary:");
            println!("-------------------------");
            let duration: Duration = start.elapsed();
            println!("Elapsed time: {} s", duration.as_secs());

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

            println!("Sketch Points: {}", len_s);
            println!("Residual Points: {}", len_r);
            println!(
                "Current Number of Samples: {} of which {} are active\n",
                self.y_data.test.shape()[0],
                self.active_samples
            );

            level -= 1;
            level_it += 1;

            if level <= min_level {
                println!("-------------------------");
                println!("\nReached lower level: {}", level);
                self.stats.residual_size = len_r;
                let min_oversamples =
                    oversample(len_s, self.options.sketching.oversampling_diag_blocks);
                println!("Minimum samples: {}", min_oversamples);

                let (tot_sampling_time, tot_id_update, tot_lu_update) = self.add_samples(
                    min_oversamples,
                    operator,
                    rsrs_factors,
                    level_it,
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
        OpImpl: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
    >(
        &mut self,
        operator: &OpImpl,
        rsrs_factors: &mut RsrsFactors<Item>,
        level_it: usize,
    ) {
        let merged_count = self
            .box_types
            .iter()
            .filter(|box_type| matches!(box_type, BoxType::Merged(_rank)))
            .count();
        println!("Number of merged boxes: {}\n", merged_count);

        let current_box_indices =
            self.sampling_step(operator, rsrs_factors, level_it == 0, level_it);
        let id_step_start: Instant = Instant::now();
        let (id_factors_res, current_box_indices, level_ind_r) =
            self.id_level_iteration(&current_box_indices);
        rsrs_factors.id_factors[level_it] = id_factors_res;
        let id_step_duration = id_step_start.elapsed();
        self.stats.tot_id_time += id_step_duration.as_millis();

        println!("ID step in {:?}", id_step_duration);

        let start_id_update: Instant = Instant::now();
        let update_type = UpdateType::Id(&rsrs_factors.id_factors[level_it]);
        self.update_samples(0, self.active_samples, level_it, &update_type);
        let update_id_time: Duration = start_id_update.elapsed();

        let mut update_times = UpdateTimes::new();
        update_times.sum(update_id_time.as_millis(), 0_u128);
        self.stats.update_times.push(update_times);

        println!("ID updated in {:?}\n", update_id_time);

        rsrs_factors.lu_factors[level_it] =
            self.lu_level_iteration(&current_box_indices, &level_ind_r, level_it);
    }

    fn sampling_step<
        OpImpl: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
    >(
        &mut self,
        operator: &OpImpl,
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

        let min_oversamples = oversample(
            self.ind_s[last_box_index].len() + self.get_near_indices(last_box_index).len(),
            self.options.sketching.oversampling,
        );

        let min_samples = if start {
            self.options
                .sketching
                .initial_num_samples
                .max(min_oversamples)
        } else {
            min_oversamples
        };

        let (tot_sampling_time, tot_id_update, tot_lu_update) =
            self.add_samples(min_samples, operator, rsrs_factors, level_it, start, 1);

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
        OpImpl: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
    >(
        &mut self,
        min_samples: usize,
        operator: &OpImpl,
        rsrs_factors: &RsrsFactors<Item>,
        level_it: usize,
        start: bool,
        _seed: u64,
    ) -> (u128, u128, u128) {
        let mut tot_sampling_time = 0_u128;
        let test_shape = self.y_data.test.shape();
        if min_samples > test_shape[0] {
            let extra_samples = min_samples.saturating_sub(self.y_data.test.shape()[0]);
            println!("Sampling step. Sampling new {} vectors", extra_samples);

            tot_sampling_time += self.y_data.add_samples(extra_samples, operator, 0_u64);

            if !self.options.hermitian {
                let tot_z_sampling_time = self.z_data.add_samples(extra_samples, operator, 0_u64);
                tot_sampling_time += tot_z_sampling_time;
            }

            println!("Total samples: {}", self.y_data.test.shape()[0]);
            println!("Sampling time: {}ms\n", tot_sampling_time);
        }
        if !start && min_samples > self.active_samples {
            let extra_active_samples = min_samples.saturating_sub(self.active_samples);
            println!(
                "Extra active samples: {}. Min samples: {}",
                extra_active_samples, min_samples
            );
            let update_start = self.active_samples;
            let (tot_id_update, tot_lu_update) = self.update_samples(
                update_start,
                extra_active_samples,
                level_it,
                &UpdateType::Both(rsrs_factors),
            );

            println!("Update times: {}ms, {}ms", tot_id_update, tot_lu_update);
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
    ) -> (u128, u128) {
        let (mut tot_id_update, mut tot_lu_update) =
            self.y_data
                .update_samples(update_start, samples_to_update, level, &update_type);

        if !self.options.hermitian {
            let (tot_z_id_update, tot_z_lu_update) =
                self.z_data
                    .update_samples(update_start, samples_to_update, level, &update_type);
            tot_id_update += tot_z_id_update;
            tot_lu_update += tot_z_lu_update;
        }

        (tot_id_update, tot_lu_update)
    }

    fn id_level_iteration(
        &mut self,
        current_box_indices: &Vec<usize>,
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

        let mut current_box_indices = current_box_indices.clone();
        current_box_indices.sort_by_key(|&box_ind| {
            let box_num = *current_near_field_ind_to_num.get(&box_ind).unwrap();
            let near_field_len = current_near_field_indices[box_num].len();
            let source_len = self.ind_s[box_ind].len();
            oversample(
                near_field_len + source_len,
                self.options.sketching.oversampling,
            )
        });
        let start = Instant::now();
        let id_level_iteration_res: Vec<_> = current_box_indices
            .par_iter()
            .map(|&box_ind| {
                let box_num = *current_near_field_ind_to_num.get(&box_ind).unwrap();
                let mut near_field_inds = &current_near_field_indices[box_num];
                let min_box_samples = oversample(
                    near_field_inds.len() + self.ind_s[box_ind].len(),
                    self.options.sketching.oversampling,
                );
                let mut skel_box = <Item as Default>::default();

                let rank = skel_box.id_step(
                    &self.box_types[box_ind],
                    &self.ind_s[box_ind],
                    &mut near_field_inds,
                    &self.y_data,
                    &self.z_data,
                    min_box_samples,
                    &self.options,
                );
                (box_ind, rank)
            })
            .collect();
        let id_level_duration = start.elapsed();
        println!("ID calculations in {:?}", id_level_duration,);

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
                Rank::Full(it_id_times) => {
                    len_full_rank += self.ind_s[box_ind].len();
                    let res_id_times = match it_id_times {
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
        println!("ID postprocessing in {:?}", id_level_duration);

        (id_level, current_box_indices, level_ind_r)
    }

    fn lu_level_iteration(
        &mut self,
        current_box_indices: &Vec<usize>,
        level_ind_r: &Vec<Vec<usize>>,
        level_it: usize,
    ) -> Vec<CommutativeFactors<Item>> {
        println!("Start LU step");

        let start: Instant = Instant::now();
        let independent_near_fields = self.group_near_fields(&current_box_indices);

        let level_near_field_inds: Vec<_> = current_box_indices
            .iter()
            .map(|&box_ind| self.get_near_indices(box_ind))
            .collect();
        let time_independent_nf = start.elapsed();

        self.stats.sorting_near_field += time_independent_nf.as_millis();

        println!("Batches computed in {:?}", time_independent_nf);

        let mut update_parallel_batch_time = 0;
        let mut lu_times = LuTimes::new();
        let mut update_times = UpdateTimes::new();

        println!(
            "Active samples vs total samples: {}, {}",
            self.active_samples,
            self.y_data.test.shape()[0]
        );
        let lu_step_start: Instant = Instant::now();
        let batches_res: Vec<_> = independent_near_fields
            .into_iter()
            .map(|batch| {
                let mut lu_batch: CommutativeFactors<Item> = CommutativeFactorsOperations::new();
                let mut lu_batch_time = LuTimes::new();
                let lu_times_and_factor: Vec<_> = batch
                    .par_iter()
                    .map(|box_num| {
                        let skel_box = <Item as Default>::default();
                        let box_ind = current_box_indices[*box_num];
                        let min_num_samples = oversample(
                            self.target_inds[box_ind].len() + level_near_field_inds[*box_num].len(),
                            self.options.sketching.oversampling,
                        );
                        let (lu_factor, lu_times) = skel_box.lu_step(
                            &self.y_data,
                            &self.z_data,
                            &mut level_ind_r[*box_num].clone(),
                            &mut level_near_field_inds[*box_num].clone(),
                            min_num_samples,
                            &self.options,
                        );

                        (lu_times, lu_factor)
                    })
                    .collect();

                lu_times_and_factor
                    .into_iter()
                    .for_each(|(lu_time, lu_factor)| {
                        lu_batch.add_factor(Factor::Lu(lu_factor));
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
            "LU step in {}ms, with updates in {}ms\n",
            lu_step_duration, update_parallel_batch_time
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

        let mut diag_box_factors: CommutativeFactors<Item> = CommutativeFactorsOperations::new();
        let mut diag_box_res: Vec<_> = self
            .ind_r
            .par_iter()
            .map(|inds| {
                DiagBoxFactor::new(
                    &mut inds.to_vec(),
                    &mut inds.to_vec(),
                    &self.y_data,
                    &self.z_data,
                    self.active_samples,
                    &BoxType::Merged(1),
                    &self.options,
                )
            })
            .collect();

        diag_box_res.push(DiagBoxFactor::new(
            &mut acc_ind_s.to_vec(),
            &mut acc_ind_s.to_vec(),
            &self.y_data,
            &self.z_data,
            self.active_samples,
            &BoxType::Merged(1),
            &self.options,
        ));

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
                    box_types[parent_index] = BoxType::Merged(min_rank);
                } else {
                    if matches!(self.options.rank_picking, RankPicking::Tol) {
                        box_types[parent_index] = BoxType::Full(self.options.id_options.tol_id);
                    }
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

            println!(
                "New {} boxes, with {} active indices.",
                self.ind_s.len(),
                active_indices,
            );

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
            println!(
                "New {} boxes, and active indices: {}",
                num_boxes, total_active
            );

            self.stats.limiting_factors.max_level = self.level_indexing.current_level;
        }
    }

    fn group_near_fields(&mut self, current_box_indices: &Vec<usize>) -> Vec<Vec<usize>> {
        // Get the next level's keys and the current level's keys

        let num_indices = current_box_indices.len();
        let mut group_contents: Vec<FxHashSet<usize>> = Vec::with_capacity(num_indices);
        let mut group_indices: Vec<Vec<usize>> = Vec::with_capacity(num_indices);

        let inds = (0..num_indices).collect::<Vec<_>>(); // optional: sort here by neighbor size

        'outer: for ind in inds {
            let current_neighbors = &self.near_inds[current_box_indices[ind]];

            for (group_set, group) in group_contents.iter_mut().zip(group_indices.iter_mut()) {
                let has_overlap = current_neighbors.iter().any(|x| group_set.contains(x));
                if !has_overlap {
                    group_set.extend(current_neighbors.iter().copied());
                    group.push(ind);
                    continue 'outer;
                }
            }

            // No compatible group found, create a new one
            let mut new_set = FxHashSet::default();
            new_set.extend(current_neighbors.iter().copied());
            group_contents.push(new_set);
            group_indices.push(vec![ind]);
        }
        group_indices
    }
}

fn pick_ranks<Item: RlstScalar>(
    rank_picking: &RankPicking,
    local_box_ranks: &Vec<BoxType<Item>>,
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

            let double_min = match (min, max) {
                (Some(min_val), Some(max_val)) => Some((2 * min_val).min(max_val)),
                _ => None,
            };

            double_min
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

            let mid = match (min, max) {
                (Some(min_val), Some(max_val)) => Some((min_val + max_val) / 2),
                _ => None,
            };

            mid
        }
        RankPicking::Tol => None,
        RankPicking::AdTol => None,
    }
}
