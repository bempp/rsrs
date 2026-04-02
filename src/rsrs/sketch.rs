use crate::rsrs::rsrs_factors::base_factors::BaseFactorOptions;
use crate::rsrs::rsrs_factors::commutative_factors::CommutativeFactors;
use crate::rsrs::rsrs_factors::commutative_factors::CommutativeFactorsOperations;
use crate::rsrs::rsrs_factors::commutative_factors::FactorType;
use crate::rsrs::rsrs_factors::commutative_factors::MulOptions;
use crate::rsrs::rsrs_factors::commutative_factors::RsrsFactors;
use crate::rsrs::rsrs_factors::rsrs_operator::FactType;
use crate::rsrs::rsrs_factors::rsrs_operator::RsrsFactorsImpl;
use crate::utils::io::IOData;
use crate::utils::linear_algebra::streaming_chunk_rows;
use mpi::traits::Communicator;
use mpi::traits::Equivalence;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::ThreadPool;
use rlst::dense::linalg::{interpolative_decomposition::MatrixIdNoSkel, lu::MatrixLu};
use rlst::operator::ConcreteElementContainer;
use rlst::{dense::array::reference::ArrayRef, dense::array::views::ArraySubView};
pub use rlst::{
    dense::{array::empty_array, tools::RandScalar},
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Instant};
pub enum UpdateType<'a, Item: RlstScalar> {
    Lu(&'a CommutativeFactors<Item>),
    Id(&'a CommutativeFactors<Item>),
    Both(&'a RsrsFactors<Item>),
}

pub enum BatchUpdateType<'a, Item: RlstScalar> {
    Single(&'a CommutativeFactors<Item>),
    Multi(&'a RsrsFactors<Item>),
}

pub struct SketchData<Item: RlstScalar> {
    pub sketch: DynamicArray<Item, 2>,
    pub test: DynamicArray<Item, 2>,
    pub dim: usize,
    pub num_samples: usize,
    pub trans: TransMode,
}

type SketchArrayRef<'a, Item> = ArrayRef<'a, Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>;
pub type SketchChunkView<'a, Item> =
    Array<Item, ArraySubView<Item, SketchArrayRef<'a, Item>, 2>, 2>;

pub struct SampleChunk<'a, Item: RlstScalar> {
    pub row_offset: usize,
    pub test: SketchChunkView<'a, Item>,
    pub sketch: SketchChunkView<'a, Item>,
}

pub struct SampleChunkIter<'a, Item: RlstScalar> {
    data: &'a SketchData<Item>,
    subs_sample_dim: usize,
    chunk_rows: usize,
    next_row: usize,
}

impl<'a, Item: RlstScalar> Iterator for SampleChunkIter<'a, Item> {
    type Item = SampleChunk<'a, Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_row >= self.subs_sample_dim {
            return None;
        }

        let row_offset = self.next_row;
        let rows = (row_offset + self.chunk_rows).min(self.subs_sample_dim) - row_offset;
        self.next_row += rows;

        Some(SampleChunk {
            row_offset,
            test: self
                .data
                .test
                .r()
                .into_subview([row_offset, 0], [rows, self.data.dim]),
            sketch: self
                .data
                .sketch
                .r()
                .into_subview([row_offset, 0], [rows, self.data.dim]),
        })
    }
}

pub struct FullBoxesData<Item: RlstScalar> {
    pub y_data: SketchData<Item>,
    pub z_data: SketchData<Item>,
    pub dim: usize,
    pub active_samples: usize,
    pub symmetric: bool,
}

pub enum SampleType {
    EquallyDistributed,
    StandardNormal,
    RealEquallyDistributed,
    RealStandardNormal,
}
pub trait SamplingSpace: LinearSpace {
    fn sampling<R: Rng>(
        &self,
        x: &mut Element<ConcreteElementContainer<Self::E>>,
        rng: &mut R,
        sample_type: SampleType,
    );

    fn zero(space: std::rc::Rc<Self>) -> Element<ConcreteElementContainer<Self::E>>;

    fn fill_array<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::F>
            + UnsafeRandomAccessMut<2, Item = Self::F>
            + Stride<2>
            + RawAccessMut<Item = Self::F>
            + Shape<2>,
    >(
        &self,
        x: &Element<ConcreteElementContainer<Self::E>>,
        other: &mut Array<Self::F, ArrayImpl, 2>,
        offset: usize,
        trans: TransMode,
    );

    fn clone_vec(
        &self,
        other: &Element<ConcreteElementContainer<Self::E>>,
    ) -> Element<ConcreteElementContainer<Self::E>>;

    fn conj_vec(
        &self,
        other: &Element<ConcreteElementContainer<Self::E>>,
    ) -> Element<ConcreteElementContainer<Self::E>>;
}

impl<Item: RlstScalar + RandScalar> SamplingSpace for ArrayVectorSpace<Item>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    fn sampling<R: Rng>(
        &self,
        x: &mut Element<ConcreteElementContainer<Self::E>>,
        rng: &mut R,
        sample_type: SampleType,
    ) {
        match sample_type {
            SampleType::EquallyDistributed => x.view_mut().fill_from_equally_distributed(rng),
            SampleType::StandardNormal => x.view_mut().fill_from_standard_normal(rng),
            SampleType::RealEquallyDistributed => {
                x.view_mut().fill_from_equally_distributed_real(rng)
            }
            SampleType::RealStandardNormal => x.view_mut().fill_from_normally_distributed_real(rng),
        };
    }

    fn zero(space: std::rc::Rc<Self>) -> Element<ConcreteElementContainer<Self::E>> {
        Element::<ConcreteElementContainer<Self::E>>::new(Self::E::new(space))
    }

    fn fill_array<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::F>
            + UnsafeRandomAccessMut<2, Item = Self::F>
            + Stride<2>
            + RawAccessMut<Item = Self::F>
            + Shape<2>,
    >(
        &self,
        x: &Element<ConcreteElementContainer<Self::E>>,
        other: &mut Array<Self::F, ArrayImpl, 2>,
        offset: usize,
        trans: TransMode,
    ) {
        match trans {
            TransMode::NoTrans => other.r_mut().slice(0, offset).fill_from(x.view()),
            TransMode::ConjNoTrans => todo!(),
            TransMode::Trans => other.r_mut().slice(0, offset).fill_from(x.view().conj()),
            TransMode::ConjTrans => todo!(),
        };
    }

    fn clone_vec(
        &self,
        other: &Element<ConcreteElementContainer<Self::E>>,
    ) -> Element<ConcreteElementContainer<Self::E>> {
        let mut new =
            Element::<ConcreteElementContainer<Self::E>>::new(Self::E::new(other.space()));
        new.view_mut().fill_from(other.view());

        new
    }

    fn conj_vec(
        &self,
        other: &Element<ConcreteElementContainer<Self::E>>,
    ) -> Element<ConcreteElementContainer<Self::E>> {
        let mut aux_array = rlst_dynamic_array2!(Item, [self.dimension(), 1]);

        aux_array.r_mut().slice(1, 0).fill_from(other.view());

        let mut new =
            Element::<ConcreteElementContainer<Self::E>>::new(Self::E::new(other.space()));

        new.view_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(i, val)| *val = aux_array.r().data()[i].conj());
        new
    }
}

impl<C: Communicator, Item: RlstScalar + RandScalar + Equivalence> SamplingSpace
    for DistributedArrayVectorSpace<'_, C, Item>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    fn sampling<R: Rng>(
        &self,
        x: &mut Element<ConcreteElementContainer<Self::E>>,
        rng: &mut R,
        sample_type: SampleType,
    ) {
        match sample_type {
            SampleType::EquallyDistributed => {
                x.view_mut().local_mut().fill_from_equally_distributed(rng)
            }
            SampleType::StandardNormal => x.view_mut().local_mut().fill_from_standard_normal(rng),
            SampleType::RealEquallyDistributed => x
                .view_mut()
                .local_mut()
                .fill_from_equally_distributed_real(rng),
            SampleType::RealStandardNormal => x
                .view_mut()
                .local_mut()
                .fill_from_normally_distributed_real(rng),
        };
    }

    fn zero(space: std::rc::Rc<Self>) -> Element<ConcreteElementContainer<Self::E>> {
        Element::<ConcreteElementContainer<Self::E>>::new(Self::E::new(space))
    }

    fn fill_array<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::F>
            + UnsafeRandomAccessMut<2, Item = Self::F>
            + Stride<2>
            + RawAccessMut<Item = Self::F>
            + Shape<2>,
    >(
        &self,
        x: &Element<ConcreteElementContainer<Self::E>>,
        other: &mut Array<Self::F, ArrayImpl, 2>,
        offset: usize,
        trans: TransMode,
    ) {
        match trans {
            TransMode::NoTrans => other
                .r_mut()
                .slice(0, offset)
                .fill_from(x.view().local().r()),
            TransMode::ConjNoTrans => todo!(),
            TransMode::Trans => other
                .r_mut()
                .slice(0, offset)
                .fill_from(x.view().local().r().conj()),
            TransMode::ConjTrans => todo!(),
        };
    }

    fn clone_vec(
        &self,
        other: &Element<ConcreteElementContainer<Self::E>>,
    ) -> Element<ConcreteElementContainer<Self::E>> {
        let mut new =
            Element::<ConcreteElementContainer<Self::E>>::new(Self::E::new(other.space()));
        new.view_mut()
            .local_mut()
            .fill_from(other.view().local().r());

        new
    }

    fn conj_vec(
        &self,
        other: &Element<ConcreteElementContainer<Self::E>>,
    ) -> Element<ConcreteElementContainer<Self::E>> {
        let mut aux_array = rlst_dynamic_array2!(Item, [self.dimension(), 1]);

        aux_array
            .r_mut()
            .slice(1, 0)
            .fill_from(other.view().local().r());

        let mut new =
            Element::<ConcreteElementContainer<Self::E>>::new(Self::E::new(other.space()));

        new.view_mut()
            .local_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(i, val)| *val = aux_array.r().data()[i].conj());
        new
    }
}

pub(crate) fn mix_seed(mut seed: u64) -> u64 {
    seed = seed.wrapping_add(0x9E3779B97F4A7C15);
    seed = (seed ^ (seed >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    seed = (seed ^ (seed >> 27)).wrapping_mul(0x94D049BB133111EB);
    seed ^ (seed >> 31)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Shift {
    True(f64), //TODO: Change to Item
    False,
}

pub(crate) fn shift_alpha(shift: &Shift) -> f64 {
    match shift {
        Shift::True(alpha) => *alpha,
        Shift::False => 0.0,
    }
}

pub(crate) fn apply_shift_delta<Item: RlstScalar>(
    sketch: &mut DynamicArray<Item, 2>,
    test: &DynamicArray<Item, 2>,
    delta: f64,
) {
    if delta.abs() <= f64::EPSILON {
        return;
    }

    let delta_item = Item::from(delta).unwrap();
    sketch
        .data_mut()
        .iter_mut()
        .zip(test.data().iter())
        .for_each(|(sketch_val, test_val)| {
            *sketch_val = *sketch_val + delta_item * *test_val;
        });
}

impl<Item: RlstScalar> SketchData<Item> {
    pub fn new(dim: usize, trans: TransMode) -> Self {
        let test: Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2> = empty_array();
        let sketch: Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2> = empty_array();
        Self {
            sketch,
            test,
            dim,
            num_samples: 0,
            trans,
        }
    }

    pub fn trans_val(&self) -> bool {
        match self.trans {
            TransMode::NoTrans => false,
            TransMode::ConjNoTrans => false,
            TransMode::Trans => true,
            TransMode::ConjTrans => true,
        }
    }

    pub fn chunk_iter(
        &self,
        subs_sample_dim: usize,
        cols_per_row: usize,
        live_buffers: usize,
    ) -> SampleChunkIter<'_, Item> {
        let subs_sample_dim = subs_sample_dim
            .min(self.test.shape()[0])
            .min(self.sketch.shape()[0]);
        let chunk_rows =
            streaming_chunk_rows::<Item>(subs_sample_dim, cols_per_row, live_buffers).max(1);

        SampleChunkIter {
            data: self,
            subs_sample_dim,
            chunk_rows,
            next_row: 0,
        }
    }
}

impl<
        Item: RlstScalar
            + RandScalar
            + MatrixId
            + MatrixIdNoSkel
            + MatrixInverse
            + MatrixPseudoInverse
            + MatrixLu
            + MatrixQr,
    > SketchData<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
    Item: IOData<Item>,
{
    pub fn add_samples<
        Space: SamplingSpace<F = Item>,
        OpImpl: AsApply<Domain = Space, Range = Space>,
    >(
        &mut self,
        extra_num_samples: usize,
        operator: Operator<OpImpl>,
        shift: &Shift,
        save_samples: bool,
        sample_storage_dir: Option<&Path>,
        seed: u64,
    ) -> u128 {
        let sampling_start: Instant = Instant::now();
        let test_shape = self.test.shape();
        let total_samples = test_shape[0] + extra_num_samples;

        // Preserve existing samples while letting the backing Vec grow amortized
        // instead of rebuilding a fresh matrix on every resize.
        self.test.resize_in_place([total_samples, self.dim]);
        self.sketch.resize_in_place([total_samples, self.dim]);

        let mut sample_generation = std::time::Duration::ZERO;
        let mut multiplication = std::time::Duration::ZERO;
        let mut filling = std::time::Duration::ZERO;
        (0..extra_num_samples).for_each(|row| {
            let start: Instant = Instant::now();
            let offset = test_shape[0] + row;
            let mut chunk_test_vec = SamplingSpace::zero(operator.r().domain());
            let row_seed = mix_seed(
                seed ^ (offset as u64).wrapping_mul(0x9E3779B97F4A7C15)
                    ^ (self.dim as u64).rotate_left(21)
                    ^ u64::from(self.trans_val()),
            );
            let mut rng = ChaCha8Rng::seed_from_u64(row_seed);
            operator.domain().sampling(
                &mut chunk_test_vec,
                &mut rng,
                SampleType::RealStandardNormal,
            );

            sample_generation += start.elapsed();

            let start: Instant = Instant::now();

            let chunk_sketch_vec = match shift {
                Shift::True(alpha) => {
                    let mut chunk_sketch_vec_stab = operator.domain().clone_vec(&chunk_test_vec);
                    chunk_sketch_vec_stab.scale_inplace(Item::from(*alpha).unwrap());
                    chunk_sketch_vec_stab
                        .sum_inplace(operator.apply(chunk_test_vec.r(), self.trans));
                    chunk_sketch_vec_stab
                }
                Shift::False => operator.apply(chunk_test_vec.r(), self.trans),
            };

            multiplication += start.elapsed();

            let start: Instant = Instant::now();
            operator
                .domain()
                .fill_array(&chunk_test_vec, &mut self.test, offset, self.trans);
            operator
                .domain()
                .fill_array(&chunk_sketch_vec, &mut self.sketch, offset, self.trans);
            filling += start.elapsed();

            if (row + 1) % 30 == 0 {
                println!("Sample generation: {sample_generation:?}");
                println!(
                    "Multiplication: {:?} ({:?} per sample) -> 30 samples",
                    multiplication,
                    multiplication / 30
                );
                println!("Filling: {filling:?}");
                println!("Current number of new samples: {}\n", row + 1);
                sample_generation = std::time::Duration::ZERO;
                multiplication = std::time::Duration::ZERO;
                filling = std::time::Duration::ZERO;
            }
        });

        if save_samples {
            let mut test_sv = empty_array();
            test_sv.r_mut().fill_from_resize(
                self.test
                    .r()
                    .into_subview([test_shape[0], 0], [extra_num_samples, self.dim]),
            );
            let mut sketch_sv: Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2> =
                empty_array();
            sketch_sv.r_mut().fill_from_resize(
                self.sketch
                    .r()
                    .into_subview([test_shape[0], 0], [extra_num_samples, self.dim]),
            );

            let (test_base, sketch_base) = if self.trans_val() {
                ("z_test_file", "z_sketch_file")
            } else {
                ("y_test_file", "y_sketch_file")
            };
            // Persist canonical unshifted sketches on disk so saved samples can be
            // reused across runs with different operator shifts.
            let current_shift = shift_alpha(shift);
            if current_shift.abs() > f64::EPSILON {
                apply_shift_delta(&mut sketch_sv, &test_sv, -current_shift);
            }

            let _ = <Item as IOData<Item>>::append_in_dir(&test_sv, test_base, sample_storage_dir);
            let _ =
                <Item as IOData<Item>>::append_in_dir(&sketch_sv, sketch_base, sample_storage_dir);

            println!("{} samples saved", test_sv.shape()[0])
        }
        let duration = sampling_start.elapsed();

        self.num_samples = test_shape[0] + extra_num_samples; //TODO: Change this to total_samples

        duration.as_millis()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_samples(
        &mut self,
        update_start: usize,
        samples_to_update: usize,
        level: usize,
        update_type: &UpdateType<Item>,
        fact_type: &FactType,
        thread_pool: &ThreadPool,
        num_threads: usize,
    ) -> (u128, u128) {
        let (factor_1, factor_2) = if self.trans_val() {
            (FactorType::S, FactorType::F)
        } else {
            (FactorType::F, FactorType::S)
        };

        let (mut sub_test, mut sub_sketch) = (
            self.test
                .r_mut()
                .into_subview([update_start, 0], [samples_to_update, self.dim]),
            self.sketch
                .r_mut()
                .into_subview([update_start, 0], [samples_to_update, self.dim]),
        );

        let mut id_time = 0_u128;
        let mut lu_time = 0_u128;

        match update_type {
            UpdateType::Lu(lu_batch) => {
                lu_time += update_lu_level(
                    &mut sub_sketch,
                    &mut sub_test,
                    level,
                    BatchUpdateType::Single(lu_batch),
                    &factor_1,
                    &factor_2,
                    self.trans,
                    thread_pool,
                    num_threads,
                );
            }
            UpdateType::Id(id_batch) => {
                lu_time += update_id_level(
                    &mut sub_sketch,
                    &mut sub_test,
                    level,
                    BatchUpdateType::Single(id_batch),
                    &factor_1,
                    &factor_2,
                    self.trans,
                    thread_pool,
                    num_threads,
                );
            }
            UpdateType::Both(rsrs_factors) => match fact_type {
                FactType::Joint => (0..level).for_each(|level_it| {
                    let (loc_id_time, loc_lu_time) = update_level(
                        &mut sub_sketch,
                        &mut sub_test,
                        level_it,
                        BatchUpdateType::Multi(rsrs_factors),
                        &factor_1,
                        &factor_2,
                        self.trans,
                    );
                    id_time += loc_id_time;
                    lu_time += loc_lu_time;
                }),
                FactType::Split => (0..level).for_each(|level_it| {
                    id_time += update_id_level(
                        &mut sub_sketch,
                        &mut sub_test,
                        level_it,
                        BatchUpdateType::Multi(rsrs_factors),
                        &factor_1,
                        &factor_2,
                        self.trans,
                        thread_pool,
                        num_threads,
                    );

                    lu_time += update_lu_level(
                        &mut sub_sketch,
                        &mut sub_test,
                        level_it,
                        BatchUpdateType::Multi(rsrs_factors),
                        &factor_1,
                        &factor_2,
                        self.trans,
                        thread_pool,
                        num_threads,
                    );
                }),
            },
        }

        (id_time, lu_time)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn update_id_level<
    Item: RlstScalar
        + RandScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessMut<2, Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>
        + std::marker::Send
        + std::marker::Sync,
>(
    sketch: &mut Array<Item, ArrayImplMut, 2>,
    test: &mut Array<Item, ArrayImplMut, 2>,
    level_it: usize,
    update_type: BatchUpdateType<Item>,
    factor_1: &FactorType,
    factor_2: &FactorType,
    trans: TransMode,
    thread_pool: &ThreadPool,
    num_threads: usize,
) -> u128
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let start = Instant::now();
    let sketch_base_options = BaseFactorOptions {
        inv: true,
        trans,
        trans_target: true,
    };
    let test_base_options = BaseFactorOptions {
        inv: false,
        trans,
        trans_target: true,
    };

    let sketch_factor_options = MulOptions {
        base_options: sketch_base_options,
        side: Side::Left,
        factor_type: factor_1.clone(),
    };
    let test_factor_options = MulOptions {
        base_options: test_base_options,
        side: Side::Left,
        factor_type: factor_2.clone(),
    };

    match update_type {
        BatchUpdateType::Single(id_batch) => {
            id_batch.mul(sketch, thread_pool, num_threads, &sketch_factor_options);
            id_batch.mul(test, thread_pool, num_threads, &test_factor_options);
        }
        BatchUpdateType::Multi(rsrs_factors) => {
            rsrs_factors.apply_id_level(sketch, &sketch_factor_options, level_it);
            rsrs_factors.apply_id_level(test, &test_factor_options, level_it);
        }
    }

    start.elapsed().as_millis()
}

#[allow(clippy::too_many_arguments)]
pub fn update_lu_level<
    Item: RlstScalar
        + RandScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessMut<2, Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>
        + std::marker::Send
        + std::marker::Sync,
>(
    sketch: &mut Array<Item, ArrayImplMut, 2>,
    test: &mut Array<Item, ArrayImplMut, 2>,
    level_it: usize,
    update_type: BatchUpdateType<Item>,
    factor_1: &FactorType,
    factor_2: &FactorType,
    trans: TransMode,
    thread_pool: &ThreadPool,
    num_threads: usize,
) -> u128
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let start = Instant::now();
    let sketch_base_options = BaseFactorOptions {
        inv: true,
        trans,
        trans_target: true,
    };
    let test_base_options = BaseFactorOptions {
        inv: false,
        trans,
        trans_target: true,
    };

    let sketch_factor_options = MulOptions {
        side: Side::Left,
        factor_type: factor_1.clone(),
        base_options: sketch_base_options,
    };
    let test_factor_options = MulOptions {
        side: Side::Left,
        factor_type: factor_2.clone(),
        base_options: test_base_options,
    };

    match update_type {
        BatchUpdateType::Single(lu_batch) => {
            lu_batch.mul(sketch, thread_pool, num_threads, &sketch_factor_options);
            lu_batch.mul(test, thread_pool, num_threads, &test_factor_options);
        }
        BatchUpdateType::Multi(rsrs_factors) => {
            rsrs_factors.apply_lu_level(sketch, &sketch_factor_options, false, level_it);
            rsrs_factors.apply_lu_level(test, &test_factor_options, false, level_it);
        }
    }

    start.elapsed().as_millis()
}

pub fn update_level<
    Item: RlstScalar
        + RandScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessMut<2, Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>
        + std::marker::Send
        + std::marker::Sync,
>(
    sketch: &mut Array<Item, ArrayImplMut, 2>,
    test: &mut Array<Item, ArrayImplMut, 2>,
    level_it: usize,
    update_type: BatchUpdateType<Item>,
    factor_1: &FactorType,
    factor_2: &FactorType,
    trans: TransMode,
) -> (u128, u128)
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let sketch_base_options = BaseFactorOptions {
        inv: true,
        trans,
        trans_target: true,
    };
    let test_base_options = BaseFactorOptions {
        inv: false,
        trans,
        trans_target: true,
    };
    let sketch_factor_options = MulOptions {
        side: Side::Left,
        factor_type: factor_1.clone(),
        base_options: sketch_base_options,
    };
    let test_factor_options = MulOptions {
        side: Side::Left,
        factor_type: factor_2.clone(),
        base_options: test_base_options,
    };

    let mut id_update_time = 0;
    let mut lu_update_time = 0;

    match update_type {
        BatchUpdateType::Single(_lu_batch) => panic!("Only implemented for multi batch updates"),
        BatchUpdateType::Multi(rsrs_factors) => {
            let (id_time, lu_time) =
                rsrs_factors.apply_level(sketch, &sketch_factor_options, false, level_it);
            id_update_time += id_time;
            lu_update_time += lu_time;

            let (id_time, lu_time) =
                rsrs_factors.apply_level(test, &test_factor_options, false, level_it);
            id_update_time += id_time;
            lu_update_time += lu_time;
        }
    }

    (id_update_time, lu_update_time)
}
