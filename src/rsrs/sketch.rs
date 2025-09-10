use super::rsrs_factors::{
    CommutativeFactors, CommutativeFactorsOperations, FactorType, MulOptions, RsrsFactors,
    RsrsFactorsImpl,
};
use mpi::traits::Communicator;
use mpi::traits::Equivalence;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::dense::linalg::lu::MatrixLu;
use rlst::operator::ConcreteElementContainer;
pub use rlst::{
    dense::{array::empty_array, tools::RandScalar},
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{cell::RefCell, time::Instant};
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
    pub trans: bool,
}

pub struct FullBoxesData<Item: RlstScalar> {
    pub y_data: SketchData<Item>,
    pub z_data: SketchData<Item>,
    pub dim: usize,
    pub active_samples: usize,
    pub hermitian: bool,
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
    ) {
        other.r_mut().slice(0, offset).fill_from(x.view());
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
    ) {
        other
            .r_mut()
            .slice(0, offset)
            .fill_from(x.view().local().r());
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

thread_local! {
    static THREAD_RNG: RefCell<ChaCha8Rng> = RefCell::new(init_rng());
}

fn init_rng() -> ChaCha8Rng {
    let time_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seed = (time_seed as u64).wrapping_mul(0x9E3779B97F4A7C15); // or add thread ID if needed
    ChaCha8Rng::seed_from_u64(seed)
}

pub fn with_thread_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut ChaCha8Rng) -> R,
{
    THREAD_RNG.with(|rng_cell| {
        let mut rng = rng_cell.borrow_mut();
        f(&mut rng)
    })
}

fn resize_rows<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + RawAccessMut<Item = Item> + Shape<2>,
>(
    arr: &Array<Item, ArrayImpl, 2>,
    new_shape: [usize; 2],
) -> DynamicArray<Item, 2> {
    let mut new_arr = rlst_dynamic_array2!(Item, new_shape);
    new_arr
        .r_mut()
        .into_subview([0, 0], arr.shape())
        .fill_from(arr.r());

    new_arr
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Stabilise {
    True(f64), //TODO: Change to Item
    False,
}

impl<
        Item: RlstScalar
            + RandScalar
            + MatrixId
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
{
    pub fn new(dim: usize, trans: bool) -> Self {
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

    pub fn add_samples<
        Space: SamplingSpace<F = Item>,
        OpImpl: AsApply<Domain = Space, Range = Space>,
    >(
        &mut self,
        extra_num_samples: usize,
        operator: Operator<OpImpl>,
        stabilise: &Stabilise,
        _seed: u64,
    ) -> u128 {
        let sampling_start: Instant = Instant::now();
        let test_shape = self.test.shape();
        let total_samples = test_shape[0] + extra_num_samples;
        let trans_mode = if self.trans {
            TransMode::ConjTrans
        } else {
            TransMode::NoTrans
        };

        self.test = resize_rows(&self.test, [total_samples, self.dim]);
        self.sketch = resize_rows(&self.sketch, [total_samples, self.dim]);

        let mut sample_generation = std::time::Duration::ZERO;
        let mut multiplication = std::time::Duration::ZERO;
        let mut filling = std::time::Duration::ZERO;
        (0..extra_num_samples).for_each(|row| {
            let start: Instant = Instant::now();
            let offset = test_shape[0] + row;
            let mut chunk_test_vec = SamplingSpace::zero(operator.r().domain());

            with_thread_rng(|rng| {
                operator.domain().sampling(
                    &mut chunk_test_vec,
                    rng,
                    SampleType::RealStandardNormal,
                );
            });

            sample_generation += start.elapsed();

            let start: Instant = Instant::now();

            let chunk_sketch_vec = match stabilise {
                Stabilise::True(alpha) => {
                    let mut chunk_sketch_vec_stab = operator.domain().clone_vec(&chunk_test_vec);
                    chunk_sketch_vec_stab.scale_inplace(Item::from(*alpha).unwrap());
                    chunk_sketch_vec_stab
                        .sum_inplace(operator.apply(chunk_test_vec.r(), trans_mode));
                    chunk_sketch_vec_stab
                }
                Stabilise::False => operator.apply(chunk_test_vec.r(), trans_mode),
            };

            multiplication += start.elapsed();

            let start: Instant = Instant::now();
            operator
                .domain()
                .fill_array(&chunk_test_vec, &mut self.test, offset);
            operator
                .domain()
                .fill_array(&chunk_sketch_vec, &mut self.sketch, offset);
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
        let duration = sampling_start.elapsed();

        self.num_samples = test_shape[0] + extra_num_samples; //TODO: Change this to total_samples

        duration.as_millis()
    }

    pub fn update_samples(
        &mut self,
        update_start: usize,
        samples_to_update: usize,
        level: usize,
        update_type: &UpdateType<Item>,
    ) -> (u128, u128) {
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

        let (factor_1, factor_2) = if !self.trans {
            (FactorType::F, FactorType::S)
        } else {
            (FactorType::S, FactorType::F)
        };

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
                );
            }
            UpdateType::Both(rsrs_factors) => {
                (0..level).for_each(|level_it| {
                    id_time += update_id_level(
                        &mut sub_sketch,
                        &mut sub_test,
                        level_it,
                        BatchUpdateType::Multi(rsrs_factors),
                        &factor_1,
                        &factor_2,
                        self.trans,
                    );

                    lu_time += update_lu_level(
                        &mut sub_sketch,
                        &mut sub_test,
                        level_it,
                        BatchUpdateType::Multi(rsrs_factors),
                        &factor_1,
                        &factor_2,
                        self.trans,
                    );
                });
            }
        }

        (id_time, lu_time)
    }
}

pub fn update_id_level<
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + MatrixLu + MatrixQr,
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
    trans: bool,
) -> u128
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let start = Instant::now();
    let sketch_factor_options = MulOptions {
        inv: true,
        trans,
        side: Side::Left,
        factor_type: factor_1.clone(),
        t_trans: true,
    };
    let test_factor_options = MulOptions {
        inv: false,
        trans,
        side: Side::Left,
        factor_type: factor_2.clone(),
        t_trans: true,
    };

    match update_type {
        BatchUpdateType::Single(id_batch) => {
            id_batch.mul(sketch, &sketch_factor_options);
            id_batch.mul(test, &test_factor_options);
        }
        BatchUpdateType::Multi(rsrs_factors) => {
            rsrs_factors.apply_id_level(sketch, &sketch_factor_options, false, level_it);
            rsrs_factors.apply_id_level(test, &test_factor_options, false, level_it);
        }
    }

    start.elapsed().as_millis()
}

pub fn update_lu_level<
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + MatrixLu + MatrixQr,
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
    trans: bool,
) -> u128
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let start = Instant::now();
    let sketch_factor_options = MulOptions {
        inv: true,
        trans,
        side: Side::Left,
        factor_type: factor_1.clone(),
        t_trans: true,
    };
    let test_factor_options = MulOptions {
        inv: false,
        trans,
        side: Side::Left,
        factor_type: factor_2.clone(),
        t_trans: true,
    };

    match update_type {
        BatchUpdateType::Single(lu_batch) => {
            lu_batch.mul(sketch, &sketch_factor_options);
            lu_batch.mul(test, &test_factor_options);
        }
        BatchUpdateType::Multi(rsrs_factors) => {
            rsrs_factors.apply_lu_level(sketch, &sketch_factor_options, false, level_it);
            rsrs_factors.apply_lu_level(test, &test_factor_options, false, level_it);
        }
    }

    start.elapsed().as_millis()
}
