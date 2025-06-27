use super::rsrs_factors::{
    CommutativeFactors, CommutativeFactorsOperations, FactorMulType, FactorOptions, FactorType,
    RsrsFactors, RsrsFactorsImpl,
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

/*pub trait SamplingSpaceElementImpl: ElementImpl{
    fn get_data<'a>(&'a self) -> &'a [Self::F];
    fn get_data_mut<'a>(&'a mut self) -> &'a mut [Self::F];
}

impl <Item: RlstScalar> SamplingSpaceElementImpl for ArrayVectorSpaceElement<Item> {
    fn get_data<'a>(&'a self) -> &'a [Self::F]{
        self.view().data()
    }

    fn get_data_mut(&mut self) -> &mut[Self::F]{
        self.view().data_mut()
    }
}*/
pub trait SamplingSpace: LinearSpace {
    fn sampling<R: Rng>(&self, x: &mut Element<ConcreteElementContainer<Self::E>>, rng: &mut R);

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
}

impl<Item: RlstScalar + RandScalar> SamplingSpace for ArrayVectorSpace<Item>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    fn sampling<R: Rng>(&self, x: &mut Element<ConcreteElementContainer<Self::E>>, rng: &mut R) {
        x.view_mut().fill_from_equally_distributed_real(rng);
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
}

impl<C: Communicator, Item: RlstScalar + RandScalar + Equivalence> SamplingSpace
    for DistributedArrayVectorSpace<'_, C, Item>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    fn sampling<R: Rng>(&self, x: &mut Element<ConcreteElementContainer<Self::E>>, rng: &mut R) {
        x.view_mut()
            .local_mut()
            .fill_from_equally_distributed_real(rng);
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
        
        /*println!("gather to rank root: {:?}, {:?}", x.view().local().len(), self.index_layout().number_of_global_indices());
        x.view().gather_to_rank_root(other
            .r_mut()
            .slice(0, offset));
        println!("Gathered");*/
        other
            .r_mut()
            .slice(0, offset)
            .fill_from(x.view().local().r());
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
        operator: &OpImpl,
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
            let mut chunk_test_vec = SamplingSpace::zero(operator.domain());

            with_thread_rng(|rng| {
                operator.domain().sampling(&mut chunk_test_vec, rng);
            });

            sample_generation += start.elapsed();

            let start: Instant = Instant::now();
            let chunk_sketch_vec: Element<ConcreteElementContainer<_>> =
                operator.apply(chunk_test_vec.r(), trans_mode);
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
                println!("Sample generation: {:?}", sample_generation);
                println!(
                    "Multiplication: {:?} ({:?} per sample) -> 30 samples",
                    multiplication,
                    multiplication / 30
                );
                println!("Filling: {:?}", filling);
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
    let sketch_factor_options = FactorOptions { inv: true, trans };
    let test_factor_options = FactorOptions { inv: false, trans };

    let sketch_mul_type = FactorMulType {
        side: Side::Left,
        factor_type: factor_1.clone(),
        right_trans: true,
    };
    let test_mul_type = FactorMulType {
        side: Side::Left,
        factor_type: factor_2.clone(),
        right_trans: true,
    };

    match update_type {
        BatchUpdateType::Single(id_batch) => {
            id_batch.mul(sketch, &sketch_factor_options, &sketch_mul_type);
            id_batch.mul(test, &test_factor_options, &test_mul_type);
        }
        BatchUpdateType::Multi(rsrs_factors) => {
            rsrs_factors.apply_id_level(sketch, &sketch_mul_type, &sketch_factor_options, level_it);
            rsrs_factors.apply_id_level(test, &test_mul_type, &test_factor_options, level_it);
        }
    }

    let id_update_time = start.elapsed().as_millis();
    id_update_time
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
    let sketch_factor_options = FactorOptions { inv: true, trans };
    let test_factor_options = FactorOptions { inv: false, trans };

    let sketch_mul_type = FactorMulType {
        side: Side::Left,
        factor_type: factor_1.clone(),
        right_trans: true,
    };
    let test_mul_type = FactorMulType {
        side: Side::Left,
        factor_type: factor_2.clone(),
        right_trans: true,
    };

    match update_type {
        BatchUpdateType::Single(lu_batch) => {
            lu_batch.mul(sketch, &sketch_factor_options, &sketch_mul_type);
            lu_batch.mul(test, &test_factor_options, &test_mul_type);
        }
        BatchUpdateType::Multi(rsrs_factors) => {
            rsrs_factors.apply_lu_level(
                sketch,
                &sketch_mul_type,
                &sketch_factor_options,
                false,
                level_it,
            );
            rsrs_factors.apply_lu_level(
                test,
                &test_mul_type,
                &test_factor_options,
                false,
                level_it,
            );
        }
    }

    let lu_update_time = start.elapsed().as_millis();
    lu_update_time
}
