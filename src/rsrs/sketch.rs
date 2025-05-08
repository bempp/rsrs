use super::rsrs_factors::{
    CommutativeFactors, CommutativeFactorsOperations, FactorMulType, FactorOptions, FactorType,
    RsrsFactors, RsrsFactorsOps,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::prelude::*;
use rlst::dense::linalg::lu::MatrixLu;
pub use rlst::{
    dense::{array::empty_array, tools::RandScalar},
    prelude::*,
};
use std::sync::Mutex;
use std::{cell::RefCell, time::Instant};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

//use rlst::operator::interface::DistributedArrayVectorSpace;

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

impl<Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + MatrixLu>
    SketchData<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
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
        OpImpl: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
    >(
        &mut self,
        extra_num_samples: usize,
        operator: &Operator<OpImpl>,
        _seed: u64,
    ) -> u128 {
        let sampling_start: Instant = Instant::now();
        let test_shape = self.test.shape();
        let total_samples = test_shape[0] + extra_num_samples;

        self.test = resize_rows(&self.test, [total_samples, self.dim]);
        self.sketch = resize_rows(&self.sketch, [total_samples, self.dim]);

        let chunk_size = 30;
        let shapes: Vec<_> = (0..extra_num_samples)
            .step_by(chunk_size)
            .map(|start| {
                let end = (start + chunk_size).min(extra_num_samples);
                let width = end - start;
                let shape = [width, self.dim];
                shape
            })
            .collect();

        let start = Instant::now();
        let chunks: Vec<_> = shapes
            .into_iter()
            .map(|shape| {
                println!("Chunking shape: {:?}", shape);
                let mut chunk_test = rlst_dynamic_array2!(Item, shape);
                let mut chunk_sketch = rlst_dynamic_array2!(Item, shape);

                (0..shape[0]).for_each(|row| {
                    let mut chunk_test_vec = ArrayVectorSpace::zero(operator.domain());
                    with_thread_rng(|rng| {
                        chunk_test_vec.view_mut().fill_from_standard_normal(rng);
                    });
                    let chunk_sketch_vec = operator.apply(chunk_test_vec.r());
                    chunk_test
                        .r_mut()
                        .slice(0, row)
                        .fill_from(chunk_test_vec.view());
                    chunk_sketch
                        .r_mut()
                        .slice(0, row)
                        .fill_from(chunk_sketch_vec.view());
                });

                (chunk_test, chunk_sketch)
            })
            .collect();

        let duration = start.elapsed();
        println!("Chunking time: {:?}", duration);

        let test_mutex = Mutex::new(&mut self.test);
        let sketch_mutex = Mutex::new(&mut self.sketch);

        let col_start = AtomicUsize::new(0);
        let start = Instant::now();
        chunks
            .into_par_iter()
            .for_each(|(chunk_test, chunk_sketch)| {
                let current_col_start =
                    col_start.fetch_add(chunk_sketch.shape()[0], Ordering::SeqCst);
                let offset = [test_shape[0] + current_col_start, 0];
                {
                    let mut test_guard = test_mutex.lock().unwrap();
                    test_guard
                        .r_mut()
                        .into_subview(offset, chunk_test.shape())
                        .fill_from(chunk_test.r());
                }
                {
                    let mut sketch_guard = sketch_mutex.lock().unwrap();
                    sketch_guard
                        .r_mut()
                        .into_subview(offset, chunk_sketch.shape())
                        .fill_from(chunk_sketch.r());
                }
            });
        let duration = start.elapsed();
        println!("Filling time: {:?}\n", duration);

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
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + MatrixLu,
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
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + MatrixLu,
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
