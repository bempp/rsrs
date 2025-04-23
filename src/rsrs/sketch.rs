use super::rsrs_factors::{
    DiagBox, CommutativeFactors, CommutativeFactorsOperations, FactorOptions, FactorType, MulType, RsrsFactors,
    RsrsFactorsOps, RsrsSide,
};
use crate::utils::{
    data_ins_ext::{ExtInsType, Extraction, MatrixExtraction},
    least_squares_and_null::right_least_squares,
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
use std::{cell::RefCell, time::Instant};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

pub enum UpdateType<'a, Item: RlstScalar> {
    Lu(&'a CommutativeFactors<Item>),
    Id(&'a CommutativeFactors<Item>),
    Both(&'a RsrsFactors<Item>),
}

pub enum BatchUpdateType<'a, Item: RlstScalar> {
    Single(&'a CommutativeFactors<Item>),
    Multi(&'a RsrsFactors<Item>),
}

pub struct BoxesData<Item: RlstScalar> {
    pub sketch: DynamicArray<Item, 2>,
    pub test: DynamicArray<Item, 2>,
    pub dim: usize,
    pub num_samples: usize,
    pub trans: bool,
}

pub struct FullBoxesData<Item: RlstScalar> {
    pub y_data: BoxesData<Item>,
    pub z_data: BoxesData<Item>,
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

pub trait SketchOps {
    type Item: RlstScalar;
    fn new<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>,
    >(
        arr: &Array<Self::Item, ArrayImpl, 2>,
        trans: bool,
    ) -> Self;
    fn add_samples(
        &mut self,
        extra_num_samples: usize,
        arr: &DynamicArray<Self::Item, 2>,
        _seed: u64,
    ) -> u128;
    fn update_samples(
        &mut self,
        update_start: usize,
        samples_to_update: usize,
        level: usize,
        update_type: &UpdateType<Self::Item>,
    ) -> (u128, u128);
    fn get_sketch_box(
        &self,
        rows: Vec<usize>,
        cols: Vec<usize>,
        active_samples: usize,
        tol_lstq: <Self::Item as RlstScalar>::Real,
    ) -> DynamicArray<Self::Item, 2>;
    fn extract_diag_boxes(
        &mut self,
        ind_r: Vec<Vec<usize>>,
        ind_s: Vec<Vec<usize>>,
        active_samples: usize,
        tol_lstq: <Self::Item as RlstScalar>::Real,
        rsrs_factors: &mut RsrsFactors<Self::Item>,
    );
}

impl<T: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + MatrixLu>
    SketchOps for BoxesData<T>
where
    StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>,
    LuDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixLuDecomposition<Item = T>,
{
    type Item = T;

    fn new<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>,
    >(
        arr: &Array<Self::Item, ArrayImpl, 2>,
        trans: bool,
    ) -> Self {
        let test: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
        let sketch: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
        let dim = arr.shape()[0];
        Self {
            sketch,
            test,
            dim,
            num_samples: 0,
            trans,
        }
    }

    fn add_samples(
        &mut self,
        extra_num_samples: usize,
        arr: &DynamicArray<Self::Item, 2>,
        _seed: u64,
    ) -> u128 {
        let sampling_start: Instant = Instant::now();
        let test_shape = self.test.shape();
        let total_cols = test_shape[1] + extra_num_samples;

        self.test.resize_in_place([self.dim, total_cols]);
        self.sketch.resize_in_place([self.dim, total_cols]);

        let chunk_size = 30;
        if extra_num_samples > chunk_size {
            let shapes: Vec<_> = (0..extra_num_samples)
                .step_by(chunk_size)
                .map(|start| {
                    let end = (start + chunk_size).min(extra_num_samples);
                    let width = end - start;
                    let shape = [self.dim, width];
                    shape
                })
                .collect();

            let start = Instant::now();
            let chunks: Vec<_> = shapes
                .par_iter()
                .map(|&shape| {
                    println!("Chunking shape: {:?}", shape);
                    let mut chunk_test = rlst_dynamic_array2!(Self::Item, shape);
                    let mut chunk_sketch = rlst_dynamic_array2!(Self::Item, shape);

                    with_thread_rng(|rng| {
                        chunk_test.fill_from_standard_normal(rng);
                    });

                    if self.trans {
                        chunk_sketch.r_mut().mult_into(
                            TransMode::Trans,
                            TransMode::NoTrans,
                            num::One::one(),
                            arr.r(), //TODO: Do this for the conjugate
                            chunk_test.r(),
                            num::Zero::zero(),
                        );
                    } else {
                        chunk_sketch
                            .r_mut()
                            .simple_mult_into(arr.r(), chunk_test.r());
                    }
                    (chunk_test, chunk_sketch)
                })
                .collect();

            let duration = start.elapsed();
            println!("Chunking time: {:?}", duration);

            use std::sync::Mutex;
            let test_mutex = Mutex::new(&mut self.test);
            let sketch_mutex = Mutex::new(&mut self.sketch);

            let col_start = AtomicUsize::new(0);
            let start = Instant::now();
            chunks
                .into_par_iter()
                .for_each(|(chunk_test, chunk_sketch)| {
                    let current_col_start =
                        col_start.fetch_add(chunk_sketch.shape()[1], Ordering::SeqCst);
                    let offset = [0, test_shape[1] + current_col_start];
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
        } else {
            println!("Simple chunk: {:?}", [self.dim, extra_num_samples]);
            let mut sub_test = self
                .test
                .r_mut()
                .into_subview([0, test_shape[1]], [self.dim, extra_num_samples]);
            let mut sub_sketch = self
                .sketch
                .r_mut()
                .into_subview([0, test_shape[1]], [self.dim, extra_num_samples]);

            with_thread_rng(|rng| {
                sub_test.fill_from_standard_normal(rng);
            });

            if !self.trans {
                sub_sketch.r_mut().simple_mult_into(arr.r(), sub_test.r());
            } else {
                sub_sketch.r_mut().mult_into(
                    TransMode::Trans,
                    TransMode::NoTrans,
                    num::One::one(),
                    arr.r(),
                    sub_test.r(),
                    num::Zero::zero(),
                );
            }
        }
        let duration = sampling_start.elapsed();

        self.num_samples = test_shape[1] + extra_num_samples;

        duration.as_millis()
    }

    fn update_samples(
        &mut self,
        update_start: usize,
        samples_to_update: usize,
        level: usize,
        update_type: &UpdateType<Self::Item>,
    ) -> (u128, u128) {
        let (mut sub_test, mut sub_sketch) = (
            self.test
                .r_mut()
                .into_subview([0, update_start], [self.dim, samples_to_update]),
            self.sketch
                .r_mut()
                .into_subview([0, update_start], [self.dim, samples_to_update]),
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

    fn get_sketch_box(
        &self,
        rows: Vec<usize>,
        cols: Vec<usize>,
        active_samples: usize,
        tol_lstq: <Self::Item as RlstScalar>::Real,
    ) -> DynamicArray<Self::Item, 2>
    where
        LuDecomposition<Self::Item, BaseArray<Self::Item, VectorContainer<Self::Item>, 2>>:
            MatrixLuDecomposition<Item = Self::Item>,
    {
        let (mut sub_test, mut sub_sketch) = (
            self.test
                .r()
                .into_subview([0, 0], [self.dim, active_samples]),
            self.sketch
                .r()
                .into_subview([0, 0], [self.dim, active_samples]),
        );

        let sketch_r: DynamicArray<Self::Item, 2> =
            <Extraction<Self::Item> as MatrixExtraction>::new(
                &mut sub_sketch,
                ExtInsType::Axis(rows, 0, false),
            )
            .unwrap()
            .ext;
        let test_c: DynamicArray<Self::Item, 2> =
            <Extraction<Self::Item> as MatrixExtraction>::new(
                &mut sub_test,
                ExtInsType::Axis(cols, 0, false),
            )
            .unwrap()
            .ext;

        right_least_squares(&test_c, &sketch_r, tol_lstq)
    }

    fn extract_diag_boxes(
        &mut self,
        ind_r: Vec<Vec<usize>>,
        ind_s: Vec<Vec<usize>>,
        active_samples: usize,
        tol_lstq: <Self::Item as RlstScalar>::Real,
        rsrs_factors: &mut RsrsFactors<Self::Item>,
    ) {
        let rows: Vec<usize> = (0..self.dim).collect();
        let mut acc_ind_s = Vec::new();
        let mut acc_ind_r = Vec::new();

        for inds in ind_s.iter() {
            acc_ind_s.extend_from_slice(inds);
        }

        for inds in ind_r.iter() {
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

        rsrs_factors.perm_factor.col_indices = cols;
        rsrs_factors.perm_factor.row_indices = rows;

        for inds in ind_r.iter() {
            let dbox = self.get_sketch_box(inds.clone(), inds.clone(), active_samples, tol_lstq);
            let diag_box = DiagBox {
                dbox,
                inv_dbox: empty_array(),
                inds: inds.to_vec(),
            };
            rsrs_factors.diag_box_factor.push(diag_box);
        }

        let dbox = self.get_sketch_box(
            acc_ind_s.clone(),
            acc_ind_s.clone(),
            active_samples,
            tol_lstq,
        );
        let diag_box = DiagBox {
            dbox,
            inv_dbox: empty_array(),
            inds: acc_ind_s.to_vec(),
        };
        rsrs_factors.diag_box_factor.push(diag_box);
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
{
    let start = Instant::now();
    let sketch_factor_options = FactorOptions { inv: true, trans };
    let test_factor_options = FactorOptions { inv: false, trans };

    let sketch_mul_type = MulType {
        side: RsrsSide::Left,
        factor_type: factor_1.clone(),
    };
    let test_mul_type = MulType {
        side: RsrsSide::Left,
        factor_type: factor_2.clone(),
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
{
    let start = Instant::now();
    let sketch_factor_options = FactorOptions { inv: true, trans };
    let test_factor_options = FactorOptions { inv: false, trans };

    let sketch_mul_type = MulType {
        side: RsrsSide::Left,
        factor_type: factor_1.clone(),
    };
    let test_mul_type = MulType {
        side: RsrsSide::Left,
        factor_type: factor_2.clone(),
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
