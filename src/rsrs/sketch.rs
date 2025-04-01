use super::rsrs_factors::{
    DiagBox, FactorOptions, FactorType, IdFactor, IdFactorOperations, LuFactor, LuFactorOperations,
    RsrsFactors, RsrsSide,
};
use crate::utils::data_ins_ext::{ExtInsType, Extraction, MatrixExtraction};
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
pub use rlst::{
    dense::{array::empty_array, tools::RandScalar},
    prelude::*,
};
use std::sync::Mutex;
use std::time::Instant;

pub struct BoxesData<Item: RlstScalar> {
    pub sketch: DynamicArray<Item, 2>,
    pub test: DynamicArray<Item, 2>,
    pub dim: usize,
    pub num_samples: usize,
    pub trans: bool,
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
        rsrs_factors: &RsrsFactors<Self::Item>,
        silent: bool,
        _seed: u64,
    ) -> (u128, u128, u128);
    fn get_sketch_box(
        &mut self,
        rows: Vec<usize>,
        cols: Vec<usize>,
        tol_lstq: <Self::Item as RlstScalar>::Real,
    ) -> DynamicArray<Self::Item, 2>;
    fn extract_diag_boxes(
        &mut self,
        ind_r: Vec<Vec<usize>>,
        ind_s: Vec<Vec<usize>>,
        tol_lstq: <Self::Item as RlstScalar>::Real,
        rsrs_factors: &mut RsrsFactors<Self::Item>,
    );
}

impl<T: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse> SketchOps
    for BoxesData<T>
where
    StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>,
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
        rsrs_factors: &RsrsFactors<Self::Item>,
        silent: bool,
        _seed: u64,
    ) -> (u128, u128, u128) {
        if extra_num_samples < 300 {
            add_samples_single_node(self, extra_num_samples, arr, rsrs_factors, silent, _seed)
        } else {
            add_samples_multi_node(self, extra_num_samples, arr, rsrs_factors, silent, _seed)
        }
    }

    fn get_sketch_box(
        &mut self,
        rows: Vec<usize>,
        cols: Vec<usize>,
        tol_lstq: <Self::Item as RlstScalar>::Real,
    ) -> DynamicArray<Self::Item, 2> {
        let sketch_r: DynamicArray<Self::Item, 2> =
            <Extraction<Self::Item> as MatrixExtraction>::new(
                &mut self.sketch,
                ExtInsType::Axis(rows, 0, false),
            )
            .unwrap()
            .ext;
        let mut test_c: DynamicArray<Self::Item, 2> =
            <Extraction<Self::Item> as MatrixExtraction>::new(
                &mut self.test,
                ExtInsType::Axis(cols, 0, false),
            )
            .unwrap()
            .ext;

        //Solve right least squares problem
        let shape = test_c.shape(); // Get shape directly
        let mut pinv = rlst_dynamic_array2!(Self::Item, [shape[1], shape[0]]); // Avoid extra allocation
        test_c
            .r_mut()
            .into_pseudo_inverse_alloc(pinv.r_mut(), tol_lstq)
            .unwrap();
        let mut sol: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
        sol.r_mut().simple_mult_into_resize(sketch_r.r(), pinv.r());
        sol
    }

    fn extract_diag_boxes(
        &mut self,
        ind_r: Vec<Vec<usize>>,
        ind_s: Vec<Vec<usize>>,
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
            let dbox = self.get_sketch_box(inds.clone(), inds.clone(), tol_lstq);
            let diag_box = DiagBox {
                dbox,
                inv_dbox: empty_array(),
                inds: inds.to_vec(),
            };
            rsrs_factors.diag_box_factor.push(diag_box);
        }

        let dbox = self.get_sketch_box(acc_ind_s.clone(), acc_ind_s.clone(), tol_lstq);
        let diag_box = DiagBox {
            dbox,
            inv_dbox: empty_array(),
            inds: acc_ind_s.to_vec(),
        };
        rsrs_factors.diag_box_factor.push(diag_box);
    }
}

fn par_batch_update<
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessMut<2, Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>
        + std::marker::Send,
>(
    lu_batch: &Vec<LuFactor<Item>>,
    sketch: &mut Array<Item, ArrayImpl, 2>,
    test: &mut Array<Item, ArrayImpl, 2>,
    factor_1: &FactorType,
    factor_2: &FactorType,
    trans: bool,
) {
    let sketch = Mutex::new(sketch);
    let test = Mutex::new(test);
    lu_batch.par_iter().for_each(|lu_factor| {
        let mut sketch = sketch.lock().unwrap();
        let mut test = test.lock().unwrap();
        update_sketch_lu(&mut sketch, &mut test, lu_factor, factor_1, factor_2, trans);
    });
}

pub fn update_samples<
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessMut<2, Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>
        + std::marker::Send,
>(
    sketch: &mut Array<Item, ArrayImpl, 2>,
    test: &mut Array<Item, ArrayImpl, 2>,
    rsrs_factors: &RsrsFactors<Item>,
    trans: bool,
) -> (u128, u128) {
    let (factor_1, factor_2) = if !trans {
        (FactorType::F, FactorType::S)
    } else {
        (FactorType::S, FactorType::F)
    };

    let full_cycle_start = Instant::now();
    let mut lu_update_time = 0;
    rsrs_factors
        .id_factors
        .iter()
        .enumerate()
        .for_each(|(level, level_id_factors)| {
            level_id_factors.iter().for_each(|id_factor| {
                update_sketch_id(sketch, test, id_factor, &factor_1, &factor_2, trans);
            });

            let start: Instant = Instant::now();
            let level_lu_batches = &rsrs_factors.lu_factors[level];
            level_lu_batches.iter().for_each(|lu_batch| {
                par_batch_update(lu_batch, sketch, test, &factor_1, &factor_2, trans);
            });
            lu_update_time += start.elapsed().as_millis();
        });
    let tot_update_duration = full_cycle_start.elapsed().as_millis();
    let id_update_time = tot_update_duration - lu_update_time;
    (id_update_time, lu_update_time)
}

pub fn update_sketch_id<
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Shape<2>
        + RawAccessMut<Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    sketch: &mut Array<Item, ArrayImplMut, 2>,
    test: &mut Array<Item, ArrayImplMut, 2>,
    factor: &IdFactor<Item>,
    factor_1: &FactorType,
    factor_2: &FactorType,
    trans: bool,
) {
    factor.mul(
        sketch,
        &FactorOptions { inv: true, trans },
        factor_1,
        &RsrsSide::Left,
    );
    factor.mul(
        test,
        &FactorOptions { inv: false, trans },
        factor_2,
        &RsrsSide::Left,
    );
}

pub fn update_sketch_lu<
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Shape<2>
        + RawAccessMut<Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    sketch: &mut Array<Item, ArrayImplMut, 2>,
    test: &mut Array<Item, ArrayImplMut, 2>,
    factor: &LuFactor<Item>,
    factor_1: &FactorType,
    factor_2: &FactorType,
    trans: bool,
) {
    factor.mul(
        sketch,
        &FactorOptions { inv: true, trans },
        factor_1,
        &RsrsSide::Left,
    );
    factor.mul(
        test,
        &FactorOptions { inv: false, trans },
        factor_2,
        &RsrsSide::Left,
    );
}

fn add_samples_multi_node<
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse,
>(
    sketch_data: &mut BoxesData<Item>,
    extra_num_samples: usize,
    arr: &DynamicArray<Item, 2>,
    rsrs_factors: &RsrsFactors<Item>,
    silent: bool,
    _seed: u64,
) -> (u128, u128, u128)
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let start: Instant = Instant::now();
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng(); // For testing: ChaCha8Rng::seed_from_u64(0);
    let test_shape: [usize; 2] = sketch_data.test.shape();

    sketch_data
        .test
        .resize_in_place([sketch_data.dim, test_shape[1] + extra_num_samples]);
    sketch_data
        .sketch
        .resize_in_place([sketch_data.dim, test_shape[1] + extra_num_samples]);

    let mut extra_test = sketch_data
        .test
        .r_mut()
        .into_subview([0, test_shape[1]], [sketch_data.dim, extra_num_samples]);
    let mut extra_sketch = sketch_data
        .sketch
        .r_mut()
        .into_subview([0, test_shape[1]], [sketch_data.dim, extra_num_samples]);

    extra_test.fill_from_standard_normal(&mut rng);

    let num_chunks = rayon::current_num_threads();
    let chunk_size = (extra_num_samples + num_chunks -1) / num_chunks;
    //let chunk_size = 31;
    //let num_chunks = (extra_num_samples + chunk_size - 1) / chunk_size;

    let mut sub: Vec<_> = (0..num_chunks)
        .into_iter()
        .map(|chunk_num| {
            let end_offset = (chunk_size * chunk_num).min(extra_num_samples);
            let offset = [0, end_offset];
            let current_chunk_size = chunk_size.min(extra_num_samples - end_offset);
            let shape = [sketch_data.dim, current_chunk_size];
            let mut sub_test = empty_array();
            sub_test.fill_from_resize(extra_test.r().into_subview(offset, shape));
            let mut sub_sketch = empty_array();
            sub_sketch.fill_from_resize(extra_sketch.r().into_subview(offset, shape));
            (sub_test, sub_sketch, offset, shape)
        })
        .collect();

    sub.par_iter_mut()
        .for_each(|(sub_test, sub_sketch, _offset, _shape)| {
            if !sketch_data.trans {
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
        });

    for (sub_test, sub_sketch, offset, shape) in sub {
        extra_sketch
            .r_mut()
            .into_subview(offset, shape)
            .fill_from(sub_sketch.r());
        extra_test
            .r_mut()
            .into_subview(offset, shape)
            .fill_from(sub_test.r());
    }
    let duration = start.elapsed();
    sketch_data.num_samples = test_shape[1] + extra_num_samples;

    if !silent {
        println!("Testing in {} ms", duration.as_millis());
    }

    let (id_update_time, lu_update_time) = update_samples(
        &mut extra_sketch,
        &mut extra_test,
        rsrs_factors,
        sketch_data.trans,
    );

    (duration.as_millis(), id_update_time, lu_update_time)
}

fn add_samples_single_node<
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse,
>(
    sketch_data: &mut BoxesData<Item>,
    extra_num_samples: usize,
    arr: &DynamicArray<Item, 2>,
    rsrs_factors: &RsrsFactors<Item>,
    silent: bool,
    _seed: u64,
) -> (u128, u128, u128)
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let start: Instant = Instant::now();
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng(); // For testing: ChaCha8Rng::seed_from_u64(0);
    let test_shape: [usize; 2] = sketch_data.test.shape();
    sketch_data
        .test
        .resize_in_place([sketch_data.dim, test_shape[1] + extra_num_samples]);
    sketch_data
        .sketch
        .resize_in_place([sketch_data.dim, test_shape[1] + extra_num_samples]);
    let mut sub_test = sketch_data
        .test
        .r_mut()
        .into_subview([0, test_shape[1]], [sketch_data.dim, extra_num_samples]);
    let mut sub_sketch = sketch_data
        .sketch
        .r_mut()
        .into_subview([0, test_shape[1]], [sketch_data.dim, extra_num_samples]);
    sub_test.fill_from_standard_normal(&mut rng);

    if !sketch_data.trans {
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
    let duration = start.elapsed();

    sketch_data.num_samples = test_shape[1] + extra_num_samples;

    if !silent {
        println!("Testing in {} ms", duration.as_millis());
    }

    let (id_update_time, lu_update_time) = update_samples(
        &mut sub_sketch,
        &mut sub_test,
        rsrs_factors,
        sketch_data.trans,
    );

    (duration.as_millis(), id_update_time, lu_update_time)
}
