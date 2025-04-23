use super::{rsrs_cycle::BoxType, sketch::BoxesData};
use crate::utils::{
    data_ins_ext::{matrix_insertion, ExtInsType, Extraction, MatrixExtraction},
    elementary_matrix::{col_ops_no_sub, col_perm, col_subs, row_ops_no_sub, row_perm, row_subs},
    least_squares_and_null::{null_space_by_projection, right_least_squares},
};
use num::One;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rlst::{
    dense::{
        linalg::{interpolative_decomposition::Accuracy, lu::MatrixLu},
        tools::RandScalar,
    },
    prelude::*,
};
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

pub struct FactorOptions {
    /// Inverse operation
    pub inv: bool,
    /// Transpose operation
    pub trans: bool,
}

pub enum OpInfo<T: RlstScalar> {
    DecFact(
        DynamicArray<T, 2>,
        DynamicArray<T, 2>,
        Vec<usize>,
        Vec<usize>,
    ),
    DiagBlocks(Vec<DynamicArray<T, 2>>),
    Perm(Vec<usize>, Vec<usize>),
}

#[derive(PartialEq)]
pub enum RsrsSide {
    Squeeze,
    Left,
    Right,
}

#[derive(Clone)]
pub enum FactorType {
    F,
    S,
}

pub struct IdFactor<T: RlstScalar> {
    data: DynamicArray<T, 2>,
    pub perm: Vec<usize>,
    pub ind_r: Vec<usize>, //row_indices
    pub ind_s: Vec<usize>, //col_indices
    pub ind_f: Vec<usize>,
}

#[derive(Debug, Serialize, Clone)]
pub struct LuTimes {
    pub extraction: u128,
    pub lu: u128,
}

#[derive(Debug, Serialize, Clone)]
pub struct IdTimes {
    pub nullification: u128,
    pub id: u128,
}

pub enum Times {
    Lu(LuTimes),
    Id(IdTimes),
}

pub trait FactorOperations: Sized {
    type Item: RlstScalar;
    fn new(
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        y_data: &BoxesData<Self::Item>,
        z_data: &BoxesData<Self::Item>,
        subs_sample_dim: usize,
        tol: <Self::Item as RlstScalar>::Real,
        rank_par: &BoxType<Real<Self::Item>>,
        hermitian: bool,
    ) -> (Option<Self>, Times)
    where
        StandardNormal: Distribution<Real<Self::Item>>,
        Standard: Distribution<Real<Self::Item>>,
        LuDecomposition<Self::Item, BaseArray<Self::Item, VectorContainer<Self::Item>, 2>>:
            MatrixLuDecomposition<Item = Self::Item>,
        QrDecomposition<Self::Item, BaseArray<Self::Item, VectorContainer<Self::Item>, 2>>:
            MatrixQrDecomposition<Item = Self::Item>;

    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        options: &FactorOptions,
        factor_type: &FactorType,
        operation_type: &RsrsSide,
    );

    fn mul_data<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &Array<Self::Item, ArrayImplMut, 2>,
        options: &FactorOptions,
        factor_type: &FactorType,
        operation_type: &RsrsSide,
    ) -> DynamicArray<Self::Item, 2>;

    fn ins_data<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        source_arr: &DynamicArray<Self::Item, 2>,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        options: &FactorOptions,
        factor_type: &FactorType,
        operation_type: &RsrsSide,
    );
}

type Real<T> = <T as rlst::RlstScalar>::Real;

fn null_sketch_near_field<
    Item: RlstScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + RandScalar + MatrixLu,
>(
    target_inds: &Vec<usize>,
    near_field_inds: &Vec<usize>,
    sketch: &DynamicArray<Item, 2>,
    test: &DynamicArray<Item, 2>,
    subs_sample_dim: usize,
    tol_null: <Item as RlstScalar>::Real,
) -> DynamicArray<Item, 2>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let row_num = test.shape()[0];
    let sub_test = test.r().into_subview([0, 0], [row_num, subs_sample_dim]);
    let sub_sketch = sketch.r().into_subview([0, 0], [row_num, subs_sample_dim]);

    let test_n = <Extraction<Item> as MatrixExtraction>::new(
        &sub_test,
        ExtInsType::Axis(near_field_inds.clone(), 0, false),
    )
    .unwrap()
    .ext;
    let sketch_t = <Extraction<Item> as MatrixExtraction>::new(
        &sub_sketch,
        ExtInsType::Axis(target_inds.clone(), 0, false),
    )
    .unwrap()
    .ext;

    let res = null_space_by_projection(&test_n, &sketch_t, tol_null);
    res

}

fn null_near_field<
    Item: RlstScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + RandScalar + MatrixLu,
>(
    target_inds: &Vec<usize>,
    near_field_inds: &Vec<usize>,
    y_data: &BoxesData<Item>,
    z_data: &BoxesData<Item>,
    subs_sample_dim: usize,
    tol_null: Real<Item>,
    hermitian: bool,
) -> DynamicArray<Item, 2>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let far_field_sketch = if hermitian {
        null_sketch_near_field(
            target_inds,
            near_field_inds,
            &y_data.sketch,
            &y_data.test,
            subs_sample_dim,
            tol_null,
        )
    } else {
        let null_y_sketch = null_sketch_near_field(
            target_inds,
            near_field_inds,
            &y_data.sketch,
            &y_data.test,
            subs_sample_dim,
            tol_null,
        );
        let null_z_sketch = null_sketch_near_field(
            target_inds,
            near_field_inds,
            &z_data.sketch,
            &z_data.test,
            subs_sample_dim,
            tol_null,
        );
        let mut sketch_sum = empty_array();
        sketch_sum.fill_from_resize(null_y_sketch.r() + null_z_sketch.r());
        sketch_sum
    };

    far_field_sketch
}

impl<Item: RlstScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + RandScalar + MatrixLu>
    FactorOperations for IdFactor<Item>
{
    type Item = Item;

    fn new(
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        y_data: &BoxesData<Self::Item>,
        z_data: &BoxesData<Self::Item>,
        subs_sample_dim: usize,
        tol_null: <Self::Item as RlstScalar>::Real,
        rank_par: &BoxType<Real<Self::Item>>,
        hermitian: bool,
    ) -> (Option<Self>, Times)
    where
        StandardNormal: Distribution<Item::Real>,
        Standard: Distribution<Item::Real>,
        LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
            MatrixLuDecomposition<Item = Item>,
    {
        let start: Instant = Instant::now();
        let far_field_sketch = null_near_field(
            &target_inds,
            &near_field_inds,
            y_data,
            z_data,
            subs_sample_dim,
            tol_null,
            hermitian,
        );
        let nullification_time: Duration = start.elapsed();
        let start: Instant = Instant::now();
        let max_rank: usize = *far_field_sketch.shape().iter().min().unwrap();
        let id_sketch = match rank_par {
            BoxType::Full(tol) => far_field_sketch.into_id_alloc(Accuracy::Tol(*tol)).unwrap(),
            BoxType::Merged(rank) => far_field_sketch
                .into_id_alloc(Accuracy::FixedRank(*rank))
                .unwrap(),
        };

        let k: usize = id_sketch.rank;
        let mut ind_r = Vec::new();
        let mut ind_s = Vec::new();
        let id_time = start.elapsed();

        let id_times = IdTimes {
            nullification: nullification_time.as_millis(),
            id: id_time.as_millis(),
        };

        let times = Times::Id(id_times);

        if id_sketch.rank < max_rank {
            let aux_indices = target_inds.clone();

            for (id, &elem) in id_sketch.perm.iter().enumerate() {
                let val = aux_indices[elem];
                target_inds[id] = val;
                near_field_inds[id] = val;
            }

            ind_r.extend_from_slice(&target_inds[k..]);
            ind_s.extend_from_slice(&target_inds[..k]);
            let ind_f = get_far_indices(y_data.dim, near_field_inds.to_vec());

            (
                Some(Self {
                    data: id_sketch.id_mat,
                    perm: id_sketch.perm,
                    ind_r,
                    ind_s,
                    ind_f,
                }),
                times,
            )
        } else {
            (None, times)
        }
    }

    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        factor_options: &FactorOptions,
        factor_type: &FactorType,
        operation_type: &RsrsSide,
    ) {
        let target_block = self.mul_data(target_arr, factor_options, factor_type, operation_type);

        let t_arr_mutex = std::sync::Mutex::new(target_arr);
        self.ins_data(
            &target_block,
            &mut *t_arr_mutex.lock().unwrap(),
            factor_options,
            factor_type,
            operation_type,
        );
    }

    fn mul_data<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &Array<Self::Item, ArrayImplMut, 2>,
        options: &FactorOptions,
        factor_type: &FactorType,
        operation_type: &RsrsSide,
    ) -> DynamicArray<Self::Item, 2> {
        let mut beta: Self::Item = <Self::Item as One>::one();
        let mut trans = options.trans;
        if options.inv {
            beta = -<Self::Item as One>::one();
        }

        match factor_type {
            FactorType::F => {}
            FactorType::S => {
                trans = !trans;
            }
        }

        if *operation_type == RsrsSide::Left {
            row_ops_no_sub(
                self.ind_s.clone(),
                self.ind_r.clone(),
                &self.data,
                target_arr,
                beta,
                trans,
            )
        } else if *operation_type == RsrsSide::Right {
            col_ops_no_sub(
                self.ind_s.clone(),
                self.ind_r.clone(),
                &self.data,
                target_arr,
                beta,
                trans,
            )
        } else {
            empty_array()
        }
    }

    fn ins_data<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        source_arr: &DynamicArray<Self::Item, 2>,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        options: &FactorOptions,
        factor_type: &FactorType,
        operation_type: &RsrsSide,
    ) {
        let mut trans = options.trans;

        match factor_type {
            FactorType::F => {}
            FactorType::S => {
                trans = !trans;
            }
        }

        if *operation_type == RsrsSide::Left {
            row_subs(
                self.ind_s.clone(),
                self.ind_r.clone(),
                source_arr,
                target_arr,
                trans,
            );
        } else if *operation_type == RsrsSide::Right {
            col_subs(
                self.ind_s.clone(),
                self.ind_r.clone(),
                source_arr,
                target_arr,
                trans,
            );
        }
    }
}

pub struct LuFactor<T: RlstScalar> {
    l_arr: DynamicArray<T, 2>,
    u_arr: DynamicArray<T, 2>,
    hermitian: bool,
    pub ind_r: Vec<usize>, //cols
    pub ind_t: Vec<usize>, //rows
}

fn near_box_extraction<Item: RlstScalar + MatrixPseudoInverse + MatrixLu>(
    ind_r: &[usize],
    near_field_inds: &[usize],
    sketch_data: &BoxesData<Item>,
    subs_sample_dim: usize,
    tol_lstq: <Item as RlstScalar>::Real,
    r_numbering: &Vec<usize>,
    t_numbering: &Vec<usize>,
) -> (
    DynamicArray<Item, 2>,
    DynamicArray<Item, 2>,
    (Duration, Duration),
)
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let row_num = sketch_data.test.shape()[0];
    let test_subview = sketch_data
        .test
        .r()
        .into_subview([0, 0], [row_num, subs_sample_dim]);
    let sketch_subview = sketch_data
        .sketch
        .r()
        .into_subview([0, 0], [row_num, subs_sample_dim]);
    let start = Instant::now();
    let sketch_r: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        &sketch_subview,
        ExtInsType::Axis(ind_r.to_vec(), 0, false),
    )
    .unwrap()
    .ext;
    let mut test_n: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        &test_subview,
        ExtInsType::Axis(near_field_inds.to_vec(), 0, false),
    )
    .unwrap()
    .ext;
    let mut lu_io_time = start.elapsed();
    let start = Instant::now();
    let mut near_box = right_least_squares(&mut test_n, &sketch_r, tol_lstq);
    let lu_b_ext_time = start.elapsed();
    let data_r: DynamicArray<Item, 2>;
    let data_n: DynamicArray<Item, 2>;
    let start = Instant::now();
    if !sketch_data.trans {
        data_r = <Extraction<Item> as MatrixExtraction>::new(
            &mut near_box,
            ExtInsType::Axis(r_numbering.to_vec(), 1, false),
        )
        .unwrap()
        .ext;
        data_n = <Extraction<Item> as MatrixExtraction>::new(
            &mut near_box,
            ExtInsType::Axis(t_numbering.to_vec(), 1, false),
        )
        .unwrap()
        .ext;
    } else {
        data_r = <Extraction<Item> as MatrixExtraction>::new(
            &mut near_box,
            ExtInsType::Axis(r_numbering.to_vec(), 1, true),
        )
        .unwrap()
        .ext;
        data_n = <Extraction<Item> as MatrixExtraction>::new(
            &mut near_box,
            ExtInsType::Axis(t_numbering.to_vec(), 1, true),
        )
        .unwrap()
        .ext;
    }
    let lu_small_io_time = start.elapsed();
    lu_io_time += lu_small_io_time;
    (data_r, data_n, (lu_io_time, lu_b_ext_time))
}

impl<T: RlstScalar + MatrixInverse + MatrixPseudoInverse + MatrixLu> FactorOperations
    for LuFactor<T>
where
    LuDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixLuDecomposition<Item = T>,
{
    type Item = T;

    fn new(
        ind_r: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        y_data: &BoxesData<Self::Item>,
        z_data: &BoxesData<Self::Item>,
        subs_sample_dim: usize,
        tol_lstq: <Self::Item as RlstScalar>::Real,
        _rank_par: &BoxType<Real<Self::Item>>,
        hermitian: bool,
    ) -> (Option<Self>, Times) {
        let mut r_numbering: Vec<usize> = Vec::new();
        let mut t_numbering: Vec<usize> = Vec::new();
        let mut ind_t = Vec::new();

        let near_field_ind_to_num: HashMap<_, _> = near_field_inds
            .iter()
            .enumerate()
            .map(|(num, ind)| (ind, num))
            .collect();

        for &elem in ind_r.iter() {
            r_numbering.push(*near_field_ind_to_num.get(&elem).unwrap());
        }
        for (pos, &elem) in near_field_inds.iter().enumerate() {
            if !ind_r.contains(&elem) {
                t_numbering.push(pos);
                ind_t.push(elem);
            }
        }

        let (y_r, mut u_arr, (_y_lu_io_time, y_lu_b_ext_time)) = near_box_extraction(
            ind_r,
            near_field_inds,
            y_data,
            subs_sample_dim,
            tol_lstq,
            &r_numbering,
            &t_numbering,
        );

        let start = Instant::now();
        let lu_y_r = y_r.into_lu_alloc().unwrap();
        let _ = <LuDecomposition<Self::Item, _> as MatrixLuDecomposition>::solve_mat(
            &lu_y_r,
            TransMode::NoTrans,
            u_arr.r_mut(),
        );

        let u_assembly = start.elapsed();

        let mut l_arr: DynamicArray<Self::Item, 2> = empty_array();

        let lu_b_ext_time;
        let lu_assembly_time;

        if !hermitian {
            let (mut z_r, z_n, (_z_lu_io_time, z_lu_b_ext_time)) = near_box_extraction(
                ind_r,
                near_field_inds,
                z_data,
                subs_sample_dim,
                tol_lstq,
                &r_numbering,
                &t_numbering,
            );
            let mut aux: DynamicArray<Self::Item, 2> = empty_array();

            let start = Instant::now();
            z_r.r_mut().into_inverse_alloc().unwrap();
            aux.r_mut().simple_mult_into_resize(z_n.r(), z_r.r());
            l_arr.r_mut().fill_from_resize(aux.r().conj());
            let l_assembly = start.elapsed();
            lu_b_ext_time = y_lu_b_ext_time + z_lu_b_ext_time;
            lu_assembly_time = u_assembly + l_assembly;
        } else {
            lu_b_ext_time = y_lu_b_ext_time;
            lu_assembly_time = u_assembly;
        }

        let lu_times = LuTimes {
            extraction: lu_b_ext_time.as_millis(),
            lu: lu_assembly_time.as_millis(),
        };

        let times = Times::Lu(lu_times);

        (
            Some(Self {
                l_arr,
                u_arr,
                hermitian,
                ind_r: ind_r.to_vec(),
                ind_t,
            }),
            times,
        )
    }

    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        factor_options: &FactorOptions,
        factor_type: &FactorType,
        operation_type: &RsrsSide,
    ) {
        let target_block = self.mul_data(target_arr, factor_options, factor_type, operation_type);

        let t_arr_mutex = std::sync::Mutex::new(target_arr);
        self.ins_data(
            &target_block,
            &mut *t_arr_mutex.lock().unwrap(),
            factor_options,
            factor_type,
            operation_type,
        );
    }

    fn mul_data<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &Array<Self::Item, ArrayImplMut, 2>,
        options: &FactorOptions,
        factor_type: &FactorType,
        operation_type: &RsrsSide,
    ) -> DynamicArray<Self::Item, 2> {
        let mut beta: Self::Item = <Self::Item as One>::one();
        let mut trans = options.trans;
        if options.inv {
            beta = -<Self::Item as One>::one();
        }

        if self.hermitian {
            match factor_type {
                FactorType::F => {
                    trans = !trans;
                }
                FactorType::S => {}
            }

            if *operation_type == RsrsSide::Left {
                row_ops_no_sub(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    &self.u_arr,
                    target_arr,
                    beta,
                    trans,
                )
            } else if *operation_type == RsrsSide::Right {
                col_ops_no_sub(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    &self.u_arr,
                    target_arr,
                    beta,
                    trans,
                )
            } else {
                empty_array()
            }
        } else {
            match factor_type {
                FactorType::F => {
                    if *operation_type == RsrsSide::Left {
                        row_ops_no_sub(
                            self.ind_r.clone(),
                            self.ind_t.clone(),
                            &self.l_arr,
                            target_arr,
                            beta,
                            options.trans,
                        )
                    } else if *operation_type == RsrsSide::Right {
                        col_ops_no_sub(
                            self.ind_r.clone(),
                            self.ind_t.clone(),
                            &self.l_arr,
                            target_arr,
                            beta,
                            options.trans,
                        )
                    } else {
                        empty_array()
                    }
                }
                FactorType::S => {
                    if *operation_type == RsrsSide::Left {
                        row_ops_no_sub(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            &self.u_arr,
                            target_arr,
                            beta,
                            options.trans,
                        )
                    } else if *operation_type == RsrsSide::Right {
                        col_ops_no_sub(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            &self.u_arr,
                            target_arr,
                            beta,
                            options.trans,
                        )
                    } else {
                        empty_array() //TODO: Add panic for this case
                    }
                }
            }
        }
    }

    fn ins_data<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        source_arr: &DynamicArray<Self::Item, 2>,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        options: &FactorOptions,
        factor_type: &FactorType,
        operation_type: &RsrsSide,
    ) {
        let mut trans = options.trans;

        if self.hermitian {
            match factor_type {
                FactorType::F => {
                    trans = !trans;
                }
                FactorType::S => {}
            }

            if *operation_type == RsrsSide::Left {
                row_subs(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    source_arr,
                    target_arr,
                    trans,
                );
            } else if *operation_type == RsrsSide::Right {
                col_subs(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    source_arr,
                    target_arr,
                    trans,
                );
            }
        } else {
            match factor_type {
                FactorType::F => {
                    if *operation_type == RsrsSide::Left {
                        row_subs(
                            self.ind_r.clone(),
                            self.ind_t.clone(),
                            source_arr,
                            target_arr,
                            options.trans,
                        );
                    } else if *operation_type == RsrsSide::Right {
                        col_subs(
                            self.ind_r.clone(),
                            self.ind_t.clone(),
                            source_arr,
                            target_arr,
                            options.trans,
                        );
                    }
                }
                FactorType::S => {
                    if *operation_type == RsrsSide::Left {
                        row_subs(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            source_arr,
                            target_arr,
                            options.trans,
                        );
                    } else if *operation_type == RsrsSide::Right {
                        col_subs(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            source_arr,
                            target_arr,
                            options.trans,
                        );
                    }
                }
            }
        }
    }
}

pub enum Factor<Item: RlstScalar> {
    Lu(LuFactor<Item>),
    Id(IdFactor<Item>),
}

pub type CommutativeFactors<Item> = Vec<Factor<Item>>;

pub trait CommutativeFactorsOperations: Sized {
    type Item: RlstScalar;
    fn new() -> Self;
    fn add_factor(&mut self, factor: Factor<Self::Item>);
    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        factor_options: &FactorOptions,
        mul_type: &MulType,
    );
}

impl<Item: RlstScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + RandScalar + MatrixLu>
    CommutativeFactorsOperations for CommutativeFactors<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    type Item = Item;

    fn new() -> Self {
        Vec::new()
    }
    fn add_factor(&mut self, factor: Factor<Self::Item>) {
        self.push(factor);
    }

    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        factor_options: &FactorOptions,
        mul_type: &MulType,
    ) where
        Self: Sized,
    {
        let updated_t_arr_blocks: Vec<_> = self
            .par_iter()
            .enumerate()
            .map(|(factor_ind, factor)| {
                let target_block = match factor {
                    Factor::Lu(lu_factor) => lu_factor.mul_data(
                        target_arr,
                        &factor_options,
                        &mul_type.factor_type,
                        &mul_type.side,
                    ),
                    Factor::Id(id_factor) => id_factor.mul_data(
                        target_arr,
                        &factor_options,
                        &mul_type.factor_type,
                        &mul_type.side,
                    ),
                };

                (factor_ind, target_block)
            })
            .collect();

        let t_arr_mutex = std::sync::Mutex::new(target_arr);
        updated_t_arr_blocks
            .par_iter()
            .for_each(|(factor_ind, target_block)| {
                let factor = &self[*factor_ind];
                match factor {
                    Factor::Lu(lu_factor) => lu_factor.ins_data(
                        target_block,
                        &mut *t_arr_mutex.lock().unwrap(),
                        &factor_options,
                        &mul_type.factor_type,
                        &mul_type.side,
                    ),
                    Factor::Id(id_factor) => id_factor.ins_data(
                        target_block,
                        &mut *t_arr_mutex.lock().unwrap(),
                        &factor_options,
                        &mul_type.factor_type,
                        &mul_type.side,
                    ),
                };
            });
    }
}

pub struct PermFactor {
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
}

pub trait PermOperations: Sized {
    fn new(row_indices: Vec<usize>, col_indices: Vec<usize>) -> RlstResult<Self>;

    fn left_mul<
        T: RlstScalar,
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = T>
            + Shape<2>
            + RawAccessMut<Item = T>
            + UnsafeRandomAccessMut<2, Item = T>
            + UnsafeRandomAccessByRef<2, Item = T>,
    >(
        &self,
        right_arr: &mut Array<T, ArrayImplMut, 2>,
        options: &FactorOptions,
    );

    fn right_mul<
        T: RlstScalar,
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = T>
            + Shape<2>
            + RawAccessMut<Item = T>
            + UnsafeRandomAccessMut<2, Item = T>
            + UnsafeRandomAccessByRef<2, Item = T>,
    >(
        &self,
        left_arr: &mut Array<T, ArrayImplMut, 2>,
        options: &FactorOptions,
    );
}

impl PermOperations for PermFactor {
    fn new(row_indices: Vec<usize>, col_indices: Vec<usize>) -> RlstResult<Self> {
        Ok(Self {
            row_indices,
            col_indices,
        })
    }

    fn left_mul<
        T: RlstScalar,
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = T>
            + Shape<2>
            + RawAccessMut<Item = T>
            + UnsafeRandomAccessMut<2, Item = T>
            + UnsafeRandomAccessByRef<2, Item = T>,
    >(
        &self,
        right_arr: &mut Array<T, ArrayImplMut, 2>,
        options: &FactorOptions,
    ) {
        assert_eq!(self.row_indices.len(), self.col_indices.len());
        let mut trans = options.trans;
        if options.inv {
            trans = !trans;
        }
        row_perm(
            self.col_indices.clone(),
            self.row_indices.clone(),
            right_arr,
            trans,
        );
    }

    fn right_mul<
        T: RlstScalar,
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = T>
            + Shape<2>
            + RawAccessMut<Item = T>
            + UnsafeRandomAccessMut<2, Item = T>
            + UnsafeRandomAccessByRef<2, Item = T>,
    >(
        &self,
        left_arr: &mut Array<T, ArrayImplMut, 2>,
        options: &FactorOptions,
    ) {
        assert_eq!(self.row_indices.len(), self.col_indices.len());
        let mut trans = options.trans;
        if options.inv {
            trans = !trans;
        }
        col_perm(
            self.col_indices.clone(),
            self.row_indices.clone(),
            left_arr,
            trans,
        );
    }
}

pub struct DiagBox<T: RlstScalar> {
    pub dbox: DynamicArray<T, 2>,
    pub inv_dbox: DynamicArray<T, 2>,
    pub inds: Vec<usize>,
}

type DiagBoxFactor<T> = Vec<DiagBox<T>>;

pub trait DiagBoxOperations: Sized {
    /// Item type
    type Item: RlstScalar;
    fn new() -> RlstResult<Self>;

    fn get_diag_inv(&mut self);

    fn left_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &mut self,
        right_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        options: &FactorOptions,
    );

    fn right_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        right_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        options: &FactorOptions,
    );
}

impl<T: RlstScalar + MatrixInverse> DiagBoxOperations for DiagBoxFactor<T> {
    type Item = T;

    fn new() -> RlstResult<Self> {
        let diag_boxes = Vec::new();
        Ok(diag_boxes)
    }

    fn get_diag_inv(&mut self) {
        if self[0].inv_dbox.is_empty() {
            self.par_iter_mut().for_each(|diag_box| {
                diag_box.inv_dbox.fill_from_resize(diag_box.dbox.r());
                diag_box.inv_dbox.r_mut().into_inverse_alloc().unwrap();
            });
        }
    }

    fn left_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &mut self,
        right_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        options: &FactorOptions,
    ) {
        //TODO: Parallelize these blocks
        self.get_diag_inv();
        if options.inv {
            self.iter().for_each(|diag_box| {
                let target_rows: DynamicArray<T, 2> = <Extraction<T> as MatrixExtraction>::new(
                    right_arr,
                    ExtInsType::Axis(diag_box.inds.clone(), 0, false),
                )
                .unwrap()
                .ext;
                let mut new_target_rows = empty_array();
                new_target_rows
                    .r_mut()
                    .simple_mult_into_resize(diag_box.inv_dbox.r(), target_rows.r()); //TODO: Allow conj transpose
                matrix_insertion(
                    right_arr,
                    &mut new_target_rows,
                    ExtInsType::Axis(diag_box.inds.clone(), 0, false),
                );
            });
        } else {
            self.iter().for_each(|diag_box| {
                let target_rows: DynamicArray<T, 2> = <Extraction<T> as MatrixExtraction>::new(
                    right_arr,
                    ExtInsType::Axis(diag_box.inds.clone(), 0, false),
                )
                .unwrap()
                .ext;
                let mut new_target_rows = empty_array();
                new_target_rows
                    .r_mut()
                    .simple_mult_into_resize(diag_box.dbox.r(), target_rows.r()); //TODO: Allow conj transpose
                matrix_insertion(
                    right_arr,
                    &mut new_target_rows,
                    ExtInsType::Axis(diag_box.inds.clone(), 0, false),
                );
            });
        }
    }

    fn right_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        left_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        options: &FactorOptions,
    ) {
        //TODO: Parallelize this block
        if options.inv {
            for diag_box in self {
                let target_rows: DynamicArray<T, 2> = <Extraction<T> as MatrixExtraction>::new(
                    left_arr,
                    ExtInsType::Axis(diag_box.inds.clone(), 1, false),
                )
                .unwrap()
                .ext;
                let mut new_target_rows = empty_array();
                new_target_rows
                    .r_mut()
                    .simple_mult_into_resize(target_rows.r(), diag_box.inv_dbox.r()); //TODO: Allow conj transpose
                matrix_insertion(
                    left_arr,
                    &mut new_target_rows,
                    ExtInsType::Axis(diag_box.inds.clone(), 1, false),
                );
            }
        } else {
            for diag_box in self {
                let target_rows: DynamicArray<T, 2> = <Extraction<T> as MatrixExtraction>::new(
                    left_arr,
                    ExtInsType::Axis(diag_box.inds.clone(), 1, false),
                )
                .unwrap()
                .ext;
                let mut new_target_rows = empty_array();
                new_target_rows
                    .r_mut()
                    .simple_mult_into_resize(target_rows.r(), diag_box.dbox.r()); //TODO: Allow conj transpose
                matrix_insertion(
                    left_arr,
                    &mut new_target_rows,
                    ExtInsType::Axis(diag_box.inds.clone(), 1, false),
                );
            }
        }
    }
}

pub struct MulType {
    pub side: RsrsSide,
    pub factor_type: FactorType,
}

type LevelLuFactors<T> = Vec<Vec<CommutativeFactors<T>>>;
type LevelIdFactors<T> = Vec<CommutativeFactors<T>>;
type LevelNearFieldInds = Vec<Vec<Vec<usize>>>;

pub struct RsrsFactors<Item: RlstScalar> {
    pub num_levels: usize,
    pub id_factors: LevelIdFactors<Item>,
    pub lu_factors: LevelLuFactors<Item>,
    pub near_field_inds: LevelNearFieldInds,
    pub perm_factor: PermFactor,
    pub diag_box_factor: DiagBoxFactor<Item>,
}

pub trait RsrsFactorsOps: Sized {
    type Item: RlstScalar;
    fn new(num_levels: usize) -> Self;

    fn apply_id_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        mul_type: &MulType,
        factor_options: &FactorOptions,
        level_it: usize,
    );

    fn apply_lu_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        mul_type: &MulType,
        factor_options: &FactorOptions,
        dec: bool,
        level_it: usize,
    );

    fn el_factors_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &mut self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        mul_type: MulType,
        factor_options: &FactorOptions,
        level: bool,
    );

    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &mut self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        mul_type: RsrsSide,
        factor_options: &FactorOptions,
    );

    fn perm_target_array<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
    );
}

fn get_far_indices(n: usize, near_indices: Vec<usize>) -> Vec<usize> {
    let near_set: HashSet<usize> = near_indices.into_iter().collect();
    (0..n).filter(|x| !near_set.contains(x)).collect()
}

impl<T: RlstScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar>
    RsrsFactorsOps for RsrsFactors<T>
where
    LuDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixLuDecomposition<Item = T>,
{
    type Item = T;

    fn new(num_levels: usize) -> Self {
        let mut id_factors = Vec::new();
        id_factors.resize_with(num_levels, || Vec::new());
        let mut lu_factors = Vec::new();
        lu_factors.resize_with(num_levels, || Vec::new());
        let mut near_field_inds = Vec::new();
        near_field_inds.resize_with(num_levels, || Vec::new());
        let row_indices = Vec::new();
        let col_indices = Vec::new();
        let perm_factor = PermFactor {
            row_indices,
            col_indices,
        };
        let diag_box_factor = DiagBoxFactor::new();
        Self {
            num_levels,
            near_field_inds,
            id_factors,
            lu_factors,
            perm_factor,
            diag_box_factor,
        }
    }

    fn apply_id_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        mul_type: &MulType,
        factor_options: &FactorOptions,
        level_it: usize,
    ) {
        if mul_type.side == RsrsSide::Squeeze {
            let left_mul_type = MulType {
                side: RsrsSide::Left,
                factor_type: FactorType::F,
            };
            let id_batch = &self.id_factors[level_it];

            id_batch.mul(target_arr, factor_options, &left_mul_type);

            let right_mul_type = MulType {
                side: RsrsSide::Right,
                factor_type: FactorType::S,
            };

            id_batch.mul(target_arr, factor_options, &right_mul_type);
        } else {
            let id_batch = &self.id_factors[level_it];
            id_batch.mul(target_arr, factor_options, mul_type);
        }
    }

    fn apply_lu_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        mul_type: &MulType,
        factor_options: &FactorOptions,
        dec: bool,
        level_it: usize,
    ) {
        let num_lu_batches = self.lu_factors[level_it].len();
        if mul_type.side == RsrsSide::Squeeze {
            (0..num_lu_batches).for_each(|batch_ind| {
                let left_mul_type = MulType {
                    side: RsrsSide::Left,
                    factor_type: FactorType::F,
                };
                let lu_batch = &self.lu_factors[level_it][batch_ind];

                lu_batch.mul(target_arr, factor_options, &left_mul_type);

                let right_mul_type = MulType {
                    side: RsrsSide::Right,
                    factor_type: FactorType::S,
                };

                lu_batch.mul(target_arr, factor_options, &right_mul_type);
            });
        } else {
            if dec {
                (0..num_lu_batches).rev().for_each(|batch_ind| {
                    let lu_batch = &self.lu_factors[level_it][batch_ind];
                    lu_batch.mul(target_arr, factor_options, mul_type);
                });
            } else {
                (0..num_lu_batches).for_each(|batch_ind| {
                    let lu_batch = &self.lu_factors[level_it][batch_ind];
                    lu_batch.mul(target_arr, factor_options, mul_type);
                });
            }
        }
    }

    fn el_factors_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &mut self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        mul_type: MulType,
        factor_options: &FactorOptions,
        dec: bool,
    ) {
        let levels = (0..self.num_levels).collect::<Vec<_>>();

        if dec {
            levels.iter().rev().for_each(|&level_it| {
                self.apply_lu_level(target_arr, &mul_type, &factor_options, dec, level_it);
                self.apply_id_level(target_arr, &mul_type, &factor_options, level_it);
            });
        } else {
            levels.iter().for_each(|&level_it| {
                self.apply_id_level(target_arr, &mul_type, &factor_options, level_it);
                self.apply_lu_level(target_arr, &mul_type, &factor_options, dec, level_it);
            });
        }
    }

    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &mut self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        side: RsrsSide,
        factor_options: &FactorOptions,
    ) {
        match side {
            RsrsSide::Squeeze => {}
            RsrsSide::Left => {
                let mul_type_1;
                let mul_type_2;
                if !factor_options.inv {
                    mul_type_1 = MulType {
                        side: RsrsSide::Left,
                        factor_type: FactorType::S,
                    };
                    mul_type_2 = MulType {
                        side: RsrsSide::Left,
                        factor_type: FactorType::F,
                    };
                } else {
                    mul_type_1 = MulType {
                        side: RsrsSide::Left,
                        factor_type: FactorType::F,
                    };
                    mul_type_2 = MulType {
                        side: RsrsSide::Left,
                        factor_type: FactorType::S,
                    };
                }

                self.el_factors_mul(target_arr, mul_type_1, factor_options, false);

                self.diag_box_factor.left_mul(target_arr, &factor_options);

                self.el_factors_mul(target_arr, mul_type_2, factor_options, true);
            }
            RsrsSide::Right => {
                let mul_type_1;
                let mul_type_2;
                if !factor_options.inv {
                    mul_type_1 = MulType {
                        side: RsrsSide::Right,
                        factor_type: FactorType::F,
                    };
                    mul_type_2 = MulType {
                        side: RsrsSide::Right,
                        factor_type: FactorType::S,
                    };
                } else {
                    mul_type_1 = MulType {
                        side: RsrsSide::Right,
                        factor_type: FactorType::S,
                    };
                    mul_type_2 = MulType {
                        side: RsrsSide::Right,
                        factor_type: FactorType::F,
                    };
                }
                self.el_factors_mul(target_arr, mul_type_1, factor_options, false);
                self.diag_box_factor.right_mul(target_arr, &factor_options);
                self.el_factors_mul(target_arr, mul_type_2, factor_options, true);
            }
        }
    }

    fn perm_target_array<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
    ) {
        self.perm_factor.left_mul(
            target_arr,
            &FactorOptions {
                inv: false,
                trans: false,
            },
        );
        self.perm_factor.right_mul(
            target_arr,
            &FactorOptions {
                inv: false,
                trans: true,
            },
        );
    }
}
