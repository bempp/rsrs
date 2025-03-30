use super::{rsrs_cycle::RsrsOptions, sketch::BoxesData};
use crate::utils::{
    data_ins_ext::{matrix_insertion, ExtInsType, Extraction, MatrixExtraction},
    elementary_matrix::{col_ops, col_perm, row_ops, row_perm},
};
use num::One;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use rlst::{
    dense::linalg::interpolative_decomposition::Accuracy, empty_array, rlst_dynamic_array2, Array,
    DynamicArray, IdDecomposition, MatrixId, MatrixInverse, MatrixPseudoInverse, MultIntoResize,
    RawAccessMut, RlstResult, RlstScalar, Shape, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use serde::Serialize;
use std::{
    sync::{Arc, Mutex},
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

pub trait IdFactorOperations: Sized {
    type Item: RlstScalar;

    fn new(
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        dim: usize,
        target_arr: DynamicArray<Self::Item, 2>,
        tol_id: <Self::Item as RlstScalar>::Real,
        options: &RsrsOptions,
    ) -> Option<Self>;

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
}

impl<T: RlstScalar + MatrixInverse + MatrixId> IdFactorOperations for IdFactor<T> {
    type Item = T;

    fn new(
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        dim: usize,
        target_arr: DynamicArray<Self::Item, 2>,
        tol_id: <Self::Item as RlstScalar>::Real,
        options: &RsrsOptions,
    ) -> Option<Self> {
        let max_rank: usize = *target_arr.shape().iter().min().unwrap();
        let id_sketch: IdDecomposition<Self::Item> = target_arr.into_id_alloc(Accuracy::Tol(tol_id)).unwrap();
        let k: usize = id_sketch.rank;
        let mut ind_r = Vec::new();
        let mut ind_s = Vec::new();

        if !options.silent {
            println!("Rank of box: {}. Max rank: {}", k, max_rank);
        }

        if id_sketch.rank < max_rank {
            let mut aux_indices: Vec<usize> = target_inds.clone();
            for (id, &elem) in id_sketch.perm.iter().enumerate() {
                *target_inds.get_mut(id).unwrap() = *aux_indices.get_mut(elem).unwrap();
                *near_field_inds.get_mut(id).unwrap() = *aux_indices.get_mut(elem).unwrap();
            }
            ind_r.append(&mut target_inds[k..].to_vec());
            ind_s.append(&mut target_inds[0..k].to_vec());

            let ind_f = get_far_indices(dim, near_field_inds.to_vec());

            Some(Self {
                data: id_sketch.id_mat,
                perm: id_sketch.perm,
                ind_r,
                ind_s,
                ind_f,
            })
        } else {
            None
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
        options: &FactorOptions,
        factor_type: &FactorType,
        operation_type: &RsrsSide,
    ) {
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
            row_ops(
                self.ind_s.clone(),
                self.ind_r.clone(),
                &self.data,
                target_arr,
                beta,
                trans,
            );
        } else if *operation_type == RsrsSide::Right {
            col_ops(
                self.ind_s.clone(),
                self.ind_r.clone(),
                &self.data,
                target_arr,
                beta,
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

fn near_box_extraction<Item: RlstScalar + MatrixPseudoInverse>(
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
) {
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

    //Solve right least squares problem
    let shape = test_n.shape();
    let mut pinv = rlst_dynamic_array2!(Item, [shape[1], shape[0]]); // Avoid extra allocation
    test_n
        .r_mut()
        .into_pseudo_inverse_alloc(pinv.r_mut(), tol_lstq)
        .unwrap();
    let mut near_box: DynamicArray<Item, 2> = empty_array();
    near_box
        .r_mut()
        .simple_mult_into_resize(sketch_r.r(), pinv.r());

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

#[derive(Debug, Serialize, Clone)]
pub struct LuTimes {
    pub extraction: u128,
    pub lu: u128,
}
pub trait LuFactorOperations: Sized {
    type Item: RlstScalar;
    fn new(
        ind_r: &[usize],
        near_field_inds: &[usize],
        y_data: &BoxesData<Self::Item>,
        z_data: &BoxesData<Self::Item>,
        subs_sample_dim: usize,
        tol_lstq: <Self::Item as RlstScalar>::Real,
        options: &RsrsOptions,
    ) -> (Self, LuTimes);
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
}

impl<T: RlstScalar + MatrixInverse + MatrixPseudoInverse> LuFactorOperations for LuFactor<T> {
    type Item = T;

    fn new(
        ind_r: &[usize],
        near_field_inds: &[usize],
        y_data: &BoxesData<Self::Item>,
        z_data: &BoxesData<Self::Item>,
        subs_sample_dim: usize,
        tol_lstq: <Self::Item as RlstScalar>::Real,
        options: &RsrsOptions,
    ) -> (Self, LuTimes) {
        let mut r_numbering: Vec<usize> = Vec::new();
        let mut t_numbering: Vec<usize> = Vec::new();
        let mut ind_t = Vec::new();

        for &elem in ind_r.iter() {
            r_numbering.push(near_field_inds.iter().position(|&y| y == elem).unwrap());
        }
        for (pos, &elem) in near_field_inds.iter().enumerate() {
            if !ind_r.contains(&elem) {
                t_numbering.push(pos);
                ind_t.push(elem);
            }
        }

        let (mut y_r, y_n, (y_lu_io_time, y_lu_b_ext_time)) = near_box_extraction(
            ind_r,
            near_field_inds,
            y_data,
            subs_sample_dim,
            tol_lstq,
            &r_numbering,
            &t_numbering,
        );

        let start = Instant::now();
        y_r.r_mut().into_inverse_alloc().unwrap();
        let u_arr = empty_array().simple_mult_into_resize(y_r.r(), y_n.r());
        let u_assembly = start.elapsed();

        let mut l_arr: DynamicArray<Self::Item, 2> = empty_array();

        let lu_io_time;
        let lu_b_ext_time;
        let lu_assembly_time;

        if !options.hermitian {
            let (mut z_r, z_n, (z_lu_io_time, z_lu_b_ext_time)) = near_box_extraction(
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
            aux.r_mut()
                .simple_mult_into_resize(z_n.r(), z_r.r());
            l_arr.r_mut().fill_from_resize(aux.r().conj());
            let l_assembly = start.elapsed();

            lu_io_time = y_lu_io_time + z_lu_io_time;
            lu_b_ext_time = y_lu_b_ext_time + z_lu_b_ext_time;
            lu_assembly_time = u_assembly + l_assembly;
        } else {
            lu_io_time = y_lu_io_time;
            lu_b_ext_time = y_lu_b_ext_time;
            lu_assembly_time = u_assembly;
        }

        if !options.silent {
            println!("LU io in {} ms", lu_io_time.as_millis());
            println!("LU block extraction in {} ms", lu_b_ext_time.as_millis());
        }

        let lu_times = LuTimes {
            extraction: lu_b_ext_time.as_millis(),
            lu: lu_assembly_time.as_millis(),
        };

        (
            Self {
                l_arr,
                u_arr,
                hermitian: options.hermitian,
                ind_r: ind_r.to_vec(),
                ind_t,
            },
            lu_times,
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
        options: &FactorOptions,
        factor_type: &FactorType,
        operation_type: &RsrsSide,
    ) {
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
                row_ops(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    &self.u_arr,
                    target_arr,
                    beta,
                    trans,
                );
            } else if *operation_type == RsrsSide::Right {
                col_ops(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    &self.u_arr,
                    target_arr,
                    beta,
                    trans,
                );
            }
        } else {
            match factor_type {
                FactorType::F => {
                    if *operation_type == RsrsSide::Left {
                        row_ops(
                            self.ind_r.clone(),
                            self.ind_t.clone(),
                            &self.l_arr,
                            target_arr,
                            beta,
                            options.trans,
                        );
                    } else if *operation_type == RsrsSide::Right {
                        col_ops(
                            self.ind_r.clone(),
                            self.ind_t.clone(),
                            &self.l_arr,
                            target_arr,
                            beta,
                            options.trans,
                        );
                    }
                }
                FactorType::S => {
                    if *operation_type == RsrsSide::Left {
                        row_ops(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            &self.u_arr,
                            target_arr,
                            beta,
                            options.trans,
                        );
                    } else if *operation_type == RsrsSide::Right {
                        col_ops(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            &self.u_arr,
                            target_arr,
                            beta,
                            options.trans,
                        );
                    }
                }
            }
        }
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
    side: RsrsSide,
    factor_type: FactorType,
}

type LevelLuFactors<T> = Vec<Vec<Vec<LuFactor<T>>>>;
type LevelIdFactors<T> = Vec<Vec<IdFactor<T>>>;
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

    fn apply_id_level(
        &self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
        mul_type: &MulType,
        factor_options: &FactorOptions,
        level_it: usize,
    );

    fn apply_lu_level(
        &self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
        mul_type: &MulType,
        factor_options: &FactorOptions,
        dec: bool,
        level_it: usize,
    );

    fn el_factors_mul(
        &mut self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
        mul_type: MulType,
        factor_options: &FactorOptions,
        level: bool,
    );

    fn mul(
        &mut self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
        mul_type: RsrsSide,
        factor_options: &FactorOptions,
    );

    fn perm_target_array(&self, target_arr: &mut DynamicArray<Self::Item, 2>);
}

fn get_far_indices(n: usize, near_indices: Vec<usize>) -> Vec<usize> {
    let mut domain: Vec<usize> = (0..n).collect();
    domain.retain(|x| !near_indices.contains(x));
    domain
}

impl<T: RlstScalar + MatrixInverse + MatrixId + MatrixPseudoInverse> RsrsFactorsOps
    for RsrsFactors<T>
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

    fn apply_id_level(
        &self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
        mul_type: &MulType,
        factor_options: &FactorOptions,
        level_it: usize,
    ) {
        let target_arr = Arc::new(Mutex::new(target_arr));
        let side_mul = |side: &RsrsSide, factor_type: &FactorType| {
            self.id_factors[level_it].par_iter().for_each(|id_factor| {
                let mut target_arr = target_arr.lock().unwrap();
                id_factor.mul(&mut target_arr, factor_options, &factor_type, &side);
            });
        };

        if mul_type.side == RsrsSide::Squeeze {
            self.id_factors[level_it].par_iter().for_each(|id_factor| {
                let mut target_arr = target_arr.lock().unwrap();
                id_factor.mul(
                    &mut target_arr,
                    factor_options,
                    &FactorType::F,
                    &RsrsSide::Left,
                );
                id_factor.mul(
                    &mut target_arr,
                    factor_options,
                    &FactorType::S,
                    &RsrsSide::Right,
                );
            });
        } else {
            side_mul(&mul_type.side, &mul_type.factor_type);
        }
    }

    fn apply_lu_level(
        &self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
        mul_type: &MulType,
        factor_options: &FactorOptions,
        dec: bool,
        level_it: usize,
    ) {
        let target_arr = Arc::new(Mutex::new(target_arr));
        let side_mul = |side: &RsrsSide, factor_type: &FactorType| {
            let batch_iter = |lu_batch: &Vec<LuFactor<Self::Item>>| {
                lu_batch.par_iter().for_each(|lu_factor| {
                    let mut target_arr = target_arr.lock().unwrap();
                    lu_factor.mul(&mut target_arr, factor_options, &factor_type, side);
                });
            };
            if dec {
                self.lu_factors[level_it].iter().rev().for_each(|lu_batch| {
                    batch_iter(lu_batch);
                });
            } else {
                self.lu_factors[level_it].iter().for_each(|lu_batch| {
                    batch_iter(lu_batch);
                });
            }
        };

        if mul_type.side == RsrsSide::Squeeze {
            self.lu_factors[level_it].iter().for_each(|lu_batch| {
                lu_batch.par_iter().for_each(|lu_factor| {
                    let mut target_arr = target_arr.lock().unwrap();
                    lu_factor.mul(
                        &mut target_arr,
                        factor_options,
                        &FactorType::F,
                        &RsrsSide::Left,
                    );
                    lu_factor.mul(
                        &mut target_arr,
                        factor_options,
                        &FactorType::S,
                        &RsrsSide::Right,
                    );
                });
            });
        } else {
            side_mul(&mul_type.side, &mul_type.factor_type);
        }
    }

    fn el_factors_mul(
        &mut self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
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

    fn mul(
        &mut self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
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

    fn perm_target_array(&self, target_arr: &mut DynamicArray<Self::Item, 2>) {
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
