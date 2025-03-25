use super::{rsrs_cycle::RsrsOptions, sketch::BoxesData};
use crate::{
    utils::{
        data_ins_ext::{matrix_insertion, ExtInsType, Extraction, MatrixExtraction},
        elementary_matrix::{col_ops, col_perm, row_ops, row_perm},
        norm_estimator::spectral_norm_estimator,
    },
    with_openblas_threads,
};
use num::One;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use rlst::{
    dense::{linalg::interpolative_decomposition::Accuracy, tools::RandScalar},
    empty_array, rlst_dynamic_array2, Array, DynamicArray, IdDecomposition, MatrixId,
    MatrixInverse, MatrixPseudoInverse, MultIntoResize, RawAccessMut, RlstResult, RlstScalar,
    Shape, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
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

pub enum DecFactorOpType {
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
        factor_type: FactorType,
        operation_type: DecFactorOpType,
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
        let id_sketch: IdDecomposition<Self::Item> =
            target_arr.into_id_alloc(Accuracy::Tol(tol_id)).unwrap();
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

            let ind_f = get_far_indices(dim, near_field_inds.to_vec()); //TODO: include dimension

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
        factor_type: FactorType,
        operation_type: DecFactorOpType,
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

        match operation_type {
            DecFactorOpType::Left => {
                row_ops(
                    self.ind_s.clone(),
                    self.ind_r.clone(),
                    &self.data,
                    target_arr,
                    beta,
                    trans,
                );
            }
            DecFactorOpType::Right => {
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
}

pub struct LuFactor<T: RlstScalar> {
    l_arr: DynamicArray<T, 2>,
    u_arr: DynamicArray<T, 2>,
    hermitian: bool,
    ind_r: Vec<usize>,     //cols
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
        .view()
        .into_subview([0, 0], [row_num, subs_sample_dim]);
    let sketch_subview = sketch_data
        .sketch
        .view()
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
        .view_mut()
        .into_pseudo_inverse_alloc(pinv.view_mut(), tol_lstq)
        .unwrap();
    let mut near_box: DynamicArray<Item, 2> = empty_array();
    near_box
        .view_mut()
        .simple_mult_into_resize(sketch_r.view(), pinv.view());

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

#[derive(Serialize, Clone)]
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
        factor_type: FactorType,
        operation_type: DecFactorOpType,
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
        y_r.view_mut().into_inverse_alloc().unwrap();
        let u_arr = empty_array().simple_mult_into_resize(y_r.view(), y_n.view());
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
            z_r.view_mut().into_inverse_alloc().unwrap();
            aux.view_mut()
                .simple_mult_into_resize(z_n.view(), z_r.view());
            l_arr.view_mut().fill_from_resize(aux.view().conj());
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
        factor_type: FactorType,
        operation_type: DecFactorOpType,
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

            match operation_type {
                DecFactorOpType::Left => {
                    row_ops(
                        self.ind_t.clone(),
                        self.ind_r.clone(),
                        &self.u_arr,
                        target_arr,
                        beta,
                        trans,
                    );
                }
                DecFactorOpType::Right => {
                    col_ops(
                        self.ind_t.clone(),
                        self.ind_r.clone(),
                        &self.u_arr,
                        target_arr,
                        beta,
                        trans,
                    );
                }
            }
        } else {
            match factor_type {
                FactorType::F => match operation_type {
                    DecFactorOpType::Left => {
                        row_ops(
                            self.ind_r.clone(),
                            self.ind_t.clone(),
                            &self.l_arr,
                            target_arr,
                            beta,
                            options.trans,
                        );
                    }
                    DecFactorOpType::Right => {
                        col_ops(
                            self.ind_r.clone(),
                            self.ind_t.clone(),
                            &self.l_arr,
                            target_arr,
                            beta,
                            options.trans,
                        );
                    }
                },
                FactorType::S => match operation_type {
                    DecFactorOpType::Left => {
                        row_ops(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            &self.u_arr,
                            target_arr,
                            beta,
                            options.trans,
                        );
                    }
                    DecFactorOpType::Right => {
                        col_ops(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            &self.u_arr,
                            target_arr,
                            beta,
                            options.trans,
                        );
                    }
                },
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

    fn get_diag_inv(&mut self, blas_cores: &usize);

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
        blas_cores: &usize,
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

    fn get_diag_inv(&mut self, blas_cores: &usize) {
        if self[0].inv_dbox.is_empty() {
            /*with_openblas_threads!(self.par_iter_mut().for_each(|diag_box| {
                diag_box.inv_dbox.fill_from_resize(diag_box.dbox.view());
                diag_box.inv_dbox.view_mut().into_inverse_alloc().unwrap();
            }), blas_cores);*/
            self.iter_mut().for_each(|diag_box| {
                diag_box.inv_dbox.fill_from_resize(diag_box.dbox.view());
                diag_box.inv_dbox.view_mut().into_inverse_alloc().unwrap();
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
        blas_cores: &usize,
    ) {
        //TODO: Parallelize this block
        self.get_diag_inv(blas_cores);
        if options.inv {
            for diag_box in self {
                let target_rows: DynamicArray<T, 2> = <Extraction<T> as MatrixExtraction>::new(
                    right_arr,
                    ExtInsType::Axis(diag_box.inds.clone(), 0, false),
                )
                .unwrap()
                .ext;
                let mut new_target_rows = empty_array();
                new_target_rows
                    .view_mut()
                    .simple_mult_into_resize(diag_box.inv_dbox.view(), target_rows.view()); //TODO: Allow conj transpose
                matrix_insertion(
                    right_arr,
                    &mut new_target_rows,
                    ExtInsType::Axis(diag_box.inds.clone(), 0, false),
                );
            }
        } else {
            for diag_box in self {
                let target_rows: DynamicArray<T, 2> = <Extraction<T> as MatrixExtraction>::new(
                    right_arr,
                    ExtInsType::Axis(diag_box.inds.clone(), 0, false),
                )
                .unwrap()
                .ext;
                let mut new_target_rows = empty_array();
                new_target_rows
                    .view_mut()
                    .simple_mult_into_resize(diag_box.dbox.view(), target_rows.view()); //TODO: Allow conj transpose
                matrix_insertion(
                    right_arr,
                    &mut new_target_rows,
                    ExtInsType::Axis(diag_box.inds.clone(), 0, false),
                );
            }
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
                    .view_mut()
                    .simple_mult_into_resize(target_rows.view(), diag_box.inv_dbox.view()); //TODO: Allow conj transpose
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
                    .view_mut()
                    .simple_mult_into_resize(target_rows.view(), diag_box.dbox.view()); //TODO: Allow conj transpose
                matrix_insertion(
                    left_arr,
                    &mut new_target_rows,
                    ExtInsType::Axis(diag_box.inds.clone(), 1, false),
                );
            }
        }
    }
}

pub enum DecFactorType<Item: RlstScalar> {
    Id(IdFactor<Item>),
    Lu(LuFactor<Item>),
}

/*pub struct DecFactors<Item: RlstScalar> {
    pub id_factor: IdFactor<Item>,
    pub lu_factor: LuFactor<Item>,
    pub near_field_inds: Vec<usize>,
}*/

pub struct RsrsFactors<Item: RlstScalar> {
    pub num_levels: usize,
    pub id_factors: LevelIdFactors<Item>,
    pub lu_factors: LevelLuFactors<Item>,
    pub near_field_inds: LevelNearFieldInds,
    pub perm_factor: PermFactor,
    pub diag_box_factor: DiagBoxFactor<Item>,
}

type LevelLuFactors<T> = Vec<Vec<Vec<LuFactor<T>>>>;
type LevelIdFactors<T> = Vec<Vec<IdFactor<T>>>;
type LevelNearFieldInds = Vec<Vec<Vec<usize>>>;
//type LevelDecFactors<T> = Vec<DecFactors<T>>;
type Errors<T> = (<T as RlstScalar>::Real, <T as RlstScalar>::Real);
type RelAbsErrors<T> = (Errors<T>, Errors<T>);
type LuErrors<T> = Vec<RelAbsErrors<T>>;
type IdErrors<T> = Vec<RelAbsErrors<T>>;

pub trait RsrsFactorsOps: Sized {
    type Item: RlstScalar;
    fn new(num_levels: usize) -> Self;

    fn apply_id_level(
        &self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
        factor_options: &FactorOptions,
        get_box_errors: bool,
        level_it: usize,
        blas_cores: &usize,
    ) -> Option<Vec<RelAbsErrors<Self::Item>>>;

    fn apply_lu_level(
        &self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
        factor_options: &FactorOptions,
        get_box_errors: bool,
        level_it: usize,
    ) -> Option<LuErrors<Self::Item>>;

    fn el_factors_mul(&self, target_arr: &mut DynamicArray<Self::Item, 2>, blas_cores: &usize);

    fn el_factors_inv_mul(
        &self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
        get_box_errors: bool,
        blas_cores: &usize,
    ) -> Option<Vec<(IdErrors<Self::Item>, LuErrors<Self::Item>)>>;

    fn perm_target_array(&self, target_arr: &mut DynamicArray<Self::Item, 2>);
}

fn get_far_indices(n: usize, near_indices: Vec<usize>) -> Vec<usize> {
    let mut domain: Vec<usize> = (0..n).collect();
    domain.retain(|x| !near_indices.contains(x));
    domain
}

type Real<T> = <T as rlst::RlstScalar>::Real;

impl<T: RlstScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + RandScalar> RsrsFactorsOps
    for RsrsFactors<T>
where
    StandardNormal: Distribution<Real<T>>,
    Standard: Distribution<Real<T>>,
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
        factor_options: &FactorOptions,
        get_box_errors: bool,
        level_it: usize,
        blas_cores: &usize,
    ) -> Option<IdErrors<Self::Item>> {
        if get_box_errors {
            let apply_box_id = |id_factor: &IdFactor<Self::Item>| {
                let (arr_rf, arr_fr) = box_errors_id(id_factor, target_arr);
                id_factor.mul(
                    target_arr,
                    factor_options,
                    FactorType::F,
                    DecFactorOpType::Left,
                );
                id_factor.mul(
                    target_arr,
                    factor_options,
                    FactorType::S,
                    DecFactorOpType::Right,
                );
                let (arr_rf_ae, arr_fr_ae) = box_errors_id(id_factor, target_arr);
                let rel_errs: Errors<Self::Item> = (arr_rf_ae / arr_rf, arr_fr_ae / arr_fr);
                let abs_errs: Errors<Self::Item> = (arr_rf_ae, arr_fr_ae);
                println!("rel_errs id, {:?}", rel_errs);
                (rel_errs, abs_errs)
            };

            let apply_box_id_mutex = std::sync::Mutex::new(apply_box_id);

            /*let errors: Vec<RelAbsErrors<Self::Item>> = with_openblas_threads!(
                self.dec_factors[level_it]
                .par_iter()
                .map(|dec_factors| {
                    let mut apply_box_id_mutex_guard = apply_box_id_mutex.lock().unwrap();
                    apply_box_id_mutex_guard(dec_factors)
                })
                .collect(),
                blas_cores
            );*/

            let errors: Vec<RelAbsErrors<Self::Item>> = self.id_factors[level_it]
                .iter()
                .map(|dec_factors| {
                    let mut apply_box_id_mutex_guard = apply_box_id_mutex.lock().unwrap();
                    apply_box_id_mutex_guard(dec_factors)
                })
                .collect();

            Some(errors)
        } else {
            let apply_box_id = |id_factor: &IdFactor<Self::Item>| {
                id_factor.mul(
                    target_arr,
                    factor_options,
                    FactorType::F,
                    DecFactorOpType::Left,
                );
                id_factor.mul(
                    target_arr,
                    factor_options,
                    FactorType::S,
                    DecFactorOpType::Right,
                );
            };

            let apply_box_id_mutex = std::sync::Mutex::new(apply_box_id);

            self.id_factors[level_it].iter().for_each(|id_factor| {
                let mut apply_box_id_mutex_guard = apply_box_id_mutex.lock().unwrap();
                apply_box_id_mutex_guard(id_factor);
            });
            /*with_openblas_threads!(self.dec_factors[level_it]
            .par_iter()
            .for_each(|dec_factors| {
                let mut apply_box_id_mutex_guard = apply_box_id_mutex.lock().unwrap();
                apply_box_id_mutex_guard(dec_factors);
            }), blas_cores);*/

            None
        }
    }

    fn apply_lu_level(
        &self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
        factor_options: &FactorOptions,
        get_box_errors: bool,
        level_it: usize,
    ) -> Option<LuErrors<Self::Item>> {
        if get_box_errors {
            let mut apply_box_lu = |lu_factor| {
                let (arr_rt, arr_tr) = box_errors_lu(lu_factor, target_arr);
                lu_factor.mul(
                    target_arr,
                    factor_options,
                    FactorType::F,
                    DecFactorOpType::Left,
                );
                lu_factor.mul(
                    target_arr,
                    factor_options,
                    FactorType::S,
                    DecFactorOpType::Right,
                );
                let (arr_rt_ae, arr_tr_ae) = box_errors_lu(lu_factor, target_arr);
                let rel_errs: Errors<Self::Item> = (arr_rt_ae / arr_rt, arr_tr_ae / arr_tr);
                let abs_errs: Errors<Self::Item> = (arr_rt_ae, arr_tr_ae);

                println!("rel_errs lu, {:?}", rel_errs);

                (rel_errs, abs_errs)
            };

            let errors: Vec<_> = self.lu_factors[level_it]
                .iter()
                .map(|lu_batch| {
                    let batch_errors: Vec<RelAbsErrors<Self::Item>> = lu_batch
                        .iter()
                        .map(|lu_factor| apply_box_lu(lu_factor))
                        .collect();
                    batch_errors
                })
                .collect();

            /*let errors: Vec<_> = self.lu_batches[level_it].iter().map(|batch|{
                let batch_errors : Vec<RelAbsErrors<Self::Item>> = batch.iter().map(|box_ind|{
                    let dec_factors = &self.dec_factors[level_it][*box_ind];
                    apply_box_lu(&dec_factors)
                }).collect();
                batch_errors
            }).collect();*/

            let errors: Vec<RelAbsErrors<Self::Item>> = errors.into_iter().flatten().collect();

            Some(errors)
        } else {
            self.lu_factors[level_it].iter().for_each(|lu_batch| {
                lu_batch.iter().for_each(|lu_factor| {
                    lu_factor.mul(
                        target_arr,
                        factor_options,
                        FactorType::F,
                        DecFactorOpType::Left,
                    );
                    lu_factor.mul(
                        target_arr,
                        factor_options,
                        FactorType::S,
                        DecFactorOpType::Right,
                    );
                });
            });
            None
        }
    }

    fn el_factors_mul(&self, target_arr: &mut DynamicArray<Self::Item, 2>, blas_cores: &usize) {
        let factor_options = FactorOptions {
            inv: false,
            trans: false,
        };

        for level_it in 0..self.num_levels {
            self.apply_id_level(target_arr, &factor_options, false, level_it, blas_cores);
            self.apply_lu_level(target_arr, &factor_options, false, level_it);
        }
    }

    fn el_factors_inv_mul(
        &self,
        target_arr: &mut DynamicArray<Self::Item, 2>,
        get_box_errors: bool,
        blas_cores: &usize,
    ) -> Option<Vec<(IdErrors<Self::Item>, LuErrors<Self::Item>)>> {
        let factor_options = FactorOptions {
            inv: true,
            trans: false,
        };
        if get_box_errors {
            let errors: Vec<(IdErrors<Self::Item>, LuErrors<Self::Item>)> = (0..self.num_levels)
                .map(|level_it| {
                    let id_errors = self
                        .apply_id_level(
                            target_arr,
                            &factor_options,
                            get_box_errors,
                            level_it,
                            blas_cores,
                        )
                        .unwrap();
                    let lu_errors = self
                        .apply_lu_level(target_arr, &factor_options, get_box_errors, level_it)
                        .unwrap();
                    (id_errors, lu_errors)
                })
                .collect();

            Some(errors)
        } else {
            (0..self.num_levels).for_each(|level_it| {
                self.apply_id_level(
                    target_arr,
                    &factor_options,
                    get_box_errors,
                    level_it,
                    blas_cores,
                );
                self.apply_lu_level(target_arr, &factor_options, get_box_errors, level_it);
            });
            None
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

fn box_errors_id<Item: RlstScalar + RandScalar>(
    id_factor: &IdFactor<Item>,
    arr: &mut DynamicArray<Item, 2>,
) -> Errors<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let ind_r = &id_factor.ind_r;
    let far_indices = &id_factor.ind_f;

    let arr_rf = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(ind_r.clone(), far_indices.clone()),
    )
    .unwrap()
    .ext;
    let arr_fr = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(far_indices.clone(), ind_r.clone()),
    )
    .unwrap()
    .ext;

    let arr_rf = spectral_norm_estimator(arr_rf, 10).unwrap();
    let arr_fr = spectral_norm_estimator(arr_fr, 10).unwrap();

    (arr_rf, arr_fr)
}

fn box_errors_lu<Item: RlstScalar + RandScalar>(
    lu_factor: &LuFactor<Item>,
    arr: &mut DynamicArray<Item, 2>,
) -> Errors<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let ind_r = &lu_factor.ind_r;
    let ind_t = &lu_factor.ind_t;

    let arr_rt = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(ind_r.clone(), ind_t.clone()),
    )
    .unwrap()
    .ext;
    let arr_tr = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(ind_t.clone(), ind_r.clone()),
    )
    .unwrap()
    .ext;

    let arr_rt = spectral_norm_estimator(arr_rt, 10).unwrap();
    let arr_tr = spectral_norm_estimator(arr_tr, 10).unwrap();

    (arr_rt, arr_tr)
}

pub fn get_diag_errors<
    Item: RlstScalar + RandScalar + rlst::MatrixId + rlst::MatrixInverse + rlst::MatrixPseudoInverse,
>(
    rsrs_factors: &RsrsFactors<Item>,
    arr: &mut DynamicArray<Item, 2>,
    blas_cores: &usize,
) -> Vec<<Item as RlstScalar>::Real>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let mut_arr = Arc::new(Mutex::new(arr));
    let exact_boxes_errors = rsrs_factors
        .diag_box_factor
        .iter()
        .map(|diag_box| {
            let mut arr = mut_arr.lock().unwrap();
            let exact_diag_box = <Extraction<Item> as MatrixExtraction>::new(
                &mut arr,
                ExtInsType::Cross(diag_box.inds.clone(), diag_box.inds.clone()),
            )
            .unwrap()
            .ext;
            let mut res: DynamicArray<Item, 2> = empty_array();
            res.fill_from_resize(exact_diag_box - diag_box.dbox.view());
            spectral_norm_estimator(res, 10).unwrap()
        })
        .collect();
    exact_boxes_errors
}
