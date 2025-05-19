use super::{
    rsrs_cycle::{BoxType, RsrsOptions},
    sketch::SketchData,
};
use crate::utils::{
    data_ins_ext::{ExtInsType, Extraction, MatrixExtraction},
    elementary_matrix::{
        col_ops_no_sub, col_perm, col_subs, ext_cols, ext_rows, row_ops_no_sub, row_perm, row_subs,
    },
    least_squares_and_null::{nullify_near_sketch, right_least_squares, NormalEquations},
};
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rlst::{
    dense::{
        linalg::{
            interpolative_decomposition::Accuracy, lu::MatrixLu,
            triangular_arrays::TriangularOperations,
        },
        tools::RandScalar,
    },
    prelude::*,
};
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

type Real<T> = <T as rlst::RlstScalar>::Real;

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

pub struct LuFactor<T: RlstScalar> {
    l_arr: DynamicArray<T, 2>,
    u_arr: DynamicArray<T, 2>,
    hermitian: bool,
    pub ind_r: Vec<usize>, //cols
    pub ind_t: Vec<usize>, //rows
}

pub struct DiagBoxArr<T: RlstScalar> {
    pub u_arr: TriangularMatrix<T>,
    pub l_arr: TriangularMatrix<T>,
    pub perm: PermFactor,
}

pub struct PermFactor {
    pub orig_indices: Vec<usize>,
    pub perm_indices: Vec<usize>,
}

pub struct DiagBoxFactor<T: RlstScalar> {
    pub arr: DiagBoxArr<T>,
    pub inds: Vec<usize>,
}

pub enum Factor<Item: RlstScalar> {
    Lu(LuFactor<Item>),
    Id(IdFactor<Item>),
    Diag(DiagBoxFactor<Item>),
}

pub struct RsrsFactors<Item: RlstScalar> {
    pub num_levels: usize,
    pub id_factors: LevelIdFactors<Item>,
    pub lu_factors: LevelLuFactors<Item>,
    pub near_field_inds: LevelNearFieldInds,
    pub perm_factor: PermFactor,
    pub diag_box_factors: DiagBoxFactors<Item>,
}

pub struct RsrsMulType {
    pub side: RsrsSide,
    pub factor_type: FactorType,
    pub right_trans: bool,
}
pub struct FactorMulType {
    pub side: Side,
    pub factor_type: FactorType,
    pub right_trans: bool,
}

type DiagBoxFactors<T> = CommutativeFactors<T>;
type LevelLuFactors<T> = Vec<Vec<CommutativeFactors<T>>>;
type LevelIdFactors<T> = Vec<CommutativeFactors<T>>;
type LevelNearFieldInds = Vec<Vec<Vec<usize>>>;
pub type CommutativeFactors<Item> = Vec<Factor<Item>>;

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

fn get_far_indices(n: usize, near_indices: Vec<usize>) -> Vec<usize> {
    let near_set: HashSet<usize> = near_indices.into_iter().collect();
    (0..n).filter(|x| !near_set.contains(x)).collect()
}

fn null_sketch_near_field<
    Item: RlstScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + RandScalar + MatrixLu + MatrixQr,
>(
    target_inds: &[usize],
    near_field_inds: &[usize],
    sketch: &DynamicArray<Item, 2>,
    test: &DynamicArray<Item, 2>,
    subs_sample_dim: usize,
    rsrs_options: &RsrsOptions<Item>,
) -> DynamicArray<Item, 2>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    QrDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixQrDecomposition<Item = Item>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let dim = test.shape()[1];
    let sub_test = test.r().into_subview([0, 0], [subs_sample_dim, dim]);
    let sub_sketch = sketch.r().into_subview([0, 0], [subs_sample_dim, dim]);
    let mut sketch_t = <Extraction<Item> as MatrixExtraction>::new(
        &sub_sketch,
        ExtInsType::Axis(target_inds.to_vec(), 1, false),
    )
    .unwrap()
    .ext;
    let test_n = <Extraction<Item> as MatrixExtraction>::new(
        &sub_test,
        ExtInsType::Axis(near_field_inds.to_vec(), 1, false),
    )
    .unwrap()
    .ext;
    nullify_near_sketch(&test_n, &mut sketch_t, &rsrs_options.id_options);
    sketch_t
}

fn null_near_field<
    Item: RlstScalar + MatrixId + MatrixInverse + MatrixPseudoInverse + RandScalar + MatrixLu + MatrixQr,
>(
    target_inds: &Vec<usize>,
    near_field_inds: &Vec<usize>,
    y_data: &SketchData<Item>,
    z_data: &SketchData<Item>,
    subs_sample_dim: usize,
    rsrs_options: &RsrsOptions<Item>,
) -> DynamicArray<Item, 2>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    QrDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixQrDecomposition<Item = Item>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let far_field_sketch = if rsrs_options.hermitian {
        null_sketch_near_field(
            target_inds,
            near_field_inds,
            &y_data.sketch,
            &y_data.test,
            subs_sample_dim,
            rsrs_options,
        )
    } else {
        let null_y_sketch = null_sketch_near_field(
            target_inds,
            near_field_inds,
            &y_data.sketch,
            &y_data.test,
            subs_sample_dim,
            rsrs_options,
        );
        let null_z_sketch = null_sketch_near_field(
            target_inds,
            near_field_inds,
            &z_data.sketch,
            &z_data.test,
            subs_sample_dim,
            rsrs_options,
        );
        let mut sketch_sum = empty_array();
        sketch_sum.fill_from_resize(null_y_sketch.r() + null_z_sketch.r());
        sketch_sum
    };

    far_field_sketch
}

fn near_box_extraction<Item: RlstScalar + MatrixPseudoInverse + MatrixLu>(
    ind_r: &[usize],
    near_field_inds: &[usize],
    sketch_data: &SketchData<Item>,
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
    let dim = sketch_data.test.shape()[1];
    let test_subview = sketch_data
        .test
        .r()
        .into_subview([0, 0], [subs_sample_dim, dim]);
    let sketch_subview = sketch_data
        .sketch
        .r()
        .into_subview([0, 0], [subs_sample_dim, dim]);
    let start = Instant::now();
    let sketch_r: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        &sketch_subview,
        ExtInsType::Axis(ind_r.to_vec(), 1, false),
    )
    .unwrap()
    .ext;
    let mut test_n: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        &test_subview,
        ExtInsType::Axis(near_field_inds.to_vec(), 1, false),
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
            ExtInsType::Axis(r_numbering.to_vec(), 0, false),
        )
        .unwrap()
        .ext;
        data_n = <Extraction<Item> as MatrixExtraction>::new(
            &mut near_box,
            ExtInsType::Axis(t_numbering.to_vec(), 0, false),
        )
        .unwrap()
        .ext;
    } else {
        data_r = <Extraction<Item> as MatrixExtraction>::new(
            &mut near_box,
            ExtInsType::Axis(r_numbering.to_vec(), 0, false),
        )
        .unwrap()
        .ext;
        data_n = <Extraction<Item> as MatrixExtraction>::new(
            &mut near_box,
            ExtInsType::Axis(t_numbering.to_vec(), 0, false),
        )
        .unwrap()
        .ext;
    }
    let lu_small_io_time = start.elapsed();
    lu_io_time += lu_small_io_time;
    (data_r, data_n, (lu_io_time, lu_b_ext_time))
}

pub trait FactorOperations: Sized {
    type Item: RlstScalar;
    fn new(
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        y_data: &SketchData<Self::Item>,
        z_data: &SketchData<Self::Item>,
        subs_sample_dim: usize,
        rank_par: &BoxType<Real<Self::Item>>,
        options: &RsrsOptions<Self::Item>,
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
        mul_type: &FactorMulType,
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
        mul_type: &FactorMulType,
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
        mul_type: &FactorMulType,
    );
}

impl<
        Item: RlstScalar
            + MatrixId
            + MatrixInverse
            + MatrixPseudoInverse
            + RandScalar
            + MatrixLu
            + MatrixQr,
    > FactorOperations for IdFactor<Item>
{
    type Item = Item;

    fn new(
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        y_data: &SketchData<Self::Item>,
        z_data: &SketchData<Self::Item>,
        subs_sample_dim: usize,
        rank_par: &BoxType<Real<Self::Item>>,
        options: &RsrsOptions<Self::Item>,
    ) -> (Option<Self>, Times)
    where
        StandardNormal: Distribution<Item::Real>,
        Standard: Distribution<Item::Real>,
        LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
            MatrixLuDecomposition<Item = Item>,
        QrDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
            MatrixQrDecomposition<Item = Item>,
    {
        let start: Instant = Instant::now();
        let test_shape = [subs_sample_dim, near_field_inds.len()];
        let sketch_shape = [subs_sample_dim, target_inds.len()];
        let null_shape = [test_shape[0] - test_shape[1], sketch_shape[1]];

        let far_field_sketch = null_near_field(
            &target_inds,
            &near_field_inds,
            y_data,
            z_data,
            subs_sample_dim,
            options,
        );

        
        let nullification_time: Duration = start.elapsed();
        let start: Instant = Instant::now();
        let max_rank: usize = *far_field_sketch.shape().iter().min().unwrap();
        let id_sketch = match rank_par {
            BoxType::Full(tol) => far_field_sketch.into_subview([0, 0], null_shape)
                .into_id_alloc(Accuracy::Tol(*tol), TransMode::Trans)
                .unwrap(),
            BoxType::Merged(rank) => far_field_sketch.into_subview([0, 0], null_shape)
                .into_id_alloc(Accuracy::FixedRank(*rank), TransMode::Trans)
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
        mul_type: &FactorMulType,
    ) {
        let target_block = self.mul_data(target_arr, factor_options, mul_type);
        let t_arr_mutex = std::sync::Mutex::new(target_arr);
        self.ins_data(
            &target_block,
            &mut *t_arr_mutex.lock().unwrap(),
            factor_options,
            mul_type,
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
        mul_type: &FactorMulType,
    ) -> DynamicArray<Self::Item, 2> {
        let mut trans = options.trans;

        match mul_type.factor_type {
            FactorType::F => {}
            FactorType::S => {
                trans = !trans;
            }
        }

        match mul_type.side {
            Side::Left => row_ops_no_sub(
                self.ind_s.clone(),
                self.ind_r.clone(),
                &self.data,
                target_arr,
                options.inv,
                trans,
                mul_type.right_trans,
            ),
            Side::Right => col_ops_no_sub(
                self.ind_s.clone(),
                self.ind_r.clone(),
                &self.data,
                target_arr,
                options.inv,
                trans,
                mul_type.right_trans,
            ),
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
        mul_type: &FactorMulType,
    ) {
        let mut trans = options.trans;

        match mul_type.factor_type {
            FactorType::F => {}
            FactorType::S => {
                trans = !trans;
            }
        }

        match mul_type.side {
            Side::Left => {
                row_subs(
                    self.ind_s.clone(),
                    self.ind_r.clone(),
                    source_arr,
                    target_arr,
                    trans,
                    mul_type.right_trans,
                );
            }
            Side::Right => {
                col_subs(
                    self.ind_s.clone(),
                    self.ind_r.clone(),
                    source_arr,
                    target_arr,
                    trans,
                    mul_type.right_trans,
                );
            }
        }
    }
}

impl<Item: RlstScalar + MatrixInverse + MatrixPseudoInverse + MatrixLu> FactorOperations
    for LuFactor<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    type Item = Item;

    fn new(
        ind_r: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        y_data: &SketchData<Self::Item>,
        z_data: &SketchData<Self::Item>,
        subs_sample_dim: usize,
        _rank_par: &BoxType<Real<Self::Item>>,
        options: &RsrsOptions<Self::Item>,
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

        let (y_r, y_n, (_y_lu_io_time, y_lu_b_ext_time)) = near_box_extraction(
            ind_r,
            near_field_inds,
            y_data,
            subs_sample_dim,
            options.tol_ext_near,
            &r_numbering,
            &t_numbering,
        );

        let start = Instant::now();
        let mut u_arr: DynamicArray<Self::Item, 2> = empty_array();
        /*y_r.r_mut().into_inverse_alloc().unwrap();

        u_arr.r_mut().mult_into_resize(
            TransMode::Trans,
            TransMode::Trans,
            num::One::one(),
            y_r.r(),
            y_n.r(),
            num::Zero::zero(),
        );*/

        let mut y_r_trans = empty_array();
        y_r_trans.fill_from_resize(y_r.r().transpose());
        let mut y_n_trans = empty_array();
        y_n_trans.fill_from_resize(y_n.r().transpose());

        let normal = NormalEquations::new(&y_r_trans, options.tol_lu);
        u_arr
            .r_mut()
            .fill_from_resize(normal.solve_normal_equations(&y_n_trans));

        let u_assembly = start.elapsed();

        let mut l_arr: DynamicArray<Self::Item, 2> = empty_array();

        let lu_b_ext_time;
        let lu_assembly_time;

        if !options.hermitian {
            let (z_r, z_n, (_z_lu_io_time, z_lu_b_ext_time)) = near_box_extraction(
                ind_r,
                near_field_inds,
                z_data,
                subs_sample_dim,
                options.tol_ext_near,
                &r_numbering,
                &t_numbering,
            );

            let start = Instant::now();
            /*
            let mut aux: DynamicArray<Self::Item, 2> = empty_array();
            z_r.r_mut().into_inverse_alloc().unwrap();
            aux.r_mut().simple_mult_into_resize(z_n.r(), z_r.r());
            l_arr.r_mut().fill_from_resize(aux.r().conj());*/

            let normal = NormalEquations::new(&z_r, options.tol_lu);
            l_arr
                .r_mut()
                .fill_from_resize(normal.solve_normal_equations(&z_n));

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

        let hermitian = options.hermitian;
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
        mul_type: &FactorMulType,
    ) {
        let target_block = self.mul_data(target_arr, factor_options, mul_type);

        let t_arr_mutex = std::sync::Mutex::new(target_arr);
        self.ins_data(
            &target_block,
            &mut *t_arr_mutex.lock().unwrap(),
            factor_options,
            mul_type,
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
        mul_type: &FactorMulType,
    ) -> DynamicArray<Self::Item, 2> {
        let mut trans = options.trans;
        if self.hermitian {
            match mul_type.factor_type {
                FactorType::F => {
                    trans = !trans;
                }
                FactorType::S => {}
            }

            match mul_type.side {
                Side::Left => row_ops_no_sub(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    &self.u_arr,
                    target_arr,
                    options.inv,
                    trans,
                    mul_type.right_trans,
                ),
                Side::Right => col_ops_no_sub(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    &self.u_arr,
                    target_arr,
                    options.inv,
                    trans,
                    mul_type.right_trans,
                ),
            }
        } else {
            match mul_type.factor_type {
                FactorType::F => match mul_type.side {
                    Side::Left => row_ops_no_sub(
                        self.ind_r.clone(),
                        self.ind_t.clone(),
                        &self.l_arr,
                        target_arr,
                        options.inv,
                        options.trans,
                        mul_type.right_trans,
                    ),
                    Side::Right => col_ops_no_sub(
                        self.ind_r.clone(),
                        self.ind_t.clone(),
                        &self.l_arr,
                        target_arr,
                        options.inv,
                        options.trans,
                        mul_type.right_trans,
                    ),
                },
                FactorType::S => match mul_type.side {
                    Side::Left => row_ops_no_sub(
                        self.ind_t.clone(),
                        self.ind_r.clone(),
                        &self.u_arr,
                        target_arr,
                        options.inv,
                        options.trans,
                        mul_type.right_trans,
                    ),
                    Side::Right => col_ops_no_sub(
                        self.ind_t.clone(),
                        self.ind_r.clone(),
                        &self.u_arr,
                        target_arr,
                        options.inv,
                        options.trans,
                        mul_type.right_trans,
                    ),
                },
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
        mul_type: &FactorMulType,
    ) {
        let mut trans = options.trans;

        if self.hermitian {
            match mul_type.factor_type {
                FactorType::F => {
                    trans = !trans;
                }
                FactorType::S => {}
            }

            match mul_type.side {
                Side::Left => row_subs(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    source_arr,
                    target_arr,
                    trans,
                    mul_type.right_trans,
                ),
                Side::Right => col_subs(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    source_arr,
                    target_arr,
                    trans,
                    mul_type.right_trans,
                ),
            };
        } else {
            match mul_type.factor_type {
                FactorType::F => {
                    match mul_type.side {
                        Side::Left => row_subs(
                            self.ind_r.clone(),
                            self.ind_t.clone(),
                            source_arr,
                            target_arr,
                            options.trans,
                            mul_type.right_trans,
                        ),
                        Side::Right => col_subs(
                            self.ind_r.clone(),
                            self.ind_t.clone(),
                            source_arr,
                            target_arr,
                            options.trans,
                            mul_type.right_trans,
                        ),
                    };
                }
                FactorType::S => {
                    match mul_type.side {
                        Side::Left => row_subs(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            source_arr,
                            target_arr,
                            options.trans,
                            mul_type.right_trans,
                        ),
                        Side::Right => col_subs(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            source_arr,
                            target_arr,
                            options.trans,
                            mul_type.right_trans,
                        ),
                    };
                }
            }
        }
    }
}

impl PermFactor {
    fn new(orig_indices: Vec<usize>, perm_indices: Vec<usize>) -> RlstResult<Self> {
        Ok(Self {
            orig_indices,
            perm_indices,
        })
    }

    pub fn left_mul<
        Item: RlstScalar,
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        right_arr: &mut Array<Item, ArrayImplMut, 2>,
        options: &FactorOptions,
    ) {
        let orig_indices: Vec<_> = (0..right_arr.shape()[0]).collect();
        assert_eq!(orig_indices.len(), self.perm_indices.len());
        let mut trans = options.trans;
        if options.inv {
            trans = !trans;
        }
        row_perm(
            orig_indices.clone(),
            self.perm_indices.clone(),
            right_arr,
            trans,
        );
    }

    pub fn right_mul<
        Item: RlstScalar,
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        left_arr: &mut Array<Item, ArrayImplMut, 2>,
        options: &FactorOptions,
    ) {
        let orig_indices: Vec<_> = (0..left_arr.shape()[1]).collect();
        assert_eq!(orig_indices.len(), self.perm_indices.len());
        let mut trans = !options.trans;
        if options.inv {
            trans = !trans;
        }
        col_perm(
            orig_indices.clone(),
            self.perm_indices.clone(),
            left_arr,
            trans,
        );
    }
}

fn add_diagonal<Item: RlstScalar>(
    arr: &mut DynamicArray<Item, 2>,
    val: <Item as rlst::RlstScalar>::Real,
) {
    let shape = arr.shape();
    let mut view = arr.r_mut();
    for i in 0..shape[0] {
        view[[i, i]] += Item::from_real(val);
    }
}

impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse> DiagBoxArr<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    fn new<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccess<Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        inds: &Vec<usize>,
        tol_lstq: <Item as RlstScalar>::Real,
        sub_test: &Array<Item, ArrayImpl, 2>,
        sub_sketch: &Array<Item, ArrayImpl, 2>,
    ) -> Self {
        let sketch_r: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            sub_sketch,
            ExtInsType::Axis(inds.to_vec(), 1, false),
        )
        .unwrap()
        .ext;
        let test_c: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            sub_test,
            ExtInsType::Axis(inds.to_vec(), 1, false),
        )
        .unwrap()
        .ext;
        let mut diag_box = right_least_squares(&test_c, &sketch_r, tol_lstq);
        let shape = diag_box.shape();
        add_diagonal(&mut diag_box, tol_lstq);
        let lu = <Item as MatrixLu>::into_lu_alloc(diag_box).unwrap();
        let mut l = rlst_dynamic_array2!(Item, shape);
        let mut u = rlst_dynamic_array2!(Item, shape);

        <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_l(&lu, l.r_mut());
        <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_u(&lu, u.r_mut());

        let perm = <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_perm(&lu);

        let orig: Vec<_> = (0..shape[1]).collect();
        Self {
            l_arr: TriangularMatrix::new(&l, TriangularType::Lower).unwrap(),
            u_arr: TriangularMatrix::new(&u, TriangularType::Upper).unwrap(),
            perm: PermFactor::new(orig, perm).unwrap(),
        }
    }

    fn left_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        right_arr: &mut Array<Item, ArrayImplMut, 2>,
        factor_options: &FactorOptions,
    ) {
        if factor_options.inv {
            if factor_options.trans {
                <TriangularMatrix<Item> as TriangularOperations>::solve(
                    &self.u_arr,
                    right_arr,
                    Side::Left,
                    TransMode::ConjTrans,
                );
                <TriangularMatrix<Item> as TriangularOperations>::solve(
                    &self.l_arr,
                    right_arr,
                    Side::Left,
                    TransMode::ConjNoTrans,
                );
                self.perm.left_mul(right_arr, factor_options);
            } else {
                self.perm.left_mul(right_arr, factor_options);
                <TriangularMatrix<Item> as TriangularOperations>::solve(
                    &self.l_arr,
                    right_arr,
                    Side::Left,
                    TransMode::NoTrans,
                );
                <TriangularMatrix<Item> as TriangularOperations>::solve(
                    &self.u_arr,
                    right_arr,
                    Side::Left,
                    TransMode::NoTrans,
                );
            }
        } else {
            if factor_options.trans {
                self.perm.left_mul(right_arr, factor_options);
                <TriangularMatrix<Item> as TriangularOperations>::mul(
                    &self.l_arr,
                    right_arr,
                    Side::Left,
                    TransMode::ConjTrans,
                );
                <TriangularMatrix<Item> as TriangularOperations>::mul(
                    &self.u_arr,
                    right_arr,
                    Side::Left,
                    TransMode::ConjTrans,
                );
            } else {
                <TriangularMatrix<Item> as TriangularOperations>::mul(
                    &self.u_arr,
                    right_arr,
                    Side::Left,
                    TransMode::NoTrans,
                );
                <TriangularMatrix<Item> as TriangularOperations>::mul(
                    &self.l_arr,
                    right_arr,
                    Side::Left,
                    TransMode::NoTrans,
                );
                self.perm.left_mul(right_arr, factor_options);
            }
        }
    }

    fn right_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        right_arr: &mut Array<Item, ArrayImplMut, 2>,
        factor_options: &FactorOptions,
    ) {
        if factor_options.inv {
            if factor_options.trans {
                self.perm.right_mul(right_arr, factor_options);
                <TriangularMatrix<Item> as TriangularOperations>::solve(
                    &self.l_arr,
                    right_arr,
                    Side::Right,
                    TransMode::ConjTrans,
                );
                <TriangularMatrix<Item> as TriangularOperations>::solve(
                    &self.u_arr,
                    right_arr,
                    Side::Right,
                    TransMode::ConjTrans,
                );
            } else {
                <TriangularMatrix<Item> as TriangularOperations>::solve(
                    &self.u_arr,
                    right_arr,
                    Side::Right,
                    TransMode::NoTrans,
                );
                <TriangularMatrix<Item> as TriangularOperations>::solve(
                    &self.l_arr,
                    right_arr,
                    Side::Right,
                    TransMode::NoTrans,
                );

                self.perm.right_mul(right_arr, factor_options);
            }
        } else {
            if factor_options.trans {
                <TriangularMatrix<Item> as TriangularOperations>::mul(
                    &self.u_arr,
                    right_arr,
                    Side::Left,
                    TransMode::ConjTrans,
                );
                <TriangularMatrix<Item> as TriangularOperations>::mul(
                    &self.l_arr,
                    right_arr,
                    Side::Left,
                    TransMode::ConjTrans,
                );
                self.perm.left_mul(right_arr, factor_options);
            } else {
                self.perm.right_mul(right_arr, factor_options);
                <TriangularMatrix<Item> as TriangularOperations>::mul(
                    &self.l_arr,
                    right_arr,
                    Side::Right,
                    TransMode::NoTrans,
                );
                <TriangularMatrix<Item> as TriangularOperations>::mul(
                    &self.u_arr,
                    right_arr,
                    Side::Right,
                    TransMode::NoTrans,
                );
            }
        }
    }

    pub fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        right_arr: &mut Array<Item, ArrayImplMut, 2>,
        side: Side,
        factor_options: &FactorOptions,
    ) {
        match side {
            Side::Left => self.left_mul(right_arr, factor_options),
            Side::Right => self.right_mul(right_arr, factor_options),
        }
    }
}

impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse> FactorOperations for DiagBoxFactor<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    type Item = Item;
    fn new(
        rows: &mut Vec<usize>,
        _cols: &mut Vec<usize>,
        y_data: &SketchData<Self::Item>,
        _z_data: &SketchData<Self::Item>,
        subs_sample_dim: usize,
        _rank_par: &BoxType<Real<Self::Item>>,
        options: &RsrsOptions<Self::Item>,
    ) -> (Option<Self>, Times) {
        let (sub_test, sub_sketch) = (
            y_data
                .test
                .r()
                .into_subview([0, 0], [subs_sample_dim, y_data.dim]),
            y_data
                .sketch
                .r()
                .into_subview([0, 0], [subs_sample_dim, y_data.dim]),
        );

        let diag_times = LuTimes {
            //TODO: change this to diag_times
            extraction: 0_u128,
            lu: 0_u128,
        };

        let times = Times::Lu(diag_times);

        (
            Some(Self {
                arr: DiagBoxArr::new(&rows, options.tol_diag_ext, &sub_test, &sub_sketch),
                inds: rows.clone(),
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
        mul_type: &FactorMulType,
    ) {
        let target_block = self.mul_data(target_arr, factor_options, mul_type);
        let t_arr_mutex = std::sync::Mutex::new(target_arr);
        self.ins_data(
            &target_block,
            &mut *t_arr_mutex.lock().unwrap(),
            factor_options,
            mul_type,
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
        mul_type: &FactorMulType,
    ) -> DynamicArray<Self::Item, 2> {
        let trans = options.trans;

        match mul_type.side {
            Side::Left => {
                let mut target_rows = ext_rows(
                    self.inds.clone(),
                    self.inds.clone(),
                    target_arr,
                    trans,
                    mul_type.right_trans,
                );
                self.arr.mul(&mut target_rows, Side::Left, options);
                target_rows
            }
            Side::Right => {
                let mut target_cols = ext_cols(
                    self.inds.clone(),
                    self.inds.clone(),
                    target_arr,
                    trans,
                    mul_type.right_trans,
                );
                self.arr.mul(&mut target_cols, Side::Right, options);
                target_cols
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
        mul_type: &FactorMulType,
    ) {
        let trans = options.trans;

        match mul_type.side {
            Side::Left => {
                row_subs(
                    self.inds.clone(),
                    self.inds.clone(),
                    source_arr,
                    target_arr,
                    trans,
                    mul_type.right_trans,
                );
            }
            Side::Right => {
                col_subs(
                    self.inds.clone(),
                    self.inds.clone(),
                    source_arr,
                    target_arr,
                    trans,
                    mul_type.right_trans,
                );
            }
        }
    }
}

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
        mul_type: &FactorMulType,
    );
}

impl<
        Item: RlstScalar
            + MatrixId
            + MatrixInverse
            + MatrixPseudoInverse
            + RandScalar
            + MatrixLu
            + MatrixQr,
    > CommutativeFactorsOperations for CommutativeFactors<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
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
        mul_type: &FactorMulType,
    ) where
        Self: Sized,
    {
        let updated_t_arr_blocks: Vec<_> = self
            .par_iter()
            .enumerate()
            .map(|(factor_ind, factor)| {
                let target_block = match factor {
                    Factor::Lu(lu_factor) => {
                        lu_factor.mul_data(target_arr, &factor_options, mul_type)
                    }
                    Factor::Id(id_factor) => {
                        id_factor.mul_data(target_arr, &factor_options, mul_type)
                    }
                    Factor::Diag(diag_factor) => {
                        diag_factor.mul_data(target_arr, &factor_options, mul_type)
                    }
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
                        mul_type,
                    ),
                    Factor::Id(id_factor) => id_factor.ins_data(
                        target_block,
                        &mut *t_arr_mutex.lock().unwrap(),
                        &factor_options,
                        mul_type,
                    ),
                    Factor::Diag(diag_factor) => diag_factor.ins_data(
                        target_block,
                        &mut *t_arr_mutex.lock().unwrap(),
                        &factor_options,
                        mul_type,
                    ),
                };
            });
    }
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
        mul_type: &FactorMulType,
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
        mul_type: &FactorMulType,
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
        mul_type: RsrsMulType,
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

impl<
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
    > RsrsFactorsOps for RsrsFactors<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    type Item = Item;

    fn new(num_levels: usize) -> Self {
        let mut id_factors = Vec::new();
        id_factors.resize_with(num_levels, || Vec::new());
        let mut lu_factors = Vec::new();
        lu_factors.resize_with(num_levels, || Vec::new());
        let mut near_field_inds = Vec::new();
        near_field_inds.resize_with(num_levels, || Vec::new());
        let orig_indices = Vec::new();
        let perm_indices = Vec::new();
        let perm_factor = PermFactor::new(orig_indices, perm_indices).unwrap();
        let diag_box_factors = DiagBoxFactors::new();
        Self {
            num_levels,
            near_field_inds,
            id_factors,
            lu_factors,
            perm_factor,
            diag_box_factors,
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
        mul_type: &FactorMulType,
        factor_options: &FactorOptions,
        level_it: usize,
    ) {
        let id_batch = &self.id_factors[level_it];
        id_batch.mul(target_arr, factor_options, &mul_type);
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
        mul_type: &FactorMulType,
        factor_options: &FactorOptions,
        dec: bool,
        level_it: usize,
    ) {
        let num_lu_batches = self.lu_factors[level_it].len();

        if dec {
            (0..num_lu_batches).rev().for_each(|batch_ind| {
                let lu_batch = &self.lu_factors[level_it][batch_ind];
                lu_batch.mul(target_arr, factor_options, &mul_type);
            });
        } else {
            (0..num_lu_batches).for_each(|batch_ind| {
                let lu_batch = &self.lu_factors[level_it][batch_ind];
                lu_batch.mul(target_arr, factor_options, &mul_type);
            });
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
        mul_type: RsrsMulType,
        factor_options: &FactorOptions,
        dec: bool,
    ) {
        let levels = (0..self.num_levels).collect::<Vec<_>>();
        if matches!(mul_type.side, RsrsSide::Squeeze) {
            let left_mul_type = FactorMulType {
                side: Side::Left,
                factor_type: FactorType::F,
                right_trans: mul_type.right_trans,
            };

            let right_mul_type = FactorMulType {
                side: Side::Right,
                factor_type: FactorType::S,
                right_trans: mul_type.right_trans,
            };

            levels.iter().for_each(|&level_it| {
                self.apply_id_level(target_arr, &left_mul_type, &factor_options, level_it);
                self.apply_id_level(target_arr, &right_mul_type, &factor_options, level_it);
                self.apply_lu_level(target_arr, &left_mul_type, &factor_options, dec, level_it);
                self.apply_lu_level(target_arr, &right_mul_type, &factor_options, dec, level_it);
            });
        } else {
            let factor_mul_type = if matches!(mul_type.side, RsrsSide::Left) {
                FactorMulType {
                    side: Side::Left,
                    factor_type: mul_type.factor_type.clone(),
                    right_trans: mul_type.right_trans,
                }
            } else {
                FactorMulType {
                    side: Side::Right,
                    factor_type: mul_type.factor_type.clone(),
                    right_trans: mul_type.right_trans,
                }
            };

            if dec {
                levels.iter().rev().for_each(|&level_it| {
                    self.apply_lu_level(
                        target_arr,
                        &factor_mul_type,
                        &factor_options,
                        dec,
                        level_it,
                    );
                    self.apply_id_level(target_arr, &factor_mul_type, &factor_options, level_it);
                });
            } else {
                levels.iter().for_each(|&level_it| {
                    self.apply_id_level(target_arr, &factor_mul_type, &factor_options, level_it);
                    self.apply_lu_level(
                        target_arr,
                        &factor_mul_type,
                        &factor_options,
                        dec,
                        level_it,
                    );
                });
            }
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
                    mul_type_1 = RsrsMulType {
                        side: RsrsSide::Left,
                        factor_type: FactorType::S,
                        right_trans: false,
                    };
                    mul_type_2 = RsrsMulType {
                        side: RsrsSide::Left,
                        factor_type: FactorType::F,
                        right_trans: false,
                    };
                } else {
                    mul_type_1 = RsrsMulType {
                        side: RsrsSide::Left,
                        factor_type: FactorType::F,
                        right_trans: false,
                    };
                    mul_type_2 = RsrsMulType {
                        side: RsrsSide::Left,
                        factor_type: FactorType::S,
                        right_trans: false,
                    };
                }

                let diag_mul_type = FactorMulType {
                    side: Side::Left,
                    factor_type: FactorType::F,
                    right_trans: false,
                }; //TODO: CHECK IF CORRECT
                self.el_factors_mul(target_arr, mul_type_1, factor_options, false);
                self.diag_box_factors
                    .mul(target_arr, &factor_options, &diag_mul_type);
                self.el_factors_mul(target_arr, mul_type_2, factor_options, true);
            }
            RsrsSide::Right => {
                let mul_type_1;
                let mul_type_2;
                if !factor_options.inv {
                    mul_type_1 = RsrsMulType {
                        side: RsrsSide::Right,
                        factor_type: FactorType::F,
                        right_trans: false,
                    };
                    mul_type_2 = RsrsMulType {
                        side: RsrsSide::Right,
                        factor_type: FactorType::S,
                        right_trans: false,
                    };
                } else {
                    mul_type_1 = RsrsMulType {
                        side: RsrsSide::Right,
                        factor_type: FactorType::S,
                        right_trans: false,
                    };
                    mul_type_2 = RsrsMulType {
                        side: RsrsSide::Right,
                        factor_type: FactorType::F,
                        right_trans: false,
                    };
                }

                let diag_mul_type = FactorMulType {
                    side: Side::Right,
                    factor_type: FactorType::F,
                    right_trans: false,
                }; //TODO: CHECK IF CORRECT

                self.el_factors_mul(target_arr, mul_type_1, factor_options, false);
                self.diag_box_factors
                    .mul(target_arr, &factor_options, &diag_mul_type);
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
