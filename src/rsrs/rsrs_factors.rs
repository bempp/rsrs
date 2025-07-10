use super::{
    rsrs_cycle::{BoxType, ExtractOptions, RsrsOptions},
    sketch::SketchData,
};
use crate::{
    rsrs::sketch::SamplingSpace,
    utils::{
        data_ins_ext::{ExtInsType, Extraction, MatrixExtraction},
        elementary_matrix::{
            col_ops_no_sub, col_perm, col_subs, ext_cols, ext_rows, row_ops_no_sub, row_perm,
            row_subs,
        },
        least_squares_and_null::{block_extraction, nullify_near_sketch},
    },
};
use itertools::min;
use mpi::{
    topology::SimpleCommunicator,
    traits::{Communicator, Equivalence},
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
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
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

pub struct LuDBox<T: RlstScalar> {
    pub u_arr: TriangularMatrix<T>,
    pub l_arr: TriangularMatrix<T>,
    pub perm: PermFactor,
}

pub struct RegDBox<T: RlstScalar> {
    pub arr: DynamicArray<T, 2>,
    pub inv_arr: DynamicArray<T, 2>,
}

pub enum DiagBoxType<T: RlstScalar> {
    Reg(RegDBox<T>),
    Lu(LuDBox<T>),
}

type DiagBoxArr<T> = DiagBoxType<T>;
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
    pub dim: usize,
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

pub fn condition_number<Item: RlstScalar + MatrixSvd>(mat: &DynamicArray<Item, 2>) -> Real<Item> {
    let shape = mat.shape();
    let dim: usize = min(shape).unwrap();
    let mut singular_values: DynamicArray<Real<Item>, 1> = rlst_dynamic_array1!(Real<Item>, [dim]);
    let mode: SvdMode = SvdMode::Reduced;
    let mut u: DynamicArray<Item, 2> = rlst_dynamic_array2!(Item, [shape[0], dim]);
    let mut vt: DynamicArray<Item, 2> = rlst_dynamic_array2!(Item, [dim, shape[1]]);

    let mut aux_data = empty_array();
    aux_data.fill_from_resize(mat.r());

    aux_data
        .r_mut()
        .into_svd_alloc(u.r_mut(), vt.r_mut(), singular_values.data_mut(), mode)
        .unwrap();

    let sigma_max = singular_values[[0]];
    let sigma_min = singular_values[[dim - 1]];

    sigma_max / sigma_min
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
    lu_options: &ExtractOptions<Item>,
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
    let mut near_box = block_extraction(&mut test_n, &sketch_r, lu_options);
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

    fn cond(&self) -> (Real<Self::Item>, Real<Self::Item>);
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
            BoxType::Full(tol) => {
                if *tol < num::One::one() {
                    far_field_sketch
                        .into_subview([0, 0], null_shape)
                        .into_id_alloc(Accuracy::Tol(*tol), TransMode::Trans)
                        .unwrap()
                } else {
                    far_field_sketch
                        .into_subview([0, 0], null_shape)
                        .into_id_alloc(
                            Accuracy::FixedRank(num::ToPrimitive::to_usize(tol).unwrap()),
                            TransMode::Trans,
                        )
                        .unwrap()
                }
            }
            BoxType::Merged(rank) => far_field_sketch
                .into_subview([0, 0], null_shape)
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

    fn cond(&self) -> (Real<Self::Item>, Real<Self::Item>) {
        (condition_number(&self.data), num::Zero::zero())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum PivotMethod {
    DirectInversion,
    Lu,
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

        let (mut y_r, y_n, (_y_lu_io_time, y_lu_b_ext_time)) = near_box_extraction(
            ind_r,
            near_field_inds,
            y_data,
            subs_sample_dim,
            &options.lu_options,
            &r_numbering,
            &t_numbering,
        );

        println!("Lu cond numbers: {}, {}", condition_number(&y_r), condition_number(&y_n));


        let start = Instant::now();
        let mut u_arr: DynamicArray<Self::Item, 2> = empty_array();

        match options.lu_options.pivot_method {
            PivotMethod::DirectInversion => {
                y_r.r_mut().into_inverse_alloc().unwrap();
                u_arr.r_mut().mult_into_resize(
                    TransMode::Trans,
                    TransMode::Trans,
                    num::One::one(),
                    y_r.r(),
                    y_n.r(),
                    num::Zero::zero(),
                );
            }
            PivotMethod::Lu => {
                //let mut y_r_trans = empty_array();
                //y_r_trans.fill_from_resize(y_r.r().transpose());
                let mut y_n_trans = empty_array();
                y_n_trans.fill_from_resize(y_n.r().transpose());

                let lu = <Item as MatrixLu>::into_lu_alloc(y_r).unwrap();

                let _ = lu.solve_mat(TransMode::Trans, y_n_trans.r_mut());

                //let normal = NormalEquations::new(&y_r_trans, tol);
                u_arr.r_mut().fill_from_resize(y_n_trans.r());
            }
        }

        let u_assembly = start.elapsed();

        let mut l_arr: DynamicArray<Self::Item, 2> = empty_array();

        let lu_b_ext_time;
        let lu_assembly_time;

        if !options.hermitian {
            let (mut z_r, mut z_n, (_z_lu_io_time, z_lu_b_ext_time)) = near_box_extraction(
                ind_r,
                near_field_inds,
                z_data,
                subs_sample_dim,
                &options.lu_options,
                &r_numbering,
                &t_numbering,
            );

            let start = Instant::now();

            match options.lu_options.pivot_method {
                PivotMethod::DirectInversion => {
                    let mut aux: DynamicArray<Self::Item, 2> = empty_array();
                    z_r.r_mut().into_inverse_alloc().unwrap();
                    aux.r_mut().simple_mult_into_resize(z_n.r(), z_r.r());
                    l_arr.r_mut().fill_from_resize(aux.r().conj());
                }
                PivotMethod::Lu => {
                    let lu = <Item as MatrixLu>::into_lu_alloc(z_r).unwrap();
                    let _ = lu.solve_mat(TransMode::NoTrans, z_n.r_mut());
                    l_arr.r_mut().fill_from_resize(z_n.r());
                }
            };

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

    fn cond(&self) -> (Real<Self::Item>, Real<Self::Item>) {
        let sigma_1 = if !self.hermitian {
            condition_number(&self.l_arr)
        } else {
            num::Zero::zero()
        };

        (sigma_1, condition_number(&self.u_arr))
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

fn _add_diagonal<Item: RlstScalar>(
    arr: &mut DynamicArray<Item, 2>,
    val: <Item as rlst::RlstScalar>::Real,
) {
    let shape = arr.shape();
    let mut view = arr.r_mut();
    for i in 0..shape[0] {
        view[[i, i]] += Item::from_real(val);
    }
}

impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse + MatrixInverse> DiagBoxArr<Item>
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
        db_ext_options: &ExtractOptions<Item>,
        sub_test: &Array<Item, ArrayImpl, 2>,
        sub_sketch: &Array<Item, ArrayImpl, 2>,
    ) -> Self {
        let sketch_r: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            sub_sketch,
            ExtInsType::Axis(inds.to_vec(), 1, false),
        )
        .unwrap()
        .ext;
        let mut test_c: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            sub_test,
            ExtInsType::Axis(inds.to_vec(), 1, false),
        )
        .unwrap()
        .ext;
        let diag_box = block_extraction(&mut test_c, &sketch_r, db_ext_options);

        match db_ext_options.pivot_method {
            PivotMethod::DirectInversion => {
                let mut inv_arr = empty_array();
                inv_arr.fill_from_resize(diag_box.r().transpose().conj());
                inv_arr.r_mut().into_inverse_alloc().unwrap();
                let reg_arr = RegDBox {
                    arr: diag_box,
                    inv_arr,
                };
                return DiagBoxType::Reg(reg_arr);
            }
            PivotMethod::Lu => {
                let shape = diag_box.shape();
                let mut inv_arr = empty_array();
                inv_arr.fill_from_resize(diag_box.r().transpose().conj());
                let lu = <Item as MatrixLu>::into_lu_alloc(inv_arr).unwrap();
                let mut l = rlst_dynamic_array2!(Item, shape);
                let mut u = rlst_dynamic_array2!(Item, shape);

                <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_l(&lu, l.r_mut());
                <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_u(&lu, u.r_mut());

                let perm = <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_perm(&lu);

                let orig: Vec<_> = (0..shape[1]).collect();

                let lu_arr = LuDBox {
                    l_arr: TriangularMatrix::new(&l, TriangularType::Lower).unwrap(),
                    u_arr: TriangularMatrix::new(&u, TriangularType::Upper).unwrap(),
                    perm: PermFactor::new(orig, perm).unwrap(),
                };
                return DiagBoxType::Lu(lu_arr);
            }
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
        match self {
            DiagBoxType::Reg(ref reg) => {
                let trans_mode = if factor_options.trans {
                    TransMode::ConjTrans
                } else {
                    TransMode::NoTrans
                };

                let mut new_right_arr = empty_array();
                if factor_options.inv {
                    new_right_arr.r_mut().mult_into_resize(
                        trans_mode,
                        TransMode::NoTrans,
                        num::One::one(),
                        reg.inv_arr.r(),
                        right_arr.r(),
                        num::Zero::zero(),
                    );
                } else {
                    new_right_arr.r_mut().mult_into_resize(
                        trans_mode,
                        TransMode::NoTrans,
                        num::One::one(),
                        reg.arr.r(),
                        right_arr.r(),
                        num::Zero::zero(),
                    );
                }
                right_arr.r_mut().fill_from(new_right_arr.r());
            }
            DiagBoxType::Lu(ref lu) => {
                if factor_options.inv {
                    if factor_options.trans {
                        <TriangularMatrix<Item> as TriangularOperations>::solve(
                            &lu.u_arr,
                            right_arr,
                            Side::Left,
                            TransMode::ConjTrans,
                        );
                        <TriangularMatrix<Item> as TriangularOperations>::solve(
                            &lu.l_arr,
                            right_arr,
                            Side::Left,
                            TransMode::ConjTrans,
                        );
                        lu.perm.left_mul(right_arr, factor_options);
                    } else {
                        lu.perm.left_mul(right_arr, factor_options);
                        <TriangularMatrix<Item> as TriangularOperations>::solve(
                            &lu.l_arr,
                            right_arr,
                            Side::Left,
                            TransMode::NoTrans,
                        );
                        <TriangularMatrix<Item> as TriangularOperations>::solve(
                            &lu.u_arr,
                            right_arr,
                            Side::Left,
                            TransMode::NoTrans,
                        );
                    }
                } else {
                    if factor_options.trans {
                        lu.perm.left_mul(right_arr, factor_options);
                        <TriangularMatrix<Item> as TriangularOperations>::mul(
                            &lu.l_arr,
                            right_arr,
                            Side::Left,
                            TransMode::ConjTrans,
                        );
                        <TriangularMatrix<Item> as TriangularOperations>::mul(
                            &lu.u_arr,
                            right_arr,
                            Side::Left,
                            TransMode::ConjTrans,
                        );
                    } else {
                        <TriangularMatrix<Item> as TriangularOperations>::mul(
                            &lu.u_arr,
                            right_arr,
                            Side::Left,
                            TransMode::NoTrans,
                        );
                        <TriangularMatrix<Item> as TriangularOperations>::mul(
                            &lu.l_arr,
                            right_arr,
                            Side::Left,
                            TransMode::NoTrans,
                        );
                        lu.perm.left_mul(right_arr, factor_options);
                    }
                }
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
        match self {
            DiagBoxType::Reg(ref reg) => {
                let trans_mode = if factor_options.trans {
                    TransMode::ConjTrans
                } else {
                    TransMode::NoTrans
                };

                let mut new_right_arr = empty_array();
                if factor_options.inv {
                    new_right_arr.r_mut().mult_into_resize(
                        TransMode::NoTrans,
                        trans_mode,
                        num::One::one(),
                        right_arr.r(),
                        reg.inv_arr.r(),
                        num::Zero::zero(),
                    );
                } else {
                    new_right_arr.r_mut().mult_into_resize(
                        TransMode::NoTrans,
                        trans_mode,
                        num::One::one(),
                        right_arr.r(),
                        reg.arr.r(),
                        num::Zero::zero(),
                    );
                }
                right_arr.r_mut().fill_from(new_right_arr.r());
            }
            DiagBoxType::Lu(ref lu) => {
                if factor_options.inv {
                    if factor_options.trans {
                        lu.perm.right_mul(right_arr, factor_options);
                        <TriangularMatrix<Item> as TriangularOperations>::solve(
                            &lu.l_arr,
                            right_arr,
                            Side::Right,
                            TransMode::ConjTrans,
                        );
                        <TriangularMatrix<Item> as TriangularOperations>::solve(
                            &lu.u_arr,
                            right_arr,
                            Side::Right,
                            TransMode::ConjTrans,
                        );
                    } else {
                        <TriangularMatrix<Item> as TriangularOperations>::solve(
                            &lu.u_arr,
                            right_arr,
                            Side::Right,
                            TransMode::NoTrans,
                        );
                        <TriangularMatrix<Item> as TriangularOperations>::solve(
                            &lu.l_arr,
                            right_arr,
                            Side::Right,
                            TransMode::NoTrans,
                        );

                        lu.perm.right_mul(right_arr, factor_options);
                    }
                } else {
                    if factor_options.trans {
                        <TriangularMatrix<Item> as TriangularOperations>::mul(
                            &lu.u_arr,
                            right_arr,
                            Side::Left,
                            TransMode::ConjTrans,
                        );
                        <TriangularMatrix<Item> as TriangularOperations>::mul(
                            &lu.l_arr,
                            right_arr,
                            Side::Left,
                            TransMode::ConjTrans,
                        );
                        lu.perm.left_mul(right_arr, factor_options);
                    } else {
                        lu.perm.right_mul(right_arr, factor_options);
                        <TriangularMatrix<Item> as TriangularOperations>::mul(
                            &lu.l_arr,
                            right_arr,
                            Side::Right,
                            TransMode::NoTrans,
                        );
                        <TriangularMatrix<Item> as TriangularOperations>::mul(
                            &lu.u_arr,
                            right_arr,
                            Side::Right,
                            TransMode::NoTrans,
                        );
                    }
                }
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

impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse + MatrixInverse> FactorOperations
    for DiagBoxFactor<Item>
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
                arr: DiagBoxArr::new(&rows, &options.extract_db_options, &sub_test, &sub_sketch),
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

    fn cond(&self) -> (Real<Self::Item>, Real<Self::Item>) {
        match &self.arr {
            DiagBoxType::Reg(reg_dbox) => {
                (condition_number(&reg_dbox.arr), num::Zero::zero())
            }
            DiagBoxType::Lu(lu_dbox) => {
                (condition_number(&lu_dbox.l_arr.tri), condition_number(&lu_dbox.u_arr.tri))
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
    fn get_condition_numbers(&self) -> Vec<(Real<Self::Item>, Real<Self::Item>)>;
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

    fn get_condition_numbers(&self) -> Vec<(Real<Self::Item>, Real<Self::Item>)> {
        let condition_numbers: Vec<_> = self
            .par_iter()
            .enumerate()
            .map(|(_factor_ind, factor)| match factor {
                Factor::Lu(lu_factor) => lu_factor.cond(),
                Factor::Id(id_factor) => id_factor.cond(),
                Factor::Diag(diag_factor) => diag_factor.cond(),
            })
            .collect();

        condition_numbers
    }
}
pub trait RsrsFactorsImpl<Item: RlstScalar>: Sized {
    //type Item: RlstScalar;
    fn new(num_levels: usize, dim: usize) -> Self;

    fn apply_id_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        mul_type: &FactorMulType,
        factor_options: &FactorOptions,
        level_it: usize,
    );

    fn apply_lu_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        mul_type: &FactorMulType,
        factor_options: &FactorOptions,
        dec: bool,
        level_it: usize,
    );

    fn el_factors_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        mul_type: RsrsMulType,
        factor_options: &FactorOptions,
        level: bool,
    );

    fn matmul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &mut self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        mul_type: RsrsSide,
        factor_options: &FactorOptions,
    );

    fn matvec(
        &self,
        x: &[Item],
        y: &mut [Item],
        mul_type: RsrsSide,
        factor_options: &mut FactorOptions,
    );

    fn perm_target_array<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
    );

    fn dim(&self) -> usize;

    fn get_condition_numbers(
        &self,
    ) -> (
        Vec<Vec<(Real<Item>, Real<Item>)>>,
        Vec<Vec<(Real<Item>, Real<Item>)>>,
        Vec<(Real<Item>, Real<Item>)>,
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
    > RsrsFactorsImpl<Item> for RsrsFactors<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    fn new(num_levels: usize, dim: usize) -> Self {
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
            dim,
        }
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn apply_id_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        mul_type: &FactorMulType,
        factor_options: &FactorOptions,
        level_it: usize,
    ) {
        let id_batch = &self.id_factors[level_it];
        id_batch.mul(target_arr, factor_options, &mul_type);
    }

    fn apply_lu_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
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
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
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

    fn matmul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &mut self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
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

    fn matvec(
        &self,
        x: &[Item],
        y: &mut [Item],
        side: RsrsSide,
        factor_options: &mut FactorOptions,
    ) {
        let target_arr = match side {
            RsrsSide::Squeeze => empty_array(),
            RsrsSide::Left => {
                let mut target_arr = rlst_dynamic_array2!(Item, [x.len(), 1]);
                for (i, val) in x.iter().enumerate() {
                    target_arr.r_mut()[[i, 0]] = *val;
                }
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
                self.el_factors_mul(&mut target_arr, mul_type_1, factor_options, false);
                self.diag_box_factors
                    .mul(&mut target_arr, &factor_options, &diag_mul_type);
                self.el_factors_mul(&mut target_arr, mul_type_2, factor_options, true);
                target_arr
            }
            RsrsSide::Right => {
                let mut target_arr = rlst_dynamic_array2!(Item, [1, x.len()]);

                for (i, val) in x.iter().enumerate() {
                    target_arr.r_mut()[[0, i]] = *val;
                }
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

                self.el_factors_mul(&mut target_arr, mul_type_1, factor_options, false);
                self.diag_box_factors
                    .mul(&mut target_arr, &factor_options, &diag_mul_type);
                self.el_factors_mul(&mut target_arr, mul_type_2, factor_options, true);
                target_arr
            }
        };

        for (i, val) in target_arr.r().iter().enumerate() {
            y[i] = val;
        }
    }

    fn perm_target_array<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
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

    fn get_condition_numbers(
        &self,
    ) -> (
        Vec<Vec<(Real<Item>, Real<Item>)>>,
        Vec<Vec<(Real<Item>, Real<Item>)>>,
        Vec<(Real<Item>, Real<Item>)>,
    ) {
        let mut id_condition_numbers = Vec::new();
        let mut lu_condition_numbers = Vec::new();
        for id_batch in self.id_factors.iter() {
            id_condition_numbers.push(id_batch.get_condition_numbers());
        }
        for lu_level_batches in self.lu_factors.iter() {
            let mut lu_level_condition_numbers = Vec::new();
            for lu_batch in lu_level_batches.iter() {
                lu_level_condition_numbers.extend_from_slice(&lu_batch.get_condition_numbers());
            }
            lu_condition_numbers.push(lu_level_condition_numbers);
        }
        let diag_condition_numbers = self.diag_box_factors.get_condition_numbers();

        (
            id_condition_numbers,
            lu_condition_numbers,
            diag_condition_numbers,
        )
    }
}

impl<Item: RlstScalar> Shape<2> for RsrsFactors<Item> {
    fn shape(&self) -> [usize; 2] {
        [self.dim, self.dim]
    }
}

pub struct RsrsOperator<
    'a,
    Item: RlstScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr,
    Space: SamplingSpace<F = Item>,
    Op: RsrsFactorsImpl<Item> + Shape<2>,
> {
    pub op: &'a Op,
    domain: Rc<Space>,
    range: Rc<Space>,
    inv: bool,
}

// Implement OperatorBase for RsrsOperator so it can be used with rlst::Operator
impl<
        'a,
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Space: SamplingSpace<F = Item> + LinearSpace,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > OperatorBase for RsrsOperator<'a, Item, Space, Op>
{
    type Domain = Space;
    type Range = Space;

    fn domain(&self) -> Rc<Self::Domain> {
        self.domain.clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.range.clone()
    }
}

impl<
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Space: SamplingSpace<F = Item>,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > std::fmt::Debug for RsrsOperator<'_, Item, Space, Op>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.op.shape();
        write!(f, "RsrsOperator: [{}x{}]", shape[0], shape[1]).unwrap();
        Ok(())
    }
}

pub trait LocalFromSpaces<
    'a,
    Item: RlstScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr,
    Space,
    Op,
>: Sized
{
    fn from_local_spaces(op: &'a Op, domain: Rc<Space>, range: Rc<Space>) -> Self;
}

pub trait Inv {
    fn inv(&mut self, inv: bool);
}

impl<
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Space: SamplingSpace<F = Item>,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > Inv for RsrsOperator<'_, Item, Space, Op>
where
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    fn inv(&mut self, inv: bool) {
        self.inv = inv;
    }
}

impl<
        'a,
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > LocalFromSpaces<'a, Item, ArrayVectorSpace<Item>, Op>
    for RsrsOperator<'a, Item, ArrayVectorSpace<Item>, Op>
where
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    fn from_local_spaces(
        op: &'a Op,
        domain: Rc<ArrayVectorSpace<Item>>,
        range: Rc<ArrayVectorSpace<Item>>,
    ) -> Self {
        RsrsOperator {
            op,
            domain: domain.clone(),
            range: range.clone(),
            inv: false,
        }
    }
}

impl<
        'a,
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr
            + Equivalence,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > LocalFromSpaces<'a, Item, DistributedArrayVectorSpace<'a, SimpleCommunicator, Item>, Op>
    for RsrsOperator<'a, Item, DistributedArrayVectorSpace<'a, SimpleCommunicator, Item>, Op>
where
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    fn from_local_spaces(
        op: &'a Op,
        domain: Rc<DistributedArrayVectorSpace<'a, SimpleCommunicator, Item>>,
        range: Rc<DistributedArrayVectorSpace<'a, SimpleCommunicator, Item>>,
    ) -> Self {
        RsrsOperator {
            op,
            domain: domain.clone(),
            range: range.clone(),
            inv: false,
        }
    }
}

impl<
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > AsApply for RsrsOperator<'_, Item, ArrayVectorSpace<Item>, Op>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
{
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        _alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        _beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
        trans_mode: TransMode,
    ) {
        match trans_mode {
            TransMode::NoTrans => {
                let mut factor_options = FactorOptions {
                    inv: self.inv,
                    trans: false,
                };

                // Reshape y to a 2D array before passing to mul
                self.op.matvec(
                    x.imp().view().data(),
                    y.imp_mut().view_mut().data_mut(),
                    RsrsSide::Left,
                    &mut factor_options,
                );
            }
            TransMode::ConjNoTrans => {
                panic!("TransMode::ConjNoTrans not supported for multiplication.")
            }
            TransMode::Trans => {
                let mut factor_options = FactorOptions {
                    inv: self.inv,
                    trans: false,
                };

                self.op.matvec(
                    x.imp().view().data(),
                    y.imp_mut().view_mut().data_mut(),
                    RsrsSide::Right,
                    &mut factor_options,
                );
            }
            TransMode::ConjTrans => {
                panic!("TransMode::ConjTrans not supported for multiplication.")
            }
        }
    }

    fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
        &self,
        x: Element<ContainerIn>,
        trans_mode: rlst::TransMode,
    ) -> rlst::operator::ElementType<<Self::Range as LinearSpace>::E> {
        let mut y = zero_element(self.range());
        self.apply_extended(
            <<Self::Range as LinearSpace>::F as num::One>::one(),
            x,
            <<Self::Range as LinearSpace>::F as num::Zero>::zero(),
            y.r_mut(),
            trans_mode,
        );
        y
    }
}

impl<
        C: Communicator,
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr
            + Equivalence,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > AsApply for RsrsOperator<'_, Item, DistributedArrayVectorSpace<'_, C, Item>, Op>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
{
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        _alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        _beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
        trans_mode: TransMode,
    ) {
        match trans_mode {
            TransMode::NoTrans => {
                let mut factor_options = FactorOptions {
                    inv: self.inv,
                    trans: false,
                };

                // Reshape y to a 2D array before passing to mul
                self.op.matvec(
                    x.imp().view().local().data(),
                    y.imp_mut().view_mut().local_mut().data_mut(),
                    RsrsSide::Left,
                    &mut factor_options,
                );
            }
            TransMode::ConjNoTrans => {
                panic!("TransMode::ConjNoTrans not supported for multiplication.")
            }
            TransMode::Trans => {
                let mut factor_options = FactorOptions {
                    inv: self.inv,
                    trans: false,
                };

                self.op.matvec(
                    x.imp().view().local().data(),
                    y.imp_mut().view_mut().local_mut().data_mut(),
                    RsrsSide::Right,
                    &mut factor_options,
                );
            }
            TransMode::ConjTrans => {
                panic!("TransMode::ConjTrans not supported for multiplication.")
            }
        }
    }

    fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
        &self,
        x: Element<ContainerIn>,
        trans_mode: rlst::TransMode,
    ) -> rlst::operator::ElementType<<Self::Range as LinearSpace>::E> {
        let mut y = zero_element(self.range());
        self.apply_extended(
            <<Self::Range as LinearSpace>::F as num::One>::one(),
            x,
            <<Self::Range as LinearSpace>::F as num::Zero>::zero(),
            y.r_mut(),
            trans_mode,
        );
        y
    }
}
