use crate::rsrs::rsrs_factors::base_factors::{
    condition_number, BaseFactorOptions, ComposedFactorData, CondType, DiagBoxType, FactorData,
    LuSMat, RectArr, RegSMat, SquareArr,
};
use crate::rsrs::rsrs_factors::null_and_extract::{
    near_box_extraction, null_near_field, ExtractOptions, IdOptions, PivotMethod,
};
use crate::rsrs::rsrs_factors::rsrs_operator::FactType;
use crate::rsrs::rsrs_factors::statistics::{IdTimes, LuTimes, Times};
use crate::rsrs::sketch::SketchData;
use crate::utils::linear_algebra::add_diagonal;
use crate::utils::{
    data_ins_ext::{ExtInsType, Extraction, MatrixExtraction},
    elementary_matrix::{col_subs, ext_cols, ext_rows, row_perm, row_subs},
    linear_algebra::block_extraction,
};
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    ThreadPoolBuilder,
};
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
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

type Real<T> = <T as rlst::RlstScalar>::Real;

#[derive(Debug, Clone)]
pub enum BoxType<Item: RlstScalar> {
    Merged(usize),
    Full(Real<Item>),
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

#[derive(Clone, PartialEq, Debug)]
pub enum FactorType {
    F,
    S,
}

#[derive(Clone, Debug)]
pub struct MulOptions {
    pub base_options: BaseFactorOptions,
    pub side: Side,
    pub factor_type: FactorType,
}

pub struct IdFactor<T: RlstScalar> {
    data: FactorData<T>,
    pub perm: Vec<usize>,
    pub ind_r: Vec<usize>, //row_indices
    pub ind_s: Vec<usize>, //col_indices
    pub ind_f: Vec<usize>,
}

pub struct LuFactor<T: RlstScalar> {
    l_arr: FactorData<T>,
    pub u_arr: FactorData<T>,
    symmetric: bool,
    pub ind_r: Vec<usize>, //cols
    pub ind_t: Vec<usize>, //rows
}

type DiagBoxArr<T> = DiagBoxType<T>;

pub struct DiagBoxFactor<T: RlstScalar> {
    pub arr: DiagBoxArr<T>,
    pub inds: Vec<usize>,
}

pub struct PermFactor {
    pub orig_indices: Vec<usize>,
    pub perm_indices: Vec<usize>,
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
    pub fact_type: FactType,
    pub dim: usize,
    pub num_threads: usize,
}

pub enum LevelIdFactors<T: RlstScalar> {
    Single(Vec<CommutativeFactors<T>>),
    Batched(BatchedFactors<T>),
}

pub type DiagBoxFactors<T> = CommutativeFactors<T>;
type BatchedFactors<T> = Vec<Vec<CommutativeFactors<T>>>;
type LevelLuFactors<T> = BatchedFactors<T>;
type LevelNearFieldInds = Vec<Vec<Vec<usize>>>;

pub type CommutativeFactors<Item> = Vec<Factor<Item>>;

fn get_far_indices(n: usize, near_indices: Vec<usize>) -> Vec<usize> {
    let near_set: HashSet<usize> = near_indices.into_iter().collect();
    (0..n).filter(|x| !near_set.contains(x)).collect()
}

pub trait FactorOperations: Sized {
    type Item: RlstScalar;

    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        //side: &Side,
        options: &MulOptions,
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
        side: &Side,
        factor_type: Option<FactorType>,
        options: &BaseFactorOptions,
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
        side: &Side,
        factor_type: Option<FactorType>,
        options: &BaseFactorOptions,
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
    > IdFactor<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    pub fn new(
        target_inds: &mut [usize],
        near_field_inds: &mut [usize],
        y_data: &SketchData<Item>,
        z_data: &SketchData<Item>,
        subs_sample_dim: usize,
        rank_par: &BoxType<Real<Item>>,
        id_options: &IdOptions<Item>,
        symmetric: bool,
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
            target_inds,
            near_field_inds,
            y_data,
            z_data,
            subs_sample_dim,
            symmetric,
            id_options,
        );

        let nullification_time: Duration = start.elapsed();
        let start: Instant = Instant::now();
        let max_rank: usize = *far_field_sketch.shape().iter().min().unwrap();
        let id_sketch = match rank_par {
            BoxType::Full(tol) => {
                if *tol < num::One::one() {
                    far_field_sketch
                        .into_subview([0, 0], null_shape)
                        .into_id_alloc(
                            Accuracy::Tol(*tol),
                            id_options.qr_method.clone(),
                            TransMode::Trans,
                        )
                        .unwrap()
                } else {
                    let loc_rank = null_shape[1].min(num::ToPrimitive::to_usize(tol).unwrap());
                    far_field_sketch
                        .into_subview([0, 0], null_shape)
                        .into_id_alloc(
                            Accuracy::FixedRank(loc_rank),
                            id_options.qr_method.clone(),
                            TransMode::Trans,
                        )
                        .unwrap()
                }
            }
            BoxType::Merged(rank) => far_field_sketch
                .into_subview([0, 0], null_shape)
                .into_id_alloc(
                    Accuracy::FixedRank(*rank),
                    id_options.qr_method.clone(),
                    TransMode::Trans,
                )
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
            let ind_f = get_far_indices(y_data.dim, near_field_inds.to_vec());
            if ind_f.len() > 0 {
                let aux_indices = target_inds.to_vec();

                for (id, &elem) in id_sketch.perm.iter().enumerate() {
                    let val = aux_indices[elem];
                    target_inds[id] = val;
                    near_field_inds[id] = val;
                }
            }
            ind_r.extend_from_slice(&target_inds[k..]);
            ind_s.extend_from_slice(&target_inds[..k]);

            (
                Some(Self {
                    data: FactorData::Reg(RectArr {
                        arr: id_sketch.id_mat,
                        //apply_transposed: false,
                    }),
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

    pub fn cond(&self) -> (CondType<Item>, Option<CondType<Item>>) {
        let (dim, max_entry) = match &self.data {
            FactorData::Comp(_composed_factor_data) => todo!(),
            FactorData::Reg(rectg) => {
                let [rows, cols] = rectg.arr.r().shape();
                let dim = rows.min(cols);
                let max_entry = rectg
                    .arr
                    .r()
                    .data()
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, |a, b| {
                        let a_c: f64 = num::NumCast::from(a).unwrap();
                        let b_c: f64 = num::NumCast::from(b).unwrap();
                        a_c.max(b_c)
                    });
                (Item::real(dim), Item::real(max_entry))
            }
        };

        (self.data.cond(), Some(((dim, max_entry), None)))
    }
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
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    type Item = Item;

    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        //side: &Side,
        options: &MulOptions,
    ) {
        let aux_options = match options.factor_type {
            FactorType::F => options.base_options.clone(),
            FactorType::S => options.base_options.transpose(),
        };
        let target_block = self.mul_data(target_arr, &options.side, None, &aux_options);
        let t_arr_mutex = std::sync::Mutex::new(target_arr);
        self.ins_data(
            &target_block,
            *t_arr_mutex.lock().unwrap(),
            &options.side,
            None,
            &aux_options,
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
        side: &Side,
        _factor_type: Option<FactorType>,
        options: &BaseFactorOptions,
    ) -> DynamicArray<Self::Item, 2> {
        self.data
            .mul(target_arr, side, options, &self.ind_s, &self.ind_r)
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
        side: &Side,
        _factor_type: Option<FactorType>,
        options: &BaseFactorOptions,
    ) {
        match side {
            Side::Left => {
                row_subs(
                    self.ind_s.clone(),
                    self.ind_r.clone(),
                    source_arr,
                    target_arr,
                    options,
                );
            }
            Side::Right => {
                col_subs(
                    self.ind_s.clone(),
                    self.ind_r.clone(),
                    source_arr,
                    target_arr,
                    options,
                );
            }
        }
    }
}

pub fn inv_diagonal<Item: RlstScalar>(arr: &DynamicArray<Item, 2>) -> DynamicArray<Item, 2> {
    let shape = arr.shape();
    let mut d_inv = rlst_dynamic_array2!(Item, shape);
    let mut view_1 = d_inv.r_mut();
    let view_2 = arr.r();
    for i in 0..shape[0] {
        view_1[[i, i]] = <Item as num::One>::one() / view_2[[i, i]];
    }
    d_inv
}

fn extract_lu_factor<Item: RlstScalar + MatrixInverse + MatrixLu>(
    data_r: DynamicArray<Item, 2>,
    data_n: DynamicArray<Item, 2>,
    pivot_method: &PivotMethod,
    //apply_transposed: bool,
) -> FactorData<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    match pivot_method {
        PivotMethod::DirectInversion => {
            let mut y_r_inv = empty_array();
            y_r_inv.r_mut().fill_from_resize(data_r.r().transpose());
            y_r_inv.r_mut().into_inverse_alloc().unwrap();
            let mut rectg = empty_array();
            rectg.fill_from_resize(data_n.transpose());

            let sq = RegSMat {
                arr: data_r,
                inv_arr: y_r_inv,
            };
            let factor = ComposedFactorData {
                sq: SquareArr::Reg(sq),
                rectg: RectArr {
                    arr: rectg,
                    //apply_transposed,
                },
                //apply_transposed,
            };
            FactorData::Comp(factor)
        }
        PivotMethod::Lu(alpha) => {
            let shape = data_r.shape();
            let mut data_r_trans = empty_array();
            data_r_trans.fill_from_resize(data_r.r().transpose());
            add_diagonal(&mut data_r_trans, Item::real(*alpha));
            let lu: LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>> =
                <Item as MatrixLu>::into_lu_alloc(data_r_trans).unwrap();
            let mut l = rlst_dynamic_array2!(Item, shape);
            let mut u = rlst_dynamic_array2!(Item, shape);
            <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_l(&lu, l.r_mut());
            <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_u(&lu, u.r_mut());

            let perm = <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_perm(&lu);

            let orig: Vec<_> = (0..shape[1]).collect();

            let lu_arr = LuSMat {
                l_arr: TriangularMatrix::new(&l, TriangularType::Lower).unwrap(),
                u_arr: TriangularMatrix::new(&u, TriangularType::Upper).unwrap(),
                perm: PermFactor::new(orig, perm).unwrap(),
            };

            let sq = SquareArr::Lu(lu_arr);
            let mut rectg = empty_array();
            rectg.fill_from_resize(data_n.transpose());
            let factor = ComposedFactorData {
                sq,
                rectg: RectArr {
                    arr: rectg,
                    //apply_transposed,
                },
                //apply_transposed,
            };
            FactorData::Comp(factor)
        }
    }
}

impl<Item: RlstScalar + MatrixInverse + MatrixPseudoInverse + MatrixLu> LuFactor<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    pub fn new(
        ind_r: &[usize],
        near_field_inds: &[usize],
        inactive_inds: &[usize],
        y_data: &SketchData<Item>,
        z_data: &SketchData<Item>,
        subs_sample_dim: usize,
        lu_options: &ExtractOptions<Item>,
        symmetric: bool,
    ) -> (Option<Self>, Times)
    where
        LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
            MatrixLuDecomposition<Item = Item>,
        TriangularMatrix<Item>: TriangularOperations<Item = Item>,
    {
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
            if !ind_r.contains(&elem) && !inactive_inds.contains(&elem) {
                t_numbering.push(pos);
                ind_t.push(elem);
            }
        }
        let (y_r, y_n, (_y_lu_io_time, y_lu_b_ext_time)) = near_box_extraction(
            ind_r,
            near_field_inds,
            y_data,
            subs_sample_dim,
            &lu_options,
            &r_numbering,
            &t_numbering,
        );
        let start = Instant::now();
        let u_arr = extract_lu_factor(y_r, y_n, &lu_options.pivot_method); //, false);
        let u_assembly = start.elapsed();

        let lu_b_ext_time;
        let lu_assembly_time;

        let l_arr = if !symmetric {
            let (z_r, z_n, (_z_lu_io_time, z_lu_b_ext_time)) = near_box_extraction(
                ind_r,
                near_field_inds,
                z_data,
                subs_sample_dim,
                &lu_options,
                &r_numbering,
                &t_numbering,
            );

            let start = Instant::now();

            let l_arr = extract_lu_factor(z_r, z_n, &lu_options.pivot_method); //, true);
            let l_assembly = start.elapsed();
            lu_b_ext_time = y_lu_b_ext_time + z_lu_b_ext_time;
            lu_assembly_time = u_assembly + l_assembly;
            l_arr
        } else {
            lu_b_ext_time = y_lu_b_ext_time;
            lu_assembly_time = u_assembly;
            FactorData::Reg(RectArr {
                arr: empty_array(),
                //apply_transposed: false,
            })
        };
        let lu_times = LuTimes {
            extraction: lu_b_ext_time.as_millis(),
            lu: lu_assembly_time.as_millis(),
        };

        let times = Times::Lu(lu_times);
        (
            Some(Self {
                l_arr,
                u_arr,
                symmetric,
                ind_r: ind_r.to_vec(),
                ind_t,
            }),
            times,
        )
    }

    pub fn cond(&self) -> (CondType<Item>, Option<CondType<Item>>) {
        if !self.symmetric {
            (self.l_arr.cond(), Some(self.u_arr.cond()))
        } else {
            (self.u_arr.cond(), None)
        }
    }
}

impl<Item: RlstScalar + MatrixInverse + MatrixPseudoInverse + MatrixLu> FactorOperations
    for LuFactor<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    type Item = Item;

    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        options: &MulOptions,
    ) {
        let aux_options = match options.factor_type {
            FactorType::F => options.base_options.transpose(),
            FactorType::S => options.base_options.clone(),
        };
        let target_block = self.mul_data(
            target_arr,
            &options.side,
            Some(options.factor_type.clone()),
            &aux_options,
        );

        let t_arr_mutex = std::sync::Mutex::new(target_arr);
        self.ins_data(
            &target_block,
            *t_arr_mutex.lock().unwrap(),
            &options.side,
            Some(options.factor_type.clone()),
            &aux_options,
        );
    }

    fn mul_data<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &Array<Self::Item, ArrayImpl, 2>,
        side: &Side,
        factor_type: Option<FactorType>,
        options: &BaseFactorOptions,
    ) -> DynamicArray<Self::Item, 2> {
        if self.symmetric {
            self.u_arr
                .mul(target_arr, side, options, &self.ind_t, &self.ind_r)
        } else {
            match factor_type {
                Some(FactorType::F) => {
                    self.l_arr
                        .mul(target_arr, side, options, &self.ind_t, &self.ind_r)
                }
                Some(FactorType::S) => {
                    self.u_arr
                        .mul(target_arr, side, options, &self.ind_t, &self.ind_r)
                }
                None => todo!(),
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
        side: &Side,
        factor_type: Option<FactorType>,
        options: &BaseFactorOptions,
    ) {
        if self.symmetric {
            match side {
                Side::Left => row_subs(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    source_arr,
                    target_arr,
                    options,
                ),
                Side::Right => col_subs(
                    self.ind_t.clone(),
                    self.ind_r.clone(),
                    source_arr,
                    target_arr,
                    options,
                ),
            };
        } else {
            match factor_type {
                Some(FactorType::F) => {
                    match side {
                        Side::Left => row_subs(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            source_arr,
                            target_arr,
                            options,
                        ),
                        Side::Right => col_subs(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            source_arr,
                            target_arr,
                            options,
                        ),
                    };
                }
                Some(FactorType::S) => {
                    match side {
                        Side::Left => row_subs(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            source_arr,
                            target_arr,
                            options,
                        ),
                        Side::Right => col_subs(
                            self.ind_t.clone(),
                            self.ind_r.clone(),
                            source_arr,
                            target_arr,
                            options,
                        ),
                    };
                }
                None => todo!(),
            }
        }
    }
}

impl PermFactor {
    pub fn new(orig_indices: Vec<usize>, perm_indices: Vec<usize>) -> RlstResult<Self> {
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
        options: &BaseFactorOptions,
    ) {
        let orig_indices: Vec<_> = (0..right_arr.shape()[0]).collect();
        assert_eq!(orig_indices.len(), self.perm_indices.len());
        let trans = if options.inv {
            let aux_options = options.transpose();
            aux_options.trans_val()
        } else {
            options.trans_val()
        };

        row_perm(
            orig_indices.clone(),
            self.perm_indices.clone(),
            right_arr,
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
        inds: &[usize],
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
                inv_arr.fill_from_resize(diag_box.r().transpose());
                inv_arr.r_mut().into_inverse_alloc().unwrap();
                let reg_arr = RegSMat {
                    arr: diag_box,
                    inv_arr,
                };
                DiagBoxType::Reg(reg_arr)
            }
            PivotMethod::Lu(alpha) => {
                let shape = diag_box.shape();
                let mut inv_arr = empty_array();
                inv_arr.fill_from_resize(diag_box.r().transpose());
                add_diagonal(&mut inv_arr, Item::real(alpha));
                let lu = <Item as MatrixLu>::into_lu_alloc(inv_arr).unwrap();
                let mut l = rlst_dynamic_array2!(Item, shape);
                let mut u = rlst_dynamic_array2!(Item, shape);

                <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_l(&lu, l.r_mut());
                <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_u(&lu, u.r_mut());

                let perm = <LuDecomposition<Item, _> as MatrixLuDecomposition>::get_perm(&lu);

                let orig: Vec<_> = (0..shape[1]).collect();

                let lu_arr = LuSMat {
                    l_arr: TriangularMatrix::new(&l, TriangularType::Lower).unwrap(),
                    u_arr: TriangularMatrix::new(&u, TriangularType::Upper).unwrap(),
                    perm: PermFactor::new(orig, perm).unwrap(),
                };
                DiagBoxType::Lu(lu_arr)
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
        factor_options: &BaseFactorOptions,
    ) {
        match self {
            DiagBoxType::Reg(ref reg) => {
                /*let trans_mode = if factor_options.trans {
                    TransMode::Trans
                } else {
                    TransMode::NoTrans
                };*/

                let mut new_right_arr = empty_array();
                if factor_options.inv {
                    new_right_arr.r_mut().mult_into_resize(
                        factor_options.trans,
                        TransMode::NoTrans,
                        num::One::one(),
                        reg.inv_arr.r(),
                        right_arr.r(),
                        num::Zero::zero(),
                    );
                } else {
                    new_right_arr.r_mut().mult_into_resize(
                        factor_options.trans,
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
                    match factor_options.trans {
                        TransMode::NoTrans => {
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
                        TransMode::ConjNoTrans => todo!(),
                        TransMode::Trans => {
                            <TriangularMatrix<Item> as TriangularOperations>::solve(
                                &lu.u_arr,
                                right_arr,
                                Side::Left,
                                TransMode::Trans,
                            );
                            <TriangularMatrix<Item> as TriangularOperations>::solve(
                                &lu.l_arr,
                                right_arr,
                                Side::Left,
                                TransMode::Trans,
                            );
                            lu.perm.left_mul(right_arr, factor_options);
                        }
                        TransMode::ConjTrans => todo!(),
                    }
                } else {
                    match factor_options.trans {
                        TransMode::NoTrans => {
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
                        TransMode::ConjNoTrans => todo!(),
                        TransMode::Trans => {
                            lu.perm.left_mul(right_arr, factor_options);
                            <TriangularMatrix<Item> as TriangularOperations>::mul(
                                &lu.l_arr,
                                right_arr,
                                Side::Left,
                                TransMode::Trans,
                            );
                            <TriangularMatrix<Item> as TriangularOperations>::mul(
                                &lu.u_arr,
                                right_arr,
                                Side::Left,
                                TransMode::Trans,
                            );
                        }
                        TransMode::ConjTrans => todo!(),
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
        arr: &mut Array<Item, ArrayImplMut, 2>,
        side: &Side,
        factor_options: &BaseFactorOptions,
    ) {
        match side {
            Side::Left => self.left_mul(arr, factor_options),
            Side::Right => {
                let mut aux_arr = empty_array();
                aux_arr.r_mut().fill_from_resize(arr.r().transpose());
                let aux_factor_options = factor_options.transpose();
                self.left_mul(&mut aux_arr, &aux_factor_options);
                arr.fill_from(aux_arr.r().transpose());
            }
        }
    }
}

impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse + MatrixInverse> DiagBoxFactor<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    pub fn new(
        rows: &mut [usize],
        y_data: &SketchData<Item>,
        subs_sample_dim: usize,
        options: &ExtractOptions<Item>,
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
                arr: DiagBoxArr::new(rows, &options, &sub_test, &sub_sketch),
                inds: rows.to_vec(),
            }),
            times,
        )
    }

    pub fn cond(&self) -> (CondType<Item>, Option<CondType<Item>>) {
        match &self.arr {
            DiagBoxType::Reg(reg_dbox) => ((condition_number(&reg_dbox.arr), None), None),
            DiagBoxType::Lu(lu_dbox) => (
                (
                    (num::Zero::zero(), num::Zero::zero()),
                    Some((
                        condition_number(&lu_dbox.l_arr.tri),
                        condition_number(&lu_dbox.u_arr.tri),
                    )),
                ),
                None,
            ),
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

    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        target_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        factor_options: &MulOptions,
    ) {
        let target_block = self.mul_data(
            target_arr,
            &factor_options.side,
            None,
            &factor_options.base_options,
        );
        let t_arr_mutex = std::sync::Mutex::new(target_arr);
        self.ins_data(
            &target_block,
            *t_arr_mutex.lock().unwrap(),
            &factor_options.side,
            None,
            &factor_options.base_options,
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
        side: &Side,
        _factor_type: Option<FactorType>,
        factor_options: &BaseFactorOptions,
    ) -> DynamicArray<Self::Item, 2> {
        match side {
            Side::Left => {
                let mut target_rows = ext_rows(
                    self.inds.clone(),
                    self.inds.clone(),
                    target_arr,
                    &factor_options,
                );
                self.arr.mul(&mut target_rows, &Side::Left, &factor_options);
                target_rows
            }
            Side::Right => {
                let mut target_cols = ext_cols(
                    self.inds.clone(),
                    self.inds.clone(),
                    target_arr,
                    &factor_options,
                );
                self.arr.mul(&mut target_cols, &Side::Right, factor_options);
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
        side: &Side,
        _factor_type: Option<FactorType>,
        factor_options: &BaseFactorOptions,
    ) {
        match side {
            Side::Left => {
                row_subs(
                    self.inds.clone(),
                    self.inds.clone(),
                    source_arr,
                    target_arr,
                    factor_options,
                );
            }
            Side::Right => {
                col_subs(
                    self.inds.clone(),
                    self.inds.clone(),
                    source_arr,
                    target_arr,
                    factor_options,
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
        num_threads: usize,
        factor_options: &MulOptions,
    );
    #[allow(clippy::type_complexity)]
    fn get_condition_numbers(&self) -> Vec<(CondType<Self::Item>, Option<CondType<Self::Item>>)>;
    fn flush(&mut self);
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
        num_threads: usize,
        factor_options: &MulOptions,
    ) where
        Self: Sized,
    {
        let pool_threads = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        let updated_t_arr_blocks: Vec<_> = pool_threads.install(|| {
            self.par_iter()
                .enumerate()
                .map(|(factor_ind, factor)| {
                    let target_block = match factor {
                        Factor::Lu(lu_factor) => {
                            let base_options = match factor_options.factor_type {
                                FactorType::F => factor_options.base_options.transpose(),
                                FactorType::S => factor_options.base_options.clone(),
                            };
                            lu_factor.mul_data(
                                target_arr,
                                &factor_options.side,
                                Some(factor_options.factor_type.clone()),
                                &base_options,
                            )
                        }
                        Factor::Id(id_factor) => {
                            let base_options = match factor_options.factor_type {
                                FactorType::F => factor_options.base_options.clone(),
                                FactorType::S => factor_options.base_options.transpose(),
                            };
                            id_factor.mul_data(
                                target_arr,
                                &factor_options.side,
                                Some(factor_options.factor_type.clone()),
                                &base_options,
                            )
                        }
                        Factor::Diag(diag_factor) => diag_factor.mul_data(
                            target_arr,
                            &factor_options.side,
                            Some(factor_options.factor_type.clone()),
                            &factor_options.base_options,
                        ),
                    };
                    (factor_ind, target_block)
                })
                .collect()
        });

        let t_arr_mutex = std::sync::Mutex::new(target_arr);
        pool_threads.install(|| {
            updated_t_arr_blocks
                .par_iter()
                .for_each(|(factor_ind, target_block)| {
                    let factor = &self[*factor_ind];
                    match factor {
                        Factor::Lu(lu_factor) => {
                            let base_options = match factor_options.factor_type {
                                FactorType::F => factor_options.base_options.transpose(),
                                FactorType::S => factor_options.base_options.clone(),
                            };
                            lu_factor.ins_data(
                                target_block,
                                *t_arr_mutex.lock().unwrap(),
                                &factor_options.side,
                                Some(factor_options.factor_type.clone()),
                                &base_options,
                            )
                        }
                        Factor::Id(id_factor) => {
                            let base_options = match factor_options.factor_type {
                                FactorType::F => factor_options.base_options.clone(),
                                FactorType::S => factor_options.base_options.transpose(),
                            };
                            id_factor.ins_data(
                                target_block,
                                *t_arr_mutex.lock().unwrap(),
                                &factor_options.side,
                                Some(factor_options.factor_type.clone()),
                                &base_options,
                            )
                        }
                        Factor::Diag(diag_factor) => diag_factor.ins_data(
                            target_block,
                            *t_arr_mutex.lock().unwrap(),
                            &factor_options.side,
                            Some(factor_options.factor_type.clone()),
                            &factor_options.base_options,
                        ),
                    };
                });
        });
    }

    fn get_condition_numbers(&self) -> Vec<(CondType<Self::Item>, Option<CondType<Self::Item>>)> {
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

    fn flush(&mut self) {
        self.clear();
        self.shrink_to_fit();
    }
}
