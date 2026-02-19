use crate::rsrs::rsrs_factors::base_factors::{
    condition_number, BaseFactorOptions, CondType, DiagBoxArr, FactorData, LuSMat, RectArr, RegSMat,
};
use crate::rsrs::rsrs_factors::null_and_extract::{
    extract_lu_factor, near_box_extraction, null_near_field, ExtractOptions, IdOptions, PivotMethod,
};
use crate::rsrs::rsrs_factors::rsrs_operator::FactType;
use crate::rsrs::sketch::SketchData;
use crate::rsrs::statistics::{IdTimes, LuTimes, Times};
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

/// BoxType marks a box as merged if it is an
/// union of boxes that have been compressed in previous levels.
#[derive(Debug, Clone)]
pub enum BoxType<Item: RlstScalar> {
    Merged(usize),
    Full(Real<Item>),
}

/// FactorType:
/// - F: first factor (E, U).
/// - S: second factor (F, L).
#[derive(Clone, PartialEq, Debug)]
pub enum FactorType {
    F,
    S,
}

/// MulOptions: Multiplication options for one or more factors.
#[derive(Clone, Debug)]
pub struct MulOptions {
    /// base_options: Indicate if a factor should be inverted, transposed or if in b= A*x x should be transposed.
    pub base_options: BaseFactorOptions,
    /// side: indicates if a factor should be applied by the left or by the right
    pub side: Side,
    /// factor_type: "first" or "second" factor
    pub factor_type: FactorType,
}

/// IdFactor: Obtained from Interpolative decomposition.
/// It represents an elementary operation: (I+/-F)
pub struct IdFactor<T: RlstScalar> {
    /// data: stores F
    data: FactorData<T>,
    /// perm: stores the permutation induced by the ID
    pub perm: Vec<usize>,
    /// ind_r: stores the residual columns or rows in A
    pub ind_r: Vec<usize>, //row_indices
    /// ind_s: stores the skeleton columns or rows in A
    pub ind_s: Vec<usize>, //col_indices
    /// stores the far field indices when necessary.
    pub ind_f: Vec<usize>,
}

/// LuFactor: Obtain from Block LU near field compression.
/// It represents an elementary operation: (I+/-F)
/// U and L are defined in the same factor.
pub struct LuFactor<T: RlstScalar> {
    /// l_arr: L in (I+/-L)
    l_arr: FactorData<T>,
    /// u_arr: U in (I+/-U)
    pub u_arr: FactorData<T>,
    /// symmetric: indicates if LU was performed in a symmetric matrix.
    /// (for a symmetric matrix we only store U)
    symmetric: bool,
    /// ind_r: residual indices
    pub ind_r: Vec<usize>, //cols
    /// ind_t: target indices
    pub ind_t: Vec<usize>, //rows
}

/// Diagonal factor after ID and LU factorisation have been applied.
pub struct DiagBoxFactor<T: RlstScalar> {
    /// arr: contains the information of the diagonal block matrix.
    pub arr: DiagBoxArr<T>,
    /// inds: contains the indices of the rows/columns to apply the block factor.
    pub inds: Vec<usize>,
}

/// Permutation factor: application that permutes selected rows/columns of a given matrix/vector.
pub struct PermFactor {
    /// orig_indices: original indices
    pub orig_indices: Vec<usize>,
    /// perm_indices: permuted indices
    pub perm_indices: Vec<usize>,
}

/// A factor can either be an ID factor, a LU factor or a diagonal block factor.
pub enum Factor<Item: RlstScalar> {
    Lu(LuFactor<Item>),
    Id(IdFactor<Item>),
    Diag(DiagBoxFactor<Item>),
}

/// RsrsFactors: collects the relevant information for the RSRS of a matrix.
pub struct RsrsFactors<Item: RlstScalar> {
    /// num_levels: depth of the tree
    pub num_levels: usize,
    /// id_factors: collection of ID factors sorted per level
    pub id_factors: MultiLevelIdFactors<Item>,
    /// lu_factors: collection of LU factors sorted per level
    pub lu_factors: MultiLevelLuFactors<Item>,
    /// perm_factor: permutation induced by RSRS
    pub perm_factor: PermFactor,
    /// diag_box_factors: block diagonal factors obtained from RSRS
    pub diag_box_factors: DiagBoxFactors<Item>,
    /// fact_type: either a split RSRS (first run all ID
    /// factors and then LU) or joint (run ID and LU together).
    /// split RSRS has better parallel properties, but introduces
    /// larger errors.
    pub fact_type: FactType,
    /// dim: dimension of the compressed operator
    pub dim: usize,
    /// num_threads: number of threads used in the factorisation
    /// and matrix-vector multiplication
    pub num_threads: usize,
}

/// CommmutativeFactors: batcn of factors of any kind that
/// can be applied in parallel
pub type CommutativeFactors<Item> = Vec<Factor<Item>>;

/// LevelFactors: batches inside of a level
type LevelFactors<T> = Vec<CommutativeFactors<T>>;
/// MultiLevelFactors: collection of the results for all levels
type MultiLevelFactors<T> = Vec<LevelFactors<T>>;

/// MultiLevelIdFactors: storage for ID factors
pub enum MultiLevelIdFactors<T: RlstScalar> {
    // Single: single batch
    Single(LevelFactors<T>),
    // Batched: collection of batches associated to a level
    Batched(MultiLevelFactors<T>),
}

/// MultilevelLuFactors: storage for LU factors
type MultiLevelLuFactors<T> = MultiLevelFactors<T>;

/// DiagBoxFactors: storage for block diagonal factors
pub type DiagBoxFactors<T> = CommutativeFactors<T>;

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

/// Helper to extract far indices when needed
fn get_far_indices(n: usize, near_indices: Vec<usize>) -> Vec<usize> {
    let near_set: HashSet<usize> = near_indices.into_iter().collect();
    (0..n).filter(|x| !near_set.contains(x)).collect()
}

/// FactorOperations: multiplication and inversion of
/// factors (ID, LU, block diagonal).
pub trait FactorOperations: Sized {
    type Item: RlstScalar;
    /// mul: manages parameters to call mul_data and ins_data
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
    );
    /// mul_data: takes information from target_data and
    /// modifies it without changing target_data
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

    /// ins_data: uses the result in mul_data to modify target_data
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

/// Constructor of ID factors
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
    /// new: nullifies the near field to perform ID decomposition and return ID factors.
    /// Arguments:
    /// - target_inds: indices of the target box in the octree
    /// - near_field_inds: indices of the near field associated to a give box
    /// - y_data: stores the test matrix Ω and the associated sketch Y=AΩ
    /// - z_data: stores the test matrix ψ and the associated sketch Z=A'ψ
    /// - subs_sample_dim: useful when using less than available samples to
    /// perform the decomposition
    /// - rank_par: indicates hot to pick the rank given that a box has been
    /// merged or not
    /// - id_options: see IdOptions
    /// - symmetric: indicates if A' should also be sketched or not
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

        // nullification of the near field
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
                // for a box that hasn't been merged yet it
                // applies ID with rank-revealing QR if a
                // rank has not been prescribed
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
            // for a box that has been merged
            // rank-revealing QR is not necessary
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

        // we check if the interactions can be compressed or if they should pass to the next level
        if id_sketch.rank < max_rank {
            let mut ind_f = get_far_indices(y_data.dim, near_field_inds.to_vec());
            if ind_f.len() > 0 {
                let aux_indices = target_inds.to_vec();

                for (id, &elem) in id_sketch.perm.iter().enumerate() {
                    let val = aux_indices[elem];
                    target_inds[id] = val;
                    near_field_inds[id] = val;
                }
            }

            // we store the residual and skeleton indices
            ind_r.extend_from_slice(&target_inds[k..]);
            ind_s.extend_from_slice(&target_inds[..k]);

            // we store the far field indices only for debugging purposes
            if !id_options.store_far {
                ind_f.clear();
                ind_f.shrink_to_fit();
            }

            (
                Some(Self {
                    data: FactorData::Reg(RectArr {
                        arr: id_sketch.id_mat,
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

    /// Returns the largest entry of the ID factor
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

/// Implementation of multiplication and inversion of ID factors
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

    /// mul: manages parameters to call mul_data and ins_data
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
        // Transpose accordingly
        // See section 4.3 in Yesypenko, A., & Martinsson, P. G. (2026). Randomized Strong Recursive Skeletonization:
        // Simultaneous Compression and LU Factorization of Hierarchical Matrices using Matrix–Vector Products:
        // A. Yesypenko, P.-G. Martinsson. Journal of Scientific Computing, 106(3), 63.
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

    /// mul_data: performs the elementary operation without changing target_arr
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
        if options.trans_val() {
            let mut aux_target_arr = empty_array();
            aux_target_arr
                .r_mut()
                .fill_from_resize(target_arr.r().conj());

            let res = self
                .data
                .mul(&aux_target_arr, side, options, &self.ind_s, &self.ind_r);

            aux_target_arr.r_mut().fill_from_resize(res.conj());
            aux_target_arr
        } else {
            self.data
                .mul(target_arr, side, options, &self.ind_s, &self.ind_r)
        }
    }

    /// ins_data: modifies the corresponding entries in target_arr
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
        let u_arr = extract_lu_factor(y_r, y_n, &lu_options.pivot_method);
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
            FactorData::Reg(RectArr { arr: empty_array() })
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
            if options.trans_val() {
                let mut aux_target_arr = empty_array();
                aux_target_arr
                    .r_mut()
                    .fill_from_resize(target_arr.r().conj());

                let res = match factor_type {
                    Some(FactorType::F) => {
                        self.l_arr
                            .mul(&aux_target_arr, side, options, &self.ind_t, &self.ind_r)
                    }
                    Some(FactorType::S) => {
                        self.u_arr
                            .mul(&aux_target_arr, side, options, &self.ind_t, &self.ind_r)
                    }
                    None => todo!(),
                };

                aux_target_arr.r_mut().fill_from_resize(res.conj());
                aux_target_arr
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
                let mut arr = empty_array();
                arr.fill_from_resize(diag_box.r().transpose());
                let reg_arr = RegSMat { arr, inv_arr };
                DiagBoxArr::Reg(reg_arr)
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
                DiagBoxArr::Lu(lu_arr)
            }
        }
    }

    fn new_no_symm<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccess<Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        inds: &[usize],
        db_ext_options: &ExtractOptions<Item>,
        y_sub_test: &Array<Item, ArrayImpl, 2>,
        y_sub_sketch: &Array<Item, ArrayImpl, 2>,
        z_sub_test: &Array<Item, ArrayImpl, 2>,
        z_sub_sketch: &Array<Item, ArrayImpl, 2>,
    ) -> Self {
        let y_sketch_r: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            y_sub_sketch,
            ExtInsType::Axis(inds.to_vec(), 1, false),
        )
        .unwrap()
        .ext;
        let mut y_test_c: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            y_sub_test,
            ExtInsType::Axis(inds.to_vec(), 1, false),
        )
        .unwrap()
        .ext;
        let y_diag_box = block_extraction(&mut y_test_c, &y_sketch_r, db_ext_options);

        let z_sketch_r: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            z_sub_sketch,
            ExtInsType::Axis(inds.to_vec(), 1, false),
        )
        .unwrap()
        .ext;
        let mut z_test_c: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            z_sub_test,
            ExtInsType::Axis(inds.to_vec(), 1, false),
        )
        .unwrap()
        .ext;
        let z_diag_box = block_extraction(&mut z_test_c, &z_sketch_r, db_ext_options);

        let mut diag_box = empty_array();
        diag_box
            .r_mut()
            .fill_from_resize(y_diag_box.r() + z_diag_box.r().transpose().conj());

        diag_box
            .r_mut()
            .scale_inplace(num::NumCast::from(0.5).unwrap());

        match db_ext_options.pivot_method {
            PivotMethod::DirectInversion => {
                let mut inv_arr = empty_array();
                inv_arr.fill_from_resize(diag_box.r().transpose());
                inv_arr.r_mut().into_inverse_alloc().unwrap();
                let mut arr = empty_array();
                arr.fill_from_resize(diag_box.r().transpose());
                let reg_arr = RegSMat { arr, inv_arr };
                DiagBoxArr::Reg(reg_arr)
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
                DiagBoxArr::Lu(lu_arr)
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
            DiagBoxArr::Reg(ref reg) => {
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
            DiagBoxArr::Lu(ref lu) => {
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

    pub fn new_no_symm(
        rows: &mut [usize],
        y_data: &SketchData<Item>,
        z_data: &SketchData<Item>,
        subs_sample_dim: usize,
        options: &ExtractOptions<Item>,
    ) -> (Option<Self>, Times) {
        let (y_sub_test, y_sub_sketch) = (
            y_data
                .test
                .r()
                .into_subview([0, 0], [subs_sample_dim, y_data.dim]),
            y_data
                .sketch
                .r()
                .into_subview([0, 0], [subs_sample_dim, y_data.dim]),
        );

        let (z_sub_test, z_sub_sketch) = (
            z_data
                .test
                .r()
                .into_subview([0, 0], [subs_sample_dim, z_data.dim]),
            z_data
                .sketch
                .r()
                .into_subview([0, 0], [subs_sample_dim, z_data.dim]),
        );

        let diag_times = LuTimes {
            //TODO: change this to diag_times
            extraction: 0_u128,
            lu: 0_u128,
        };

        let times = Times::Lu(diag_times);

        (
            Some(Self {
                arr: DiagBoxArr::new_no_symm(
                    rows,
                    &options,
                    &y_sub_test,
                    &y_sub_sketch,
                    &z_sub_test,
                    &z_sub_sketch,
                ),
                inds: rows.to_vec(),
            }),
            times,
        )
    }

    pub fn cond(&self) -> (CondType<Item>, Option<CondType<Item>>) {
        match &self.arr {
            DiagBoxArr::Reg(reg_dbox) => ((condition_number(&reg_dbox.arr), None), None),
            DiagBoxArr::Lu(lu_dbox) => (
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
                //println!("left");
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

                if factor_options.trans_val() {
                    let mut aux_target_cols = empty_array();
                    aux_target_cols
                        .r_mut()
                        .fill_from_resize(target_cols.r().conj());

                    self.arr
                        .mul(&mut aux_target_cols, &Side::Right, factor_options);

                    target_cols.r_mut().fill_from_resize(aux_target_cols.conj());
                    target_cols
                } else {
                    self.arr.mul(&mut target_cols, &Side::Right, factor_options);
                    target_cols
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
