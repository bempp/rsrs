use crate::rsrs::args::Symmetry;
use crate::rsrs::rsrs_factors::base_factors::{
    assert_transpose_only_mode, condition_number, conjugate_array_in_place, factor_apply_layout,
    BaseFactorOptions, CondType, DiagBoxArr, FactorApplyScratch, FactorData, LuSMat, RectArr,
    RegSMat, SquareArr,
};
use crate::rsrs::rsrs_factors::null_and_extract::{
    extract_lu_factor_from_blocks, near_box_extraction, null_near_field_into, ExtractOptions,
    ExtractionScratch, IdOptions, PivotMethod,
};
use crate::rsrs::rsrs_factors::rsrs_operator::FactType;
use crate::rsrs::sketch::SketchData;
use crate::rsrs::statistics::{IdTimes, LuTimes, Times};
use crate::utils::linear_algebra::add_diagonal;
use crate::utils::{
    data_ins_ext::{extract_axis_into, raw_matrix_mut, RawMatrixMut},
    elementary_matrix::{
        col_delta, col_delta_raw, col_perm, col_subs, ext_cols, ext_rows, row_delta, row_delta_raw,
        row_perm, row_subs,
    },
    linear_algebra::{
        block_extraction_into, streaming_chunk_rows, BlockExtractionMethod,
        NormalEquationAccumulator,
    },
    memory::{matrix_bytes, trace_memory_event, trace_memory_growth},
};
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    ThreadPool,
};
use rlst::{
    dense::{
        linalg::{
            interpolative_decomposition::{Accuracy, MatrixIdNoSkel},
            lu::{MatrixLu, SquareLuFactors},
            triangular_arrays::TriangularOperations,
        },
        tools::RandScalar,
    },
    prelude::*,
};
use std::{
    collections::{HashMap, HashSet},
    mem::{size_of, size_of_val},
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

/// Selects which half of an elementary RSRS update is being applied.
///
/// The names stay abstract on purpose: an ID factor and an LU factor map `F`
/// and `S` to different stored arrays, but the higher-level elimination logic
/// can use one common ordering.
#[derive(Clone, PartialEq, Debug)]
pub enum FactorType {
    F,
    S,
}

/// Application options shared by a batch of commutative factors.
#[derive(Clone)]
pub struct MulOptions {
    /// Basic orientation and inversion flags for the factor application.
    pub base_options: BaseFactorOptions,
    /// Whether the factor is applied to a column-vector/matrix (`Left`) or to
    /// a row-vector/matrix (`Right`).
    pub side: Side,
    /// Which half of the split elimination is currently being traversed.
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
    /// type of symmetry of the factor
    pub symmetry: Symmetry,
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
    symmetry: Symmetry,
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

pub(crate) struct DiagExtractionScratch<Item: RlstScalar> {
    primary: DynamicArray<Item, 2>,
    secondary: DynamicArray<Item, 2>,
    tertiary: DynamicArray<Item, 2>,
    quaternary: DynamicArray<Item, 2>,
    quinary: DynamicArray<Item, 2>,
}

impl<Item: RlstScalar> DiagExtractionScratch<Item> {
    pub(crate) fn new() -> Self {
        Self {
            primary: empty_array(),
            secondary: empty_array(),
            tertiary: empty_array(),
            quaternary: empty_array(),
            quinary: empty_array(),
        }
    }
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

#[derive(Debug, Clone, Copy, Default)]
pub struct FactorMemoryBreakdown {
    pub id_bytes: u64,
    pub lu_bytes: u64,
    pub diag_bytes: u64,
    pub perm_bytes: u64,
    pub id_count: usize,
    pub lu_count: usize,
    pub diag_count: usize,
}

impl FactorMemoryBreakdown {
    pub fn total_bytes(&self) -> u64 {
        self.id_bytes + self.lu_bytes + self.diag_bytes + self.perm_bytes
    }

    fn add_factor<Item: RlstScalar>(&mut self, factor: &Factor<Item>) {
        match factor {
            Factor::Id(id_factor) => {
                self.id_bytes += id_factor_bytes(id_factor);
                self.id_count += 1;
            }
            Factor::Lu(lu_factor) => {
                self.lu_bytes += lu_factor_bytes(lu_factor);
                self.lu_count += 1;
            }
            Factor::Diag(diag_factor) => {
                self.diag_bytes += diag_factor_bytes(diag_factor);
                self.diag_count += 1;
            }
        }
    }
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

fn dynamic_array_bytes<Item: RlstScalar>(arr: &DynamicArray<Item, 2>) -> u64 {
    (arr.shape()[0] as u64) * (arr.shape()[1] as u64) * (size_of::<Item>() as u64)
}

fn usize_vec_bytes(values: &[usize]) -> u64 {
    (values.len() as u64) * (size_of::<usize>() as u64)
}

fn perm_factor_bytes(perm: &PermFactor) -> u64 {
    usize_vec_bytes(&perm.orig_indices) + usize_vec_bytes(&perm.perm_indices)
}

fn square_arr_bytes<Item: RlstScalar>(arr: &SquareArr<Item>) -> u64 {
    match arr {
        SquareArr::Reg(reg) => dynamic_array_bytes(&reg.arr) + dynamic_array_bytes(&reg.inv_arr),
        SquareArr::Lu(lu) => size_of_val(lu) as u64,
    }
}

fn diag_box_arr_bytes<Item: RlstScalar>(arr: &DiagBoxArr<Item>) -> u64 {
    match arr {
        DiagBoxArr::Reg(reg) => dynamic_array_bytes(&reg.arr) + dynamic_array_bytes(&reg.inv_arr),
        DiagBoxArr::Lu(lu) => size_of_val(lu) as u64,
    }
}

fn factor_data_bytes<Item: RlstScalar>(data: &FactorData<Item>) -> u64 {
    match data {
        FactorData::Comp(comp) => square_arr_bytes(&comp.sq) + dynamic_array_bytes(&comp.rectg.arr),
        FactorData::Reg(rect) => dynamic_array_bytes(&rect.arr),
    }
}

fn id_factor_bytes<Item: RlstScalar>(factor: &IdFactor<Item>) -> u64 {
    factor_data_bytes(&factor.data)
        + usize_vec_bytes(&factor.perm)
        + usize_vec_bytes(&factor.ind_r)
        + usize_vec_bytes(&factor.ind_s)
        + usize_vec_bytes(&factor.ind_f)
}

fn lu_factor_bytes<Item: RlstScalar>(factor: &LuFactor<Item>) -> u64 {
    factor_data_bytes(&factor.l_arr)
        + factor_data_bytes(&factor.u_arr)
        + usize_vec_bytes(&factor.ind_r)
        + usize_vec_bytes(&factor.ind_t)
}

fn diag_factor_bytes<Item: RlstScalar>(factor: &DiagBoxFactor<Item>) -> u64 {
    diag_box_arr_bytes(&factor.arr) + usize_vec_bytes(&factor.inds)
}

fn prefer_direct_diag_extraction<Item: RlstScalar>(
    rows_len: usize,
    subs_sample_dim: usize,
    nonsymmetric_buffers: usize,
) -> bool {
    const DIRECT_DIAG_BYTES_BUDGET: u64 = 8 * 1024 * 1024;

    let extracted =
        matrix_bytes::<Item>(subs_sample_dim, rows_len).saturating_mul(nonsymmetric_buffers as u64);
    let square = matrix_bytes::<Item>(rows_len, rows_len)
        .saturating_mul((nonsymmetric_buffers.saturating_sub(1)) as u64);

    extracted.saturating_add(square) <= DIRECT_DIAG_BYTES_BUDGET
}

fn symmetrize_square_in_place<Item: RlstScalar>(arr: &mut DynamicArray<Item, 2>, adjoint: bool) {
    let shape = arr.shape();
    debug_assert_eq!(shape[0], shape[1]);
    let half = Item::real(0.5);

    for row in 0..shape[0] {
        for col in row..shape[1] {
            let mirror = if adjoint {
                arr[[col, row]].conj()
            } else {
                arr[[col, row]]
            };
            let avg = (arr[[row, col]] + mirror).mul_real(half);
            arr[[row, col]] = avg;
            arr[[col, row]] = if adjoint { avg.conj() } else { avg };
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
        assert_transpose_only_mode(options.trans, "PermFactor::left_mul");
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
        options: &BaseFactorOptions,
    ) {
        assert_transpose_only_mode(options.trans, "PermFactor::right_mul");
        let orig_indices: Vec<_> = (0..left_arr.shape()[1]).collect();
        assert_eq!(orig_indices.len(), self.perm_indices.len());
        let trans = if options.inv {
            let aux_options = options.transpose();
            aux_options.trans_val()
        } else {
            options.trans_val()
        };

        col_perm(
            orig_indices.clone(),
            self.perm_indices.clone(),
            left_arr,
            trans,
        );
    }

    pub fn stored_bytes(&self) -> u64 {
        perm_factor_bytes(self)
    }
}

impl<Item: RlstScalar> RsrsFactors<Item> {
    pub fn memory_breakdown(&self) -> FactorMemoryBreakdown {
        let mut breakdown = FactorMemoryBreakdown {
            perm_bytes: self.perm_factor.stored_bytes(),
            ..FactorMemoryBreakdown::default()
        };

        match &self.id_factors {
            MultiLevelIdFactors::Single(levels) => {
                for batch in levels {
                    for factor in batch {
                        breakdown.add_factor(factor);
                    }
                }
            }
            MultiLevelIdFactors::Batched(levels) => {
                for level in levels {
                    for batch in level {
                        for factor in batch {
                            breakdown.add_factor(factor);
                        }
                    }
                }
            }
        }

        for level in &self.lu_factors {
            for batch in level {
                for factor in batch {
                    breakdown.add_factor(factor);
                }
            }
        }

        for factor in &self.diag_box_factors {
            breakdown.add_factor(factor);
        }

        breakdown
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
            + MatrixIdNoSkel
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
    ///   perform the decomposition
    /// - rank_par: indicates hot to pick the rank given that a box has been
    ///   merged or not
    /// - id_options: see IdOptions
    /// - symmetric: indicates if A' should also be sketched or not
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scratch: &mut ExtractionScratch<Item>,
        target_inds: &mut [usize],
        near_field_inds: &mut [usize],
        y_data: &SketchData<Item>,
        z_data: &SketchData<Item>,
        subs_sample_dim: usize,
        fixed_rank: bool,
        rank_par: &BoxType<Real<Item>>,
        id_options: &IdOptions<Item>,
        symmetry: &Symmetry,
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
        null_near_field_into(
            target_inds,
            near_field_inds,
            y_data,
            z_data,
            subs_sample_dim,
            symmetry,
            fixed_rank,
            id_options,
            &mut scratch.primary,
            &mut scratch.secondary,
            &mut scratch.tertiary,
            &mut scratch.normal,
        );

        let nullification_time: Duration = start.elapsed();
        let start: Instant = Instant::now();
        let max_rank: usize = *scratch.primary.shape().iter().min().unwrap();
        if max_rank <= 1 {
            let id_times = IdTimes {
                nullification: nullification_time.as_millis(),
                id: 0,
            };
            return (None, Times::Id(id_times));
        }
        let id_sketch = match rank_par {
            BoxType::Full(tol) => {
                // for a box that hasn't been merged yet it
                // applies ID with rank-revealing QR if a
                // rank has not been prescribed
                if *tol < num::One::one() {
                    scratch
                        .primary
                        .r_mut()
                        .into_subview([0, 0], null_shape)
                        .into_id_alloc_no_skel(
                            Accuracy::Tol(*tol),
                            id_options.qr_method.clone(),
                            TransMode::Trans,
                        )
                        .unwrap()
                } else {
                    let loc_rank = max_rank.min(num::ToPrimitive::to_usize(tol).unwrap());
                    scratch
                        .primary
                        .r_mut()
                        .into_subview([0, 0], null_shape)
                        .into_id_alloc_no_skel(
                            Accuracy::FixedRank(loc_rank),
                            id_options.qr_method.clone(),
                            TransMode::Trans,
                        )
                        .unwrap()
                }
            }
            // for a box that has been merged
            // rank-revealing QR is not necessary
            BoxType::Merged(rank) => scratch
                .primary
                .r_mut()
                .into_subview([0, 0], null_shape)
                .into_id_alloc_no_skel(
                    Accuracy::FixedRank((*rank).min(max_rank)),
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
        let factor_symmetry = if symmetry.complex_symmetric_val::<Item>() {
            Symmetry::NoSymm
        } else {
            symmetry.clone()
        };

        // we check if the interactions can be compressed or if they should pass to the next level
        if id_sketch.rank < max_rank {
            let mut ind_f = get_far_indices(y_data.dim, near_field_inds.to_vec());
            if !ind_f.is_empty() {
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
                        arr: Box::new(id_sketch.id_mat),
                    }),
                    perm: id_sketch.perm,
                    symmetry: factor_symmetry,
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

    /// Returns whether this factor application needs an explicit conjugation.
    ///
    /// `BaseFactorOptions` only tracks whether the factor is transposed. For
    /// Hermitian and complex nonsymmetric factors, a transpose also changes
    /// whether the extracted sketch data must be conjugated.
    pub fn conj_val(&self, trans_val: bool) -> bool {
        match self.symmetry {
            Symmetry::NoSymm => !trans_val,
            // Symmetric means A^T = A, not A^H = A, so no conjugation is needed.
            Symmetry::Symmetric => false,
            Symmetry::Hermitian => trans_val,
        }
    }

    fn fill_delta_with_scratch<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &Array<Item, ArrayImplMut, 2>,
        side: &Side,
        options: &BaseFactorOptions,
        scratch: &mut FactorApplyScratch<Item>,
    ) {
        let conj_target = self.conj_val(options.trans_val());
        if conj_target {
            let layout = factor_apply_layout(side, options, &self.ind_s, &self.ind_r);
            let shape = target_arr.shape();
            trace_memory_event(
                &format!(
                    "id_factor conj copy target_arr (rows={}, cols={})",
                    shape[0], shape[1]
                ),
                Some(matrix_bytes::<Item>(shape[0], shape[1])),
            );
            trace_memory_growth(
                "id_factor conj source slice",
                Some(matrix_bytes::<Item>(
                    if layout.axis == 0 {
                        layout.source_indices.len()
                    } else {
                        shape[0]
                    },
                    if layout.axis == 0 {
                        shape[1]
                    } else {
                        layout.source_indices.len()
                    },
                )),
            );
        }

        self.data.delta_with_scratch(
            target_arr,
            side,
            options,
            &self.ind_s,
            &self.ind_r,
            conj_target,
            scratch,
        );

        if conj_target {
            conjugate_array_in_place(&mut scratch.result);
        }
    }

    fn apply_delta_with_scratch<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        side: &Side,
        options: &BaseFactorOptions,
        scratch: &mut FactorApplyScratch<Item>,
    ) {
        self.fill_delta_with_scratch(target_arr, side, options, scratch);
        match side {
            Side::Left => row_delta(
                &self.ind_s,
                &self.ind_r,
                &scratch.result,
                target_arr,
                options,
                options.inv,
            ),
            Side::Right => col_delta(
                &self.ind_s,
                &self.ind_r,
                &scratch.result,
                target_arr,
                options,
                options.inv,
            ),
        }
    }

    unsafe fn apply_delta_with_scratch_raw<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        read_target_arr: &Array<Item, ArrayImplMut, 2>,
        raw_target_arr: RawMatrixMut<Item>,
        side: &Side,
        options: &BaseFactorOptions,
        scratch: &mut FactorApplyScratch<Item>,
    ) {
        self.fill_delta_with_scratch(read_target_arr, side, options, scratch);
        match side {
            Side::Left => unsafe {
                row_delta_raw(
                    &self.ind_s,
                    &self.ind_r,
                    &scratch.result,
                    raw_target_arr,
                    options,
                    options.inv,
                )
            },
            Side::Right => unsafe {
                col_delta_raw(
                    &self.ind_s,
                    &self.ind_r,
                    &scratch.result,
                    raw_target_arr,
                    options,
                    options.inv,
                )
            },
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
            + MatrixIdNoSkel
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
        // For ID factors the stored "second" half is the transpose partner of
        // the "first" half, so applying `S` means toggling the factor
        // orientation before we touch the low-level delta kernels.
        let aux_options = match options.factor_type {
            FactorType::F => options.base_options.clone(),
            FactorType::S => options.base_options.transpose(),
        };
        let mut scratch = FactorApplyScratch::new();
        self.apply_delta_with_scratch(target_arr, &options.side, &aux_options, &mut scratch);
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
        let mut scratch = FactorApplyScratch::new();
        self.fill_delta_with_scratch(target_arr, side, options, &mut scratch);
        let layout = factor_apply_layout(side, options, &self.ind_s, &self.ind_r);
        let mut subarr_target = empty_array();
        extract_axis_into(
            &mut subarr_target,
            target_arr,
            layout.target_indices,
            layout.axis,
            layout.transposed,
        );
        if options.inv {
            subarr_target.sub_into(scratch.result.r());
        } else {
            subarr_target.sum_into(scratch.result.r());
        }
        subarr_target
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _scratch: &mut ExtractionScratch<Item>,
        ind_r: &[usize],
        near_field_inds: &[usize],
        inactive_inds: &[usize],
        y_data: &SketchData<Item>,
        z_data: &SketchData<Item>,
        subs_sample_dim: usize,
        fixed_rank: bool,
        lu_options: &ExtractOptions<Item>,
        symmetry: &Symmetry,
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
            if !ind_r.contains(&elem) && inactive_inds.binary_search(&elem).is_err() {
                t_numbering.push(pos);
                ind_t.push(elem);
            }
        }
        let (mut y_data_r, mut y_data_n, (_y_lu_io_time, y_lu_b_ext_time)) = near_box_extraction(
            ind_r,
            near_field_inds,
            y_data,
            subs_sample_dim,
            fixed_rank,
            false,
            lu_options,
            &r_numbering,
            &t_numbering,
        );
        let start = Instant::now();
        let u_arr =
            extract_lu_factor_from_blocks(&mut y_data_r, &mut y_data_n, &lu_options.pivot_method);
        let u_assembly = start.elapsed();

        let lu_b_ext_time;
        let lu_assembly_time;

        let l_arr =
            if !symmetry.factor_symm_val::<Item>() || symmetry.complex_symmetric_val::<Item>() {
                let (secondary_data, conjugate_data) = if symmetry.complex_symmetric_val::<Item>() {
                    (y_data, true)
                } else {
                    (z_data, false)
                };
                let (mut z_data_r, mut z_data_n, (_z_lu_io_time, z_lu_b_ext_time)) =
                    near_box_extraction(
                        ind_r,
                        near_field_inds,
                        secondary_data,
                        subs_sample_dim,
                        fixed_rank,
                        conjugate_data,
                        lu_options,
                        &r_numbering,
                        &t_numbering,
                    );

                let start = Instant::now();

                let l_arr = extract_lu_factor_from_blocks(
                    &mut z_data_r,
                    &mut z_data_n,
                    &lu_options.pivot_method,
                );
                let l_assembly = start.elapsed();
                lu_b_ext_time = y_lu_b_ext_time + z_lu_b_ext_time;
                lu_assembly_time = u_assembly + l_assembly;
                l_arr
            } else {
                lu_b_ext_time = y_lu_b_ext_time;
                lu_assembly_time = u_assembly;
                FactorData::Reg(RectArr {
                    arr: Box::new(empty_array()),
                })
            };

        let lu_times = LuTimes {
            extraction: lu_b_ext_time.as_millis(),
            lu: lu_assembly_time.as_millis(),
        };

        let times = Times::Lu(lu_times);
        let factor_symmetry = if symmetry.complex_symmetric_val::<Item>() {
            Symmetry::NoSymm
        } else {
            symmetry.clone()
        };
        (
            Some(Self {
                l_arr,
                u_arr,
                symmetry: factor_symmetry,
                ind_r: ind_r.to_vec(),
                ind_t,
            }),
            times,
        )
    }

    /// Returns whether this LU update needs explicit conjugation.
    ///
    /// Symmetric factors reuse the same triangular data without conjugation,
    /// while Hermitian factors use the same storage path but must conjugate
    /// whenever the transpose orientation is requested.
    pub fn conj_val(&self, trans_val: bool) -> bool {
        match self.symmetry {
            Symmetry::NoSymm => trans_val,
            Symmetry::Symmetric => false,
            Symmetry::Hermitian => trans_val,
        }
    }

    fn fill_delta_with_scratch<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &Array<Item, ArrayImpl, 2>,
        side: &Side,
        factor_type: Option<FactorType>,
        options: &BaseFactorOptions,
        scratch: &mut FactorApplyScratch<Item>,
    ) {
        let conj_target = self.conj_val(options.trans_val());
        if conj_target {
            let layout = factor_apply_layout(side, options, &self.ind_t, &self.ind_r);
            let shape = target_arr.shape();
            trace_memory_event(
                &format!(
                    "lu_factor conj copy target_arr (rows={}, cols={})",
                    shape[0], shape[1]
                ),
                Some(matrix_bytes::<Item>(shape[0], shape[1])),
            );
            trace_memory_growth(
                "lu_factor conj source slice",
                Some(matrix_bytes::<Item>(
                    if layout.axis == 0 {
                        layout.source_indices.len()
                    } else {
                        shape[0]
                    },
                    if layout.axis == 0 {
                        shape[1]
                    } else {
                        layout.source_indices.len()
                    },
                )),
            );
        }

        if self.symmetry.factor_symm_val::<Item>() {
            self.u_arr.delta_with_scratch(
                target_arr,
                side,
                options,
                &self.ind_t,
                &self.ind_r,
                conj_target,
                scratch,
            );
        } else {
            match factor_type {
                Some(FactorType::F) => self.l_arr.delta_with_scratch(
                    target_arr,
                    side,
                    options,
                    &self.ind_t,
                    &self.ind_r,
                    conj_target,
                    scratch,
                ),
                Some(FactorType::S) => self.u_arr.delta_with_scratch(
                    target_arr,
                    side,
                    options,
                    &self.ind_t,
                    &self.ind_r,
                    conj_target,
                    scratch,
                ),
                None => todo!(),
            };
        }

        if conj_target {
            conjugate_array_in_place(&mut scratch.result);
        }
    }

    fn apply_delta_with_scratch<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        side: &Side,
        factor_type: Option<FactorType>,
        options: &BaseFactorOptions,
        scratch: &mut FactorApplyScratch<Item>,
    ) {
        self.fill_delta_with_scratch(target_arr, side, factor_type, options, scratch);
        match side {
            Side::Left => row_delta(
                &self.ind_t,
                &self.ind_r,
                &scratch.result,
                target_arr,
                options,
                options.inv,
            ),
            Side::Right => col_delta(
                &self.ind_t,
                &self.ind_r,
                &scratch.result,
                target_arr,
                options,
                options.inv,
            ),
        }
    }

    unsafe fn apply_delta_with_scratch_raw<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        read_target_arr: &Array<Item, ArrayImplMut, 2>,
        raw_target_arr: RawMatrixMut<Item>,
        side: &Side,
        factor_type: Option<FactorType>,
        options: &BaseFactorOptions,
        scratch: &mut FactorApplyScratch<Item>,
    ) {
        self.fill_delta_with_scratch(read_target_arr, side, factor_type, options, scratch);
        match side {
            Side::Left => unsafe {
                row_delta_raw(
                    &self.ind_t,
                    &self.ind_r,
                    &scratch.result,
                    raw_target_arr,
                    options,
                    options.inv,
                )
            },
            Side::Right => unsafe {
                col_delta_raw(
                    &self.ind_t,
                    &self.ind_r,
                    &scratch.result,
                    raw_target_arr,
                    options,
                    options.inv,
                )
            },
        }
    }

    pub fn cond(&self) -> (CondType<Item>, Option<CondType<Item>>) {
        if !self.symmetry.factor_symm_val::<Item>() {
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
        // LU factors store the transpose relationship on the opposite half from
        // ID factors, so `F` is the branch that flips orientation here.
        let aux_options = match options.factor_type {
            FactorType::F => options.base_options.transpose(),
            FactorType::S => options.base_options.clone(),
        };
        let mut scratch = FactorApplyScratch::new();
        self.apply_delta_with_scratch(
            target_arr,
            &options.side,
            Some(options.factor_type.clone()),
            &aux_options,
            &mut scratch,
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
        let mut scratch = FactorApplyScratch::new();
        self.fill_delta_with_scratch(target_arr, side, factor_type, options, &mut scratch);
        let layout = factor_apply_layout(side, options, &self.ind_t, &self.ind_r);
        let mut subarr_target = empty_array();
        extract_axis_into(
            &mut subarr_target,
            target_arr,
            layout.target_indices,
            layout.axis,
            layout.transposed,
        );
        if options.inv {
            subarr_target.sub_into(scratch.result.r());
        } else {
            subarr_target.sum_into(scratch.result.r());
        }
        subarr_target
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
        if self.symmetry.factor_symm_val::<Item>() {
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
    fn from_extracted(
        diag_box: &DynamicArray<Item, 2>,
        db_ext_options: &ExtractOptions<Item>,
    ) -> Self {
        let shape = diag_box.shape();
        let transposed_bytes = matrix_bytes::<Item>(shape[1], shape[0]);

        match db_ext_options.pivot_method {
            PivotMethod::DirectInversion => {
                let mut arr = empty_array();
                trace_memory_event(
                    &format!(
                        "diag_box_arr direct transpose diag_box -> arr (box={}, samples={})",
                        shape[0], shape[1]
                    ),
                    Some(transposed_bytes),
                );
                arr.fill_from_resize(diag_box.r().transpose());

                trace_memory_event(
                    &format!(
                        "diag_box_arr direct clone arr -> inv_arr (box={}, samples={})",
                        shape[1], shape[0]
                    ),
                    Some(transposed_bytes),
                );
                let mut inv_arr = empty_array();
                inv_arr.fill_from_resize(arr.r());
                inv_arr.r_mut().into_inverse_alloc().unwrap();
                let reg_arr = RegSMat { arr, inv_arr };
                DiagBoxArr::Reg(reg_arr)
            }
            PivotMethod::Lu(alpha) => {
                let mut lu_input = empty_array();
                trace_memory_event(
                    &format!(
                        "diag_box_arr lu transpose diag_box -> lu_input (box={}, samples={})",
                        shape[0], shape[1]
                    ),
                    Some(transposed_bytes),
                );
                lu_input.fill_from_resize(diag_box.r().transpose());
                add_diagonal(&mut lu_input, Item::real(alpha));
                let lu = <Item as MatrixLu>::into_lu_alloc(lu_input).unwrap();
                let square_factors = SquareLuFactors::from_lu(&lu).unwrap();
                let lu_arr = LuSMat { square_factors };
                DiagBoxArr::Lu(lu_arr)
            }
            PivotMethod::LuHybrid(alpha) => {
                let mut lu_input = empty_array();
                trace_memory_event(
                    &format!(
                        "diag_box_arr lu hybrid transpose diag_box -> lu_input (box={}, samples={})",
                        shape[0], shape[1]
                    ),
                    Some(transposed_bytes),
                );
                lu_input.fill_from_resize(diag_box.r().transpose());
                add_diagonal(&mut lu_input, Item::real(alpha));
                let mut arr = empty_array();
                arr.fill_from_resize(lu_input.r());
                let lu = <Item as MatrixLu>::into_lu_alloc(lu_input).unwrap();
                let mut inv_arr = rlst_dynamic_array2!(Item, [shape[1], shape[1]]);
                add_diagonal(&mut inv_arr, num::One::one());
                <LuDecomposition<Item, _> as MatrixLuDecomposition>::solve_mat(
                    &lu,
                    TransMode::NoTrans,
                    inv_arr.r_mut(),
                )
                .unwrap();
                DiagBoxArr::Reg(RegSMat { arr, inv_arr })
            }
        }
    }

    fn streamed_extraction_from_data(
        inds: &[usize],
        tol_lstsq: Real<Item>,
        sketch_data: &SketchData<Item>,
        subs_sample_dim: usize,
        conjugate_data: bool,
        test_chunk: &mut DynamicArray<Item, 2>,
        sketch_chunk: &mut DynamicArray<Item, 2>,
    ) -> DynamicArray<Item, 2> {
        let capped_samples = subs_sample_dim.min(sketch_data.test.shape()[0]);
        let chunk_rows = streaming_chunk_rows::<Item>(capped_samples, inds.len() * 2, 2).max(1);
        let mut accumulator = NormalEquationAccumulator::<Item>::new(inds.len(), inds.len());

        for chunk in sketch_data.chunk_iter(subs_sample_dim, inds.len() * 2, 2) {
            extract_axis_into(test_chunk, &chunk.test, inds, 1, false);
            extract_axis_into(sketch_chunk, &chunk.sketch, inds, 1, false);
            if conjugate_data {
                conjugate_array_in_place(test_chunk);
                conjugate_array_in_place(sketch_chunk);
            }
            accumulator.add_chunk(test_chunk, sketch_chunk);
        }

        trace_memory_growth(
            &format!(
                "diag_box_extraction streamed chunks (size={}, samples={}, chunk_rows={chunk_rows})",
                inds.len(),
                capped_samples,
            ),
            Some(
                matrix_bytes::<Item>(chunk_rows, inds.len()) * 2
                    + matrix_bytes::<Item>(inds.len(), inds.len()) * 2,
            ),
        );

        accumulator.solve(tol_lstsq)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_scratch<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccess<Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        inds: &[usize],
        db_ext_options: &ExtractOptions<Item>,
        sub_test: &Array<Item, ArrayImpl, 2>,
        sub_sketch: &Array<Item, ArrayImpl, 2>,
        test_c: &mut DynamicArray<Item, 2>,
        sketch_r: &mut DynamicArray<Item, 2>,
        diag_box: &mut DynamicArray<Item, 2>,
        symmetrize_adjoint: Option<bool>,
    ) -> Self {
        extract_axis_into(sketch_r, sub_sketch, inds, 1, false);
        extract_axis_into(test_c, sub_test, inds, 1, false);
        block_extraction_into(test_c, sketch_r, db_ext_options, diag_box);
        if let Some(adjoint) = symmetrize_adjoint {
            symmetrize_square_in_place(diag_box, adjoint);
        }
        trace_memory_growth(
            &format!(
                "diag_box_extraction symmetric (size={}, samples={})",
                inds.len(),
                sub_test.shape()[0]
            ),
            Some(
                matrix_bytes::<Item>(sub_test.shape()[0], inds.len()) * 2
                    + matrix_bytes::<Item>(inds.len(), inds.len()),
            ),
        );
        Self::from_extracted(diag_box, db_ext_options)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_no_symm_with_scratch<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccess<Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        inds: &[usize],
        db_ext_options: &ExtractOptions<Item>,
        y_sub_test: &Array<Item, ArrayImpl, 2>,
        y_sub_sketch: &Array<Item, ArrayImpl, 2>,
        z_sub_test: &Array<Item, ArrayImpl, 2>,
        z_sub_sketch: &Array<Item, ArrayImpl, 2>,
        y_conjugate: bool,
        z_conjugate: bool,
        y_test_c: &mut DynamicArray<Item, 2>,
        y_sketch_r: &mut DynamicArray<Item, 2>,
        z_test_c: &mut DynamicArray<Item, 2>,
        z_sketch_r: &mut DynamicArray<Item, 2>,
        diag_box: &mut DynamicArray<Item, 2>,
    ) -> Self {
        extract_axis_into(y_sketch_r, y_sub_sketch, inds, 1, false);
        extract_axis_into(y_test_c, y_sub_test, inds, 1, false);
        if y_conjugate {
            conjugate_array_in_place(y_sketch_r);
            conjugate_array_in_place(y_test_c);
        }
        block_extraction_into(y_test_c, y_sketch_r, db_ext_options, diag_box);

        extract_axis_into(z_sketch_r, z_sub_sketch, inds, 1, false);
        extract_axis_into(z_test_c, z_sub_test, inds, 1, false);
        if z_conjugate {
            conjugate_array_in_place(z_sketch_r);
            conjugate_array_in_place(z_test_c);
        }
        block_extraction_into(z_test_c, z_sketch_r, db_ext_options, y_test_c);

        diag_box.sum_into(y_test_c.r().transpose().conj());
        trace_memory_growth(
            &format!(
                "diag_box_extraction nonsymmetric (size={}, samples={})",
                inds.len(),
                y_sub_test.shape()[0]
            ),
            Some(
                matrix_bytes::<Item>(y_sub_test.shape()[0], inds.len()) * 4
                    + matrix_bytes::<Item>(inds.len(), inds.len()) * 3,
            ),
        );

        diag_box
            .r_mut()
            .scale_inplace(num::NumCast::from(0.5).unwrap());

        Self::from_extracted(diag_box, db_ext_options)
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
        assert_transpose_only_mode(factor_options.trans, "DiagBoxArr::left_mul");
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
                    lu.square_factors
                        .solve_mat(factor_options.trans, right_arr.r_mut())
                        .unwrap();
                } else {
                    lu.square_factors
                        .mul_mat(factor_options.trans, right_arr.r_mut())
                        .unwrap();
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
        assert_transpose_only_mode(factor_options.trans, "DiagBoxArr::mul");
        match side {
            Side::Left => self.left_mul(arr, factor_options),
            Side::Right => {
                let shape = arr.shape();
                trace_memory_event(
                    &format!(
                        "diag_factor right transpose copy (rows={}, cols={})",
                        shape[1], shape[0]
                    ),
                    Some(matrix_bytes::<Item>(shape[1], shape[0])),
                );
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
        fixed_rank: bool,
        options: &ExtractOptions<Item>,
    ) -> (Option<Self>, Times) {
        let mut scratch = DiagExtractionScratch::new();
        Self::new_with_scratch(
            rows.to_vec(),
            y_data,
            subs_sample_dim,
            fixed_rank,
            options,
            &mut scratch,
        )
    }

    pub(crate) fn new_with_scratch(
        rows: Vec<usize>,
        y_data: &SketchData<Item>,
        subs_sample_dim: usize,
        fixed_rank: bool,
        options: &ExtractOptions<Item>,
        scratch: &mut DiagExtractionScratch<Item>,
    ) -> (Option<Self>, Times) {
        Self::new_with_scratch_impl(
            rows,
            y_data,
            subs_sample_dim,
            fixed_rank,
            options,
            scratch,
            None,
        )
    }

    pub(crate) fn new_symm_with_scratch(
        rows: Vec<usize>,
        y_data: &SketchData<Item>,
        subs_sample_dim: usize,
        fixed_rank: bool,
        options: &ExtractOptions<Item>,
        scratch: &mut DiagExtractionScratch<Item>,
        adjoint: bool,
    ) -> (Option<Self>, Times) {
        Self::new_with_scratch_impl(
            rows,
            y_data,
            subs_sample_dim,
            fixed_rank,
            options,
            scratch,
            Some(adjoint),
        )
    }

    fn new_with_scratch_impl(
        rows: Vec<usize>,
        y_data: &SketchData<Item>,
        subs_sample_dim: usize,
        fixed_rank: bool,
        options: &ExtractOptions<Item>,
        scratch: &mut DiagExtractionScratch<Item>,
        symmetrize_adjoint: Option<bool>,
    ) -> (Option<Self>, Times) {
        let diag_times = LuTimes {
            //TODO: change this to diag_times
            extraction: 0_u128,
            lu: 0_u128,
        };

        let times = Times::Lu(diag_times);

        (
            Some(Self {
                arr: if fixed_rank
                    && matches!(
                        options.block_extraction_method,
                        BlockExtractionMethod::LuLstSq
                    )
                    && !prefer_direct_diag_extraction::<Item>(rows.len(), subs_sample_dim, 2)
                {
                    {
                        let mut diag_box = DiagBoxArr::streamed_extraction_from_data(
                            &rows,
                            options.tol_lstsq,
                            y_data,
                            subs_sample_dim,
                            false,
                            &mut scratch.primary,
                            &mut scratch.secondary,
                        );
                        if let Some(adjoint) = symmetrize_adjoint {
                            symmetrize_square_in_place(&mut diag_box, adjoint);
                        }
                        DiagBoxArr::from_extracted(&diag_box, options)
                    }
                } else {
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
                    DiagBoxArr::new_with_scratch(
                        &rows,
                        options,
                        &sub_test,
                        &sub_sketch,
                        &mut scratch.primary,
                        &mut scratch.secondary,
                        &mut scratch.tertiary,
                        symmetrize_adjoint,
                    )
                },
                inds: rows,
            }),
            times,
        )
    }

    pub fn new_no_symm(
        rows: &mut [usize],
        y_data: &SketchData<Item>,
        z_data: &SketchData<Item>,
        subs_sample_dim: usize,
        fixed_rank: bool,
        options: &ExtractOptions<Item>,
    ) -> (Option<Self>, Times) {
        let mut scratch = DiagExtractionScratch::new();
        Self::new_no_symm_with_scratch(
            rows.to_vec(),
            y_data,
            z_data,
            subs_sample_dim,
            fixed_rank,
            options,
            &mut scratch,
        )
    }

    pub(crate) fn new_no_symm_with_scratch(
        rows: Vec<usize>,
        y_data: &SketchData<Item>,
        z_data: &SketchData<Item>,
        subs_sample_dim: usize,
        fixed_rank: bool,
        options: &ExtractOptions<Item>,
        scratch: &mut DiagExtractionScratch<Item>,
    ) -> (Option<Self>, Times) {
        let diag_times = LuTimes {
            //TODO: change this to diag_times
            extraction: 0_u128,
            lu: 0_u128,
        };

        let times = Times::Lu(diag_times);

        (
            Some(Self {
                arr: if fixed_rank
                    && matches!(
                        options.block_extraction_method,
                        BlockExtractionMethod::LuLstSq
                    )
                    && !prefer_direct_diag_extraction::<Item>(rows.len(), subs_sample_dim, 4)
                {
                    let y_diag_box = DiagBoxArr::streamed_extraction_from_data(
                        &rows,
                        options.tol_lstsq,
                        y_data,
                        subs_sample_dim,
                        false,
                        &mut scratch.primary,
                        &mut scratch.secondary,
                    );
                    let z_diag_box = DiagBoxArr::streamed_extraction_from_data(
                        &rows,
                        options.tol_lstsq,
                        z_data,
                        subs_sample_dim,
                        false,
                        &mut scratch.tertiary,
                        &mut scratch.quaternary,
                    );
                    let mut diag_box = y_diag_box;
                    diag_box.sum_into(z_diag_box.r().transpose().conj());
                    diag_box
                        .r_mut()
                        .scale_inplace(num::NumCast::from(0.5).unwrap());
                    DiagBoxArr::from_extracted(&diag_box, options)
                } else {
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

                    DiagBoxArr::new_no_symm_with_scratch(
                        &rows,
                        options,
                        &y_sub_test,
                        &y_sub_sketch,
                        &z_sub_test,
                        &z_sub_sketch,
                        false,
                        false,
                        &mut scratch.primary,
                        &mut scratch.secondary,
                        &mut scratch.tertiary,
                        &mut scratch.quaternary,
                        &mut scratch.quinary,
                    )
                },
                inds: rows,
            }),
            times,
        )
    }

    pub(crate) fn new_complex_symm_with_scratch(
        rows: Vec<usize>,
        y_data: &SketchData<Item>,
        subs_sample_dim: usize,
        fixed_rank: bool,
        options: &ExtractOptions<Item>,
        scratch: &mut DiagExtractionScratch<Item>,
    ) -> (Option<Self>, Times) {
        let diag_times = LuTimes {
            extraction: 0_u128,
            lu: 0_u128,
        };

        let times = Times::Lu(diag_times);

        (
            Some(Self {
                arr: if fixed_rank
                    && matches!(
                        options.block_extraction_method,
                        BlockExtractionMethod::LuLstSq
                    )
                    && !prefer_direct_diag_extraction::<Item>(rows.len(), subs_sample_dim, 4)
                {
                    let y_diag_box = DiagBoxArr::streamed_extraction_from_data(
                        &rows,
                        options.tol_lstsq,
                        y_data,
                        subs_sample_dim,
                        false,
                        &mut scratch.primary,
                        &mut scratch.secondary,
                    );
                    let z_diag_box = DiagBoxArr::streamed_extraction_from_data(
                        &rows,
                        options.tol_lstsq,
                        y_data,
                        subs_sample_dim,
                        true,
                        &mut scratch.tertiary,
                        &mut scratch.quaternary,
                    );
                    let mut diag_box = y_diag_box;
                    diag_box.sum_into(z_diag_box.r().transpose().conj());
                    diag_box
                        .r_mut()
                        .scale_inplace(num::NumCast::from(0.5).unwrap());
                    DiagBoxArr::from_extracted(&diag_box, options)
                } else {
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

                    DiagBoxArr::new_no_symm_with_scratch(
                        &rows,
                        options,
                        &y_sub_test,
                        &y_sub_sketch,
                        &y_sub_test,
                        &y_sub_sketch,
                        false,
                        true,
                        &mut scratch.primary,
                        &mut scratch.secondary,
                        &mut scratch.tertiary,
                        &mut scratch.quaternary,
                        &mut scratch.quinary,
                    )
                },
                inds: rows,
            }),
            times,
        )
    }

    pub fn cond(&self) -> (CondType<Item>, Option<CondType<Item>>) {
        match &self.arr {
            DiagBoxArr::Reg(reg_dbox) => ((condition_number(&reg_dbox.arr), None), None),
            DiagBoxArr::Lu(_) => (((num::Zero::zero(), num::Zero::zero()), None), None),
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
                    factor_options,
                );
                self.arr.mul(&mut target_rows, &Side::Left, factor_options);
                target_rows
            }
            Side::Right => {
                let mut target_cols = ext_cols(
                    self.inds.clone(),
                    self.inds.clone(),
                    target_arr,
                    factor_options,
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
        thread_pool: &ThreadPool,
        num_threads: usize,
        factor_options: &MulOptions,
    );
    #[allow(clippy::type_complexity)]
    fn get_condition_numbers(&self) -> Vec<(CondType<Self::Item>, Option<CondType<Self::Item>>)>;
    fn flush(&mut self);
}

#[cfg(debug_assertions)]
fn indices_disjoint(a: &[usize], b: &[usize]) -> bool {
    let a_set: HashSet<usize> = a.iter().copied().collect();
    b.iter().all(|idx| !a_set.contains(idx))
}

#[cfg(debug_assertions)]
fn factor_apply_options<Item: RlstScalar>(
    factor: &Factor<Item>,
    factor_options: &MulOptions,
) -> BaseFactorOptions {
    match factor {
        Factor::Lu(_) => match factor_options.factor_type {
            FactorType::F => factor_options.base_options.transpose(),
            FactorType::S => factor_options.base_options.clone(),
        },
        Factor::Id(_) => match factor_options.factor_type {
            FactorType::F => factor_options.base_options.clone(),
            FactorType::S => factor_options.base_options.transpose(),
        },
        Factor::Diag(_) => factor_options.base_options.clone(),
    }
}

#[cfg(debug_assertions)]
fn factor_read_write_indices<Item: RlstScalar>(
    factor: &Factor<Item>,
    side: &Side,
    base_options: &BaseFactorOptions,
) -> (Vec<usize>, Vec<usize>) {
    let layout = match factor {
        Factor::Lu(lu_factor) => {
            factor_apply_layout(side, base_options, &lu_factor.ind_t, &lu_factor.ind_r)
        }
        Factor::Id(id_factor) => {
            factor_apply_layout(side, base_options, &id_factor.ind_s, &id_factor.ind_r)
        }
        Factor::Diag(diag_factor) => {
            factor_apply_layout(side, base_options, &diag_factor.inds, &diag_factor.inds)
        }
    };
    (
        layout.source_indices.to_vec(),
        layout.target_indices.to_vec(),
    )
}

#[cfg(debug_assertions)]
fn assert_chunk_noninterfering<Item: RlstScalar>(
    factors: &[Factor<Item>],
    factor_options: &MulOptions,
) {
    let footprints: Vec<_> = factors
        .iter()
        .map(|factor| {
            let base_options = factor_apply_options(factor, factor_options);
            factor_read_write_indices(factor, &factor_options.side, &base_options)
        })
        .collect();

    for left in 0..footprints.len() {
        for right in (left + 1)..footprints.len() {
            let (left_reads, left_writes) = &footprints[left];
            let (right_reads, right_writes) = &footprints[right];
            debug_assert!(
                indices_disjoint(left_writes, right_writes),
                "factor write sets overlap in parallel chunk"
            );
            debug_assert!(
                indices_disjoint(left_reads, right_writes),
                "factor read/write sets overlap in parallel chunk"
            );
            debug_assert!(
                indices_disjoint(left_writes, right_reads),
                "factor write/read sets overlap in parallel chunk"
            );
        }
    }
}

impl<
        Item: RlstScalar
            + MatrixId
            + MatrixIdNoSkel
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
        thread_pool: &ThreadPool,
        num_threads: usize,
        factor_options: &MulOptions,
    ) where
        Self: Sized,
    {
        let chunk_size = num_threads.max(1);

        for chunk_start in (0..self.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(self.len());
            let factors = &self[chunk_start..chunk_end];
            let direct_update_chunk = factors
                .iter()
                .all(|factor| !matches!(factor, Factor::Diag(_)));

            if direct_update_chunk {
                #[cfg(debug_assertions)]
                assert_chunk_noninterfering(factors, factor_options);

                let raw_target = raw_matrix_mut(target_arr);
                let read_target: &Array<Self::Item, ArrayImplMut, 2> =
                    unsafe { &*(target_arr as *const Array<Self::Item, ArrayImplMut, 2>) };

                thread_pool.install(|| {
                    factors.par_iter().for_each_init(
                        FactorApplyScratch::<Item>::new,
                        |scratch, factor| match factor {
                            Factor::Lu(lu_factor) => {
                                let base_options = match factor_options.factor_type {
                                    FactorType::F => factor_options.base_options.transpose(),
                                    FactorType::S => factor_options.base_options.clone(),
                                };
                                unsafe {
                                    lu_factor.apply_delta_with_scratch_raw(
                                        read_target,
                                        raw_target,
                                        &factor_options.side,
                                        Some(factor_options.factor_type.clone()),
                                        &base_options,
                                        scratch,
                                    )
                                };
                            }
                            Factor::Id(id_factor) => {
                                let base_options = match factor_options.factor_type {
                                    FactorType::F => factor_options.base_options.clone(),
                                    FactorType::S => factor_options.base_options.transpose(),
                                };
                                unsafe {
                                    id_factor.apply_delta_with_scratch_raw(
                                        read_target,
                                        raw_target,
                                        &factor_options.side,
                                        &base_options,
                                        scratch,
                                    )
                                };
                            }
                            Factor::Diag(_) => unreachable!("diag factors use the fallback path"),
                        },
                    )
                });
            } else {
                let updated_t_arr_blocks: Vec<_> = thread_pool.install(|| {
                    factors
                        .par_iter()
                        .enumerate()
                        .map(|(offset, factor)| {
                            let factor_ind = chunk_start + offset;
                            let target_block = match factor {
                                Factor::Lu(lu_factor) => lu_factor.mul_data(
                                    target_arr,
                                    &factor_options.side,
                                    Some(factor_options.factor_type.clone()),
                                    &match factor_options.factor_type {
                                        FactorType::F => factor_options.base_options.transpose(),
                                        FactorType::S => factor_options.base_options.clone(),
                                    },
                                ),
                                Factor::Id(id_factor) => id_factor.mul_data(
                                    target_arr,
                                    &factor_options.side,
                                    None,
                                    &match factor_options.factor_type {
                                        FactorType::F => factor_options.base_options.clone(),
                                        FactorType::S => factor_options.base_options.transpose(),
                                    },
                                ),
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

                for (factor_ind, target_block) in updated_t_arr_blocks {
                    let factor = &self[factor_ind];
                    match factor {
                        Factor::Lu(lu_factor) => {
                            let base_options = match factor_options.factor_type {
                                FactorType::F => factor_options.base_options.transpose(),
                                FactorType::S => factor_options.base_options.clone(),
                            };
                            lu_factor.ins_data(
                                &target_block,
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
                            id_factor.ins_data(
                                &target_block,
                                target_arr,
                                &factor_options.side,
                                Some(factor_options.factor_type.clone()),
                                &base_options,
                            )
                        }
                        Factor::Diag(diag_factor) => diag_factor.ins_data(
                            &target_block,
                            target_arr,
                            &factor_options.side,
                            Some(factor_options.factor_type.clone()),
                            &factor_options.base_options,
                        ),
                    };
                }
            }
        }
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
