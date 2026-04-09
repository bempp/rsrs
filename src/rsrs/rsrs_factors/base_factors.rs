use crate::utils::data_ins_ext::extract_axis_into;
use itertools::min;
use rlst::{
    dense::linalg::lu::{MatrixLu, SquareLuFactors},
    prelude::*,
};

type Real<T> = <T as rlst::RlstScalar>::Real;
type CNTuple<T> = (Real<T>, Real<T>); //TODO: Remove before releasing
pub type CondType<T> = (CNTuple<T>, Option<(CNTuple<T>, CNTuple<T>)>); //TODO: Remove before releasing

pub fn condition_number<Item: RlstScalar + MatrixSvd>(
    //TODO: Remove before releasing
    mat: &DynamicArray<Item, 2>,
) -> CNTuple<Item> {
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

    (sigma_max / sigma_min, sigma_max)
}

/// Basic flags used when applying an elementary RSRS factor.
///
/// `trans` describes the orientation of the factor itself, while `trans_target`
/// keeps track of whether the current input is viewed through a transposed
/// row/column layout.
#[derive(Clone)]
pub struct BaseFactorOptions {
    /// Inverse operation
    pub inv: bool,
    /// Transpose operation
    pub trans: TransMode,
    /// Transpose vector or matrix when applying factor
    pub trans_target: bool,
}

pub(crate) struct FactorApplyScratch<Item: RlstScalar> {
    pub source: DynamicArray<Item, 2>,
    pub result: DynamicArray<Item, 2>,
}

pub(crate) struct FactorApplyLayout<'a> {
    pub source_indices: &'a [usize],
    pub target_indices: &'a [usize],
    pub axis: usize,
    pub transposed: bool,
}

impl<Item: RlstScalar> FactorApplyScratch<Item> {
    pub(crate) fn new() -> Self {
        Self {
            source: empty_array(),
            result: empty_array(),
        }
    }
}

pub(crate) fn factor_apply_layout<'a>(
    side: &Side,
    factor_options: &BaseFactorOptions,
    c_indices: &'a [usize],
    r_indices: &'a [usize],
) -> FactorApplyLayout<'a> {
    // The low-level delta kernels operate on "source" entries that are read
    // and "target" entries that are updated. Left/right application, factor
    // transpose, and transposed vector views all permute those roles, so we
    // normalize the layout once here.
    match side {
        Side::Left => {
            let (source_indices, target_indices) = if !factor_options.trans_val() {
                (c_indices, r_indices)
            } else {
                (r_indices, c_indices)
            };

            if factor_options.trans_target {
                FactorApplyLayout {
                    source_indices,
                    target_indices,
                    axis: 1,
                    transposed: true,
                }
            } else {
                FactorApplyLayout {
                    source_indices,
                    target_indices,
                    axis: 0,
                    transposed: false,
                }
            }
        }
        Side::Right => {
            let (source_indices, target_indices) = if !factor_options.trans_val() {
                (r_indices, c_indices)
            } else {
                (c_indices, r_indices)
            };

            if factor_options.trans_target {
                FactorApplyLayout {
                    source_indices,
                    target_indices,
                    axis: 0,
                    transposed: true,
                }
            } else {
                FactorApplyLayout {
                    source_indices,
                    target_indices,
                    axis: 1,
                    transposed: false,
                }
            }
        }
    }
}

pub(crate) fn conjugate_array_in_place<Item: RlstScalar>(arr: &mut DynamicArray<Item, 2>) {
    arr.data_mut()
        .iter_mut()
        .for_each(|elem| *elem = elem.conj());
}

pub(crate) fn assert_transpose_only_mode(trans: TransMode, context: &str) {
    match trans {
        TransMode::NoTrans | TransMode::Trans => {}
        TransMode::ConjNoTrans | TransMode::ConjTrans => {
            panic!(
                "{context} received a conjugating transposition mode. Conjugation must be handled above the low-level factor kernels."
            );
        }
    }
}

/// Handler to manage the factor basic options
impl BaseFactorOptions {
    /// Returns whether the factor is applied in transposed orientation.
    ///
    /// This only answers the layout question used to swap source and target
    /// indices. Any conjugation required for complex or Hermitian factors is
    /// handled separately in the concrete factor implementations.
    pub fn trans_val(&self) -> bool {
        match self.trans {
            TransMode::NoTrans => false,
            TransMode::ConjNoTrans => false,
            TransMode::Trans => true,
            TransMode::ConjTrans => true,
        }
    }

    /// Flip the factor orientation between `NoTrans` and `Trans`.
    ///
    /// Conjugating transpose modes are normalized before this helper is used,
    /// so they are intentionally left unsupported here.
    pub fn transpose(&self) -> Self {
        let mut new_options = self.clone();
        match self.trans {
            TransMode::NoTrans => new_options.trans = TransMode::Trans,
            TransMode::ConjNoTrans => panic!(
                "BaseFactorOptions::transpose cannot be used with conjugating modes. Conjugation must be normalized before reaching low-level factors."
            ),
            TransMode::Trans => new_options.trans = TransMode::NoTrans,
            TransMode::ConjTrans => panic!(
                "BaseFactorOptions::transpose cannot be used with conjugating modes. Conjugation must be normalized before reaching low-level factors."
            ),
        };

        new_options
    }

    /// Toggle between applying the factor and its inverse.
    pub fn invert(&self) -> Self {
        let mut new_options = self.clone();
        new_options.inv = !self.inv;
        new_options
    }
}

/// Lu Decomposition of a square box.
pub struct LuSMat<T: RlstScalar> {
    /// Reusable LU factors for trusted multiplications and solves.
    pub square_factors: SquareLuFactors<T>,
}

/// Stores a square matrix in a regular array.
pub struct RegSMat<T: RlstScalar> {
    /// Storage for A
    pub arr: DynamicArray<T, 2>,
    /// Storage for A^-1
    pub inv_arr: DynamicArray<T, 2>,
}
// A square matrix can either be stored in a LU factor or as a dense matrix with its inverse.
pub enum SquareArr<T: RlstScalar> {
    Reg(RegSMat<T>),
    Lu(LuSMat<T>),
}

/// Likewise, diagonal blocks can either be stored in a LU factor or as a dense matrix with its inverse.
pub enum DiagBoxArr<T: RlstScalar> {
    Reg(RegSMat<T>),
    Lu(LuSMat<T>),
}

/// Implementation of the multiplication operations of SquareArr.
impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse + MatrixInverse> SquareArr<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    /// Implementation of SquareArr * x (multiplication by the left).
    fn left_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        factor_options: &BaseFactorOptions,
    ) {
        assert_transpose_only_mode(factor_options.trans, "SquareArr::left_mul");
        match self {
            SquareArr::Reg(ref reg) => {
                let mut new_target_arr = empty_array();
                if factor_options.inv {
                    // Transposition is built in.
                    // A / x when A is stored as a dense matrix.
                    new_target_arr.r_mut().mult_into_resize(
                        factor_options.trans,
                        TransMode::NoTrans,
                        num::One::one(),
                        reg.inv_arr.r(),
                        target_arr.r(),
                        num::Zero::zero(),
                    );
                } else {
                    // A * x when A is stored as a dense matrix.
                    new_target_arr.r_mut().mult_into_resize(
                        factor_options.trans,
                        TransMode::NoTrans,
                        num::One::one(),
                        reg.arr.r(),
                        target_arr.r(),
                        num::Zero::zero(),
                    );
                }
                target_arr.r_mut().fill_from(new_target_arr.r());
            }
            SquareArr::Lu(ref lu) => {
                if factor_options.inv {
                    lu.square_factors
                        .solve_mat(factor_options.trans, target_arr.r_mut())
                        .unwrap();
                } else {
                    lu.square_factors
                        .mul_mat(factor_options.trans, target_arr.r_mut())
                        .unwrap();
                }
            }
        }
    }

    /// Multiplication by a square matrix. Right multiplication and inversion are defined as (A'*x')' and (A'\x')'
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
        side: Side,
        factor_options: &BaseFactorOptions,
    ) {
        assert_transpose_only_mode(factor_options.trans, "SquareArr::mul");
        match side {
            Side::Left => self.left_mul(arr, factor_options),
            Side::Right => match self {
                SquareArr::Reg(ref reg) => {
                    let mut new_target_arr = empty_array();
                    if factor_options.inv {
                        new_target_arr.r_mut().mult_into_resize(
                            TransMode::NoTrans,
                            factor_options.trans,
                            num::One::one(),
                            arr.r(),
                            reg.inv_arr.r(),
                            num::Zero::zero(),
                        );
                    } else {
                        new_target_arr.r_mut().mult_into_resize(
                            TransMode::NoTrans,
                            factor_options.trans,
                            num::One::one(),
                            arr.r(),
                            reg.arr.r(),
                            num::Zero::zero(),
                        );
                    }
                    arr.fill_from(new_target_arr.r());
                }
                SquareArr::Lu(ref lu) => {
                    let mut aux_arr = empty_array();
                    aux_arr.r_mut().fill_from_resize(arr.r().transpose());
                    let aux_factor_options = factor_options.transpose();
                    if factor_options.inv {
                        lu.square_factors
                            .solve_mat(aux_factor_options.trans, aux_arr.r_mut())
                            .unwrap();
                    } else {
                        lu.square_factors
                            .mul_mat(aux_factor_options.trans, aux_arr.r_mut())
                            .unwrap();
                    }
                    arr.fill_from(aux_arr.r().transpose());
                }
            },
        }
    }

    /// Compute condition number of square matrix.
    pub fn cond(&self) -> (CNTuple<Item>, CNTuple<Item>) {
        //TODO: This function should not be included when releasing.
        match self {
            SquareArr::Reg(reg_dbox) => (
                condition_number(&reg_dbox.arr),
                (num::Zero::zero(), num::Zero::zero()),
            ),
            SquareArr::Lu(_) => (
                (num::Zero::zero(), num::Zero::zero()),
                (num::Zero::zero(), num::Zero::zero()),
            ),
        }
    }
}

/// When computing LU factorisation in RSRS,
/// we can either store A = P / R (where P
/// is the pivot extracted from the LU factorisation)or store
/// P and R independently, to then operate
/// them on the go. The second option should be better
/// if the pivot P is ill-conditioned.
///
/// Composed elementary matrix stores `X_rr` and `X_rn` independently.
pub struct ComposedFactorData<T: RlstScalar> {
    /// Pivot
    pub sq: SquareArr<T>,
    /// Rectangular part of the factor
    pub rectg: RectArr<T>,
}

/// RectArray store A = P / R
pub struct RectArr<T: RlstScalar> {
    // TODO: Maybe change to just an array.
    /// Rectangular array.
    pub arr: Box<DynamicArray<T, 2>>,
}

/// Multiplication by a composed factor.
impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse + MatrixInverse> ComposedFactorData<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    fn left_mul_into(
        &self,
        target_arr: &mut DynamicArray<Item, 2>,
        factor_options: &BaseFactorOptions,
        res_mul: &mut DynamicArray<Item, 2>,
    ) {
        let mut sq_factor_options = factor_options.clone();
        sq_factor_options.inv = true;

        if !factor_options.trans_val() {
            res_mul.r_mut().mult_into_resize(
                TransMode::NoTrans,
                TransMode::NoTrans,
                num::One::one(),
                self.rectg.arr.r(),
                target_arr.r(),
                num::Zero::zero(),
            );
            self.sq.mul(res_mul, Side::Left, &sq_factor_options);
        } else {
            self.sq.mul(target_arr, Side::Left, &sq_factor_options);
            res_mul.r_mut().mult_into_resize(
                TransMode::Trans,
                TransMode::NoTrans,
                num::One::one(),
                self.rectg.arr.r(),
                target_arr.r(),
                num::Zero::zero(),
            );
        }
    }

    /// Multiplication by Composed factor. Right multiplication is defined as (A'*x')'
    fn mul_into(
        &self,
        target_arr: &mut DynamicArray<Item, 2>,
        side: &Side,
        factor_options: &BaseFactorOptions,
        res_mul: &mut DynamicArray<Item, 2>,
    ) {
        match side {
            Side::Left => self.left_mul_into(target_arr, factor_options, res_mul),
            Side::Right => {
                let mut sq_factor_options = factor_options.clone();
                sq_factor_options.inv = true;

                if !factor_options.trans_val() {
                    self.sq.mul(target_arr, Side::Right, &sq_factor_options);
                    res_mul.r_mut().mult_into_resize(
                        TransMode::NoTrans,
                        TransMode::NoTrans,
                        num::One::one(),
                        target_arr.r(),
                        self.rectg.arr.r(),
                        num::Zero::zero(),
                    );
                } else {
                    res_mul.r_mut().mult_into_resize(
                        TransMode::NoTrans,
                        TransMode::Trans,
                        num::One::one(),
                        target_arr.r(),
                        self.rectg.arr.r(),
                        num::Zero::zero(),
                    );
                    self.sq.mul(res_mul, Side::Right, &sq_factor_options);
                }
            }
        }
    }

    /// Compute condition number of R and P
    #[allow(clippy::type_complexity)]
    pub fn cond(&self) -> CondType<Item> {
        //TODO: Remove before releasing
        (condition_number(&self.rectg.arr), Some(self.sq.cond()))
    }
}

/// Multiplication by a rectangular factor.
impl<Item: RlstScalar + MatrixSvd> RectArr<Item> {
    /// Implementation of b=A*x. Returns the result in a new array.
    fn left_mul_into(
        &self,
        target_arr: &DynamicArray<Item, 2>,
        factor_options: &BaseFactorOptions,
        res_mul: &mut DynamicArray<Item, 2>,
    ) {
        assert_transpose_only_mode(factor_options.trans, "RectArr::left_mul_into");
        res_mul.r_mut().mult_into_resize(
            factor_options.trans,
            TransMode::NoTrans,
            num::One::one(),
            self.arr.r(),
            target_arr.r(),
            num::Zero::zero(),
        );
    }
    /// Multiplication by a Rectangular factor. Right multiplication is defined as (A'*x')'
    fn mul_into(
        &self,
        target_arr: &DynamicArray<Item, 2>,
        side: &Side,
        factor_options: &BaseFactorOptions,
        res_mul: &mut DynamicArray<Item, 2>,
    ) {
        assert_transpose_only_mode(factor_options.trans, "RectArr::mul_into");
        match side {
            Side::Left => self.left_mul_into(target_arr, factor_options, res_mul),
            Side::Right => {
                res_mul.r_mut().mult_into_resize(
                    TransMode::NoTrans,
                    factor_options.trans,
                    num::One::one(),
                    target_arr.r(),
                    self.arr.r(),
                    num::Zero::zero(),
                );
            }
        }
    }

    /// Compute condition number of the rectangular array.
    #[allow(clippy::type_complexity)]
    pub fn cond(&self) -> CondType<Item> {
        //TODO: Remove before releasing.
        (condition_number(&self.arr), None)
    }
}

/// An elementary factor can either be defined by a composed or a rectangular
/// factor depending on the user settings. Factor data stores F in the application I+/-F.
pub enum FactorData<T: RlstScalar> {
    Comp(ComposedFactorData<T>),
    Reg(RectArr<T>),
}

/// Implementation of the Elementary Matrix multiplication and inversion.
impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse + MatrixInverse> FactorData<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    /// Multiplication (b=(I+/-F)*x). Instead of using full I and full F,
    /// we only use and operate on the bits of data relevant for this operation.
    ///
    /// Arguments:
    /// target_arr: x
    /// side: factor can either multiply by the left or the right.
    /// factor_options: A can be inverted or transposed and x can also be transposed.
    /// c_indices: domain indices (rows to extract in x).
    /// r_indices: range indices (rows to modify in x).
    pub fn mul<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &Array<Item, ArrayImpl, 2>,
        side: &Side,
        factor_options: &BaseFactorOptions,
        c_indices: &[usize],
        r_indices: &[usize],
    ) -> DynamicArray<Item, 2> {
        let mut scratch = FactorApplyScratch::new();
        self.mul_with_scratch(
            target_arr,
            side,
            factor_options,
            c_indices,
            r_indices,
            &mut scratch,
        )
    }

    pub(crate) fn mul_with_scratch<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &Array<Item, ArrayImpl, 2>,
        side: &Side,
        factor_options: &BaseFactorOptions,
        c_indices: &[usize],
        r_indices: &[usize],
        scratch: &mut FactorApplyScratch<Item>,
    ) -> DynamicArray<Item, 2> {
        let layout = self.delta_with_scratch(
            target_arr,
            side,
            factor_options,
            c_indices,
            r_indices,
            false,
            scratch,
        );
        let mut subarr_target = empty_array();
        extract_axis_into(
            &mut subarr_target,
            target_arr,
            layout.target_indices,
            layout.axis,
            layout.transposed,
        );

        if factor_options.inv {
            // res = (I-F)*x
            subarr_target.sub_into(scratch.result.r());
        } else {
            // res = (I+F)*x
            subarr_target.sum_into(scratch.result.r());
        }

        subarr_target
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn delta_with_scratch<
        'a,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &Array<Item, ArrayImpl, 2>,
        side: &Side,
        factor_options: &BaseFactorOptions,
        c_indices: &'a [usize],
        r_indices: &'a [usize],
        conjugate_source: bool,
        scratch: &mut FactorApplyScratch<Item>,
    ) -> FactorApplyLayout<'a> {
        let layout = factor_apply_layout(side, factor_options, c_indices, r_indices);

        extract_axis_into(
            &mut scratch.source,
            target_arr,
            layout.source_indices,
            layout.axis,
            layout.transposed,
        );

        if conjugate_source {
            conjugate_array_in_place(&mut scratch.source);
        }

        match self {
            FactorData::Comp(composed_factor_data) => composed_factor_data.mul_into(
                &mut scratch.source,
                side,
                factor_options,
                &mut scratch.result,
            ),
            FactorData::Reg(rectg) => {
                rectg.mul_into(&scratch.source, side, factor_options, &mut scratch.result)
            }
        }

        layout
    }

    /// Compute condition number of the application (I+/-F)
    pub fn cond(&self) -> CondType<Item> {
        //TODO: Remove before releasing
        match self {
            FactorData::Comp(composed_factor_data) => composed_factor_data.cond(),
            FactorData::Reg(rectg) => rectg.cond(),
        }
    }
}
