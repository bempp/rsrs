use crate::{
    rsrs::rsrs_factors::commutative_factors::PermFactor,
    utils::data_ins_ext::{ExtInsType, Extraction, MatrixExtraction},
};
use itertools::min;
use rlst::{dense::linalg::lu::MatrixLu, prelude::*};

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

/// Basic options for each factor:
/// inversion, transposition or transposing the right side.
#[derive(Clone, Debug)]
pub struct BaseFactorOptions {
    /// Inverse operation
    pub inv: bool,
    /// Transpose operation
    pub trans: TransMode,
    /// Transpose vector or matrix when applying factor
    pub trans_target: bool,
}

/// Handler to manage the factor basic options
impl BaseFactorOptions {
    /// Returns true if the factor is transposed and no it it isn't.
    /// Conjugations of the factor are not implemented for simplicity.
    pub fn trans_val(&self) -> bool {
        match self.trans {
            TransMode::NoTrans => false,
            TransMode::ConjNoTrans => todo!(),
            TransMode::Trans => true,
            TransMode::ConjTrans => todo!(),
        }
    }

    /// Transpose the factor.
    pub fn transpose(&self) -> Self {
        let mut new_options = self.clone();
        match self.trans {
            TransMode::NoTrans => new_options.trans = TransMode::Trans,
            TransMode::ConjNoTrans => todo!(),
            TransMode::Trans => new_options.trans = TransMode::NoTrans,
            TransMode::ConjTrans => todo!(),
        };
        new_options
    }

    /// Invert the factor.
    pub fn invert(&self) -> Self {
        let mut new_options = self.clone();
        new_options.inv = !self.inv;
        new_options
    }
}

/// Lu Decomposition of a square box.
pub struct LuSMat<T: RlstScalar> {
    /// - U: upper triangular (Stored in a triangular matrix object).
    pub u_arr: TriangularMatrix<T>,
    /// - L: lower triangular (Stored in a triangular matrix object).
    pub l_arr: TriangularMatrix<T>,
    /// - P: permutation associated to the LU factorisation (PermFactor object).
    pub perm: PermFactor,
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
                    match factor_options.trans {
                        TransMode::NoTrans => {
                            // Returns b = A / x when A = PLU, wit P^-1=P^T.
                            // a_1 = P * x
                            lu.perm.left_mul(target_arr, factor_options);
                            // a_2 = L /a_1
                            <TriangularMatrix<Item> as TriangularOperations>::solve(
                                &lu.l_arr,
                                target_arr,
                                Side::Left,
                                TransMode::NoTrans,
                            );
                            // b = U / a_2
                            <TriangularMatrix<Item> as TriangularOperations>::solve(
                                &lu.u_arr,
                                target_arr,
                                Side::Left,
                                TransMode::NoTrans,
                            );
                        }
                        TransMode::Trans => {
                            // Returns b = A' / x when A = PLU, wit P^-1=P^T.
                            // a_1 = U' / b
                            <TriangularMatrix<Item> as TriangularOperations>::solve(
                                &lu.u_arr,
                                target_arr,
                                Side::Left,
                                TransMode::Trans,
                            );
                            // a_2 = L' /a_1
                            <TriangularMatrix<Item> as TriangularOperations>::solve(
                                &lu.l_arr,
                                target_arr,
                                Side::Left,
                                TransMode::Trans,
                            );
                            // b = P' * x
                            lu.perm.left_mul(target_arr, factor_options);
                        }
                        TransMode::ConjNoTrans => todo!(),
                        TransMode::ConjTrans => todo!(),
                    }
                } else {
                    match factor_options.trans {
                        // Returns b = A * x when A = PLU, wit P^-1=P^T.
                        TransMode::NoTrans => {
                            // a_1 = U * a_2
                            <TriangularMatrix<Item> as TriangularOperations>::mul(
                                &lu.u_arr,
                                target_arr,
                                Side::Left,
                                TransMode::NoTrans,
                            );
                            // a_2 = L * a_1
                            <TriangularMatrix<Item> as TriangularOperations>::mul(
                                &lu.l_arr,
                                target_arr,
                                Side::Left,
                                TransMode::NoTrans,
                            );
                            // b = P * x
                            lu.perm.left_mul(target_arr, factor_options);
                        }
                        TransMode::Trans => {
                            // Returns b = A' * x when A = PLU, wit P^-1=P^T.
                            // a_1 = P' * x
                            lu.perm.left_mul(target_arr, factor_options);
                            // a_2 = L' * a_1
                            <TriangularMatrix<Item> as TriangularOperations>::mul(
                                &lu.l_arr,
                                target_arr,
                                Side::Left,
                                TransMode::Trans,
                            );
                            // b = U' * a_2
                            <TriangularMatrix<Item> as TriangularOperations>::mul(
                                &lu.u_arr,
                                target_arr,
                                Side::Left,
                                TransMode::Trans,
                            );
                        }
                        TransMode::ConjNoTrans => todo!(),
                        TransMode::ConjTrans => todo!(),
                    }
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
        match side {
            Side::Left => self.left_mul(arr, factor_options),
            Side::Right => {
                // Transpose x
                let mut aux_arr = empty_array();
                aux_arr.r_mut().fill_from_resize(arr.r().transpose());
                let mut trans_factor_options = factor_options.clone();

                // Transpose A
                trans_factor_options.trans = match factor_options.trans {
                    TransMode::NoTrans => TransMode::Trans,
                    TransMode::ConjNoTrans => todo!(),
                    TransMode::Trans => TransMode::NoTrans,
                    TransMode::ConjTrans => todo!(),
                };

                // Compute A'*x' = b'
                self.left_mul(&mut aux_arr, &trans_factor_options);

                // Transpose b'
                arr.fill_from(aux_arr.r().transpose());
            }
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
            SquareArr::Lu(lu_dbox) => (
                condition_number(&lu_dbox.l_arr.tri),
                condition_number(&lu_dbox.u_arr.tri),
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
//// Composed elementary matrix stores X_rr and X_rn independently
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
    pub arr: DynamicArray<T, 2>,
}

/// Multiplication by a composed factor.
impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse + MatrixInverse> ComposedFactorData<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    /// Implementation of b=P/(R*x). Returns the result in a new array.
    fn left_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &Array<Item, ArrayImplMut, 2>,
        factor_options: &BaseFactorOptions,
    ) -> DynamicArray<Item, 2> {
        let mut res_mul: DynamicArray<Item, 2> = empty_array::<Item, 2>();

        // Instructions for pivot. The pivot is inverted by default.
        let mut sq_factor_options = factor_options.clone();
        sq_factor_options.inv = true;

        match factor_options.trans {
            TransMode::NoTrans => {
                // y = R*x
                res_mul.r_mut().mult_into_resize(
                    TransMode::NoTrans,
                    TransMode::NoTrans,
                    num::One::one(),
                    self.rectg.arr.r(),
                    target_arr.r(),
                    num::Zero::zero(),
                );
                // b = P / y
                self.sq.mul(&mut res_mul, Side::Left, &sq_factor_options);
            }
            TransMode::ConjNoTrans => todo!(),
            TransMode::Trans => {
                // y = P' / x
                let mut aux_target_arr = empty_array();
                aux_target_arr.r_mut().fill_from_resize(target_arr.r());
                self.sq
                    .mul(&mut aux_target_arr.r_mut(), Side::Left, &sq_factor_options);

                // b = R'*y
                res_mul.r_mut().mult_into_resize(
                    TransMode::Trans,
                    TransMode::NoTrans,
                    num::One::one(),
                    self.rectg.arr.r(),
                    aux_target_arr.r(),
                    num::Zero::zero(),
                );
            }
            TransMode::ConjTrans => todo!(),
        };

        res_mul
    }

    /// Multiplication by Composed factor. Right multiplication is defined as (A'*x')'
    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &Array<Item, ArrayImplMut, 2>,
        side: &Side,
        factor_options: &BaseFactorOptions,
    ) -> DynamicArray<Item, 2> {
        match side {
            Side::Left => self.left_mul(target_arr, factor_options),
            Side::Right => {
                // Transpose x
                let mut aux_arr = empty_array();
                aux_arr.r_mut().fill_from_resize(target_arr.r().transpose());
                let mut trans_factor_options = factor_options.clone();

                // Transpose P^1 and R
                trans_factor_options.trans = match factor_options.trans {
                    TransMode::NoTrans => TransMode::Trans,
                    TransMode::ConjNoTrans => todo!(),
                    TransMode::Trans => TransMode::NoTrans,
                    TransMode::ConjTrans => todo!(),
                };

                // b'=R'*(P'/x').
                let aux_arr = self.left_mul(&aux_arr, &trans_factor_options);

                // Transpose b
                let mut res_mul = empty_array();
                res_mul.r_mut().fill_from_resize(aux_arr.r().transpose());
                res_mul
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
    fn left_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &Array<Item, ArrayImplMut, 2>,
        factor_options: &BaseFactorOptions,
    ) -> DynamicArray<Item, 2> {
        let mut res_mul: DynamicArray<Item, 2> = empty_array::<Item, 2>();
        res_mul.r_mut().mult_into_resize(
            factor_options.trans,
            TransMode::NoTrans,
            num::One::one(),
            self.arr.r(),
            target_arr.r(),
            num::Zero::zero(),
        );

        res_mul
    }
    /// Multiplication by a Rectangular factor. Right multiplication is defined as (A'*x')'
    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &Array<Item, ArrayImplMut, 2>,
        side: &Side,
        factor_options: &BaseFactorOptions,
    ) -> DynamicArray<Item, 2> {
        match side {
            Side::Left => self.left_mul(target_arr, factor_options),
            Side::Right => {
                // Transpose x
                let mut aux_arr = empty_array();
                aux_arr.r_mut().fill_from_resize(target_arr.r().transpose());

                // Transpose A
                let mut trans_factor_options = factor_options.clone();
                trans_factor_options.trans = match factor_options.trans {
                    TransMode::NoTrans => TransMode::Trans,
                    TransMode::ConjNoTrans => todo!(),
                    TransMode::Trans => TransMode::NoTrans,
                    TransMode::ConjTrans => todo!(),
                };

                // Compute b=A*x
                let aux_arr = self.left_mul(&aux_arr, &trans_factor_options);
                let mut res_mul = empty_array();

                // Transpose b
                res_mul.r_mut().fill_from_resize(aux_arr.r().transpose());
                res_mul
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
        let row_indices: Vec<usize>;
        let col_indices: Vec<usize>;

        // axis indicates if extraction should be done either row-wise or column-wise.
        let (axis, transposed) = match side {
            Side::Left => {
                // Transposition of the operation implies swapping domain by range.
                match factor_options.trans_val() {
                    false => {
                        col_indices = c_indices.to_vec();
                        row_indices = r_indices.to_vec();
                    }
                    true => {
                        col_indices = r_indices.to_vec();
                        row_indices = c_indices.to_vec();
                    }
                }

                if factor_options.trans_target {
                    (1, true)
                } else {
                    (0, false)
                }
            }
            Side::Right => {
                // Right operation => All operations are transposed.
                match factor_options.trans_val() {
                    false => {
                        col_indices = r_indices.to_vec();
                        row_indices = c_indices.to_vec();
                    }
                    true => {
                        col_indices = c_indices.to_vec();
                        row_indices = r_indices.to_vec();
                    }
                }

                if factor_options.trans_target {
                    (0, true)
                } else {
                    (1, false)
                }
            }
        };

        // Rows or columns extracted from x. These elements are modified.
        let mut subarr_target: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            target_arr,
            ExtInsType::Axis(row_indices.clone(), axis, transposed),
        )
        .unwrap()
        .ext;

        // Rows or columns extracted from x. These elements only serve as data.
        let subarr_source: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            target_arr,
            ExtInsType::Axis(col_indices.clone(), axis, transposed),
        )
        .unwrap()
        .ext;

        // Computation of F*x
        let res_mul = match self {
            FactorData::Comp(composed_factor_data) => {
                composed_factor_data.mul(&subarr_source, &side, &factor_options)
            }
            FactorData::Reg(rectg) => rectg.mul(&subarr_source, &side, &factor_options),
        };

        if factor_options.inv {
            // res = (I-F)*x
            subarr_target.sub_into(res_mul.r());
        } else {
            // res = (I+F)*x
            subarr_target.sum_into(res_mul.r());
        }

        subarr_target
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
