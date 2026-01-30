use crate::{
    rsrs::rsrs_factors::commutative_factors::PermFactor,
    utils::data_ins_ext::{ExtInsType, Extraction, MatrixExtraction},
};
use itertools::min;
use rlst::{dense::linalg::lu::MatrixLu, prelude::*};

type Real<T> = <T as rlst::RlstScalar>::Real;

type CNTuple<T> = (Real<T>, Real<T>); //TODO: Remove before publishing
pub type CondType<T> = (CNTuple<T>, Option<(CNTuple<T>, CNTuple<T>)>); //TODO: Remove before publishing

pub fn condition_number<Item: RlstScalar + MatrixSvd>(
    //TODO: Remove before publishing
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

#[derive(Clone, Debug)]
pub struct BaseFactorOptions {
    /// Inverse operation
    pub inv: bool,
    /// Transpose operation
    pub trans: TransMode,
    pub trans_target: bool,
}

impl BaseFactorOptions {
    pub fn trans_val(&self) -> bool {
        match self.trans {
            TransMode::NoTrans => false,
            TransMode::ConjNoTrans => todo!(),
            TransMode::Trans => true,
            TransMode::ConjTrans => todo!(),
        }
    }

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

    pub fn invert(&self) -> Self {
        let mut new_options = self.clone();
        new_options.inv = !self.inv;
        new_options
    }
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

pub enum SquareArr<T: RlstScalar> {
    Reg(RegDBox<T>),
    Lu(LuDBox<T>),
}

impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse + MatrixInverse> SquareArr<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
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
            SquareArr::Reg(ref reg) => {
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
            SquareArr::Lu(ref lu) => {
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
                        TransMode::ConjNoTrans => todo!(),
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
                        TransMode::ConjNoTrans => todo!(),
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
        side: Side,
        factor_options: &BaseFactorOptions,
    ) {
        match side {
            Side::Left => self.left_mul(arr, factor_options),
            Side::Right => {
                let mut aux_arr = empty_array();
                aux_arr.r_mut().fill_from_resize(arr.r().transpose());
                let mut trans_factor_options = factor_options.clone();

                trans_factor_options.trans = match factor_options.trans {
                    TransMode::NoTrans => TransMode::Trans,
                    TransMode::ConjNoTrans => todo!(),
                    TransMode::Trans => TransMode::NoTrans,
                    TransMode::ConjTrans => todo!(),
                };

                self.left_mul(&mut aux_arr, &trans_factor_options);
                arr.fill_from(aux_arr.r().transpose());
            }
        }
    }

    pub fn cond(&self) -> (CNTuple<Item>, CNTuple<Item>) {
        //TODO: Remove before publishing
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

pub struct ComposedFactorData<T: RlstScalar> {
    pub sq: SquareArr<T>,
    pub rectg: RectArr<T>,
    pub apply_transposed: bool,
}

pub struct RectArr<T: RlstScalar> {
    pub arr: DynamicArray<T, 2>,
    pub apply_transposed: bool,
}

impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse + MatrixInverse> ComposedFactorData<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
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
        let mut sq_factor_options = factor_options.clone();
        sq_factor_options.inv = true;

        match factor_options.trans {
            TransMode::NoTrans => {
                res_mul.r_mut().mult_into_resize(
                    TransMode::NoTrans,
                    TransMode::NoTrans,
                    num::One::one(),
                    self.rectg.arr.r(),
                    target_arr.r(),
                    num::Zero::zero(),
                );
                self.sq.mul(&mut res_mul, Side::Left, &sq_factor_options);
            }
            TransMode::ConjNoTrans => todo!(),
            TransMode::Trans => {
                let mut aux_target_arr = empty_array();
                aux_target_arr.r_mut().fill_from_resize(target_arr.r());
                self.sq
                    .mul(&mut aux_target_arr.r_mut(), Side::Left, &sq_factor_options);
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
                let mut aux_arr = empty_array();
                aux_arr.r_mut().fill_from_resize(target_arr.r().transpose());
                let mut trans_factor_options = factor_options.clone();

                trans_factor_options.trans = match factor_options.trans {
                    TransMode::NoTrans => TransMode::Trans,
                    TransMode::ConjNoTrans => todo!(),
                    TransMode::Trans => TransMode::NoTrans,
                    TransMode::ConjTrans => todo!(),
                };

                let aux_arr = self.left_mul(&aux_arr, &trans_factor_options);
                let mut res_mul = empty_array();
                res_mul.r_mut().fill_from_resize(aux_arr.r().transpose());
                res_mul
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn cond(&self) -> CondType<Item> {
        //TODO: Remove before publishing
        (condition_number(&self.rectg.arr), Some(self.sq.cond()))
    }
}

impl<Item: RlstScalar + MatrixSvd> RectArr<Item> {
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
                let mut aux_arr = empty_array();
                aux_arr.r_mut().fill_from_resize(target_arr.r().transpose());
                let mut trans_factor_options = factor_options.clone();
                trans_factor_options.trans = match factor_options.trans {
                    TransMode::NoTrans => TransMode::Trans,
                    TransMode::ConjNoTrans => todo!(),
                    TransMode::Trans => TransMode::NoTrans,
                    TransMode::ConjTrans => todo!(),
                };
                let aux_arr = self.left_mul(&aux_arr, &trans_factor_options);
                let mut res_mul = empty_array();
                res_mul.r_mut().fill_from_resize(aux_arr.r().transpose());
                res_mul
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn cond(&self) -> CondType<Item> {
        //TODO: Remove before publishing
        (condition_number(&self.arr), None)
    }
}

pub enum FactorData<T: RlstScalar> {
    Comp(ComposedFactorData<T>),
    Reg(RectArr<T>),
}

impl<Item: RlstScalar + MatrixLu + MatrixPseudoInverse + MatrixInverse> FactorData<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    pub fn mul<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        right_arr: &Array<Item, ArrayImpl, 2>,
        side: &Side,
        factor_options: &BaseFactorOptions,
        c_indices: &[usize],
        r_indices: &[usize],
    ) -> DynamicArray<Item, 2> {
        let row_indices: Vec<usize>;
        let col_indices: Vec<usize>;
        let (axis, transposed) = match side {
            Side::Left => {
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

        let mut subarr_target: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            right_arr,
            ExtInsType::Axis(row_indices.clone(), axis, transposed),
        )
        .unwrap()
        .ext;

        let subarr_source: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
            right_arr,
            ExtInsType::Axis(col_indices.clone(), axis, transposed),
        )
        .unwrap()
        .ext;

        let res_mul = match self {
            FactorData::Comp(composed_factor_data) => {
                composed_factor_data.mul(&subarr_source, &side, &factor_options)
            }
            FactorData::Reg(rectg) => rectg.mul(&subarr_source, &side, &factor_options),
        };

        if factor_options.inv {
            subarr_target.sub_into(res_mul.r());
        } else {
            subarr_target.sum_into(res_mul.r());
        }

        subarr_target
    }

    pub fn cond(&self) -> CondType<Item> {
        //TODO: Remove before publishing
        match self {
            FactorData::Comp(composed_factor_data) => composed_factor_data.cond(),
            FactorData::Reg(rectg) => rectg.cond(),
        }
    }
}
