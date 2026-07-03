use num::{Float, NumCast, One, Zero};
use rlst::dense::linalg::{lu::MatrixLu, null_space::Method};
pub use rlst::prelude::*;
use serde::Deserialize;

use crate::rsrs::rsrs_factors::null_and_extract::{ExtractOptions, IdOptions};

type Real<T> = <T as rlst::RlstScalar>::Real;

fn solve_svd<
    Item: RlstScalar + MatrixPseudoInverse,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>,
>(
    test_mat: &mut Array<Item, ArrayImpl, 2>,
    sketch_mat: &Array<Item, ArrayImpl, 2>,
    tol_lstq: <Item as rlst::RlstScalar>::Real,
) -> DynamicArray<Item, 2> {
    let mut test_mat_t = empty_array();
    test_mat_t
        .r_mut()
        .fill_from_resize(test_mat.r().transpose().conj());
    let shape = test_mat_t.shape();
    let mut pinv = rlst_dynamic_array2!(Item, [shape[1], shape[0]]);
    test_mat_t
        .r_mut()
        .into_pseudo_inverse_alloc(pinv.r_mut(), tol_lstq)
        .unwrap();
    let mut sol: DynamicArray<Item, 2> = empty_array();
    sol.r_mut().mult_into_resize(
        TransMode::ConjTrans,
        TransMode::NoTrans,
        num::One::one(),
        pinv.r(),
        sketch_mat.r(),
        num::Zero::zero(),
    );
    sol
}

pub struct NormalEquations<
    'a,
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + RawAccessMut<Item = Item> + Shape<2>,
> {
    pub arr: &'a Array<Item, ArrayImpl, 2>,
    pub normal: LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>,
}

pub struct NormalEquationAccumulator<Item: RlstScalar> {
    normal: DynamicArray<Item, 2>,
    rhs: DynamicArray<Item, 2>,
}

pub struct NormalEquationScratch<Item: RlstScalar> {
    solution: DynamicArray<Item, 2>,
    projected_rhs: DynamicArray<Item, 2>,
}

pub fn streaming_chunk_rows<Item: RlstScalar>(
    max_rows: usize,
    cols_per_row: usize,
    live_buffers: usize,
) -> usize {
    const STREAMING_TARGET_BYTES: usize = 2 * 1024 * 1024;

    if max_rows == 0 {
        return 0;
    }

    let bytes_per_row = cols_per_row
        .saturating_mul(live_buffers)
        .saturating_mul(std::mem::size_of::<Item>());

    if bytes_per_row == 0 {
        return max_rows;
    }

    (STREAMING_TARGET_BYTES / bytes_per_row).clamp(1, max_rows)
}

pub fn add_diagonal<Item: RlstScalar>(
    arr: &mut DynamicArray<Item, 2>,
    val: <Item as rlst::RlstScalar>::Real,
) {
    let shape = arr.shape();
    let mut view = arr.r_mut();
    for i in 0..shape[0] {
        view[[i, i]] += Item::from_real(val);
    }
}

pub fn fixed_rank_relative_tol<Item: RlstScalar>() -> Real<Item>
where
    Real<Item>: NumCast,
{
    if std::mem::size_of::<Real<Item>>() <= std::mem::size_of::<f32>() {
        NumCast::from(1e-5f64).unwrap()
    } else {
        NumCast::from(1e-10f64).unwrap()
    }
}

fn normal_equation_scale<Item: RlstScalar>(normal: &DynamicArray<Item, 2>) -> Real<Item> {
    let shape = normal.shape();
    let view = normal.r();
    let mut scale = Real::<Item>::zero();

    for i in 0..shape[0] {
        scale = scale.max(view[[i, i]].re());
    }

    scale
}

fn normal_equation_regularization<Item: RlstScalar>(
    normal: &DynamicArray<Item, 2>,
    tol_lstq: Real<Item>,
) -> Real<Item> {
    let tol_abs = Float::abs(tol_lstq);
    let tol_sq = tol_abs * tol_abs;
    let factor: Real<Item> = Float::max(Real::<Item>::epsilon(), tol_sq);
    let scale: Real<Item> = normal_equation_scale::<Item>(normal);
    let scale = if scale > Real::<Item>::zero() {
        scale
    } else {
        Real::<Item>::one()
    };
    factor * scale
}

impl<
        'a,
        Item: RlstScalar + MatrixLu,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>,
    > NormalEquations<'a, Item, ArrayImpl>
{
    pub fn new(
        arr: &'a Array<Item, ArrayImpl, 2>,
        tol_lstq: <Item as rlst::RlstScalar>::Real,
    ) -> Self {
        let shape = arr.shape();

        let mut normal = rlst_dynamic_array2!(Item, [shape[1], shape[1]]);
        normal.r_mut().mult_into(
            TransMode::ConjTrans,
            TransMode::NoTrans,
            <Item as num::One>::one(),
            arr.r(),
            arr.r(),
            <Item as num::Zero>::zero(),
        );
        let regularization = normal_equation_regularization::<Item>(&normal, tol_lstq);
        add_diagonal(&mut normal, regularization);

        let lu = <Item as MatrixLu>::into_lu_alloc(normal).unwrap();
        Self { arr, normal: lu }
    }

    pub fn solve_normal_equations(&self, rhs: &Array<Item, ArrayImpl, 2>) -> DynamicArray<Item, 2>
    where
        LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
            MatrixLuDecomposition<Item = Item>,
    {
        let mut solution = empty_array();
        self.solve_normal_equations_into(rhs, &mut solution);
        solution
    }

    pub fn solve_normal_equations_into(
        &self,
        rhs: &Array<Item, ArrayImpl, 2>,
        solution: &mut DynamicArray<Item, 2>,
    ) where
        LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
            MatrixLuDecomposition<Item = Item>,
    {
        solution.r_mut().mult_into_resize(
            TransMode::ConjTrans,
            TransMode::NoTrans,
            num::One::one(),
            self.arr.r(),
            rhs.r(),
            num::Zero::zero(),
        );
        let _ = <LuDecomposition<Item, _> as MatrixLuDecomposition>::solve_mat(
            &self.normal,
            TransMode::NoTrans,
            solution.r_mut(),
        );
    }

    pub fn apply_null_projector_with_scratch(
        &self,
        rhs: &mut Array<Item, ArrayImpl, 2>,
        scratch: &mut NormalEquationScratch<Item>,
    ) where
        LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
            MatrixLuDecomposition<Item = Item>,
    {
        self.solve_normal_equations_into(rhs, &mut scratch.solution);
        scratch.projected_rhs.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::NoTrans,
            num::One::one(),
            self.arr.r(),
            scratch.solution.r(),
            num::Zero::zero(),
        );
        rhs.r_mut().sub_into(scratch.projected_rhs.r());
    }
}

impl<Item: RlstScalar> NormalEquationAccumulator<Item> {
    pub fn new(lhs_cols: usize, rhs_cols: usize) -> Self {
        let mut normal = rlst_dynamic_array2!(Item, [lhs_cols, lhs_cols]);
        let mut rhs = rlst_dynamic_array2!(Item, [lhs_cols, rhs_cols]);
        normal.r_mut().set_zero();
        rhs.r_mut().set_zero();

        Self { normal, rhs }
    }

    pub fn add_chunk(&mut self, lhs: &DynamicArray<Item, 2>, rhs: &DynamicArray<Item, 2>) {
        self.normal.r_mut().mult_into(
            TransMode::ConjTrans,
            TransMode::NoTrans,
            num::One::one(),
            lhs.r(),
            lhs.r(),
            num::One::one(),
        );
        self.rhs.r_mut().mult_into(
            TransMode::ConjTrans,
            TransMode::NoTrans,
            num::One::one(),
            lhs.r(),
            rhs.r(),
            num::One::one(),
        );
    }
}

impl<Item: RlstScalar + MatrixLu> NormalEquationAccumulator<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    pub fn solve(mut self, tol_lstq: <Item as rlst::RlstScalar>::Real) -> DynamicArray<Item, 2> {
        let regularization = normal_equation_regularization::<Item>(&self.normal, tol_lstq);
        add_diagonal(&mut self.normal, regularization);
        let lu = <Item as MatrixLu>::into_lu_alloc(self.normal).unwrap();
        let _ = <LuDecomposition<Item, _> as MatrixLuDecomposition>::solve_mat(
            &lu,
            TransMode::NoTrans,
            self.rhs.r_mut(),
        );
        self.rhs
    }
}

impl<Item: RlstScalar> NormalEquationScratch<Item> {
    pub fn new() -> Self {
        Self {
            solution: empty_array(),
            projected_rhs: empty_array(),
        }
    }
}

impl<Item: RlstScalar> Default for NormalEquationScratch<Item> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum BlockExtractionMethod {
    ///SVD
    Svd,
    ///Projection
    LuLstSq,
}

pub fn block_extraction<
    Item: RlstScalar + MatrixPseudoInverse + MatrixLu,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>,
>(
    test_mat: &mut Array<Item, ArrayImpl, 2>,
    sketch_mat: &Array<Item, ArrayImpl, 2>,
    ext_options: &ExtractOptions<Item>,
) -> DynamicArray<Item, 2>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let mut out = empty_array();
    block_extraction_into(test_mat, sketch_mat, ext_options, &mut out);
    out
}

pub fn block_extraction_into<
    Item: RlstScalar + MatrixPseudoInverse + MatrixLu,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>,
>(
    test_mat: &mut Array<Item, ArrayImpl, 2>,
    sketch_mat: &Array<Item, ArrayImpl, 2>,
    ext_options: &ExtractOptions<Item>,
    out: &mut DynamicArray<Item, 2>,
) where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    match ext_options.block_extraction_method {
        BlockExtractionMethod::Svd => {
            *out = solve_svd(test_mat, sketch_mat, ext_options.tol_lstsq);
        }
        BlockExtractionMethod::LuLstSq => {
            let normal = NormalEquations::new(test_mat, ext_options.tol_lstsq);
            normal.solve_normal_equations_into(sketch_mat, out);
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum NullMethod {
    ///SVD
    Svd,
    ///QR
    Qr,
    ///Projection
    Projection,
}

pub fn nullify_near_sketch<
    Item: RlstScalar + MatrixPseudoInverse + MatrixLu + MatrixQr,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + rlst::ResizeInPlace<2>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    sub_test: &Array<Item, ArrayImpl, 2>,
    sub_sketch: &mut Array<Item, ArrayImpl, 2>,
    id_options: &IdOptions<Item>,
    normal_scratch: &mut NormalEquationScratch<Item>,
) where
    QrDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixQrDecomposition<Item = Item>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    match id_options.null_method {
        NullMethod::Svd => {
            let shape = sub_test.shape();
            let mut sub_test_t = rlst_dynamic_array2!(Item, [shape[1], shape[0]]);
            sub_test_t.fill_from(sub_test.r().transpose());
            let mut sub_sketch_t =
                rlst_dynamic_array2!(Item, [sub_sketch.shape()[1], sub_sketch.shape()[0]]);
            sub_sketch_t.fill_from(sub_sketch.r().transpose());
            let null_res = sub_test_t
                .into_null_alloc(id_options.tol_null, Method::Svd)
                .unwrap();
            let null_arr = null_res.null_space_arr;
            let mut res = empty_array();
            res.r_mut()
                .simple_mult_into_resize(sub_sketch_t.r(), null_arr.r());
            sub_sketch.r_mut().fill_from_resize(res.r().transpose());
        }
        NullMethod::Qr => {
            let shape = sub_test.shape();
            let sub_sketch_shape = sub_sketch.shape();
            let mut sub_test_base = rlst_dynamic_array2!(Item, [shape[0], shape[1]]);
            sub_test_base.fill_from(sub_test.r());
            let mut sub_sketch_t =
                rlst_dynamic_array2!(Item, [sub_sketch_shape[1], sub_sketch_shape[0]]);
            sub_sketch_t.fill_from(sub_sketch.r().transpose());
            let null_res = sub_test_base
                .into_null_alloc(id_options.tol_null, Method::Qr)
                .unwrap();
            let null_arr = null_res.null_space_arr;
            let mut res = empty_array();
            res.r_mut()
                .simple_mult_into_resize(sub_sketch_t.r(), null_arr.r());
            sub_sketch.r_mut().fill_from_resize(res.r().transpose());
        }
        NullMethod::Projection => {
            let normal = NormalEquations::new(sub_test, id_options.tol_null);
            normal.apply_null_projector_with_scratch(sub_sketch, normal_scratch);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_normal_matrix_still_gets_positive_regularization() {
        let normal = rlst_dynamic_array2!(f64, [3, 3]);
        let regularization = normal_equation_regularization::<f64>(&normal, 1e-8);
        assert!(regularization > 0.0);
    }

    #[test]
    fn normal_equations_accept_zero_test_matrix() {
        let test_mat = rlst_dynamic_array2!(f64, [4, 3]);
        let normal = NormalEquations::new(&test_mat, 1e-8);
        let rhs = rlst_dynamic_array2!(f64, [4, 2]);
        let solution = normal.solve_normal_equations(&rhs);

        assert_eq!(solution.shape(), [3, 2]);
        assert!(solution
            .data()
            .iter()
            .all(|entry| Float::abs(*entry) <= f64::EPSILON));
    }

    #[test]
    fn fixed_rank_relative_tol_depends_on_precision() {
        assert_eq!(fixed_rank_relative_tol::<f32>(), 1e-5f32);
        assert_eq!(fixed_rank_relative_tol::<f64>(), 1e-10f64);
    }
}
