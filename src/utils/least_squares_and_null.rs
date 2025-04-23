use rlst::dense::linalg::{lu::MatrixLu, null_space::Method};
pub use rlst::prelude::*;

use super::data_ins_ext::{ExtInsType, Extraction, MatrixExtraction};

fn _solve_svd<
    Item: RlstScalar + MatrixPseudoInverse,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + RawAccessMut<Item = Item> + Shape<2>,
>(
    test_mat: &Array<Item, ArrayImpl, 2>,
    sketch_mat: &Array<Item, ArrayImpl, 2>,
    tol_lstq: <Item as rlst::RlstScalar>::Real,
) -> DynamicArray<Item, 2> {
    let shape = test_mat.shape();
    let mut test_mat_copy = empty_array();
    test_mat_copy.fill_from_resize(test_mat.r());
    let mut pinv = rlst_dynamic_array2!(Item, [shape[1], shape[0]]); // Avoid extra allocation
    test_mat_copy
        .r_mut()
        .into_pseudo_inverse_alloc(pinv.r_mut(), tol_lstq)
        .unwrap();
    let mut sol: DynamicArray<Item, 2> = empty_array();
    sol.r_mut()
        .simple_mult_into_resize(sketch_mat.r(), pinv.r());

    sol
}

fn _null_space_near_box<
    Item: RlstScalar + MatrixSvd + MatrixQr,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    mut sub_test: Array<Item, ArrayImpl, 2>,
    near_field_inds: &Vec<usize>,
    method: &Method,
    tol_null: <Item as RlstScalar>::Real,
) -> DynamicArray<Item, 2>
where
    QrDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixQrDecomposition<Item = Item>,
{
    let test_mat: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        &mut sub_test,
        ExtInsType::Axis(near_field_inds.clone(), 0, false),
    )
    .unwrap()
    .ext;
    _null_space(&test_mat, method, tol_null).null_space_arr
}

fn _null_space<Item: RlstScalar + MatrixSvd + MatrixQr>(
    sub_test: &DynamicArray<Item, 2>,
    method: &Method,
    tol_null: <Item as RlstScalar>::Real,
) -> rlst::NullSpace<Item>
where
    QrDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixQrDecomposition<Item = Item>,
{
    match method {
        Method::Svd => {
            let shape = sub_test.shape();
            let mut sub_test_copy = rlst_dynamic_array2!(Item, shape);
            sub_test_copy.fill_from(sub_test.r());
            let null_res = sub_test_copy
                .into_null_alloc(tol_null, Method::Svd)
                .unwrap();
            null_res
        }
        Method::Qr => {
            let shape = sub_test.shape();
            let mut sub_test_trans = rlst_dynamic_array2!(Item, [shape[1], shape[0]]);
            sub_test_trans.fill_from(sub_test.r().conj().transpose());
            let null_res = sub_test_trans
                .into_null_alloc(tol_null, Method::Qr)
                .unwrap();
            null_res
        }
    }
}

pub struct NormalEquations<'a, Item: RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
+ Stride<2>
+ RawAccessMut<Item = Item>
+ Shape<2>> {
    pub arr: &'a Array<Item, ArrayImpl, 2>,
    pub normal: LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>,
}

fn add_diagonal<Item: RlstScalar>(
    arr: &mut DynamicArray<Item, 2>,
    val: <Item as rlst::RlstScalar>::Real,
) {
    let shape = arr.shape();
    let mut view = arr.r_mut();
    for i in 0..shape[0] {
        view[[i,i]] += Item::from_real(val);
    }
}

impl<'a, Item: RlstScalar + MatrixLu, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
+ Stride<2>
+ RawAccessMut<Item = Item>
+ Shape<2>> NormalEquations<'a, Item, ArrayImpl> {
    fn new(
        arr: &'a Array<Item, ArrayImpl, 2>,
        tol_lstq: <Item as rlst::RlstScalar>::Real,
    ) -> Self {
        let shape = arr.shape();
        let mut normal = rlst_dynamic_array2!(Item, [shape[0], shape[0]]);
        normal.r_mut().mult_into(
            TransMode::NoTrans,
            TransMode::ConjTrans,
            <Item as num::One>::one(),
            arr.r(),
            arr.r(),
            <Item as num::Zero>::zero(),
        );

        add_diagonal(&mut normal, tol_lstq);//Regularisation
        let lu = <Item as MatrixLu>::into_lu_alloc(normal).unwrap();

        Self {
            arr,
            normal: lu,
        }
    }

    fn solve_normal_equations(
        &self,
        rhs: &Array<Item, ArrayImpl, 2>,
    ) -> DynamicArray<Item, 2>
    where
        LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
            MatrixLuDecomposition<Item = Item>,
    {
        let rhs_shape = rhs.shape();
        let arr_shape = self.arr.shape();

        let mut new_rhs = rlst_dynamic_array2!(Item, [arr_shape[0], arr_shape[0]]);
        new_rhs.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::ConjTrans,
            num::One::one(),
            self.arr.r(),
            rhs.r(),
            num::Zero::zero(),
        );

        let _ = <LuDecomposition<Item, _> as MatrixLuDecomposition>::solve_mat(
            &self.normal,
            TransMode::NoTrans,
            new_rhs.r_mut(),
        );

        let mut sol = rlst_dynamic_array2!(Item, [rhs_shape[0], arr_shape[0]]);

        sol.fill_from_resize(new_rhs.r().transpose());
        sol
    }

    fn apply_null_projector(
        &self,
        rhs: &Array<Item, ArrayImpl, 2>,
    ) -> DynamicArray<Item, 2>
    where
        LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
            MatrixLuDecomposition<Item = Item>,
    {
        let proj = self.solve_normal_equations(rhs);
        let mut proj_rhs = rlst_dynamic_array2!(Item, rhs.shape());
        proj_rhs
            .r_mut()
            .simple_mult_into(proj.r(), self.arr.r());
        let mut res = rlst_dynamic_array2!(Item, rhs.shape());
        res.fill_from(rhs.r() - proj_rhs.r());
        res
    }
}

pub fn null_space_by_projection<
    Item: RlstScalar + MatrixPseudoInverse + MatrixLu,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    sub_test: &Array<Item, ArrayImpl, 2>,
    sub_sketch: &Array<Item, ArrayImpl, 2>,
    tol_null: <Item as RlstScalar>::Real,
) -> DynamicArray<Item, 2>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let normal = NormalEquations::new(&sub_test, tol_null);
    normal.apply_null_projector(sub_sketch)
}

pub fn right_least_squares<
    Item: RlstScalar + MatrixPseudoInverse + MatrixLu,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>,
>(
    test_mat: &Array<Item, ArrayImpl, 2>,
    sketch_mat: &Array<Item, ArrayImpl, 2>,
    tol_lstq: <Item as rlst::RlstScalar>::Real,
) -> DynamicArray<Item, 2>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    
    let normal = NormalEquations::new(&test_mat, tol_lstq);
    normal.solve_normal_equations(sketch_mat)
}