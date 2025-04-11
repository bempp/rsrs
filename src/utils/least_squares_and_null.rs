use rlst::dense::{
    array::{reference::ArrayRef, views::ArraySubView},
    linalg::{lu::MatrixLu, null_space::Method},
};
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

fn solve_lu<
    Item: RlstScalar + MatrixLu,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + RawAccessMut<Item = Item> + Shape<2>,
>(
    test_mat: &Array<Item, ArrayImpl, 2>,
    sketch_mat: &Array<Item, ArrayImpl, 2>,
    tol_lstq: <Item as rlst::RlstScalar>::Real,
) -> DynamicArray<Item, 2>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let test_shape = test_mat.shape();
    let sketch_shape = sketch_mat.shape();
    if test_shape[0] < test_shape[1] {
        let mut normal = rlst_dynamic_array2!(Item, [test_shape[0], test_shape[0]]);
        let mut id: DynamicArray<Item, 2> = rlst_dynamic_array2!(Item, normal.shape());
        id.set_identity();
        id.scale_inplace(Item::from_real(tol_lstq));
        normal.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::Trans,
            num::One::one(),
            test_mat.r(),
            test_mat.r(),
            num::Zero::zero(),
        );

        normal.sum_into(id); //Regularisation

        let mut rhs = rlst_dynamic_array2!(Item, [test_shape[0], test_shape[0]]);
        rhs.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::Trans,
            num::One::one(),
            test_mat.r(),
            sketch_mat.r(),
            num::Zero::zero(),
        );
        let lu = <Item as MatrixLu>::into_lu_alloc(normal).unwrap();
        let _ = <LuDecomposition<Item, _> as MatrixLuDecomposition>::solve_mat(
            &lu,
            TransMode::NoTrans,
            rhs.r_mut(),
        );
        let mut sol = rlst_dynamic_array2!(Item, [sketch_shape[0], test_shape[0]]);

        sol.fill_from(rhs.r().transpose());
        sol
    } else {
        let mut test_mat_trans = empty_array();
        test_mat_trans.fill_from_resize(test_mat.r().transpose());
        let mut normal = rlst_dynamic_array2!(Item, [test_shape[1], test_shape[1]]);
        let mut id: DynamicArray<Item, 2> = rlst_dynamic_array2!(Item, normal.shape());
        id.set_identity();
        id.scale_inplace(Item::from_real(tol_lstq));

        normal
            .r_mut()
            .simple_mult_into(test_mat_trans.r(), test_mat.r());
        normal.sum_into(id); //Regularisation

        let lu = <Item as MatrixLu>::into_lu_alloc(normal).unwrap();
        let _ = <LuDecomposition<Item, _> as MatrixLuDecomposition>::solve_mat(
            &lu,
            TransMode::NoTrans,
            test_mat_trans.r_mut(),
        );

        let mut sol = rlst_dynamic_array2!(Item, [sketch_shape[0], test_shape[0]]);
        sol.r_mut()
            .simple_mult_into(sketch_mat.r(), test_mat_trans.r());

        sol
    }
}

pub fn right_least_squares<
    Item: RlstScalar + MatrixPseudoInverse + MatrixLu,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + RawAccessMut<Item = Item> + Shape<2>,
>(
    test_mat: &Array<Item, ArrayImpl, 2>,
    sketch_mat: &Array<Item, ArrayImpl, 2>,
    tol_lstq: <Item as rlst::RlstScalar>::Real,
) -> DynamicArray<Item, 2>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    /*if test_mat.shape()[0] > 8 * test_mat.shape()[1] {
        solve_svd(test_mat, sketch_mat, tol_lstq)
    } else {
        solve_lu(test_mat, sketch_mat, tol_lstq)
    }*/
    solve_lu(test_mat, sketch_mat, tol_lstq)
}

pub fn null_space<Item: RlstScalar + MatrixSvd + MatrixQr>(
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

pub fn null_space_near_box<Item: RlstScalar + MatrixSvd + MatrixQr>(
    mut test_subview: Array<
        Item,
        ArraySubView<Item, ArrayRef<'_, Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>, 2>,
        2,
    >,
    near_field_inds: &Vec<usize>,
    method: &Method,
    tol_null: <Item as RlstScalar>::Real,
) -> DynamicArray<Item, 2>
where
    QrDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixQrDecomposition<Item = Item>,
{
    let test_mat: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        &mut test_subview,
        ExtInsType::Axis(near_field_inds.clone(), 0, false),
    )
    .unwrap()
    .ext;
    null_space(&test_mat, method, tol_null).null_space_arr
}
