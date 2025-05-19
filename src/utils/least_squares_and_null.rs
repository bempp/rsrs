use crate::rsrs::rsrs_cycle::IdOptions;
use rlst::dense::linalg::{lu::MatrixLu, null_space::Method};
pub use rlst::prelude::*;

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

pub struct NormalEquations<
    'a,
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + RawAccessMut<Item = Item> + Shape<2>,
> {
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
        view[[i, i]] += Item::from_real(val);
    }
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
        add_diagonal(&mut normal, tol_lstq); //Regularisation

        let lu = <Item as MatrixLu>::into_lu_alloc(normal).unwrap();
        Self { arr, normal: lu }
    }

    pub fn solve_normal_equations(&self, rhs: &Array<Item, ArrayImpl, 2>) -> DynamicArray<Item, 2>
    where
        LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
            MatrixLuDecomposition<Item = Item>,
    {
        let arr_shape = self.arr.shape();
        let mut new_rhs = rlst_dynamic_array2!(Item, [arr_shape[1], arr_shape[1]]);
        new_rhs.r_mut().mult_into_resize(
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
            new_rhs.r_mut(),
        );
        new_rhs
    }

    fn apply_null_projector(&self, rhs: &mut Array<Item, ArrayImpl, 2>)
    where
        LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
            MatrixLuDecomposition<Item = Item>,
    {
        let proj = self.solve_normal_equations(rhs);
        let mut proj_rhs = rlst_dynamic_array2!(Item, rhs.shape());
        proj_rhs.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::NoTrans,
            num::One::one(),
            self.arr.r(),
            proj.r(),
            num::Zero::zero(),
        );
        rhs.r_mut().sub_into(proj_rhs.r());
    }
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

#[derive(Debug, Clone)]
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
            let shape = sub_test.shape();
            let sub_sketch_shape = sub_sketch.shape();
            let normal = NormalEquations::new(&sub_test, id_options.tol_null);
            normal.apply_null_projector(sub_sketch);
            let mut sub_sketch_copy = empty_array();
            sub_sketch_copy.r_mut().fill_from_resize(sub_sketch.r());
            sub_sketch.r_mut().fill_from_resize(
                sub_sketch_copy
                    .r()
                    .into_subview([0, 0], [shape[0] - shape[1], sub_sketch_shape[1]]),
            );
        }
    };
}
