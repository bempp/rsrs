


pub use rlst::prelude::*;


pub trait LeastSquares: RlstScalar {
    type Item: RlstScalar;
    fn ls(left: &DynamicArray<Self::Item, 2>, right: &DynamicArray<Self::Item, 2>, sol: &mut DynamicArray<Self::Item, 2>);
    fn solve_left(a: &DynamicArray<Self::Item, 2>, b: &DynamicArray<Self::Item, 2>)->DynamicArray<Self::Item, 2>;
    fn solve_right(a: &DynamicArray<Self::Item, 2>, b: &DynamicArray<Self::Item, 2>)->DynamicArray<Self::Item, 2>;
}


macro_rules! impl_ls {
    ($scalar:ty) => {
        impl LeastSquares for $scalar {
            type Item = $scalar;

            fn ls(left: &DynamicArray<Self::Item, 2>, right: &DynamicArray<Self::Item, 2>, sol: &mut DynamicArray<Self::Item, 2>){
                let tol = 1e-15;
                let mut arr = empty_array();
                arr.fill_from_resize(left.view());
                let shape = arr.shape();
                let mut pinv = rlst_dynamic_array2!($scalar, [shape[1], shape[0]]);
                arr.into_pseudo_inverse_alloc(pinv.view_mut(), tol).unwrap();
                let mut res = empty_array();
                res.view_mut().simple_mult_into_resize(pinv.view(), right.view());
                sol.fill_from_resize(res.view_mut());
            }

            fn solve_left(a: &DynamicArray<Self::Item, 2>, b: &DynamicArray<Self::Item, 2>)->DynamicArray<Self::Item, 2>{
                let mut sol = empty_array();
                Self::ls(&a, &b, &mut sol);
                sol
            }

            fn solve_right(a: &DynamicArray<Self::Item, 2>, b: &DynamicArray<Self::Item, 2>)->DynamicArray<Self::Item, 2>{
                let mut ah = empty_array();
                let mut bh = empty_array();
                let mut res = empty_array();
                let mut sol = empty_array();
                ah.fill_from_resize(a.view().transpose().conj());
                bh.fill_from_resize(b.view().transpose().conj());
                Self::ls(&bh, &ah, &mut res);
                sol.fill_from_resize(res.view());
                sol
            }
        }
    }
}

impl_ls!(f64);
impl_ls!(f32);
impl_ls!(c32);
impl_ls!(c64);


pub trait MatrixExt: RlstScalar {
    ///This method allocates space for ID
    fn into_ext_alloc<ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self>
    + Shape<2>
    + Stride<2>
    + UnsafeRandomAccessMut<2, Item = Self>
    + RawAccessMut<Item = Self>
    >(
       arr: Array<Self, ArrayImplMut, 2>, inds: &[usize], axis: usize
    ) -> RlstResult<Extraction<Self>>;
}

macro_rules! implement_into_ext {
    ($scalar:ty) => {
        impl MatrixExt for $scalar {
            fn into_ext_alloc<
            ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self>
            + Shape<2>
            + Stride<2>
            + UnsafeRandomAccessMut<2, Item = Self>
            + RawAccessMut<Item = Self>
            >(
                arr: Array<Self, ArrayImplMut, 2>, inds: &[usize], axis: usize
            ) -> RlstResult<Extraction<Self>> {
                Extraction::<$scalar>::new(arr, inds, axis)
            }
        }
    };
}

implement_into_ext!(f32);
implement_into_ext!(f64);
implement_into_ext!(c32);
implement_into_ext!(c64);

pub struct Extraction<
    Item: RlstScalar
> {
    pub ext: DynamicArray<Item, 2>,
}


pub trait MatrixExtraction: Sized {
    type Item: RlstScalar;
    fn new<
    ArrayImpl: UnsafeRandomAccessByValue<2, Item=Self::Item>
        + Stride<2>
        + Shape<2>
        + UnsafeRandomAccessMut<2, Item = Self::Item>
        + RawAccessMut<Item =Self::Item>>(arr: Array<Self::Item, ArrayImpl, 2>, inds: &[usize], axis: usize) -> RlstResult<Self>;
}


macro_rules! impl_ext {
    ($scalar:ty) => {
        impl MatrixExtraction for Extraction<$scalar>
        {
            type Item = $scalar;
            fn new<
            ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
            + Shape<2>
            + Stride<2>
            + UnsafeRandomAccessMut<2, Item = $scalar>
            + RawAccessMut<Item = $scalar>
                >(mut source_arr: Array<$scalar, ArrayImplMut, 2>, inds: &[usize], axis: usize) -> RlstResult<Self>{

                if axis == 0{
                    let mut target_arr: DynamicArray<$scalar, 2> = rlst_dynamic_array2!($scalar, [inds.len(), source_arr.shape()[1]]);
                    for col in 0..source_arr.shape()[1]{
                        for row in inds{
                            *target_arr.get_mut([*row, col]).unwrap() = *source_arr.get_mut([*row, col]).unwrap();
                        }
                    }

                    Ok(Self{ext: target_arr})
                }
                else{
                    let mut target_arr: DynamicArray<$scalar, 2> = rlst_dynamic_array2!($scalar, [source_arr.shape()[0], inds.len()]);
                    for col in inds{
                        for row in 0..source_arr.shape()[0]{
                            *target_arr.get_mut([row, *col]).unwrap() = *source_arr.get_mut([row, *col]).unwrap();
                        }
                    }

                    Ok(Self{ext: target_arr})
                }
            }
        }
    };
}

impl_ext!(f64);
impl_ext!(f32);
impl_ext!(c32);
impl_ext!(c64);
