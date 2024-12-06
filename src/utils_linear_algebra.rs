


pub use rlst::prelude::*;

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
