pub use rlst::prelude::*;

pub struct Extraction<Item: RlstScalar> {
    pub ext: DynamicArray<Item, 2>,
}

pub trait MatrixExtraction: Sized {
    type Item: RlstScalar;
    fn new<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccess<Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        source_arr: &Array<Self::Item, ArrayImplMut, 2>,
        indices: ExtInsType,
    ) -> RlstResult<Self>;
}

pub enum ExtInsType {
    /// Indices and axis of extraction
    Axis(Vec<usize>, usize, bool),
    /// Rows and cols indices
    Cross(Vec<usize>, Vec<usize>),
}

impl<T: RlstScalar> MatrixExtraction for Extraction<T> {
    type Item = T;
    fn new<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccess<Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        source_arr: &Array<Self::Item, ArrayImplMut, 2>,
        indices: ExtInsType,
    ) -> RlstResult<Self> {
        match indices {
            ExtInsType::Axis(inds, axis, exchange_axis) => {
                if axis == 0 {
                    let target_arr: DynamicArray<Self::Item, 2> =
                        get_rows(inds, source_arr, exchange_axis);
                    Ok(Self { ext: target_arr })
                } else {
                    let target_arr: DynamicArray<Self::Item, 2> =
                        get_cols(inds, source_arr, exchange_axis);
                    Ok(Self { ext: target_arr })
                }
            }
            ExtInsType::Cross(rows, cols) => {
                let mut target_arr: DynamicArray<Self::Item, 2> =
                    rlst_dynamic_array2!(Self::Item, [rows.len(), cols.len()]);
                for (col_ind, col) in cols.iter().enumerate() {
                    for (row_ind, row) in rows.iter().enumerate() {
                        *target_arr.get_mut([row_ind, col_ind]).unwrap() =
                            *source_arr.get([*row, *col]).unwrap();
                    }
                }
                Ok(Self { ext: target_arr })
            }
        }
    }
}

fn get_rows<
    T: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = T>
        + Shape<2>
        + RawAccess<Item = T>
        + UnsafeRandomAccessByRef<2, Item = T>,
>(
    inds: Vec<usize>,
    source_arr: &Array<T, ArrayImpl, 2>,
    exchange_axis: bool,
) -> DynamicArray<T, 2> {
    let mut target_arr: DynamicArray<T, 2>;
    if exchange_axis {
        target_arr = rlst_dynamic_array2!(T, [source_arr.shape()[1], inds.len()]);
        for col in 0..source_arr.shape()[1] {
            for (row_ind, row) in inds.iter().enumerate() {
                *target_arr.get_mut([col, row_ind]).unwrap() =
                    (*source_arr.get([*row, col]).unwrap()).conj();
            }
        }
    } else {
        target_arr = rlst_dynamic_array2!(T, [inds.len(), source_arr.shape()[1]]);
        for col in 0..source_arr.shape()[1] {
            for (row_ind, row) in inds.iter().enumerate() {
                *target_arr.get_mut([row_ind, col]).unwrap() =
                    *source_arr.get([*row, col]).unwrap();
            }
        }
    }
    target_arr
}

fn get_cols<
    T: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = T>
        + Shape<2>
        + RawAccess<Item = T>
        + UnsafeRandomAccessByRef<2, Item = T>,
>(
    inds: Vec<usize>,
    source_arr: &Array<T, ArrayImpl, 2>,
    exchange_axis: bool,
) -> DynamicArray<T, 2> {
    let mut target_arr: DynamicArray<T, 2>;
    if exchange_axis {
        target_arr = rlst_dynamic_array2!(T, [inds.len(), source_arr.shape()[0]]);
        for (col_ind, col) in inds.iter().enumerate() {
            for row in 0..source_arr.shape()[0] {
                *target_arr.get_mut([col_ind, row]).unwrap() =
                    (*source_arr.get([row, *col]).unwrap()).conj();
            }
        }
    } else {
        target_arr = rlst_dynamic_array2!(T, [source_arr.shape()[0], inds.len()]);
        for (col_ind, col) in inds.iter().enumerate() {
            for row in 0..source_arr.shape()[0] {
                *target_arr.get_mut([row, col_ind]).unwrap() =
                    *source_arr.get([row, *col]).unwrap();
            }
        }
    }
    target_arr
}

pub fn matrix_insertion<
    T: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = T>
        + Shape<2>
        + Stride<2>
        + UnsafeRandomAccessMut<2, Item = T>
        + UnsafeRandomAccessByRef<2, Item = T>
        + RawAccessMut<Item = T>
        + Shape<2>,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = T>
        + Shape<2>
        + RawAccessMut<Item = T>
        + UnsafeRandomAccessMut<2, Item = T>
        + UnsafeRandomAccessByRef<2, Item = T>,
>(
    target_arr: &mut Array<T, ArrayImplMut, 2>,
    source_arr: &Array<T, ArrayImpl, 2>,
    indices: ExtInsType,
) {
    match indices {
        ExtInsType::Axis(inds, axis, _exchange_axis) => {
            if axis == 0 {
                for col in 0..source_arr.shape()[1] {
                    for (row_ind, row) in inds.iter().enumerate() {
                        *target_arr.get_mut([*row, col]).unwrap() =
                            *source_arr.get([row_ind, col]).unwrap();
                    }
                }
            } else {
                for (col_ind, col) in inds.iter().enumerate() {
                    for row in 0..source_arr.shape()[0] {
                        *target_arr.get_mut([row, *col]).unwrap() =
                            *source_arr.get([row, col_ind]).unwrap();
                    }
                }
            }
        }
        ExtInsType::Cross(rows, cols) => {
            for (col_ind, col) in cols.iter().enumerate() {
                for (row_ind, row) in rows.iter().enumerate() {
                    *target_arr.get_mut([row_ind, col_ind]).unwrap() =
                        *source_arr.get([*row, *col]).unwrap();
                }
            }
        }
    }
}
