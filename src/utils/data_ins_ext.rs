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
                let mut view_1 = target_arr.r_mut();
                let view_2 = source_arr.r();

                for (col_ind, col) in cols.iter().enumerate() {
                    for (row_ind, row) in rows.iter().enumerate() {
                        view_1[[row_ind, col_ind]] = view_2[[*row, *col]];
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
    let num_cols = source_arr.shape()[1];
    let num_rows = inds.len();
    let view_2 = source_arr.r();

    let mut target_arr = if exchange_axis {
        rlst_dynamic_array2!(T, [num_cols, num_rows])
    } else {
        rlst_dynamic_array2!(T, [num_rows, num_cols])
    };
    let mut view_1 = target_arr.r_mut();

    if exchange_axis{
        for (row_ind, row) in inds.iter().enumerate() {
            view_1.r_mut().into_subview([0, row_ind], [num_cols, 1]).fill_from(view_2.r().into_subview([*row, 0], [num_cols, 1]).conj());
        }
    }
    else{
        for (row_ind, row) in inds.iter().enumerate() {
            view_1.r_mut().into_subview([row_ind, 0], [1, num_cols]).fill_from(view_2.r().into_subview([*row, 0], [1, num_cols]));
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
    let num_rows = source_arr.shape()[0];
    let num_cols = inds.len();
    let view_2 = source_arr.r();

    let mut target_arr = if exchange_axis {
        rlst_dynamic_array2!(T, [num_cols, num_rows])
    } else {
        rlst_dynamic_array2!(T, [num_rows, num_cols])
    };
    let mut view_1 = target_arr.r_mut();

    if exchange_axis{
        for (col_ind, col) in inds.iter().enumerate() {
            view_1.r_mut().into_subview([col_ind, 0], [1, num_rows]).fill_from(view_2.r().into_subview([0, *col], [num_rows, 1]).conj());
        }
    }
    else{
        for (col_ind, col) in inds.iter().enumerate() {
            view_1.r_mut().into_subview([0, col_ind], [num_rows, 1]).fill_from(view_2.r().into_subview([0, *col], [num_rows, 1]));
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
    let mut view_1 = target_arr.r_mut();
    let view_2 = source_arr.r();
    match indices {
        ExtInsType::Axis(inds, axis, _exchange_axis) => {
            if axis == 0 {
                for (row_ind, row) in inds.iter().enumerate() {
                    view_1.r_mut().into_subview([*row, 0], [1, source_arr.shape()[1]]).fill_from(view_2.r().into_subview([row_ind, 0], [1, source_arr.shape()[1]]));
                }
            } else {
                for (col_ind, col) in inds.iter().enumerate() {
                    view_1.r_mut().into_subview([0, *col], [source_arr.shape()[0], 1]).fill_from(view_2.r().into_subview([0, col_ind], [source_arr.shape()[0], 1]));
                }

            }
        }
        ExtInsType::Cross(rows, cols) => {
            for (col_ind, col) in cols.iter().enumerate() {
                for (row_ind, row) in rows.iter().enumerate() {
                    view_1[[row_ind, col_ind]] = view_2[[*row, *col]];
                }
            }
        }
    }
}

pub fn extract_axis<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + Stride<2>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    mat: &Array<Item, ArrayImpl, 2>,
    inds: &Vec<usize>,
    axis: usize,
    trans: bool,
) -> DynamicArray<Item, 2> {
    <Extraction<Item> as MatrixExtraction>::new(mat, ExtInsType::Axis(inds.clone(), axis, trans))
        .unwrap()
        .ext
}
