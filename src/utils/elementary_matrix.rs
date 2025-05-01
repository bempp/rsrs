//! Elementary matrices (row swapping, row multiplication and row addition)
use super::data_ins_ext::{matrix_insertion, ExtInsType, Extraction, MatrixExtraction};
use num::One;
use rlst::{
    dense::{
        traits::{MultIntoResize, RawAccessMut, Shape},
        types::{RlstResult, RlstScalar},
    },
    empty_array, Array, DynamicArray, TransMode, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
pub enum RowOpType {
    /// Row addition
    Add,
    /// Row substraction
    Sub,
}

/// Type of Elementary matrix
pub enum OpType<T: RlstScalar> {
    /// Row operation (addition or substraction)
    Row(DynamicArray<T, 2>),
    /// Row scaling
    Mul(T),
    /// Row permutation
    Perm,
}

pub struct ElMatOptions {
    /// Inverse operation
    pub inv: bool,
    /// Transpose operation
    pub trans: bool,
    /// Left or right operation
    pub left: bool,
}

pub trait ElementaryOperations: Sized {
    /// Item type
    type Item: RlstScalar;
    /// We create the Elementary matrix (E), so it can be applied to a matrix A. In other words, we implement E(A).
    /// arr: is needed when row additions/substractions on A are performed.
    /// dim: is the dimension of the elementary matrix, given by the row dimension of A.
    /// row_indices and col_indices are respectively the domain and range of this application; i.e. we take the rows of A
    /// corresponding to col_indices of A and we place the result of this operation in row_indices of A.
    /// op_type: indicates if we are adding/substracting rows of A, scaling rows of A by a scalar or if we are permuting rows of A.
    /// trans: indicates if we are applying E^T
    fn new(
        dim: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        op_type: OpType<Self::Item>,
        trans: bool,
    ) -> RlstResult<Self>;
    ///Obtain the conjugate transposed elementary metrix
    fn get_conj_transpose(&self) -> RlstResult<ElementaryMatrix<Self::Item>>;
    /// This method performs E(A). Here:
    /// right_arr: matrix A.
    /// row_op_type: indicates substraction or addition of rows
    /// alpha: is the scaling parameter of a scaling is applied
    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        right_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        options: ElMatOptions,
    );
}

pub struct ElementaryMatrix<Item: RlstScalar> {
    dim: usize,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    op_type: OpType<Item>,
    trans: bool,
}

impl<T: RlstScalar> ElementaryOperations for ElementaryMatrix<T> {
    type Item = T;

    fn new(
        dim: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        op_type: OpType<Self::Item>,
        trans: bool,
    ) -> RlstResult<Self> {
        Ok(Self {
            dim,
            row_indices,
            col_indices,
            op_type,
            trans,
        })
    }

    fn get_conj_transpose(&self) -> RlstResult<ElementaryMatrix<Self::Item>> {
        let op_type: OpType<Self::Item> = match &self.op_type {
            OpType::Row(arr) => {
                let mut aux_arr = empty_array();
                aux_arr.fill_from_resize(arr.r());
                OpType::Row(aux_arr)
            }
            OpType::Mul(alpha) => OpType::Mul(*alpha),
            OpType::Perm => OpType::Perm,
        };

        <ElementaryMatrix<Self::Item> as ElementaryOperations>::new(
            self.dim,
            self.row_indices.clone(),
            self.col_indices.clone(),
            op_type,
            !self.trans,
        )
    }

    fn mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        right_arr: &mut Array<Self::Item, ArrayImplMut, 2>,
        options: ElMatOptions,
    ) {
        let mut trans = self.trans;

        if options.trans {
            trans = !trans;
        }

        match &self.op_type {
            OpType::Row(arr) => {
                if options.left {
                    row_ops(
                        self.col_indices.clone(),
                        self.row_indices.clone(),
                        arr,
                        right_arr,
                        options.inv,
                        trans,
                    )
                } else {
                    col_ops(
                        self.col_indices.clone(),
                        self.row_indices.clone(),
                        arr,
                        right_arr,
                        options.inv,
                        trans,
                    )
                }
            }
            OpType::Mul(alpha) => {
                assert_eq!(self.row_indices.len(), self.col_indices.len());
                if options.inv {
                    row_mul(self, right_arr, <Self::Item as One>::one() / (*alpha))
                } else {
                    row_mul(self, right_arr, *alpha)
                }
            }

            OpType::Perm => {
                assert_eq!(self.row_indices.len(), self.col_indices.len());
                if options.left {
                    row_perm(
                        self.col_indices.clone(),
                        self.row_indices.clone(),
                        right_arr,
                        trans,
                    );
                } else {
                    col_perm(
                        self.col_indices.clone(),
                        self.row_indices.clone(),
                        right_arr,
                        trans,
                    );
                }
            }
        }
    }
}

///This method implements the row addition/substraction
pub fn row_ops<
    Item: RlstScalar,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Shape<2>
        + RawAccessMut<Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    c_indices: Vec<usize>,
    r_indices: Vec<usize>,
    arr: &DynamicArray<Item, 2>,
    right_arr: &mut Array<Item, ArrayImplMut, 2>,
    sub: bool,
    trans: bool,
) {
    let row_indices: Vec<usize>;
    let col_indices: Vec<usize>;

    if trans {
        col_indices = r_indices;
        row_indices = c_indices;
    } else {
        col_indices = c_indices;
        row_indices = r_indices;
    }

    let mut subarr_rows: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        right_arr,
        ExtInsType::Axis(row_indices.clone(), 0, false),
    )
    .unwrap()
    .ext;
    let mut subarr_cols: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        right_arr,
        ExtInsType::Axis(col_indices.clone(), 0, false),
    )
    .unwrap()
    .ext;

    let mut res_mul: DynamicArray<Item, 2> = empty_array::<Item, 2>();

    if trans {
        res_mul.r_mut().mult_into_resize(
            TransMode::Trans,
            TransMode::NoTrans,
            num::One::one(),
            arr.r(),
            subarr_cols.r_mut(),
            num::Zero::zero(),
        );
    } else {
        res_mul.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::NoTrans,
            num::One::one(),
            arr.r(),
            subarr_cols.r_mut(),
            num::Zero::zero(),
        );
    }

    if sub {
        subarr_rows.sub_into(res_mul.r());
    } else {
        subarr_rows.sum_into(res_mul.r());
    }

    matrix_insertion(
        right_arr,
        &subarr_rows,
        ExtInsType::Axis(row_indices.clone(), 0, false),
    );
}

///This method implements the row addition/substraction
pub fn col_ops<
    Item: RlstScalar,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Shape<2>
        + RawAccessMut<Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    c_indices: Vec<usize>,
    r_indices: Vec<usize>,
    arr: &DynamicArray<Item, 2>,
    right_arr: &mut Array<Item, ArrayImplMut, 2>,
    sub: bool,
    trans: bool,
) {
    let row_indices: Vec<usize>;
    let col_indices: Vec<usize>;

    if trans {
        col_indices = r_indices;
        row_indices = c_indices;
    } else {
        col_indices = c_indices;
        row_indices = r_indices;
    }

    let mut subarr_rows: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        right_arr,
        ExtInsType::Axis(row_indices.clone(), 1, false),
    )
    .unwrap()
    .ext;
    let mut subarr_cols: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        right_arr,
        ExtInsType::Axis(col_indices.clone(), 1, false),
    )
    .unwrap()
    .ext;

    let mut res_mul: DynamicArray<Item, 2> = empty_array::<Item, 2>();

    if trans {
        res_mul.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::Trans,
            num::One::one(),
            subarr_rows.r_mut(),
            arr.r(),
            num::Zero::zero(),
        );
    } else {
        res_mul.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::NoTrans,
            num::One::one(),
            subarr_rows.r_mut(),
            arr.r(),
            num::Zero::zero(),
        );
    }

    if sub {
        subarr_cols.sub_into(res_mul.r());
    } else {
        subarr_cols.sum_into(res_mul.r());
    }

    matrix_insertion(
        right_arr,
        &subarr_cols,
        ExtInsType::Axis(col_indices.clone(), 1, false),
    );
}

pub fn ext_rows<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    c_indices: Vec<usize>,
    r_indices: Vec<usize>,
    right_arr: &Array<Item, ArrayImpl, 2>,
    trans: bool,
    trans_right_arr: bool,
) -> DynamicArray<Item, 2> {
    let row_indices: Vec<usize>;

    if trans {
        row_indices = c_indices.clone();
    } else {
        row_indices = r_indices.clone();
    }

    let (axis, transposed) = if trans_right_arr {
        (1, true)
    } else {
        (0, false)
    };

    let subarr_rows: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        right_arr,
        ExtInsType::Axis(row_indices.clone(), axis, transposed),
    )
    .unwrap()
    .ext;

    subarr_rows
}

pub fn ext_cols<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    c_indices: Vec<usize>,
    r_indices: Vec<usize>,
    right_arr: &Array<Item, ArrayImpl, 2>,
    trans: bool,
    trans_right_arr: bool,
) -> DynamicArray<Item, 2> {
    let col_indices: Vec<usize>;

    if trans {
        col_indices = r_indices;
    } else {
        col_indices = c_indices;
    }

    let (axis, transposed) = if trans_right_arr {
        (0, true)
    } else {
        (1, false)
    };

    let subarr_cols: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        right_arr,
        ExtInsType::Axis(col_indices.clone(), axis, transposed),
    )
    .unwrap()
    .ext;

    subarr_cols
}

pub fn row_ops_no_sub<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    c_indices: Vec<usize>,
    r_indices: Vec<usize>,
    arr: &DynamicArray<Item, 2>,
    right_arr: &Array<Item, ArrayImpl, 2>,
    sub: bool,
    trans: bool,
    trans_right_arr: bool,
) -> DynamicArray<Item, 2> {
    let row_indices: Vec<usize>;
    let col_indices: Vec<usize>;

    if trans {
        col_indices = r_indices;
        row_indices = c_indices;
    } else {
        col_indices = c_indices;
        row_indices = r_indices;
    }

    let (axis, transposed) = if trans_right_arr {
        (1, true)
    } else {
        (0, false)
    };

    let mut subarr_rows: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        right_arr,
        ExtInsType::Axis(row_indices.clone(), axis, transposed),
    )
    .unwrap()
    .ext;

    let mut subarr_cols: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        right_arr,
        ExtInsType::Axis(col_indices.clone(), axis, transposed),
    )
    .unwrap()
    .ext;

    let mut res_mul: DynamicArray<Item, 2> = empty_array::<Item, 2>();

    if trans {
        res_mul.r_mut().mult_into_resize(
            TransMode::Trans,
            TransMode::NoTrans,
            num::One::one(),
            arr.r(),
            subarr_cols.r_mut(),
            num::Zero::zero(),
        );
    } else {
        res_mul.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::NoTrans,
            num::One::one(),
            arr.r(),
            subarr_cols.r_mut(),
            num::Zero::zero(),
        );
    }

    if sub {
        subarr_rows.sub_into(res_mul.r());
    } else {
        subarr_rows.sum_into(res_mul.r());
    }

    subarr_rows
}

pub fn row_subs<
    Item: RlstScalar,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Shape<2>
        + RawAccessMut<Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    c_indices: Vec<usize>,
    r_indices: Vec<usize>,
    source_arr: &DynamicArray<Item, 2>,
    target_arr: &mut Array<Item, ArrayImplMut, 2>,
    trans: bool,
    trans_subs: bool,
) {
    let row_indices: Vec<usize>;

    if trans {
        row_indices = c_indices;
    } else {
        row_indices = r_indices;
    }

    if trans_subs {
        matrix_insertion(
            target_arr,
            source_arr,
            ExtInsType::Axis(row_indices.clone(), 0, true),
        );
    } else {
        matrix_insertion(
            target_arr,
            source_arr,
            ExtInsType::Axis(row_indices.clone(), 0, false),
        );
    }
}

///This method implements the row addition/substraction
pub fn col_ops_no_sub<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
        + RawAccessMut<Item = Item>
        + Shape<2>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    c_indices: Vec<usize>,
    r_indices: Vec<usize>,
    arr: &DynamicArray<Item, 2>,
    right_arr: &Array<Item, ArrayImpl, 2>,
    sub: bool,
    trans: bool,
    trans_right_arr: bool,
) -> DynamicArray<Item, 2> {
    let row_indices: Vec<usize>;
    let col_indices: Vec<usize>;

    if trans {
        col_indices = r_indices;
        row_indices = c_indices;
    } else {
        col_indices = c_indices;
        row_indices = r_indices;
    }

    let (axis, transposed) = if trans_right_arr {
        (0, true)
    } else {
        (1, false)
    };

    let mut subarr_rows: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        right_arr,
        ExtInsType::Axis(row_indices.clone(), axis, transposed),
    )
    .unwrap()
    .ext;

    let mut subarr_cols: DynamicArray<Item, 2> = <Extraction<Item> as MatrixExtraction>::new(
        right_arr,
        ExtInsType::Axis(col_indices.clone(), axis, transposed),
    )
    .unwrap()
    .ext;

    let mut res_mul: DynamicArray<Item, 2> = empty_array::<Item, 2>();

    if trans {
        res_mul.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::Trans,
            num::One::one(),
            subarr_rows.r_mut(),
            arr.r(),
            num::Zero::zero(),
        );
    } else {
        res_mul.r_mut().mult_into_resize(
            TransMode::NoTrans,
            TransMode::NoTrans,
            num::One::one(),
            subarr_rows.r_mut(),
            arr.r(),
            num::Zero::zero(),
        );
    }

    if sub {
        subarr_cols.sub_into(res_mul.r());
    } else {
        subarr_cols.sum_into(res_mul.r());
    }

    subarr_cols
}

pub fn col_subs<
    Item: RlstScalar,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Shape<2>
        + RawAccessMut<Item = Item>
        + UnsafeRandomAccessMut<2, Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    c_indices: Vec<usize>,
    r_indices: Vec<usize>,
    source_arr: &DynamicArray<Item, 2>,
    target_arr: &mut Array<Item, ArrayImplMut, 2>,
    trans: bool,
    trans_subs: bool,
) {
    let col_indices: Vec<usize>;

    if trans {
        col_indices = r_indices;
    } else {
        col_indices = c_indices;
    }

    if trans_subs {
        matrix_insertion(
            target_arr,
            source_arr,
            ExtInsType::Axis(col_indices.clone(), 1, true),
        );
    } else {
        matrix_insertion(
            target_arr,
            source_arr,
            ExtInsType::Axis(col_indices.clone(), 1, false),
        );
    }
}

///This method implements the row permutation
pub fn row_perm<
    Item: RlstScalar,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Shape<2>
        + UnsafeRandomAccessMut<2, Item = Item>
        + RawAccessMut<Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    orig_indices: Vec<usize>,
    perm_indices: Vec<usize>,
    arr: &mut Array<Item, ArrayImplMut, 2>,
    trans: bool,
) {
    let num_cols: usize = arr.r().shape()[1];
    let p_indices: Vec<usize>;
    let o_indices: Vec<usize>;
    if trans {
        p_indices = orig_indices;
        o_indices = perm_indices;
    } else {
        p_indices = perm_indices;
        o_indices = orig_indices;
    }

    let mut copy_arr = empty_array();
    copy_arr.fill_from_resize(arr.r()); //Copy matrix
    let mut view_1 = arr.r_mut();
    let view_2 = copy_arr.r();

    for (&p_ind, &o_ind) in p_indices.iter().zip(o_indices.iter()) {
        for col in 0..num_cols {
            view_1[[p_ind, col]] = view_2[[o_ind, col]];
        }
    }
}

///This method implements the columm permutation
pub fn col_perm<
    Item: RlstScalar,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Shape<2>
        + UnsafeRandomAccessMut<2, Item = Item>
        + RawAccessMut<Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    orig_indices: Vec<usize>,
    perm_indices: Vec<usize>,
    arr: &mut Array<Item, ArrayImplMut, 2>,
    trans: bool,
) {
    let num_rows: usize = arr.r().shape()[0];
    let p_indices: Vec<usize>;
    let o_indices: Vec<usize>;

    if trans {
        p_indices = orig_indices;
        o_indices = perm_indices;
    } else {
        p_indices = perm_indices;
        o_indices = orig_indices;
    }

    let mut copy_arr = empty_array();
    copy_arr.fill_from_resize(arr.r()); //Copy matrix
    let mut view_1 = arr.r_mut();
    let view_2 = copy_arr.r();

    for (&p_ind, &o_ind) in p_indices.iter().zip(o_indices.iter()) {
        for row in 0..num_rows {
            view_1[[row, p_ind]] = view_2[[row, o_ind]];
        }
    }
}

///This method implements the row scaling
pub fn row_mul<
    Item: RlstScalar,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Shape<2>
        + UnsafeRandomAccessMut<2, Item = Item>
        + RawAccessMut<Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    el_mat: &ElementaryMatrix<Item>,
    right_arr: &mut Array<Item, ArrayImplMut, 2>,
    alpha: Item,
) {
    let right_arr_shape: [usize; 2] = right_arr.r().shape();
    let dim: usize = el_mat.dim;
    let row_indices: Vec<usize> = el_mat.row_indices.clone();
    for col in 0..dim {
        for &elem in row_indices.iter() {
            right_arr.data_mut()[col * right_arr_shape[0] + elem] =
                alpha * right_arr.data_mut()[col * right_arr_shape[0] + elem]
        }
    }
}
