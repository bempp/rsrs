//! Elementary matrices (row swapping, row multiplication and row addition)
use super::data_ins_ext::{matrix_insertion, ExtInsType, Extraction, MatrixExtraction};
use num::One;
use rlst::{
    dense::{
        traits::{MultIntoResize, RawAccessMut, Shape},
        types::{RlstResult, RlstScalar},
    },
    empty_array, rlst_dynamic_array2, Array, DynamicArray, TransMode, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

//use crate::linear_algebra::{matrix_insertion, ExtInsType, Extraction, MatrixExtraction};
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
                let mut beta: Self::Item = <Self::Item as One>::one();
                if options.inv {
                    beta = -<Self::Item as One>::one();
                }

                if options.left {
                    row_ops(
                        self.col_indices.clone(),
                        self.row_indices.clone(),
                        arr,
                        right_arr,
                        beta,
                        trans,
                    )
                } else {
                    col_ops(
                        self.col_indices.clone(),
                        self.row_indices.clone(),
                        arr,
                        right_arr,
                        beta,
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
    beta: Item,
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

    subarr_rows.sum_into(res_mul.r().scalar_mul(beta));
    matrix_insertion(
        right_arr,
        &mut subarr_rows,
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
    beta: Item,
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

    subarr_cols.sum_into(res_mul.r().scalar_mul(beta));
    matrix_insertion(
        right_arr,
        &mut subarr_cols,
        ExtInsType::Axis(col_indices.clone(), 1, false),
    );
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
    beta: Item,
    trans: bool,
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

    subarr_rows.sum_into(res_mul.r().scalar_mul(beta));

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
) {
    let row_indices: Vec<usize>;

    if trans {
        row_indices = c_indices;
    } else {
        row_indices = r_indices;
    }

    matrix_insertion(
        target_arr,
        source_arr,
        ExtInsType::Axis(row_indices.clone(), 0, false),
    );
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
    beta: Item,
    trans: bool,
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

    subarr_cols.sum_into(res_mul.r().scalar_mul(beta));

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
) {
    let col_indices: Vec<usize>;

    if trans {
        col_indices = r_indices;
    } else {
        col_indices = c_indices;
    }
    matrix_insertion(
        target_arr,
        source_arr,
        ExtInsType::Axis(col_indices.clone(), 1, false),
    );
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
    c_indices: Vec<usize>,
    r_indices: Vec<usize>,
    right_arr: &mut Array<Item, ArrayImplMut, 2>,
    trans: bool,
) {
    let col_dim: usize = right_arr.r().shape()[1];
    let row_indices: Vec<usize>;
    let col_indices: Vec<usize>;

    if trans {
        col_indices = r_indices.clone();
        row_indices = c_indices.clone();
    } else {
        col_indices = c_indices.clone();
        row_indices = r_indices.clone();
    }

    let mut subarr_cols: Array<Item, rlst::BaseArray<Item, rlst::VectorContainer<Item>, 2>, 2> =
        rlst_dynamic_array2!(Item, [col_indices.len(), col_dim]);
    let mut view_1 = subarr_cols.r_mut();
    let mut view_2 = right_arr.r_mut();

    for col in 0..col_dim {
        for (row, &elem) in col_indices.iter().enumerate() {
            view_1[[row, col]] = view_2[[elem, col]];
        }
    }

    for col in 0..col_dim {
        for (row, &elem) in row_indices.iter().enumerate() {
            view_2[[elem, col]] = view_1[[row, col]];
        }
    }
}

pub fn col_perm<
    Item: RlstScalar,
    ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
        + Shape<2>
        + UnsafeRandomAccessMut<2, Item = Item>
        + RawAccessMut<Item = Item>
        + UnsafeRandomAccessByRef<2, Item = Item>,
>(
    c_indices: Vec<usize>,
    r_indices: Vec<usize>,
    right_arr: &mut Array<Item, ArrayImplMut, 2>,
    trans: bool,
) {
    let row_dim: usize = right_arr.r().shape()[0];
    let row_indices: Vec<usize>;
    let col_indices: Vec<usize>;

    if trans {
        col_indices = r_indices.clone();
        row_indices = c_indices.clone();
    } else {
        col_indices = c_indices.clone();
        row_indices = r_indices.clone();
    }

    let mut subarr_cols: Array<Item, rlst::BaseArray<Item, rlst::VectorContainer<Item>, 2>, 2> =
        rlst_dynamic_array2!(Item, [row_indices.len(), row_dim]);

    let mut view_1 = subarr_cols.r_mut();
    let mut view_2 = right_arr.r_mut();
    for row in 0..row_dim {
        for (col, &elem) in row_indices.iter().enumerate() {
            view_1[[row, col]] = view_2[[row, elem]];
        }
    }

    for row in 0..row_dim {
        for (col, &elem) in col_indices.iter().enumerate() {
            view_2[[row, elem]] = view_1[[row, col]];
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
