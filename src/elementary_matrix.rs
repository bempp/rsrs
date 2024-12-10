//! Elementary matrices (row swapping, row multiplication and row addition)
use rlst::dense::traits::{RawAccessMut, Shape, MultIntoResize};
use rlst::dense::types::{c32, c64, RlstResult, RlstScalar};
use rlst::{empty_array, rlst_dynamic_array2, DynamicArray};
use rlst::dense::traits::accessors::RandomAccessMut;
use rlst::UnsafeRandomAccessByValue;
use rlst::Stride;
use rlst::Array;
pub enum RowOpType {
    /// Row addition
    Add,
    /// Row substraction
    Sub,
}

/// Type of Elementary matrix
pub enum OpType<T:RlstScalar> {
    /// Row operation (addition or substraction)
    Row(DynamicArray<T, 2>),
    /// Row scaling
    Mul(T),
    /// Row permutation
    Perm,
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
    fn new(dim:usize, row_indices: Vec<usize>, col_indices: Vec<usize>, op_type: OpType<Self::Item>) -> RlstResult<Self>;
    ///Obtain the conjugate transposed elementary metrix
    fn get_conj_transpose(&self)-> RlstResult<ElementaryMatrix<Self::Item>>;
    /// This method performs E(A). Here:
    /// right_arr: matrix A.
    /// row_op_type: indicates substraction or addition of rows
    /// alpha: is the scaling parameter of a scaling is applied
    fn mul<
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + Shape<2>
        + RawAccessMut<Item = Self::Item>,
        >(&self, right_arr: Array<Self::Item, ArrayImpl, 2>, inv: bool, trans: bool);
}

pub struct ElementaryMatrix<Item: RlstScalar> 
{
    dim: usize,
    row_indices: Vec<usize>, 
    col_indices: Vec<usize>,
    op_type: OpType<Item>,
}

macro_rules! impl_el_mat {
    ($scalar:ty) => {
        impl ElementaryOperations for ElementaryMatrix<$scalar>
        {
            type Item = $scalar;

            fn new(dim:usize, row_indices: Vec<usize>, col_indices: Vec<usize>, op_type: OpType<Self::Item>) -> RlstResult<Self> {
                
                Ok(Self{dim, row_indices, col_indices, op_type})
            }

            fn get_conj_transpose(&self)-> RlstResult<ElementaryMatrix<Self::Item>>{
                match &self.op_type{
                    OpType::Row(arr) => {
                        let mut arr_conj = empty_array();
                        arr_conj.fill_from_resize(arr.view().conj().transpose());
                        return <ElementaryMatrix<Self::Item> as ElementaryOperations>::new(self.dim, self.col_indices.clone(), self.row_indices.clone(), OpType::Row(arr_conj));
                    },
                    OpType::Mul(alpha) => {
                        let alpha_conj = alpha.conj();
                        return <ElementaryMatrix<Self::Item> as ElementaryOperations>::new(self.dim, self.col_indices.clone(), self.row_indices.clone(), OpType::Mul(alpha_conj));
                    },
                    OpType::Perm => {
                        return <ElementaryMatrix<Self::Item> as ElementaryOperations>::new(self.dim, self.col_indices.clone(), self.row_indices.clone(), OpType::Perm);
                    }
                };
            }

            fn mul<
            ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
                + Stride<2>
                + Shape<2>
                + RawAccessMut<Item = Self::Item>,
                >(&self, right_arr: Array<Self::Item, ArrayImpl, 2>, inv: bool, trans: bool){

                match &self.op_type{
                    OpType::Row(arr) => {
                        if inv 
                        {
                            let beta: Self::Item = (-1.0).into();
                            row_ops(self.col_indices.clone(), self.row_indices.clone(), self.dim, arr, right_arr, beta, trans)
                        }
                        else{
                            let beta: Self::Item = 1.0.into();
                            row_ops(self.col_indices.clone(), self.row_indices.clone(), self.dim, arr, right_arr, beta, trans)   
                        }
                    },
                    OpType::Mul(alpha) => {
                        assert_eq!(self.row_indices.len(), self.col_indices.len());
                        if inv 
                        {
                            row_mul(self, right_arr, 1.0/alpha)
                        }
                        else{
                            row_mul(self, right_arr, *alpha)
                        }
                    },
                    
                    OpType::Perm => {
                        assert_eq!(self.row_indices.len(), self.col_indices.len());
                        if inv{
                            row_perm(self, right_arr, true)
                        }
                        else{
                            row_perm(self, right_arr, trans)
                        }
                    }
                }
            }

        }
    }
}

///This method implements the row addition/substraction
fn row_ops<Item:RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
+ Stride<2>
+ Shape<2>
+ RawAccessMut<Item = Item>>(c_indices: Vec<usize>, r_indices: Vec<usize>, dim: usize, arr: &DynamicArray<Item, 2>, mut right_arr: Array<Item, ArrayImpl, 2>, beta: Item, trans: bool){
    
    let mut aux_arr: DynamicArray<Item, 2> = empty_array();

    let row_indices: Vec<usize>;
    let col_indices: Vec<usize>;

    if trans{
        aux_arr.fill_from_resize(arr.view().conj().transpose());
        col_indices = r_indices;
        row_indices = c_indices;
    }
    else{
        aux_arr.fill_from_resize(arr.view());
        col_indices = c_indices;
        row_indices = r_indices;
    }

    let mut subarr_cols: DynamicArray<Item, 2>= rlst_dynamic_array2!(Item, [col_indices.len(), dim]);
    let mut subarr_rows: DynamicArray<Item, 2>= rlst_dynamic_array2!(Item, [row_indices.len(), dim]);
    
    
    let right_arr_shape: [usize; 2] = right_arr.view().shape();

    for col in 0..dim{
        for (row, &elem) in col_indices.iter().enumerate(){
            *subarr_cols.get_mut([row, col]).unwrap() = right_arr.data_mut()[col*right_arr_shape[0] + elem];
        }
        for (row, &elem) in row_indices.iter().enumerate(){
            *subarr_rows.get_mut([row, col]).unwrap() = right_arr.data_mut()[col*right_arr_shape[0] + elem];
        }
    }

    let res_mul: DynamicArray<Item, 2> = empty_array::<Item, 2>().simple_mult_into_resize(aux_arr, subarr_cols.view_mut());
    let mut add_res: DynamicArray<Item, 2>= rlst_dynamic_array2!(Item, subarr_rows.shape());
    add_res.fill_from(subarr_rows.view() + res_mul.view().scalar_mul(beta));

    for col in 0..dim{
        for (row, &elem) in row_indices.iter().enumerate(){
            right_arr.data_mut()[col*right_arr_shape[0] + elem] = *add_res.get_mut([row, col]).unwrap();
        }
    }
}

///This method implements the row scaling
fn row_mul<Item:RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
+ Stride<2>
+ Shape<2>
+ RawAccessMut<Item = Item>>(el_mat: &ElementaryMatrix<Item>, mut right_arr: Array<Item, ArrayImpl, 2>, alpha: Item){
    let right_arr_shape = right_arr.view().shape();
    let dim = el_mat.dim;
    let row_indices = el_mat.row_indices.clone();
    for col in 0..dim{
        for &elem in row_indices.iter(){
            right_arr.data_mut()[col*right_arr_shape[0] + elem] = alpha*right_arr.data_mut()[col*right_arr_shape[0] + elem]
        }
    }
}

///This method implements the row permutation
fn row_perm<Item:RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
+ Stride<2>
+ Shape<2>
+ RawAccessMut<Item = Item>>(el_mat: &ElementaryMatrix<Item>, mut right_arr: Array<Item, ArrayImpl, 2>, trans: bool){
    let dim = el_mat.dim;
    let row_indices: Vec<usize>;
    let col_indices: Vec<usize>;

    if trans{
        col_indices = el_mat.row_indices.clone();
        row_indices = el_mat.col_indices.clone();
    }
    else{
        col_indices = el_mat.col_indices.clone();
        row_indices = el_mat.row_indices.clone();
    }

    let right_arr_shape = right_arr.view().shape();
    let mut subarr_cols = rlst_dynamic_array2!(Item, [col_indices.len(), dim]);
    for col in 0..dim{
        for (row, &elem) in col_indices.iter().enumerate(){
            *subarr_cols.get_mut([row, col]).unwrap() = right_arr.data_mut()[col*right_arr_shape[0] + elem];
        }
    }
    for col in 0..dim{
        for (row, &elem) in row_indices.iter().enumerate(){
            right_arr.data_mut()[col*right_arr_shape[0] + elem] = *subarr_cols.get_mut([row, col]).unwrap();
        }
    }
}

impl_el_mat!(f64);
impl_el_mat!(f32);
impl_el_mat!(c32);
impl_el_mat!(c64);
