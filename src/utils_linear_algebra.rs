


pub use rlst::prelude::*;



pub fn lstsq<T:RlstScalar + MatrixPseudoInverse> (left: &DynamicArray<T, 2>, right: &DynamicArray<T, 2>, sol: &mut DynamicArray<T, 2>, tol: <T as RlstScalar>::Real){
    let mut arr: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
    arr.fill_from_resize(left.view());
    let shape: [usize; 2] = arr.shape();
    let mut pinv:Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = rlst_dynamic_array2!(T, [shape[1], shape[0]]);
    arr.into_pseudo_inverse_alloc(pinv.view_mut(), tol).unwrap();
    sol.view_mut().simple_mult_into_resize(pinv.view(), right.view());
}

pub fn solve_left<T: RlstScalar + MatrixPseudoInverse>(a: &DynamicArray<T, 2>, b: &DynamicArray<T, 2>, tol: <T as RlstScalar>::Real)->DynamicArray<T, 2>{
    let mut sol: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
    lstsq(a, b, &mut sol, tol);
    sol
}

pub fn solve_right<T: RlstScalar + MatrixPseudoInverse>(a: &DynamicArray<T, 2>, b: &DynamicArray<T, 2>, tol: <T as RlstScalar>::Real)->DynamicArray<T, 2>{
    let mut ah: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
    let mut bh: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
    let mut res: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
    let mut sol: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
    ah.fill_from_resize(a.view().transpose().conj());
    bh.fill_from_resize(b.view().transpose().conj());
    lstsq(&bh, &ah, &mut res, tol);
    sol.fill_from_resize(res.view().conj().transpose());
    sol
}

pub struct Extraction<
    Item: RlstScalar
> {
    pub ext: DynamicArray<Item, 2>,
}

pub trait MatrixExtraction: Sized {
    type Item: RlstScalar;
    fn new(source_arr: &mut DynamicArray<Self::Item, 2>, indices: ExtInsType) -> RlstResult<Self>;
}
pub trait MPIMatrixExtraction: Sized {
    type Item: RlstScalar;
    fn new(source_arr: &mut DynamicArray<Self::Item, 2>, indices: ExtInsType) -> RlstResult<Self>;
}
pub enum ExtInsType {
    /// Indices and axis of extraction
    Axis(Vec<usize>, usize, bool),
    /// Rows and cols indices
    Cross(Vec<usize>, Vec<usize>),
}

impl <T:RlstScalar>MatrixExtraction for Extraction<T>
{
    type Item = T;
    fn new(source_arr: &mut DynamicArray<Self::Item, 2>, indices: ExtInsType) -> RlstResult<Self>{

        match indices {
            ExtInsType::Axis(inds, axis, exchange_axis) => {
                let target_arr: DynamicArray<Self::Item, 2>;

                if axis == 0{
                    target_arr = get_rows(inds, source_arr, exchange_axis);
                }
                else{
                    target_arr= get_cols(inds, source_arr, exchange_axis);
                    
                }

                Ok(Self{ext: target_arr})

            },
            ExtInsType::Cross(rows, cols) => {
                let mut target_arr: DynamicArray<Self::Item, 2> = rlst_dynamic_array2!(Self::Item, [rows.len(), cols.len()]);
                for (col_ind, col) in cols.iter().enumerate(){
                    for (row_ind, row) in rows.iter().enumerate(){
                        *target_arr.get_mut([row_ind, col_ind]).unwrap() = *source_arr.get_mut([*row, *col]).unwrap();
                    }
                }
                Ok(Self{ext: target_arr})
            },
        }
    }
}


fn get_rows<T: RlstScalar>(inds: Vec<usize>, source_arr: &mut DynamicArray<T, 2>, exchange_axis: bool)-> DynamicArray<T, 2>{
    let mut target_arr: DynamicArray<T, 2>;
    if exchange_axis{
        target_arr = rlst_dynamic_array2!(T, [source_arr.shape()[1], inds.len()]);
        for col in 0..source_arr.shape()[1]{
            for (row_ind, row) in inds.iter().enumerate(){
                *target_arr.get_mut([col, row_ind]).unwrap() = (*source_arr.get_mut([*row, col]).unwrap()).conj();
            }
        }
    }
    else{
        target_arr = rlst_dynamic_array2!(T, [inds.len(), source_arr.shape()[1]]);
        for col in 0..source_arr.shape()[1]{
            for (row_ind, row) in inds.iter().enumerate(){
                *target_arr.get_mut([row_ind, col]).unwrap() = *source_arr.get_mut([*row, col]).unwrap();
            }
        }
    }
    target_arr
}

fn get_cols<T: RlstScalar>(inds: Vec<usize>, source_arr: &mut DynamicArray<T, 2>, exchange_axis: bool)-> DynamicArray<T, 2>{
    let mut target_arr: DynamicArray<T, 2>;
    if exchange_axis{
        target_arr = rlst_dynamic_array2!(T, [inds.len(), source_arr.shape()[0]]);
        for (col_ind, col) in inds.iter().enumerate(){
            for row in 0..source_arr.shape()[0]{
                *target_arr.get_mut([col_ind, row]).unwrap() = (*source_arr.get_mut([row, *col]).unwrap()).conj();
            }
        }

    }
    else{
        target_arr = rlst_dynamic_array2!(T, [source_arr.shape()[0], inds.len()]);
        for (col_ind, col) in inds.iter().enumerate(){
            for row in 0..source_arr.shape()[0]{
                *target_arr.get_mut([row, col_ind]).unwrap() = *source_arr.get_mut([row, *col]).unwrap();
            }
        }
    }
    target_arr
}


pub fn matrix_insertion<T: RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = T>
+ Shape<2>
+ Stride<2>
+ UnsafeRandomAccessMut<2, Item = T>
+ UnsafeRandomAccessByRef<2, Item = T>
+ RawAccessMut<Item = T>
+ Shape<2>>(target_arr: &mut DynamicArray<T, 2>, source_arr: &mut Array<T, ArrayImpl, 2>, indices: ExtInsType){
    match indices {
        ExtInsType::Axis(inds, axis, exchange_axis) => {
            if axis == 0{
                for col in 0..source_arr.shape()[1]{
                    for (row_ind, row) in inds.iter().enumerate(){
                        *target_arr.get_mut([*row, col]).unwrap() = *source_arr.get_mut([row_ind, col]).unwrap();
                    }
                }
            }
            else{
                for (col_ind, col) in inds.iter().enumerate(){
                    for row in 0..source_arr.shape()[0]{
                        *target_arr.get_mut([row, *col]).unwrap() = *source_arr.get_mut([row, col_ind]).unwrap();
                    }
                }
            }

        },
        ExtInsType::Cross(rows, cols) => {
            for (col_ind, col) in cols.iter().enumerate(){
                for (row_ind, row) in rows.iter().enumerate(){
                    *target_arr.get_mut([row_ind, col_ind]).unwrap() = *source_arr.get_mut([*row, *col]).unwrap();
                }
            }
        },
    }
}



impl <T:RlstScalar>MPIMatrixExtraction for Extraction<T>
{
    type Item = T;
    fn new(source_arr: &mut DynamicArray<Self::Item, 2>, indices: ExtInsType) -> RlstResult<Self>{
        match indices {
            ExtInsType::Axis(inds, axis, exchange_axis) => {
                if axis == 0{
                    let mut target_arr: DynamicArray<Self::Item, 2> = rlst_dynamic_array2!(Self::Item, [inds.len(), source_arr.shape()[1]]);

                    for (row_ind, row) in inds.iter().enumerate(){
                        target_arr.view_mut().slice(0, row_ind).fill_from(source_arr.view().slice(0, *row));
                    };

                    /*for col in 0..source_arr.shape()[1]{
                        for (row_ind, row) in inds.iter().enumerate(){
                            *target_arr.get_mut([row_ind, col]).unwrap() = *source_arr.get_mut([*row, col]).unwrap();
                        }
                    }*/
                    Ok(Self{ext: target_arr})
                }
                else{
                    let mut target_arr: DynamicArray<Self::Item, 2> = rlst_dynamic_array2!(Self::Item, [source_arr.shape()[0], inds.len()]);
                    /*for (col_ind, col) in inds.iter().enumerate(){
                        for row in 0..source_arr.shape()[0]{
                            *target_arr.get_mut([row, col_ind]).unwrap() = *source_arr.get_mut([row, *col]).unwrap();
                        }
                    }*/

                    for (col_ind, col) in inds.iter().enumerate(){
                        target_arr.view_mut().slice(1, col_ind).fill_from(source_arr.view().slice(1, *col));
                    }
                    Ok(Self{ext: target_arr})
                }

            },
            ExtInsType::Cross(rows, cols) => {
                let mut aux: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MPIMatrixExtraction>::new(source_arr, ExtInsType::Axis(rows, 0, false)).unwrap().ext;
                let target_arr: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MPIMatrixExtraction>::new(&mut aux, ExtInsType::Axis(cols, 1, false)).unwrap().ext;
                Ok(Self{ext: target_arr})
            },
        }
    }
}
 

pub fn mpi_matrix_insertion<T: RlstScalar>(target_arr: &mut DynamicArray<T, 2>, source_arr: &mut DynamicArray<T, 2>, indices: ExtInsType){
    match indices {
        ExtInsType::Axis(inds, axis, exchange_axis) => {
            if axis == 0{
                for col in 0..source_arr.shape()[1]{
                    for (row_ind, row) in inds.iter().enumerate(){
                        *target_arr.get_mut([*row, col]).unwrap() = *source_arr.get_mut([row_ind, col]).unwrap();
                    }
                }
            }
            else{
                for (col_ind, col) in inds.iter().enumerate(){
                    for row in 0..source_arr.shape()[0]{
                        *target_arr.get_mut([row, *col]).unwrap() = *source_arr.get_mut([row, col_ind]).unwrap();
                    }
                }
            }

        },
        ExtInsType::Cross(rows, cols) => {
            for (col_ind, col) in cols.iter().enumerate(){
                for (row_ind, row) in rows.iter().enumerate(){
                    *target_arr.get_mut([row_ind, col_ind]).unwrap() = *source_arr.get_mut([*row, *col]).unwrap();
                }
            }
        },
    }
}
