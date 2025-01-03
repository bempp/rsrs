use itertools::min;
pub use rlst::prelude::*;
use num::{One, Zero};


type RealScalar<T> = <T as RlstScalar>::Real;
/// QR decomposition
pub struct NullSpace<
    Item: RlstScalar
> {
    ///Computed null space
    pub null_space_arr: Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>,
}

pub trait NullSpaceComputation{
    type Item: RlstScalar;
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;
    fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, tol: RealScalar<Self::Item>) -> Self;
    fn find_matrix_rank(singular_values: &mut DynamicArray<RealScalar<Self::Item>, 1>, dim: usize, tol:RealScalar<Self::Item>)->usize;
}

type ArrayImpl<Item> = BaseArray<Item, VectorContainer<Item>, 2>;

impl <T: RlstScalar + MatrixSvd> NullSpaceComputation for NullSpace<T> {
    type Item = T;

    type ArrayImpl = ArrayImpl<T>;

   fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, tol: RealScalar<Self::Item>) -> Self {

        let shape: [usize; 2] = arr.shape();
        let dim: usize = min(shape).unwrap();
        let mut singular_values: DynamicArray<RealScalar<Self::Item>, 1> = rlst_dynamic_array1!(RealScalar<Self::Item>, [dim]);
        let mode: SvdMode = rlst::dense::linalg::svd::SvdMode::Full;
        let mut u: DynamicArray<Self::Item, 2> = rlst_dynamic_array2!(Self::Item, [shape[0], shape[0]]);
        let mut vt: DynamicArray<Self::Item, 2> = rlst_dynamic_array2!(Self::Item, [shape[1], shape[1]]);

        arr.into_svd_alloc(u.view_mut(), vt.view_mut(), singular_values.data_mut(), mode).unwrap();

        //For a full rank rectangular matrix, then rank = dim. 
        //find_matrix_rank checks if the matrix is full rank and recomputes the rank.
        let rank: usize = Self::find_matrix_rank(&mut singular_values, dim, tol);

        //The null space is given by the last shape[1]-rank columns of V
        let mut null_space_arr: DynamicArray<Self::Item, 2> = empty_array();
        null_space_arr.fill_from_resize(vt.conj().transpose().into_subview([0, rank], [shape[1], shape[1]-rank]));

        Self {null_space_arr}

    }

    fn find_matrix_rank(singular_values: &mut DynamicArray<RealScalar<Self::Item>, 1>, dim: usize, tol:<Self::Item as RlstScalar>::Real)->usize{
        //We compute the rank of the matrix by expecting the values of the elements in the diagonal of R.
        let max: RealScalar<Self::Item> = singular_values.view().iter().max_by(|a, b| (a.abs().partial_cmp(&b.abs())).unwrap()).unwrap().abs();
        let mut rank: usize = dim;

        if max.re() > <RealScalar<Self::Item> as Zero>::zero(){
            let alpha: RealScalar<Self::Item> = <RealScalar<Self::Item> as One>::one()/max;
            singular_values.scale_inplace(alpha);
            let aux_vec: Vec<RealScalar<Self::Item>> = singular_values.iter().filter(|el| el.abs() > tol ).collect::<Vec<RealScalar<Self::Item>>>();
            rank = aux_vec.len();
        }

        rank

    }
}


