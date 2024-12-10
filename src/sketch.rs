pub use rlst::prelude::*;
pub use rlst::dense::array::empty_array;
use crate::elementary_matrix::{ElementaryOperations, ElementaryMatrix};

type ArrayImpl<Item> = BaseArray<Item, VectorContainer<Item>, 2>;
pub struct BoxesData<Item: RlstScalar> 
{
    pub sketch: DynamicArray<Item, 2>,
    pub test: DynamicArray<Item, 2>,
    pub inds: Vec<DynamicArray<usize, 1>>,
    pub perm_inds: Vec<DynamicArray<usize, 1>>
}

pub trait SketchUpdates{
    type Item: RlstScalar;
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;
    fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, inds: Vec<DynamicArray<usize, 1>>, num_samples: usize)->Self;
    fn update_sketch(&mut self, factor_mat1: &ElementaryMatrix<Self::Item>, factor_mat2: &ElementaryMatrix<Self::Item>, trans: bool);
    fn permute_indices();

}

macro_rules! impl_skel_box{
    ($scalar:ty) => {
        impl SketchUpdates for BoxesData<$scalar> {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl<$scalar>;

            fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, inds: Vec<DynamicArray<usize, 1>>, num_samples: usize)->Self{
                let perm_inds = Vec::<DynamicArray<usize, 1>>::with_capacity(inds.len());
                let mut rng = rand::thread_rng();
                let mut test = rlst_dynamic_array2!(Self::Item, [arr.shape()[1], num_samples]);
                test.fill_from_standard_normal(&mut rng); //How to set range properly
                let sketch = empty_array().simple_mult_into_resize(arr.view(), test.view());
                Self{sketch, test, inds, perm_inds}
            }

            fn update_sketch(&mut self, factor_mat1: &ElementaryMatrix<Self::Item>, factor_mat2: &ElementaryMatrix<Self::Item>, trans: bool){
                factor_mat1.mul(self.sketch.view_mut(), true, trans);
                factor_mat2.mul(self.test.view_mut(), false, trans);
            }
            fn permute_indices(){}

        }
    }
}

impl_skel_box!(f64);
impl_skel_box!(f32);