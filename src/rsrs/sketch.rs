use std::time::Instant;
use rand_distr::{Distribution, Standard, StandardNormal};
pub use rlst::{prelude::*, dense::{tools::RandScalar, array::empty_array}};
use crate::utils::elementary_matrix::{ElMatOptions, ElementaryMatrix, ElementaryOperations};

use super::rsrs_cycle::DecoupledBoxData;

//type ArrayImpl<Item> = BaseArray<Item, VectorContainer<Item>, 2>;

pub struct BoxesData<Item: RlstScalar> 
{
    pub sketch: DynamicArray<Item, 2>,
    pub test: DynamicArray<Item, 2>,
    pub dim: usize,
    pub num_samples: usize,
    pub trans: bool
}

pub trait SketchOps{
    type Item: RlstScalar;
    fn new<ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Stride<2>
    + RawAccessMut<Item = Self::Item>
    + Shape<2>>(arr: &Array<Self::Item, ArrayImpl, 2>, num_samples: usize, trans: bool)->Self;
    fn update_sketch(&mut self, factor_mat1: &ElementaryMatrix<Self::Item>, factor_mat2: &ElementaryMatrix<Self::Item>, trans: bool);
    fn add_samples<ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Stride<2>
    + RawAccessMut<Item = Self::Item>
    + Shape<2>>(&mut self, extra_num_samples: usize, arr: &Array<Self::Item, ArrayImpl, 2>, dec_boxes: &Vec<DecoupledBoxData<Self::Item>>);

}

impl <T:RlstScalar + RandScalar + mpi::datatype::Equivalence>SketchOps for BoxesData<T> 
where StandardNormal: Distribution<T::Real>,
Standard: Distribution<T::Real>,
{
    type Item = T;
    //type ArrayImpl = ArrayImpl<T>;

    fn new<ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Stride<2>
    + RawAccessMut<Item = Self::Item>
    + Shape<2>>(arr: &Array<Self::Item, ArrayImpl, 2>, num_samples: usize, trans: bool)->Self{
        let mut test: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
        let mut sketch: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = rlst_dynamic_array2!(Self::Item, [arr.shape()[0], num_samples]);
        testing(num_samples, arr, &mut sketch, &mut test, trans);
        let dim = arr.shape()[0];
        Self{sketch, test, dim, num_samples, trans}
    }

    fn update_sketch(&mut self, factor_mat1: &ElementaryMatrix<Self::Item>, factor_mat2: &ElementaryMatrix<Self::Item>, trans: bool){
        _update_sketch(&mut self.sketch, &mut self.test, factor_mat1, factor_mat2, trans);
        //factor_mat1.mul(&mut self.sketch, ElMatOptions{inv: true, trans, left: true});
        //factor_mat2.mul(&mut self.test, ElMatOptions{inv: false, trans, left: true});
    }

    fn add_samples<ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Stride<2>
    + RawAccessMut<Item = Self::Item>
    + Shape<2>>(&mut self, extra_num_samples: usize, arr: &Array<Self::Item, ArrayImpl, 2>, dec_boxes: &Vec<DecoupledBoxData<Self::Item>>){
        let start: Instant = Instant::now();
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng(); // For testing: ChaCha8Rng::seed_from_u64(0);
        let test_shape: [usize; 2] = self.test.shape();
        self.test.resize_in_place([test_shape[0], test_shape[1] + extra_num_samples]);
        self.sketch.resize_in_place([test_shape[0], test_shape[1] + extra_num_samples]);
        let mut sub_test = self.test.view_mut().into_subview([0, test_shape[1]], [test_shape[0], extra_num_samples]);
        let mut sub_sketch = self.sketch.view_mut().into_subview([0, test_shape[1]], [test_shape[0], extra_num_samples]);
        sub_test.fill_from_standard_normal(&mut rng); 
        
        if !self.trans{
            sub_sketch.view_mut().simple_mult_into(arr.view(), sub_test.view());

            for dec_box in dec_boxes{
                let fact_id = &dec_box.operators.fact_id;
                let fact_lu = &dec_box.operators.fact_lu;
                _update_sketch(&mut sub_sketch, &mut sub_test, &fact_id.left, &fact_id.right, self.trans);
                _update_sketch(&mut sub_sketch, &mut sub_test, &fact_lu.left, &fact_lu.right, self.trans);
            }

        }else{
            sub_sketch.view_mut().mult_into(TransMode::Trans, TransMode::NoTrans, num::One::one(),  arr.view(), sub_test.view(), num::Zero::zero());
            for dec_box in dec_boxes{
                let fact_id = &dec_box.operators.fact_id;
                let fact_lu = &dec_box.operators.fact_lu;
                _update_sketch(&mut sub_sketch, &mut sub_test, &fact_id.right, &fact_id.left, self.trans);
                _update_sketch(&mut sub_sketch, &mut sub_test, &fact_lu.right, &fact_lu.left, self.trans);
            }
        }

        
        self.num_samples = test_shape[1] + extra_num_samples;
        let duration = start.elapsed();

        println!(
            "Testing in {} ms",
            duration.as_millis()
        );

    }
}


fn testing<T:RlstScalar + RandScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = T>
+ Stride<2>
+ RawAccessMut<Item = T>
+ Shape<2>> (num_samples: usize, arr: &Array<T, ArrayImpl, 2>, sketch: &mut DynamicArray<T, 2>, test: &mut DynamicArray<T, 2>, trans: bool)
where StandardNormal: Distribution<T::Real>, Standard: Distribution<T::Real>
{
    let start: Instant = Instant::now();
    let mut rng : rand::prelude::ThreadRng = rand::thread_rng();//ChaCha8Rng::seed_from_u64(0);//: rand::prelude::ThreadRng = rand::thread_rng(); // For testing: ChaCha8Rng::seed_from_u64(0);
    test.resize_in_place([arr.shape()[1], num_samples]);
    test.fill_from_standard_normal(&mut rng); 

    if !trans{
        sketch.view_mut().simple_mult_into(arr.view(), test.view());
    }else{
        sketch.view_mut().mult_into(TransMode::Trans, TransMode::NoTrans, num::One::one(),  arr.view(), test.view(), num::Zero::zero());
    }
    let duration: std::time::Duration = start.elapsed();

    println!(
        "Testing in {} ms",
        duration.as_millis()
    );
    
}

fn _update_sketch<Item:RlstScalar + RandScalar + mpi::datatype::Equivalence, 
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>>
    (sketch: &mut  Array<Item, ArrayImplMut, 2>, test: &mut Array<Item, ArrayImplMut, 2>, factor_mat1: &ElementaryMatrix<Item>, factor_mat2: &ElementaryMatrix<Item>, trans: bool){
    factor_mat1.mul(sketch, ElMatOptions{inv: true, trans, left: true});
    factor_mat2.mul(test, ElMatOptions{inv: false, trans, left: true});
}