use mpi::traits::{CommunicatorCollectives};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
//use rand::SeedableRng;
//use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::dense::tools::RandScalar;
pub use rlst::prelude::*;
pub use rlst::dense::array::empty_array;
use crate::elementary_matrix::{ElMatOptions, ElementaryMatrix, ElementaryOperations};
//use core::slice::SlicePattern;
use std::time::Instant;
//use mpi::datatype::DynBufferMut;

type ArrayImpl<Item> = BaseArray<Item, VectorContainer<Item>, 2>;

pub struct BoxesData<Item: RlstScalar> 
{
    pub sketch: DynamicArray<Item, 2>,
    pub test: DynamicArray<Item, 2>,
    pub num_samples: usize,
    pub trans: bool
}

pub trait SketchOps{
    type Item: RlstScalar;
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;
    fn new<C: CommunicatorCollectives>(arr: &Array<Self::Item, Self::ArrayImpl, 2>, num_samples: usize, trans: bool, comm: &C)->Self;
    fn update_sketch(&mut self, factor_mat1: &ElementaryMatrix<Self::Item>, factor_mat2: &ElementaryMatrix<Self::Item>, trans: bool);
    fn add_samples(&mut self, extra_num_samples: usize, arr: &Array<Self::Item, ArrayImpl<Self::Item>, 2>);

}


impl <T:RlstScalar + RandScalar + mpi::datatype::Equivalence>SketchOps for BoxesData<T> 
where StandardNormal: Distribution<T::Real>,
Standard: Distribution<T::Real>,
{
    type Item = T;
    type ArrayImpl = ArrayImpl<T>;

    fn new<C: CommunicatorCollectives>(arr: &Array<Self::Item, Self::ArrayImpl, 2>, num_samples: usize, trans: bool, comm: &C)->Self{
        let mut test: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
        let mut sketch: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = rlst_dynamic_array2!(Self::Item, [arr.shape()[0], num_samples]);//empty_array();
        
        
        testing(num_samples, arr, &mut sketch, &mut test, trans);

        Self{sketch, test, num_samples, trans}
    }

    fn update_sketch(&mut self, factor_mat1: &ElementaryMatrix<Self::Item>, factor_mat2: &ElementaryMatrix<Self::Item>, trans: bool){
        factor_mat1.mul(&mut self.sketch, ElMatOptions{inv: true, trans, left: true});
        factor_mat2.mul(&mut self.test, ElMatOptions{inv: false, trans, left: true});
    }

    fn add_samples(&mut self, extra_num_samples: usize, arr: &Array<T, ArrayImpl<T>, 2>){
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
        }else{
            sub_sketch.view_mut().mult_into(TransMode::Trans, TransMode::NoTrans, num::One::one(),  arr.view(), sub_test.view(), num::Zero::zero());
        }
        self.num_samples = test_shape[1] + extra_num_samples;
        let duration = start.elapsed();

        println!(
            "Testing in {} ms",
            duration.as_millis()
        );
    }
}


fn testing<T:RlstScalar + RandScalar> (num_samples: usize, arr: &Array<T, ArrayImpl<T>, 2>, sketch: &mut Array<T, ArrayImpl<T>, 2>, test: &mut Array<T, ArrayImpl<T>, 2>, trans: bool)
where StandardNormal: Distribution<T::Real>, Standard: Distribution<T::Real>
{
    let start: Instant = Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(0);//: rand::prelude::ThreadRng = rand::thread_rng(); // For testing: ChaCha8Rng::seed_from_u64(0);
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
