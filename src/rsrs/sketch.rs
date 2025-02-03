use std::time::Instant;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
pub use rlst::{prelude::*, dense::{tools::RandScalar, array::empty_array}};
use crate::utils::{elementary_matrix::{ElMatOptions, ElementaryMatrix, ElementaryOperations, OpType}, data_ins_ext::{solve_right, ExtInsType, Extraction, MatrixExtraction}};

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
    + Shape<2>>(&mut self, extra_num_samples: usize, arr: &Array<Self::Item, ArrayImpl, 2>, dec_boxes: &Vec<DecoupledBoxData<Self::Item>>, seed: u64);
    fn get_sketch_box(&mut self, rows: Vec<usize>, cols: Vec<usize>, tol_lstq: <Self::Item as RlstScalar>::Real)->DynamicArray<Self::Item, 2>;
    fn extract_diag_boxes(&mut self, ind_r: Vec<Vec<usize>>, ind_s: Vec<Vec<usize>>, tol_lstq: <Self::Item as RlstScalar>::Real)-> (Vec<DynamicArray<Self::Item, 2>>, ElementaryMatrix<Self::Item>);

}

impl <T:RlstScalar + RandScalar + mpi::datatype::Equivalence + rlst::MatrixPseudoInverse>SketchOps for BoxesData<T> 
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
    + Shape<2>>(&mut self, extra_num_samples: usize, arr: &Array<Self::Item, ArrayImpl, 2>, dec_boxes: &Vec<DecoupledBoxData<Self::Item>>, seed: u64){
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

    fn get_sketch_box(&mut self, rows: Vec<usize>, cols: Vec<usize>, tol_lstq: <Self::Item as RlstScalar>::Real)->DynamicArray<Self::Item, 2>{
        let sketch_r: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(&mut self.sketch, ExtInsType::Axis(rows, 0, false)).unwrap().ext;
        let test_c: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(&mut self.test, ExtInsType::Axis(cols, 0, false)).unwrap().ext;
        solve_right(&sketch_r, &test_c, tol_lstq)
    }

    fn extract_diag_boxes(&mut self, ind_r: Vec<Vec<usize>>, ind_s: Vec<Vec<usize>>, tol_lstq: <Self::Item as RlstScalar>::Real)-> (Vec<DynamicArray<Self::Item, 2>>, ElementaryMatrix<Self::Item>){
        let mut diag_boxes : Vec<DynamicArray<Self::Item, 2>> = Vec::new();
        let rows: Vec<usize> = (0..self.dim).collect();
        let mut acc_ind_s = Vec::new();
        let mut acc_ind_r = Vec::new();
        let mut count = 0;
        
        for inds in ind_s.iter(){
            acc_ind_s.extend_from_slice(inds);
        }

        for inds in ind_r.iter(){
            acc_ind_r.extend_from_slice(inds);
        }

        let mut cols = acc_ind_r;
        cols.extend_from_slice(&acc_ind_s);

        let remaining_indices = rows.clone().into_iter().filter(|&el| !cols.contains(&el)).collect::<Vec<_>>();
        cols.extend_from_slice(&remaining_indices);

        let p_matrix: ElementaryMatrix<Self::Item> = <ElementaryMatrix<Self::Item> as ElementaryOperations>::new(self.dim, rows, cols, OpType::Perm, false).unwrap();

        p_matrix.mul(&mut self.sketch, ElMatOptions{inv: false, trans: false, left: true});
        p_matrix.mul(&mut self.test, ElMatOptions{inv: false, trans: false, left: true});

        for inds in ind_r.iter(){
            let num_els = inds.len();
            let pinds_r: Vec<usize> = (count..(num_els + count)).collect();
            diag_boxes.push(self.get_sketch_box(pinds_r.clone(),pinds_r, tol_lstq));
            count += num_els
        }

        let pinds_s: Vec<usize> = (count..self.dim).collect();

        diag_boxes.push(self.get_sketch_box(pinds_s.clone(),pinds_s, tol_lstq));
        (diag_boxes, p_matrix)
    }
}


fn testing<T:RlstScalar + RandScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = T>
+ Stride<2>
+ RawAccessMut<Item = T>
+ Shape<2>> (num_samples: usize, arr: &Array<T, ArrayImpl, 2>, sketch: &mut DynamicArray<T, 2>, test: &mut DynamicArray<T, 2>, trans: bool)
where StandardNormal: Distribution<T::Real>, Standard: Distribution<T::Real>
{
    let start: Instant = Instant::now();
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng(); // For testing: ChaCha8Rng::seed_from_u64(0);
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