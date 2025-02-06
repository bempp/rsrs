use super::rsrs_factors::{DecFactorOpType, FactorOptions, FactorType, IdFactor, IdFactorOperations, LuFactor, LuFactorOperations, PermOperations, RsrsFactors};
use crate::utils::data_ins_ext::{solve_right, ExtInsType, Extraction, MatrixExtraction};
pub use rlst::{prelude::*, dense::{tools::RandScalar, array::empty_array}};
use rand_distr::{Distribution, Standard, StandardNormal};
use std::time::{Duration, Instant};
//use rand_chacha::ChaCha8Rng;
//use rand::SeedableRng;

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
    + Shape<2>>(arr: &Array<Self::Item, ArrayImpl, 2>, trans: bool)->Self;
    fn add_samples<ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Stride<2>
    + RawAccessMut<Item = Self::Item>
    + Shape<2>>(&mut self, extra_num_samples: usize, arr: &Array<Self::Item, ArrayImpl, 2>, rsrs_factors: &RsrsFactors<Self::Item>, silent: bool, _seed:u64)->Duration;
    fn get_sketch_box(&mut self, rows: Vec<usize>, cols: Vec<usize>, tol_lstq: <Self::Item as RlstScalar>::Real)->DynamicArray<Self::Item, 2>;
    fn extract_diag_boxes(&mut self, ind_r: Vec<Vec<usize>>, ind_s: Vec<Vec<usize>>, tol_lstq: <Self::Item as RlstScalar>::Real, rsrs_factors: &mut RsrsFactors<Self::Item>);

}

impl <T:RlstScalar + RandScalar + rlst::MatrixPseudoInverse + MatrixId + MatrixInverse>SketchOps for BoxesData<T> 
where StandardNormal: Distribution<T::Real>,
Standard: Distribution<T::Real>,
{
    type Item = T;

    fn new<ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Stride<2>
    + RawAccessMut<Item = Self::Item>
    + Shape<2>>(arr: &Array<Self::Item, ArrayImpl, 2>, trans: bool)->Self{
        let test: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
        let sketch: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2> = empty_array();
        let dim = arr.shape()[0];
        Self{sketch, test, dim, num_samples: 0, trans}
    }

    fn add_samples<ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
    + Stride<2>
    + RawAccessMut<Item = Self::Item>
    + Shape<2>>(&mut self, extra_num_samples: usize, arr: &Array<Self::Item, ArrayImpl, 2>, rsrs_factors: &RsrsFactors<Self::Item>, silent: bool, _seed: u64)->Duration{
        let start: Instant = Instant::now();
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng(); // For testing: ChaCha8Rng::seed_from_u64(0);
        let test_shape: [usize; 2] = self.test.shape();
        self.test.resize_in_place([self.dim, test_shape[1] + extra_num_samples]);
        self.sketch.resize_in_place([self.dim, test_shape[1] + extra_num_samples]);
        let mut sub_test = self.test.view_mut().into_subview([0, test_shape[1]], [self.dim, extra_num_samples]);
        let mut sub_sketch = self.sketch.view_mut().into_subview([0, test_shape[1]], [self.dim, extra_num_samples]);
        sub_test.fill_from_standard_normal(&mut rng); 
        
        if !self.trans{
            sub_sketch.view_mut().simple_mult_into(arr.view(), sub_test.view());
            for dec_factor in &rsrs_factors.dec_factors{
                let fact_id = &dec_factor.id_factor;
                let fact_lu = &dec_factor.lu_factor;
                update_sketch_id(&mut sub_sketch, &mut sub_test, fact_id, FactorType::F, FactorType::S, self.trans);
                update_sketch_lu(&mut sub_sketch, &mut sub_test, fact_lu, FactorType::F, FactorType::S, self.trans);
            }

        }else{
            sub_sketch.view_mut().mult_into(TransMode::Trans, TransMode::NoTrans, num::One::one(),  arr.view(), sub_test.view(), num::Zero::zero());
            for dec_factor in &rsrs_factors.dec_factors{
                let fact_id = &dec_factor.id_factor;
                let fact_lu = &dec_factor.lu_factor;
                update_sketch_id(&mut sub_sketch, &mut sub_test, fact_id, FactorType::S, FactorType::F, self.trans);
                update_sketch_lu(&mut sub_sketch, &mut sub_test, fact_lu, FactorType::S, FactorType::F, self.trans);
            }
        }

        
        self.num_samples = test_shape[1] + extra_num_samples;
        let duration = start.elapsed();

        if !silent{
            println!(
                "Testing in {} ms",
                duration.as_millis()
            );
        }

        duration

    }

    fn get_sketch_box(&mut self, rows: Vec<usize>, cols: Vec<usize>, tol_lstq: <Self::Item as RlstScalar>::Real)->DynamicArray<Self::Item, 2>{
        let sketch_r: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(&mut self.sketch, ExtInsType::Axis(rows, 0, false)).unwrap().ext;
        let test_c: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(&mut self.test, ExtInsType::Axis(cols, 0, false)).unwrap().ext;
        solve_right(&sketch_r, &test_c, tol_lstq)
    }

    fn extract_diag_boxes(&mut self, ind_r: Vec<Vec<usize>>, ind_s: Vec<Vec<usize>>, tol_lstq: <Self::Item as RlstScalar>::Real, rsrs_factors: &mut RsrsFactors<Self::Item>){
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

        rsrs_factors.perm_factor.col_indices = cols;
        rsrs_factors.perm_factor.row_indices = rows;
        

        rsrs_factors.perm_factor.left_mul(&mut self.sketch, &FactorOptions{ inv: false, trans: false});
        rsrs_factors.perm_factor.left_mul(&mut self.test, &FactorOptions{ inv: false, trans: false});

        for inds in ind_r.iter(){
            let num_els = inds.len();
            let pinds_r: Vec<usize> = (count..(num_els + count)).collect();
            rsrs_factors.diag_box_factor.diag_boxes.push(self.get_sketch_box(pinds_r.clone(),pinds_r, tol_lstq));
            count += num_els
        }

        let pinds_s: Vec<usize> = (count..self.dim).collect();

        rsrs_factors.diag_box_factor.diag_boxes.push(self.get_sketch_box(pinds_s.clone(),pinds_s, tol_lstq));
    }
}



pub fn update_sketch_id<Item:RlstScalar + RandScalar + MatrixId + MatrixInverse, 
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>>
    (sketch: &mut  Array<Item, ArrayImplMut, 2>, test: &mut Array<Item, ArrayImplMut, 2>, factor: &IdFactor<Item>, factor1: FactorType, factor2: FactorType, trans: bool){
    factor.mul(sketch, &FactorOptions{ inv: true, trans}, factor1, DecFactorOpType::Left);
    factor.mul(test, &FactorOptions{ inv: false, trans}, factor2, DecFactorOpType::Left);
}


pub fn update_sketch_lu<Item:RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse, 
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>>
    (sketch: &mut  Array<Item, ArrayImplMut, 2>, test: &mut Array<Item, ArrayImplMut, 2>, factor: &LuFactor<Item>, factor1: FactorType, factor2: FactorType, trans: bool){
    factor.mul(sketch, &FactorOptions{ inv: true, trans}, factor1, DecFactorOpType::Left);
    factor.mul(test, &FactorOptions{ inv: false, trans}, factor2, DecFactorOpType::Left);
}