use crate::{rsrs::{rsrs_factors::{DecFactors, FactorType, IdFactor, IdFactorOperations, LuFactor, LuFactorOperations}, sketch::{update_sketch_id, update_sketch_lu, BoxesData}}, utils::{data_ins_ext::{ExtInsType, Extraction, MatrixExtraction}, elementary_matrix::ElementaryMatrix}};
use super::{rsrs_cycle::RsrsOptions, rsrs_factors::RsrsFactors};
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::dense::tools::RandScalar;
use std::time::{Duration, Instant};
pub use rlst::prelude::*;

//type ArrayImpl<Item> = BaseArray<Item, VectorContainer<Item>, 2>;

pub struct Tols<T:RlstScalar> {
    pub id: <T as RlstScalar>::Real,
    pub null: <T as RlstScalar>::Real,
    pub lstq: <T as RlstScalar>::Real,
}
pub struct BoxNearField
{
    pub near_field_inds: Vec<usize>,
}

pub trait SkelBox<T: RlstScalar>{
    type Item: RlstScalar;
    fn new(near_field_inds: Vec<usize>)->Self;
    fn null_sketch_near_field(&self, ff_sketch: &mut DynamicArray<Self::Item, 2>, target_inds: &mut Vec<usize>, sketch: &mut DynamicArray<Self::Item, 2>, test: &mut DynamicArray<Self::Item, 2>, tol_null: <Self::Item as RlstScalar>::Real);
    fn null_near_field(&mut self, target_inds: &mut Vec<usize>, far_field_sketch: &mut DynamicArray<Self::Item, 2>, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, tol_null: <Self::Item as RlstScalar>::Real, hermitian: bool);  
    fn decouple(&mut self, target_inds: &mut Vec<usize>, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, rsrs_factors: &mut RsrsFactors<Self::Item>, tols: &Tols<Self::Item>, options: &RsrsOptions)->Rank;
}

pub struct Factor<T: RlstScalar>
{
    pub left: ElementaryMatrix<T>,
    pub right: ElementaryMatrix<T>,
}

pub struct DecoupledBox<T: RlstScalar>{
    pub fact_id: Factor<T>,
    pub fact_lu: Factor<T>,
    pub perm: Vec<usize>

}

pub enum Rank{
    Low(Vec<usize>, Vec<usize>, Duration, Duration, Duration, Duration, Duration),
    Full(Duration, Duration),
}

impl <T: RlstScalar +
    MatrixId + MatrixNull + 
    MatrixInverse + MatrixPseudoInverse + 
    RandScalar> SkelBox<T> for BoxNearField
    where StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>,
{
    type Item = T;

    fn new(near_field_inds: Vec<usize>)->Self{
        Self{near_field_inds}
    }

    fn null_sketch_near_field(&self, ff_sketch: &mut DynamicArray<Self::Item, 2>, target_inds: &mut Vec<usize>, sketch: &mut DynamicArray<Self::Item, 2>, test: &mut DynamicArray<Self::Item, 2>, tol_null: <Self::Item as RlstScalar>::Real){
        let null_dim = test.shape()[1]- self.near_field_inds.len();
        let mut sub_test: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(test, ExtInsType::Axis(self.near_field_inds.clone(), 0, false)).unwrap().ext;
        let mut sub_sketch: DynamicArray<Self::Item, 2> = <Extraction<Self::Item> as MatrixExtraction>::new(sketch, ExtInsType::Axis(target_inds.clone(), 0, false)).unwrap().ext;
        let null_near_field: NullSpace<Self::Item> = sub_test.view_mut().into_null_alloc(tol_null).unwrap();
        let shape = null_near_field.null_space_arr.shape(); 
        ff_sketch.view_mut().simple_mult_into_resize(sub_sketch.view_mut(), null_near_field.null_space_arr.into_subview([0, shape[1]-null_dim], [shape[0], null_dim]));
    }

    fn null_near_field(&mut self, target_inds: &mut Vec<usize>, far_field_sketch: &mut DynamicArray<Self::Item, 2>, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, tol_null: <Self::Item as RlstScalar>::Real, hermitian: bool){
        if !hermitian{
            let mut null_y_sketch : DynamicArray<Self::Item, 2> = empty_array();
            let mut null_z_sketch : DynamicArray<Self::Item, 2> = empty_array();
            self.null_sketch_near_field(&mut null_y_sketch, target_inds, &mut y_data.sketch, & mut y_data.test, tol_null);
            self.null_sketch_near_field(&mut null_z_sketch, target_inds, &mut z_data.sketch, & mut z_data.test, tol_null);
            far_field_sketch.fill_from_resize(null_y_sketch.view() + null_z_sketch.view());// See if AXPY can be applied here
        }
        else{
            self.null_sketch_near_field(far_field_sketch, target_inds, &mut y_data.sketch, & mut y_data.test, tol_null);
        }
    }

    fn decouple(&mut self, target_inds: &mut Vec<usize>, y_data: &mut BoxesData<Self::Item>, z_data: &mut BoxesData<Self::Item>, rsrs_factors: &mut RsrsFactors<Self::Item>, tols: &Tols<Self::Item>,  options: &RsrsOptions)->Rank{
        let mut far_field_sketch: DynamicArray<Self::Item, 2> = empty_array();
        let start: Instant = Instant::now();
        self.null_near_field(target_inds, &mut far_field_sketch, y_data, z_data, tols.null, options.hermitian);
        let nullification_time: Duration = start.elapsed();
        if !options.silent{
            println!("Nullification in {} ms", nullification_time.as_millis());
        }
        let mut aux_target_inds = target_inds.clone();
        let start: Instant = Instant::now();
        let id_factor = <IdFactor<Self::Item> as IdFactorOperations>::new(&mut aux_target_inds, &mut self.near_field_inds, far_field_sketch, tols.id, options);
        let id_time: Duration = start.elapsed();

        if!options.silent{
            println!("ID in {} ms", id_time.as_millis());
        }

        match id_factor{
            Some(low_rank_factor) =>{
                if !low_rank_factor.ind_r.is_empty(){
                    target_inds.clone_from(&aux_target_inds);

                    let start: Instant = Instant::now();

                    update_sketch_id(&mut y_data.sketch, &mut y_data.test, &low_rank_factor, FactorType::F, FactorType::S, false);
                    if !options.hermitian{
                        update_sketch_id(&mut z_data.sketch, &mut z_data.test, &low_rank_factor, FactorType::S, FactorType::F, true);
                    }

                    let update_id_time: Duration = start.elapsed();

                    if !options.silent{
                        println!("Update from ID factors in {} ms", update_id_time.as_millis());
                    }

                    let start: Instant = Instant::now();

                    let lu_factors = <LuFactor<Self::Item> as LuFactorOperations>::new(&low_rank_factor.ind_r, &self.near_field_inds, y_data, z_data, tols.lstq, options);
                    
                    let lu_time: Duration = start.elapsed();

                    if !options.silent{
                        println!("LU in {} ms", lu_time.as_millis());
                    }

                    let start: Instant = Instant::now();

                    update_sketch_lu(&mut y_data.sketch, &mut y_data.test, &lu_factors, FactorType::F, FactorType::S, false);
                    if !options.hermitian{
                        update_sketch_lu(&mut z_data.sketch, &mut z_data.test, &lu_factors, FactorType::S, FactorType::F, true);
                    }

                    let update_lu_time: Duration = start.elapsed();

                    if !options.silent{
                        println!("Update from LU in {} ms", update_lu_time.as_millis());
                    }

                    let ind_r = low_rank_factor.ind_r.clone();
                    let ind_s = low_rank_factor.ind_s.clone();
                    let dec_factor = DecFactors{ id_factor: low_rank_factor, lu_factor: lu_factors, near_field_inds: self.near_field_inds.clone()};
                    rsrs_factors.dec_factors.push(dec_factor);
                    Rank::Low(ind_r, ind_s, nullification_time, id_time, lu_time, update_id_time, update_lu_time)
                }
                else{
                    Rank::Full(nullification_time, id_time)
                }

            },
            None => {
                Rank::Full(nullification_time, id_time)
            },
        }
    }

}
 


