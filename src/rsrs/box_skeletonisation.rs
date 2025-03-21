use super::{
    rsrs_cycle::{BoxType, RsrsOptions},
    rsrs_factors::LuTimes,
};
use crate::{
    rsrs::{
        rsrs_factors::{
            DecFactors, FactorType, IdFactor, IdFactorOperations, LuFactor, LuFactorOperations,
        },
        sketch::{update_sketch_id, update_sketch_lu, BoxesData},
    },
    utils::{
        data_ins_ext::{ExtInsType, Extraction, MatrixExtraction},
        elementary_matrix::ElementaryMatrix,
    },
};
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::dense::tools::RandScalar;
pub use rlst::prelude::*;
use serde::Serialize;
use std::time::{Duration, Instant};

//type ArrayImpl<Item> = BaseArray<Item, VectorContainer<Item>, 2>;

pub struct Tols<T: RlstScalar> {
    pub id: <T as RlstScalar>::Real,
    pub id_2: <T as RlstScalar>::Real,
    pub null: <T as RlstScalar>::Real,
    pub lstq: <T as RlstScalar>::Real,
}
/*pub struct Box
{
    pub near_field_inds: Vec<usize>,
}*/

pub trait Skel<T: RlstScalar> {
    type Item: RlstScalar;
    fn null_sketch_near_field(
        &self,
        ff_sketch: &mut DynamicArray<Self::Item, 2>,
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        sketch: &mut DynamicArray<Self::Item, 2>,
        test: &mut DynamicArray<Self::Item, 2>,
        subs_sample_dim: usize,
        tol_null: <Self::Item as RlstScalar>::Real,
    );
    fn null_near_field(
        &mut self,
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        far_field_sketch: &mut DynamicArray<Self::Item, 2>,
        y_data: &mut BoxesData<Self::Item>,
        z_data: &mut BoxesData<Self::Item>,
        subs_sample_dim: usize,
        tol_null: <Self::Item as RlstScalar>::Real,
        hermitian: bool,
    );
    fn id_step(
        &mut self,
        box_type: &BoxType,
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        y_data: &mut BoxesData<Self::Item>,
        z_data: &mut BoxesData<Self::Item>,
        rsrs_factors: &mut Vec<DecFactors<Self::Item>>,
        subs_sample_dim: usize,
        tols: &Tols<Self::Item>,
        options: &RsrsOptions,
    ) -> Rank;
    fn lu_step(
        &self,
        y_data: &mut BoxesData<Self::Item>,
        z_data: &mut BoxesData<Self::Item>,
        rsrs_factor: &mut DecFactors<Self::Item>,
        subs_sample_dim: usize,
        tols: &Tols<Self::Item>,
        options: &RsrsOptions,
    ) -> (LuTimes, u128);
    fn id_and_lu_steps(
        &mut self,
        box_type: &BoxType,
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        y_data: &mut BoxesData<Self::Item>,
        z_data: &mut BoxesData<Self::Item>,
        rsrs_factors: &mut Vec<DecFactors<Self::Item>>,
        subs_sample_dim: usize,
        tols: &Tols<Self::Item>,
        options: &RsrsOptions,
    ) -> BoxStats;
}

pub struct Factor<T: RlstScalar> {
    pub left: ElementaryMatrix<T>,
    pub right: ElementaryMatrix<T>,
}

pub struct DecoupledBox<T: RlstScalar> {
    pub fact_id: Factor<T>,
    pub fact_lu: Factor<T>,
    pub perm: Vec<usize>,
}

pub enum BoxStats {
    Low(DecTimes),
    Full(IdTimes),
}

pub enum Rank {
    Low(IdTimes),
    Full(IdTimes),
}

#[derive(Serialize, Clone)]
pub struct IdTimes {
    pub nullification: u128,
    pub id: u128,
}

#[derive(Serialize, Clone)]
pub struct UpdateTimes {
    pub id: u128,
    pub lu: u128,
}

pub struct DecTimes {
    pub id_times: IdTimes,
    pub lu_times: LuTimes,
    pub update_times: UpdateTimes,
}

impl<T: RlstScalar + MatrixId + MatrixNull + MatrixInverse + MatrixPseudoInverse + RandScalar>
    Skel<T> for T
where
    StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>,
{
    type Item = T;

    fn null_sketch_near_field(
        &self,
        ff_sketch: &mut DynamicArray<Self::Item, 2>,
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        sketch: &mut DynamicArray<Self::Item, 2>,
        test: &mut DynamicArray<Self::Item, 2>,
        subs_sample_dim: usize,
        tol_null: <Self::Item as RlstScalar>::Real,
    ) {
        let row_num = test.shape()[0];
        let mut test_subview = test
            .view_mut()
            .into_subview([0, 0], [row_num, subs_sample_dim]);
        let mut sketch_subview = sketch
            .view_mut()
            .into_subview([0, 0], [row_num, subs_sample_dim]);
        let null_dim = test_subview.shape()[1] - near_field_inds.len();
        let mut sub_test: DynamicArray<Self::Item, 2> =
            <Extraction<Self::Item> as MatrixExtraction>::new(
                &mut test_subview,
                ExtInsType::Axis(near_field_inds.clone(), 0, false),
            )
            .unwrap()
            .ext;
        let mut sub_sketch: DynamicArray<Self::Item, 2> =
            <Extraction<Self::Item> as MatrixExtraction>::new(
                &mut sketch_subview,
                ExtInsType::Axis(target_inds.clone(), 0, false),
            )
            .unwrap()
            .ext;
        let null_near_field: NullSpace<Self::Item> =
            sub_test.view_mut().into_null_alloc(tol_null).unwrap();
        let shape = null_near_field.null_space_arr.shape();
        ff_sketch.view_mut().simple_mult_into_resize(
            sub_sketch.view_mut(),
            null_near_field
                .null_space_arr
                .into_subview([0, 0], [shape[0], null_dim]),
        );
    }

    fn null_near_field(
        &mut self,
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        far_field_sketch: &mut DynamicArray<Self::Item, 2>,
        y_data: &mut BoxesData<Self::Item>,
        z_data: &mut BoxesData<Self::Item>,
        subs_sample_dim: usize,
        tol_null: <Self::Item as RlstScalar>::Real,
        hermitian: bool,
    ) {
        if !hermitian {
            let mut null_y_sketch: DynamicArray<Self::Item, 2> = empty_array();
            let mut null_z_sketch: DynamicArray<Self::Item, 2> = empty_array();
            self.null_sketch_near_field(
                &mut null_y_sketch,
                target_inds,
                near_field_inds,
                &mut y_data.sketch,
                &mut y_data.test,
                subs_sample_dim,
                tol_null,
            );
            self.null_sketch_near_field(
                &mut null_z_sketch,
                target_inds,
                near_field_inds,
                &mut z_data.sketch,
                &mut z_data.test,
                subs_sample_dim,
                tol_null,
            );
            far_field_sketch.fill_from_resize(null_y_sketch.view() + null_z_sketch.view());
        // See if AXPY can be applied here
        } else {
            self.null_sketch_near_field(
                far_field_sketch,
                target_inds,
                near_field_inds,
                &mut y_data.sketch,
                &mut y_data.test,
                subs_sample_dim,
                tol_null,
            );
        }
    }

    fn id_step(
        &mut self,
        box_type: &BoxType,
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        y_data: &mut BoxesData<Self::Item>,
        z_data: &mut BoxesData<Self::Item>,
        rsrs_factors: &mut Vec<DecFactors<Self::Item>>,
        subs_sample_dim: usize,
        tols: &Tols<Self::Item>,
        options: &RsrsOptions,
    ) -> Rank {
        let mut far_field_sketch: DynamicArray<Self::Item, 2> = empty_array();
        let start: Instant = Instant::now();
        self.null_near_field(
            target_inds,
            near_field_inds,
            &mut far_field_sketch,
            y_data,
            z_data,
            subs_sample_dim,
            tols.null,
            options.hermitian,
        );
        let nullification_time: Duration = start.elapsed();
        if !options.silent {
            println!("Nullification in {} ms", nullification_time.as_millis());
        }
        let mut aux_target_inds = target_inds.clone();

        let tol_id = if options.adaptive_tol {
            match box_type {
                BoxType::New => tols.id,
                BoxType::Merged => tols.id_2,
            }
        } else {
            tols.id
        };

        let start: Instant = Instant::now();
        let id_factor = <IdFactor<Self::Item> as IdFactorOperations>::new(
            &mut aux_target_inds,
            near_field_inds,
            far_field_sketch,
            tol_id,
            options,
        );
        let id_time: Duration = start.elapsed();
        if !options.silent {
            println!("ID in {} ms", id_time.as_millis());
        }
        let id_times = IdTimes {
            nullification: nullification_time.as_millis(),
            id: id_time.as_millis(),
        };
        match id_factor {
            Some(low_rank_factor) => {
                if !low_rank_factor.ind_r.is_empty() {
                    target_inds.clone_from(&aux_target_inds);
                    let dec_factor = DecFactors {
                        id_factor: low_rank_factor,
                        lu_factor: None,
                        near_field_inds: near_field_inds.clone(),
                    };
                    rsrs_factors.push(dec_factor);
                    Rank::Low(id_times)
                } else {
                    Rank::Full(id_times)
                }
            }
            None => Rank::Full(id_times),
        }
    }

    fn lu_step(
        &self,
        y_data: &mut BoxesData<Self::Item>,
        z_data: &mut BoxesData<Self::Item>,
        rsrs_factor: &mut DecFactors<Self::Item>,
        subs_sample_dim: usize,
        tols: &Tols<Self::Item>,
        options: &RsrsOptions,
    ) -> (LuTimes, u128) {
        let start: Instant = Instant::now();

        let (lu_factors, lu_times) = <LuFactor<Self::Item> as LuFactorOperations>::new(
            &rsrs_factor.id_factor.ind_r,
            &rsrs_factor.near_field_inds,
            y_data,
            z_data,
            subs_sample_dim,
            tols.lstq,
            options,
        );

        let lu_time: Duration = start.elapsed();

        if !options.silent {
            println!("LU in {} ms", lu_time.as_millis());
        }

        let start: Instant = Instant::now();

        update_sketch_lu(
            &mut y_data.sketch,
            &mut y_data.test,
            &lu_factors,
            FactorType::F,
            FactorType::S,
            false,
        );
        if !options.hermitian {
            update_sketch_lu(
                &mut z_data.sketch,
                &mut z_data.test,
                &lu_factors,
                FactorType::S,
                FactorType::F,
                true,
            );
        }

        let update_lu_time: Duration = start.elapsed();

        if !options.silent {
            println!("Update from LU in {} ms", update_lu_time.as_millis());
        }

        rsrs_factor.lu_factor = Some(lu_factors);

        (lu_times, update_lu_time.as_millis())
    }

    fn id_and_lu_steps(
        &mut self,
        box_type: &BoxType,
        target_inds: &mut Vec<usize>,
        near_field_inds: &mut Vec<usize>,
        y_data: &mut BoxesData<Self::Item>,
        z_data: &mut BoxesData<Self::Item>,
        rsrs_factors: &mut Vec<DecFactors<Self::Item>>,
        subs_sample_dim: usize,
        tols: &Tols<Self::Item>,
        options: &RsrsOptions,
    ) -> BoxStats {
        let rank = self.id_step(
            box_type,
            target_inds,
            near_field_inds,
            y_data,
            z_data,
            rsrs_factors,
            subs_sample_dim,
            tols,
            options,
        );

        match rank {
            Rank::Low(id_times) => {
                let mut last_factor = rsrs_factors.last_mut().unwrap();

                let start: Instant = Instant::now();
                update_sketch_id(
                    &mut y_data.sketch,
                    &mut y_data.test,
                    &last_factor.id_factor,
                    FactorType::F,
                    FactorType::S,
                    false,
                );
                if !options.hermitian {
                    update_sketch_id(
                        &mut z_data.sketch,
                        &mut z_data.test,
                        &last_factor.id_factor,
                        FactorType::S,
                        FactorType::F,
                        true,
                    );
                }
                let update_id_time: Duration = start.elapsed();

                let (lu_times, update_lu_time) = self.lu_step(
                    y_data,
                    z_data,
                    &mut last_factor,
                    subs_sample_dim,
                    tols,
                    options,
                );
                let update_times = UpdateTimes {
                    id: update_id_time.as_millis(),
                    lu: update_lu_time,
                };
                let dec_times = DecTimes {
                    id_times,
                    lu_times,
                    update_times,
                };
                BoxStats::Low(dec_times)
            }
            Rank::Full(id_times) => BoxStats::Full(id_times),
        }
    }
}
