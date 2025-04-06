use super::{
    rsrs_cycle::{BoxType, RsrsOptions},
    rsrs_factors::LuTimes,
};
use crate::{
    rsrs::{
        rsrs_factors::{IdFactor, IdFactorOperations, LuFactor, LuFactorOperations},
        sketch::BoxesData,
    },
    utils::{
        data_ins_ext::{ExtInsType, Extraction, MatrixExtraction},
        least_squares_and_null::null_space,
    },
};
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::dense::{
    linalg::{lu::MatrixLu, null_space::Method},
    tools::RandScalar,
};
pub use rlst::prelude::*;
use serde::Serialize;
use std::time::{Duration, Instant};

pub struct Tols<T: RlstScalar> {
    pub id: <T as RlstScalar>::Real,
    pub null: <T as RlstScalar>::Real,
    pub lstq: <T as RlstScalar>::Real,
}

pub struct LowRankResult<Item: RlstScalar> {
    pub id_factor: IdFactor<Item>,
    pub near_field_inds: Vec<usize>,
    pub target_inds: Vec<usize>,
    pub id_times: IdTimes,
}

pub enum Rank<Item: RlstScalar> {
    Low(LowRankResult<Item>),
    Full(IdTimes),
}

#[derive(Debug, Serialize, Clone)]
pub struct IdTimes {
    pub nullification: u128,
    pub id: u128,
}

#[derive(Debug, Serialize, Clone)]
pub struct UpdateTimes {
    pub id: u128,
    pub lu: u128,
}

macro_rules! impl_times_operations {
    ($struct_name:ident, $trait_name:ident, $arg_1:ident, $arg_2:ident) => {
        pub trait $trait_name {
            fn new() -> Self;
            fn sum(&mut self, $arg_1: u128, $arg_2: u128);
        }

        impl $trait_name for $struct_name {
            fn new() -> Self {
                Self {
                    $arg_1: 0_u128,
                    $arg_2: 0_u128,
                }
            }

            fn sum(&mut self, $arg_1: u128, $arg_2: u128) {
                self.$arg_1 += $arg_1;
                self.$arg_2 += $arg_2;
            }
        }
    };
}

impl_times_operations!(IdTimes, IdTimesOperations, nullification, id);
impl_times_operations!(LuTimes, LuTimesOperations, extraction, lu);
impl_times_operations!(UpdateTimes, UpdateTimesOperations, id, lu);

type Real<T> = <T as rlst::RlstScalar>::Real;
pub trait Skel<T: RlstScalar>
where
    QrDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixQrDecomposition<Item = T>,
{
    type Item: RlstScalar;
    fn null_sketch_near_field(
        &self,
        ff_sketch: &mut DynamicArray<Self::Item, 2>,
        target_inds: &Vec<usize>,
        near_field_inds: &Vec<usize>,
        sketch: &DynamicArray<Self::Item, 2>,
        test: &DynamicArray<Self::Item, 2>,
        subs_sample_dim: usize,
        tol_null: <Self::Item as RlstScalar>::Real,
    );
    fn null_near_field(
        &mut self,
        target_inds: &Vec<usize>,
        near_field_inds: &Vec<usize>,
        far_field_sketch: &mut DynamicArray<Self::Item, 2>,
        y_data: &BoxesData<Self::Item>,
        z_data: &BoxesData<Self::Item>,
        subs_sample_dim: usize,
        tol_null: <Self::Item as RlstScalar>::Real,
        hermitian: bool,
    );
    fn id_step(
        &mut self,
        box_type: &BoxType<Real<Self::Item>>,
        target_inds: &Vec<usize>,
        near_field_inds: &Vec<usize>,
        y_data: &BoxesData<Self::Item>,
        z_data: &BoxesData<Self::Item>,
        subs_sample_dim: usize,
        tols: &Tols<Self::Item>,
        options: &RsrsOptions,
    ) -> Rank<Self::Item>;
    fn lu_step(
        &self,
        y_data: &BoxesData<Self::Item>,
        z_data: &BoxesData<Self::Item>,
        ind_r: &Vec<usize>,
        near_field_inds: &Vec<usize>,
        subs_sample_dim: usize,
        tols: &Tols<Self::Item>,
        options: &RsrsOptions,
    ) -> (LuFactor<T>, LuTimes);
}

impl<
        T: RlstScalar
            + MatrixId
            + MatrixNull
            + MatrixInverse
            + MatrixPseudoInverse
            + RandScalar
            + MatrixLu
            + MatrixQr,
    > Skel<T> for T
where
    StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>,
    LuDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixLuDecomposition<Item = T>,
    QrDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixQrDecomposition<Item = T>,
{
    type Item = T;

    fn null_sketch_near_field(
        &self,
        ff_sketch: &mut DynamicArray<Self::Item, 2>,
        target_inds: &Vec<usize>,
        near_field_inds: &Vec<usize>,
        sketch: &DynamicArray<Self::Item, 2>,
        test: &DynamicArray<Self::Item, 2>,
        subs_sample_dim: usize,
        tol_null: <Self::Item as RlstScalar>::Real,
    ) {
        let row_num = test.shape()[0];
        let mut test_subview = test.r().into_subview([0, 0], [row_num, subs_sample_dim]);
        let mut sketch_subview = sketch.r().into_subview([0, 0], [row_num, subs_sample_dim]);
        let null_dim = test_subview.shape()[1] - near_field_inds.len();
        let sub_test: DynamicArray<Self::Item, 2> =
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
        let null_near_field = null_space(sub_test, Method::Svd, tol_null);
        let shape = null_near_field.null_space_arr.shape();
        ff_sketch.r_mut().simple_mult_into_resize(
            sub_sketch.r_mut(),
            null_near_field
                .null_space_arr
                .into_subview([0, 0], [shape[0], null_dim]),
        );
    }

    fn null_near_field(
        &mut self,
        target_inds: &Vec<usize>,
        near_field_inds: &Vec<usize>,
        far_field_sketch: &mut DynamicArray<Self::Item, 2>,
        y_data: &BoxesData<Self::Item>,
        z_data: &BoxesData<Self::Item>,
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
                &y_data.sketch,
                &y_data.test,
                subs_sample_dim,
                tol_null,
            );
            self.null_sketch_near_field(
                &mut null_z_sketch,
                target_inds,
                near_field_inds,
                &z_data.sketch,
                &z_data.test,
                subs_sample_dim,
                tol_null,
            );
            far_field_sketch.fill_from_resize(null_y_sketch.r() + null_z_sketch.r());
        // See if AXPY can be applied here
        } else {
            self.null_sketch_near_field(
                far_field_sketch,
                target_inds,
                near_field_inds,
                &y_data.sketch,
                &y_data.test,
                subs_sample_dim,
                tol_null,
            );
        }
    }

    fn id_step(
        &mut self,
        box_type: &BoxType<Real<Self::Item>>,
        target_inds: &Vec<usize>,
        near_field_inds: &Vec<usize>,
        y_data: &BoxesData<Self::Item>,
        z_data: &BoxesData<Self::Item>,
        subs_sample_dim: usize,
        tols: &Tols<Self::Item>,
        options: &RsrsOptions,
    ) -> Rank<Self::Item> {
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

        let mut local_target_inds = target_inds.clone();
        let mut local_near_field_inds = near_field_inds.clone();

        let start: Instant = Instant::now();
        let id_factor = <IdFactor<Self::Item> as IdFactorOperations>::new(
            &mut local_target_inds,
            &mut local_near_field_inds,
            y_data.dim,
            far_field_sketch,
            box_type,
        );
        let id_time: Duration = start.elapsed();

        let id_times = IdTimes {
            nullification: nullification_time.as_millis(),
            id: id_time.as_millis(),
        };
        match id_factor {
            Some(low_rank_factor) => {
                if !low_rank_factor.ind_r.is_empty() {
                    let low_rank_result: LowRankResult<Self::Item> = LowRankResult {
                        id_factor: low_rank_factor,
                        near_field_inds: local_near_field_inds,
                        id_times,
                        target_inds: local_target_inds,
                    };
                    Rank::Low(low_rank_result)
                } else {
                    Rank::Full(id_times)
                }
            }
            None => Rank::Full(id_times),
        }
    }

    fn lu_step(
        &self,
        y_data: &BoxesData<Self::Item>,
        z_data: &BoxesData<Self::Item>,
        ind_r: &Vec<usize>,
        near_field_inds: &Vec<usize>,
        subs_sample_dim: usize,
        tols: &Tols<Self::Item>,
        options: &RsrsOptions,
    ) -> (LuFactor<T>, LuTimes) {
        let (lu_factors, lu_times) = <LuFactor<Self::Item> as LuFactorOperations>::new(
            &ind_r,
            &near_field_inds,
            y_data,
            z_data,
            subs_sample_dim,
            tols.lstq,
            options,
        );

        (lu_factors, lu_times)
    }
}
