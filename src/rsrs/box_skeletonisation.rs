use super::{
    rsrs_cycle::{BoxType, RsrsOptions},
    rsrs_factors::{IdTimes, LuTimes, Times},
};
use crate::rsrs::sketch::SamplingSpace;
use crate::rsrs::{
    rsrs_factors::{IdFactor, LuFactor},
    sketch::SketchData,
};
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::dense::{linalg::lu::MatrixLu, tools::RandScalar};
pub use rlst::prelude::*;
use serde::Serialize;
pub struct Tols<T: RlstScalar> {
    pub id: <T as RlstScalar>::Real,
    pub null: <T as RlstScalar>::Real,
    pub lstq: <T as RlstScalar>::Real,
}

pub struct LowRankResult<Item: RlstScalar> {
    pub id_factor: IdFactor<Item>,
    pub near_field_inds: Vec<usize>,
    pub target_inds: Vec<usize>,
    pub id_times: Times,
}

#[allow(clippy::large_enum_variant)]
pub enum Rank<Item: RlstScalar> {
    Low(LowRankResult<Item>),
    Full(Times),
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
pub trait Skel<T: RlstScalar, Space: SamplingSpace<F = T>>
where
    QrDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixQrDecomposition<Item = T>,
{
    type Item: RlstScalar;
    #[allow(clippy::too_many_arguments)]
    fn id_step(
        &mut self,
        box_type: &BoxType<Real<Self::Item>>,
        target_inds: &[usize],
        near_field_inds: &[usize],
        y_data: &SketchData<Self::Item>,
        z_data: &SketchData<Self::Item>,
        subs_sample_dim: usize,
        options: &RsrsOptions<Self::Item>,
    ) -> Rank<Self::Item>;
    fn lu_step(
        &self,
        y_data: &SketchData<Self::Item>,
        z_data: &SketchData<Self::Item>,
        ind_r: &mut [usize],
        near_field_inds: &mut [usize],
        subs_sample_dim: usize,
        options: &RsrsOptions<Self::Item>,
    ) -> Option<(LuFactor<T>, Times)>;
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
        Space: SamplingSpace<F = T>,
    > Skel<T, Space> for T
where
    StandardNormal: Distribution<T::Real>,
    Standard: Distribution<T::Real>,
    LuDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixLuDecomposition<Item = T>,
    QrDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixQrDecomposition<Item = T>,
    TriangularMatrix<T>: TriangularOperations<Item = T>,
{
    type Item = T;
    fn id_step(
        &mut self,
        box_type: &BoxType<Real<Self::Item>>,
        target_inds: &[usize],
        near_field_inds: &[usize],
        y_data: &SketchData<Self::Item>,
        z_data: &SketchData<Self::Item>,
        subs_sample_dim: usize,
        options: &RsrsOptions<Self::Item>,
    ) -> Rank<Self::Item> {
        if target_inds.len() <= options.min_rank {
            let id_times = IdTimes {
                nullification: 0_u128,
                id: 0_u128,
            };

            let times = Times::Id(id_times);
            return Rank::Full(times);
        }
        let mut local_target_inds = target_inds.to_vec();
        let mut local_near_field_inds = near_field_inds.to_vec();

        let (id_factor, id_times) = IdFactor::new(
            &mut local_target_inds,
            &mut local_near_field_inds,
            y_data,
            z_data,
            subs_sample_dim,
            box_type,
            options,
        );

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
        y_data: &SketchData<Self::Item>,
        z_data: &SketchData<Self::Item>,
        ind_r: &mut [usize],
        near_field_inds: &mut [usize],
        subs_sample_dim: usize,
        options: &RsrsOptions<Self::Item>,
    ) -> Option<(LuFactor<T>, Times)> {
        if near_field_inds.len() > ind_r.len() {
            let (lu_factors, lu_times) = LuFactor::new(
                ind_r,
                near_field_inds,
                y_data,
                z_data,
                subs_sample_dim,
                options,
            );
            Some((lu_factors.unwrap(), lu_times))
        } else {
            None
        }
    }
}
