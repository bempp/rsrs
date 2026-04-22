use crate::rsrs::args::RsrsOptions;
use crate::rsrs::rsrs_factors::commutative_factors::BoxType;
use crate::rsrs::rsrs_factors::commutative_factors::IdFactor;
use crate::rsrs::rsrs_factors::commutative_factors::LuFactor;
use crate::rsrs::rsrs_factors::null_and_extract::ExtractionScratch;
use crate::rsrs::sketch::SamplingSpace;
use crate::rsrs::sketch::SketchData;
use crate::rsrs::statistics::IdTimes;
use crate::rsrs::statistics::Times;
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::dense::{
    linalg::{interpolative_decomposition::MatrixIdNoSkel, lu::MatrixLu},
    tools::RandScalar,
};
pub use rlst::prelude::*;
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

pub struct FullRankResult {
    pub len_near_field_inds: usize,
    pub len_target_inds: usize,
    pub id_times: Times,
}

#[allow(clippy::large_enum_variant)]
pub enum Rank<Item: RlstScalar> {
    Low(LowRankResult<Item>),
    Full(FullRankResult),
}

type Real<T> = <T as rlst::RlstScalar>::Real;
pub trait Skel<T: RlstScalar, Space: SamplingSpace<F = T>>
where
    QrDecomposition<T, BaseArray<T, VectorContainer<T>, 2>>: MatrixQrDecomposition<Item = T>,
{
    type Item: RlstScalar;
    #[allow(clippy::too_many_arguments)]
    fn id_step(
        &mut self,
        scratch: &mut ExtractionScratch<Self::Item>,
        box_type: &BoxType<Real<Self::Item>>,
        target_inds: &[usize],
        near_field_inds: &[usize],
        y_data: &SketchData<Self::Item>,
        z_data: &SketchData<Self::Item>,
        subs_sample_dim: usize,
        options: &RsrsOptions<Self::Item>,
    ) -> Rank<Self::Item>;
    #[allow(clippy::too_many_arguments)]
    fn lu_step(
        &self,
        scratch: &mut ExtractionScratch<Self::Item>,
        y_data: &SketchData<Self::Item>,
        z_data: &SketchData<Self::Item>,
        ind_r: &[usize],
        near_field_inds: &[usize],
        inactive_inds: &[usize],
        subs_sample_dim: usize,
        options: &RsrsOptions<Self::Item>,
    ) -> Option<(LuFactor<T>, Times)>;
}

impl<
        T: RlstScalar
            + MatrixId
            + MatrixIdNoSkel
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
        scratch: &mut ExtractionScratch<Self::Item>,
        box_type: &BoxType<Real<Self::Item>>,
        target_inds: &[usize],
        near_field_inds: &[usize],
        y_data: &SketchData<Self::Item>,
        z_data: &SketchData<Self::Item>,
        subs_sample_dim: usize,
        options: &RsrsOptions<Self::Item>,
    ) -> Rank<Self::Item> {
        let fixed_rank = if options.id_options.tol_id > num::One::one() {
            Some(num::ToPrimitive::to_usize(&options.id_options.tol_id).unwrap())
        } else {
            None
        };
        let skip_id = target_inds.len() <= options.min_rank
            || fixed_rank.is_some_and(|rank| target_inds.len() <= rank);
        if skip_id {
            let id_times = IdTimes {
                nullification: 0_u128,
                id: 0_u128,
            };

            let times = Times::Id(id_times);
            let full_rank_result: FullRankResult = FullRankResult {
                len_near_field_inds: near_field_inds.to_vec().len(),
                id_times: times,
                len_target_inds: target_inds.len(),
            };
            return Rank::Full(full_rank_result);
        }
        let mut local_target_inds = target_inds.to_vec();
        let mut local_near_field_inds = near_field_inds.to_vec();

        let (id_factor, id_times) = IdFactor::new(
            scratch,
            &mut local_target_inds,
            &mut local_near_field_inds,
            y_data,
            z_data,
            subs_sample_dim,
            options.id_options.tol_id > num::One::one(),
            box_type,
            &options.id_options,
            &options.symmetry,
        );

        match id_factor {
            Some(low_rank_factor) if !low_rank_factor.ind_r.is_empty() => {
                let low_rank_result: LowRankResult<Self::Item> = LowRankResult {
                    id_factor: low_rank_factor,
                    near_field_inds: local_near_field_inds,
                    id_times,
                    target_inds: local_target_inds,
                };
                Rank::Low(low_rank_result)
            }
            None => {
                let full_rank_result: FullRankResult = FullRankResult {
                    len_near_field_inds: near_field_inds.to_vec().len(),
                    id_times,
                    len_target_inds: target_inds.len(),
                };
                Rank::Full(full_rank_result)
            }
            Some(_) => {
                let full_rank_result: FullRankResult = FullRankResult {
                    len_near_field_inds: near_field_inds.to_vec().len(),
                    id_times,
                    len_target_inds: target_inds.len(),
                };
                Rank::Full(full_rank_result)
            }
        }
    }

    fn lu_step(
        &self,
        scratch: &mut ExtractionScratch<Self::Item>,
        y_data: &SketchData<Self::Item>,
        z_data: &SketchData<Self::Item>,
        ind_r: &[usize],
        near_field_inds: &[usize],
        inactive_inds: &[usize],
        subs_sample_dim: usize,
        options: &RsrsOptions<Self::Item>,
    ) -> Option<(LuFactor<T>, Times)> {
        if ind_r.is_empty() {
            return None;
        }
        if near_field_inds.len() > ind_r.len() {
            let (lu_factors, lu_times) = LuFactor::new(
                scratch,
                ind_r,
                near_field_inds,
                inactive_inds,
                y_data,
                z_data,
                subs_sample_dim,
                options.id_options.tol_id > num::One::one(),
                &options.lu_options,
                &options.symmetry,
            );
            Some((lu_factors.unwrap(), lu_times))
        } else {
            None
        }
    }
}
