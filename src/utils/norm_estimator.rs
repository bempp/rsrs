use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rlst::dense::tools::RandScalar;
pub use rlst::prelude::*;

pub fn spectral_norm_estimator<Item: RlstScalar + RandScalar>(
    arr: DynamicArray<Item, 2>,
    sample_size: usize,
) -> std::option::Option<<Item as rlst::RlstScalar>::Real>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    //let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();
    let dim = arr.shape()[1];

    let max_err = (0..sample_size)
        .into_par_iter()
        .map(|_sample_ind| {
            let mut test_vec = rlst_dynamic_array1!(Item, [dim]);
            let mut local_rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();
            test_vec.fill_from_standard_normal(&mut local_rng);
            let mut res_vec = empty_array();
            res_vec
                .view_mut()
                .simple_mult_into_resize(arr.view(), test_vec.view());
            res_vec.norm_2() / test_vec.norm_2()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap());

    max_err
}
