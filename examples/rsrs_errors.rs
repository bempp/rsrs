use bempp_octree::{generate_random_points, Octree};
use bempp_rsrs::{
    rsrs::{
        box_skeletonisation::Tols,
        rsrs_cycle::{Rsrs, RsrsData, RsrsOptions},
        rsrs_factors::{
            Factor, FactorOperations, FactorOptions, FactorType, IdFactor, LuFactor, RsrsFactors,
            RsrsFactorsOps, RsrsSide,
        },
    },
    utils::data_ins_ext::{ExtInsType, Extraction, MatrixExtraction},
};
use mpi::{topology::SimpleCommunicator, traits::CommunicatorCollectives};
use num::NumCast;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
    prelude::*,
};
use std::sync::{Arc, Mutex};

type Real<T> = <T as rlst::RlstScalar>::Real;

type Errors<T> = (Real<T>, Real<T>);
type LuErrors<T> = Vec<Errors<T>>;
type IdErrors<T> = Vec<Errors<T>>;

// Error functions

pub fn spectral_norm_estimator<Item: RlstScalar + RandScalar>(
    arr: DynamicArray<Item, 2>,
    sample_size: usize,
) -> std::option::Option<<Item as rlst::RlstScalar>::Real>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let dim = arr.shape()[1];

    let max_err = (0..sample_size)
        .into_iter()
        .map(|_sample_ind| {
            let mut test_vec = rlst_dynamic_array1!(Item, [dim]);
            let mut local_rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();
            test_vec.fill_from_standard_normal(&mut local_rng);
            let mut res_vec = empty_array();
            res_vec
                .r_mut()
                .simple_mult_into_resize(arr.r(), test_vec.r());
            res_vec.norm_2() / test_vec.norm_2()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap());

    max_err
}

pub fn app_inv_error<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu,
>(
    target_arr: &DynamicArray<Item, 2>,
    rsrs_factors: &mut RsrsFactors<Item>,
    sample_size: usize,
    side: RsrsSide,
) -> Real<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let dim = target_arr.shape()[1];
    let mut sample_mat_1 = empty_array();
    let mut sample_mat_2 = empty_array();
    let mut local_rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();
    let factor_options = FactorOptions {
        inv: true,
        trans: false,
    };

    let view_shape;
    let view_offset = match side {
        RsrsSide::Left => |ind| [0, ind],
        RsrsSide::Right => |ind| [ind, 0],
        RsrsSide::Squeeze => |_ind| [0, 0],
    };

    match side {
        RsrsSide::Left => {
            sample_mat_1.resize_in_place([dim, sample_size]);
            sample_mat_1.fill_from_standard_normal(&mut local_rng);
            sample_mat_2
                .r_mut()
                .simple_mult_into_resize(target_arr.r(), sample_mat_1.r());
            view_shape = [dim, 1];
        }
        RsrsSide::Right => {
            sample_mat_1.resize_in_place([sample_size, dim]);
            sample_mat_1.fill_from_standard_normal(&mut local_rng);
            sample_mat_2
                .r_mut()
                .simple_mult_into_resize(sample_mat_1.r(), target_arr.r());
            view_shape = [1, dim];
        }
        RsrsSide::Squeeze => {
            view_shape = [0, 0];
        }
    }

    rsrs_factors.mul(&mut sample_mat_2, side, &factor_options);

    let mut res = empty_array();
    res.fill_from_resize(sample_mat_2.r() - sample_mat_1.r());

    let max_err = (0..sample_size)
        .into_iter()
        .map(|sample_ind| {
            let binding = res.r().into_subview(view_offset(sample_ind), view_shape);
            let res_view = binding.view_flat();
            let binding = sample_mat_1
                .r()
                .into_subview(view_offset(sample_ind), view_shape);
            let sample_vec = binding.view_flat();
            res_view.norm_2() / sample_vec.norm_2()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap());

    max_err.unwrap()
}

pub fn app_error<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu,
>(
    target_arr: &DynamicArray<Item, 2>,
    rsrs_factors: &mut RsrsFactors<Item>,
    sample_size: usize,
    side: RsrsSide,
) -> Real<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let dim = target_arr.shape()[1];

    let mut sample_mat_1 = empty_array();
    let mut sample_mat_2 = empty_array();
    let mut local_rng: rand::rngs::StdRng = rand::SeedableRng::from_entropy();
    let factor_options = FactorOptions {
        inv: false,
        trans: false,
    };

    let view_shape;
    let view_offset = match side {
        RsrsSide::Left => |ind| [0, ind],
        RsrsSide::Right => |ind| [ind, 0],
        RsrsSide::Squeeze => |_ind| [0, 0],
    };

    match side {
        RsrsSide::Left => {
            sample_mat_1.resize_in_place([dim, sample_size]);
            sample_mat_1.fill_from_standard_normal(&mut local_rng);
            sample_mat_2
                .r_mut()
                .simple_mult_into_resize(target_arr.r(), sample_mat_1.r());
            view_shape = [dim, 1];
        }
        RsrsSide::Right => {
            sample_mat_1.resize_in_place([sample_size, dim]);
            sample_mat_1.fill_from_standard_normal(&mut local_rng);
            sample_mat_2
                .r_mut()
                .simple_mult_into_resize(sample_mat_1.r(), target_arr.r());
            view_shape = [1, dim];
        }
        RsrsSide::Squeeze => {
            view_shape = [0, 0];
        }
    }

    rsrs_factors.mul(&mut sample_mat_1, side, &factor_options);

    let mut res = empty_array();
    res.fill_from_resize(sample_mat_2.r() - sample_mat_1.r());

    let max_err = (0..sample_size)
        .into_iter()
        .map(|sample_ind| {
            let binding = res.r().into_subview(view_offset(sample_ind), view_shape);
            let res_view = binding.view_flat();
            let binding = sample_mat_2
                .r()
                .into_subview(view_offset(sample_ind), view_shape);
            let sample_vec = binding.view_flat();
            res_view.norm_2() / sample_vec.norm_2()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap());

    max_err.unwrap()
}

pub fn rsrs_error_estimator<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu,
>(
    target_arr: &DynamicArray<Item, 2>,
    rsrs_factors: &mut RsrsFactors<Item>,
    sample_size: usize,
) -> (Real<Item>, Real<Item>, Real<Item>, Real<Item>)
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let app_inv_err_left = app_inv_error(target_arr, rsrs_factors, sample_size, RsrsSide::Left);
    let app_inv_err_right = app_inv_error(target_arr, rsrs_factors, sample_size, RsrsSide::Right);
    let app_err_left = app_error(target_arr, rsrs_factors, sample_size, RsrsSide::Left);
    let app_err_right = app_error(target_arr, rsrs_factors, sample_size, RsrsSide::Right);

    (
        app_inv_err_left,
        app_inv_err_right,
        app_err_left,
        app_err_right,
    )
}

fn box_errors_id<Item: RlstScalar + RandScalar>(
    id_factor: &IdFactor<Item>,
    arr: &mut DynamicArray<Item, 2>,
) -> Errors<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let ind_r = &id_factor.ind_r;
    let far_indices = &id_factor.ind_f;

    let arr_rf = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(ind_r.clone(), far_indices.clone()),
    )
    .unwrap()
    .ext;
    let arr_fr = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(far_indices.clone(), ind_r.clone()),
    )
    .unwrap()
    .ext;

    let arr_rf = spectral_norm_estimator(arr_rf, 10).unwrap();
    let arr_fr = spectral_norm_estimator(arr_fr, 10).unwrap();

    (arr_rf, arr_fr)
}

fn box_errors_lu<Item: RlstScalar + RandScalar>(
    lu_factor: &LuFactor<Item>,
    arr: &mut DynamicArray<Item, 2>,
) -> Errors<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let ind_r = &lu_factor.ind_r;
    let ind_t = &lu_factor.ind_t;

    let arr_rt = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(ind_r.clone(), ind_t.clone()),
    )
    .unwrap()
    .ext;
    let arr_tr = <Extraction<Item> as MatrixExtraction>::new(
        arr,
        ExtInsType::Cross(ind_t.clone(), ind_r.clone()),
    )
    .unwrap()
    .ext;

    let arr_rt = spectral_norm_estimator(arr_rt, 10).unwrap();
    let arr_tr = spectral_norm_estimator(arr_tr, 10).unwrap();

    (arr_rt, arr_tr)
}

pub fn get_diag_errors<
    Item: RlstScalar + RandScalar + rlst::MatrixId + rlst::MatrixInverse + rlst::MatrixPseudoInverse,
>(
    rsrs_factors: &RsrsFactors<Item>,
    arr: &mut DynamicArray<Item, 2>,
) -> Vec<<Item as RlstScalar>::Real>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let mut_arr = Arc::new(Mutex::new(arr));
    let exact_boxes_errors = rsrs_factors
        .diag_box_factor
        .iter()
        .map(|diag_box| {
            let mut arr = mut_arr.lock().unwrap();
            let exact_diag_box = <Extraction<Item> as MatrixExtraction>::new(
                &mut arr,
                ExtInsType::Cross(diag_box.inds.clone(), diag_box.inds.clone()),
            )
            .unwrap()
            .ext;
            let mut res: DynamicArray<Item, 2> = empty_array();
            res.fill_from_resize(exact_diag_box.r() - diag_box.dbox.r());
            spectral_norm_estimator(res, 10).unwrap()
                / spectral_norm_estimator(exact_diag_box, 10).unwrap()
        })
        .collect();
    exact_boxes_errors
}

fn apply_lu_level_error<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixPseudoInverse + MatrixLu,
>(
    rsrs_factors: &RsrsFactors<Item>,
    target_arr: &mut DynamicArray<Item, 2>,
    factor_options: &FactorOptions,
    level_it: usize,
) -> LuErrors<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let target_arr = Arc::new(Mutex::new(target_arr));
    let errors: Vec<_> = rsrs_factors.lu_factors[level_it]
        .iter()
        .map(|lu_batch| {
            let batch_errors: Vec<Errors<Item>> = lu_batch
                .par_iter()
                .map(|lu_factor| {
                    let mut target_arr = target_arr.lock().unwrap();
                    match lu_factor {
                        Factor::Lu(lu_factor) => {
                            let (arr_rt, arr_tr) = box_errors_lu(lu_factor, &mut target_arr);
                            lu_factor.mul(
                                &mut target_arr,
                                factor_options,
                                &FactorType::F,
                                &RsrsSide::Left,
                            );
                            lu_factor.mul(
                                &mut target_arr,
                                factor_options,
                                &FactorType::S,
                                &RsrsSide::Right,
                            );
                            let (arr_rt_ae, arr_tr_ae) = box_errors_lu(lu_factor, &mut target_arr);
                            let rel_errs: Errors<Item> = (arr_rt_ae / arr_rt, arr_tr_ae / arr_tr);
                            rel_errs
                        }
                        Factor::Id(_id_factor) => {
                            let rel_errs: Errors<Item> = (num::Zero::zero(), num::Zero::zero());
                            rel_errs
                        }
                    }
                })
                .collect();
            batch_errors
        })
        .collect();

    let errors: Vec<Errors<Item>> = errors.into_iter().flatten().collect();

    errors
}

fn apply_id_level_error<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu,
>(
    rsrs_factors: &RsrsFactors<Item>,
    target_arr: &mut DynamicArray<Item, 2>,
    factor_options: &FactorOptions,
    level_it: usize,
) -> IdErrors<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let target_arr = Arc::new(Mutex::new(target_arr));
    let errors: Vec<Errors<Item>> = rsrs_factors.id_factors[level_it]
        .par_iter()
        .map(|id_factor| {
            let mut target_arr = target_arr.lock().unwrap();
            match id_factor {
                Factor::Lu(_lu_factor) => {
                    let rel_errs: Errors<Item> = (num::Zero::zero(), num::Zero::zero());
                    rel_errs
                }
                Factor::Id(id_factor) => {
                    let (arr_rf, arr_fr) = box_errors_id(id_factor, &mut target_arr);
                    id_factor.mul(
                        &mut target_arr,
                        factor_options,
                        &FactorType::F,
                        &RsrsSide::Left,
                    );
                    id_factor.mul(
                        &mut target_arr,
                        factor_options,
                        &FactorType::S,
                        &RsrsSide::Right,
                    );
                    let (arr_rf_ae, arr_fr_ae) = box_errors_id(id_factor, &mut target_arr);
                    let rel_errs: Errors<Item> = (arr_rf_ae / arr_rf, arr_fr_ae / arr_fr);
                    rel_errs
                }
            }
        })
        .collect();

    errors
}

type ErrorStats<T> = (Real<T>, Real<T>, Real<T>, Real<T>);

fn el_factors_inv_mul_errors<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu,
>(
    rsrs_factors: &RsrsFactors<Item>,
    target_arr: &mut DynamicArray<Item, 2>,
) -> (Vec<ErrorStats<Item>>, Vec<ErrorStats<Item>>)
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let factor_options = FactorOptions {
        inv: true,
        trans: false,
    };
    let errors: Vec<(IdErrors<Item>, LuErrors<Item>)> = (0..rsrs_factors.num_levels)
        .map(|level_it| {
            let id_errors =
                apply_id_level_error(rsrs_factors, target_arr, &factor_options, level_it);
            let lu_errors =
                apply_lu_level_error(rsrs_factors, target_arr, &factor_options, level_it);
            (id_errors, lu_errors)
        })
        .collect();

    let stats = |errors_vec: Vec<Errors<Item>>| {
        let mut mu_1: Real<Item> = NumCast::from(0.0).unwrap();
        let mut mu_2: Real<Item> = NumCast::from(0.0).unwrap();
        let mut std_dev_1: Real<Item> = NumCast::from(0.0).unwrap();
        let mut std_dev_2: Real<Item> = NumCast::from(0.0).unwrap();

        errors_vec.iter().for_each(|(errors_1, errors_2)| {
            mu_1 += *errors_1;
            mu_2 += *errors_2;
        });

        mu_1 /= NumCast::from(errors_vec.len()).unwrap();
        mu_2 /= NumCast::from(errors_vec.len()).unwrap();

        errors_vec.iter().for_each(|(errors_1, errors_2)| {
            std_dev_1 += (*errors_1 - mu_1).powi(2);
            std_dev_2 += (*errors_2 - mu_2).powi(2);
        });

        std_dev_1 /= NumCast::from(errors_vec.len()).unwrap();
        std_dev_2 /= NumCast::from(errors_vec.len()).unwrap();

        (mu_1, mu_2, std_dev_1.sqrt(), std_dev_2.sqrt())
    };

    let mut id_stats = Vec::new();
    let mut lu_stats = Vec::new();
    errors
        .iter()
        .for_each(|(id_level_errors, lu_level_errors)| {
            if !id_level_errors.is_empty() {
                id_stats.push(stats(id_level_errors.to_vec()));
                lu_stats.push(stats(lu_level_errors.to_vec()));
            }
        });

    (id_stats, lu_stats)
}

fn get_boxes_errors<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixPseudoInverse + MatrixId + MatrixLu,
>(
    kernel_mat: &mut DynamicArray<Item, 2>,
    rsrs_factors: &mut RsrsFactors<Item>,
    tol: Real<Item>,
) where
    Real<Item>: for<'a> std::iter::Sum<&'a Real<Item>>,
    StandardNormal: Distribution<Real<Item>>,
    Standard: Distribution<Real<Item>>,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
{
    let (id_error_stats, lu_error_stats) = &el_factors_inv_mul_errors(rsrs_factors, kernel_mat);

    id_error_stats
        .iter()
        .enumerate()
        .for_each(|(level, stats)| {
            let (mu_1, mu_2, std_dev_1, std_dev_2) = stats;
            println!(
                "Errors ID, level {} : ({} +/- {}, {} +/- {})",
                level, mu_1, std_dev_1, mu_2, std_dev_2
            );
        });

    lu_error_stats
        .iter()
        .enumerate()
        .for_each(|(level, stats)| {
            let (mu_1, mu_2, std_dev_1, std_dev_2) = stats;
            println!(
                "Errors LU, level {} : ({} +/- {}, {} +/- {})",
                level, mu_1, std_dev_1, mu_2, std_dev_2
            );
            assert!(*mu_1 <= tol && *mu_2 <= tol);
        });

    println!("\n");

    let diag_re = get_diag_errors(rsrs_factors, kernel_mat);
    let diag_re_r;
    let diag_re_s;

    if diag_re.len() > 1 {
        diag_re_r = &diag_re[0..diag_re.len() - 2];
        diag_re_s = diag_re[diag_re.len() - 1];
    } else {
        diag_re_r = &[];
        diag_re_s = diag_re[0];
    }

    let diag_re_r_sum = diag_re_r.iter().sum::<<Item as rlst::RlstScalar>::Real>();
    let len: Real<Item> = NumCast::from(diag_re_r.len()).unwrap();
    let diag_re_r_mean = diag_re_r_sum / len;

    println!(
        "Mean residual diagonal blocks errors : {}, sketch block error: {}",
        diag_re_r_mean, diag_re_s
    );

    assert!(diag_re_r_mean <= tol && diag_re_s <= tol);
}

//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.

pub trait TestFramework: RlstScalar {
    fn test_rsrs_geometry(
        geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point>,
        kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>,
        npoints: usize,
        kappa: <Self as RlstScalar>::Real,
        id_tols: &[<Self as RlstScalar>::Real],
        comm: &SimpleCommunicator,
    );
    fn run_test(
        geometry: &str,
        kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>,
        npoints: &[usize],
        kappa: <Self as RlstScalar>::Real,
    );
}

//Geometry building

pub fn sphere_surface<C: CommunicatorCollectives>(
    npoints: usize,
    comm: &C,
) -> std::vec::Vec<bempp_octree::Point> {
    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(0);
    let mut points: Vec<bempp_octree::Point> = generate_random_points(npoints, &mut rng, comm);

    // Find centre points
    let x: Vec<f64> = points.iter().map(|point| point.coords()[0]).collect();
    let y: Vec<f64> = points.iter().map(|point| point.coords()[0]).collect();
    let z: Vec<f64> = points.iter().map(|point| point.coords()[0]).collect();
    let x_min = x
        .clone()
        .into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let y_min = y
        .clone()
        .into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let z_min = z
        .clone()
        .into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let x_max = x
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let y_max = y
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let z_max = z
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let centre = bempp_octree::Point::new(
        [
            (x_min + x_max) * 0.5,
            (y_min + y_max) * 0.5,
            (z_min + z_max) * 0.5,
        ],
        000,
    );

    // Make sure that the points live on the unit sphere.
    for point in points.iter_mut() {
        let mut aux: [f64; 3] = [
            point.coords()[0] - centre.coords()[0],
            point.coords()[1] - centre.coords()[1],
            point.coords()[2] - centre.coords()[2],
        ];
        let len: f64 = (aux[0] * aux[0] + aux[1] * aux[1] + aux[2] * aux[2]).sqrt();
        aux[0] /= len;
        aux[1] /= len;
        aux[2] /= len;
        point.coords_mut()[0] = aux[0] + centre.coords()[0];
        point.coords_mut()[1] = aux[1] + centre.coords()[1];
        point.coords_mut()[2] = aux[2] + centre.coords()[2];
    }

    points
}

//Matrix building

fn laplace_kernel(dist: f64, npoints: usize) -> f64 {
    let pi = std::f64::consts::PI;
    let n: f64 = num::NumCast::from(npoints).unwrap();
    1.0 / (4.0 * pi * n * dist)
}

fn get_laplace_matrix(points_x: &[bempp_octree::Point]) -> DynamicArray<f64, 2> {
    let n: usize = points_x.len();
    let mut arr: DynamicArray<f64, 2> = rlst_dynamic_array2!(f64, [n, n]);
    let mut view = arr.r_mut();
    for (i, point_x) in points_x.iter().enumerate() {
        for (j, point_y) in points_x.iter().enumerate() {
            let coords_x: [f64; 3] = point_x.coords();
            let coords_y: [f64; 3] = point_y.coords();
            let dist: <f64 as RlstScalar>::Real = num::NumCast::from(
                ((coords_x[0] - coords_y[0]).powi(2)
                    + (coords_x[1] - coords_y[1]).powi(2)
                    + (coords_x[2] - coords_y[2]).powi(2))
                .sqrt(),
            )
            .unwrap();
            if dist > 0.0 {
                view[[i, j]] = laplace_kernel(dist, n);
            } else {
                //If points are equal, set the value to 1
                view[[i, j]] = 1.0.into();
            }
        }
    }
    arr
}

pub fn main() {
    //Error testing
    let universe: mpi::environment::Universe = mpi::initialize().unwrap();
    let comm: SimpleCommunicator = universe.world();
    let max_level: usize = 16;
    let max_leaf_points: usize = 50;

    let id_tols = [1e-2]; //[1e-2, 1e-4];
    let npoints_vec = [5000];

    for npts in npoints_vec {
        for &id_tol in id_tols.iter() {
            let points: Vec<bempp_octree::Point> = sphere_surface(npts, &comm);
            let tree: Octree<'_, SimpleCommunicator> =
                Octree::new(&points, max_level, max_leaf_points, &comm);
            println!("Test: {} points, tol:{}", npts, id_tol);
            let tols: Tols<f64> = Tols {
                id: id_tol,
                null: num::Zero::zero(),
                lstq: num::Zero::zero(),
            };
            let mut kernel_mat: DynamicArray<f64, 2> = get_laplace_matrix(&points);
            let mut rsrs_algo: RsrsData<f64> =
                <RsrsData<f64> as Rsrs>::new(&kernel_mat, tols, &tree, true);

            let options = RsrsOptions {
                oversampling: 8,
                adaptive_tol: true,
                initial_num_samples: 400,
            };

            let mut rsrs_factors =
                rsrs_algo.tree_cycle_and_diag_block_extraction(&kernel_mat, &options);

            let mul_errors = rsrs_error_estimator(&kernel_mat, &mut rsrs_factors, 10);

            println!("Multiplication errors: {:?}\n", mul_errors);

            assert!(
                mul_errors.0 <= id_tol
                    && mul_errors.1 <= id_tol
                    && mul_errors.2 <= id_tol
                    && mul_errors.3 <= id_tol
            );

            get_boxes_errors(&mut kernel_mat, &mut rsrs_factors, id_tol);
        }
    }
}
