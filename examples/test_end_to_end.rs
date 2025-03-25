use bempp_octree::Octree;
use bempp_rsrs::{
    rsrs::{
        box_skeletonisation::Tols,
        rsrs_cycle::{Rsrs, RsrsData, RsrsOptions, Termination},
        rsrs_factors::{
            DecFactorOpType, DiagBoxOperations, FactorOptions, FactorType, IdFactor,
            IdFactorOperations, LuFactor, LuFactorOperations, RsrsFactors,
        },
    },
    utils::{
        data_ins_ext::{ExtInsType, Extraction, MatrixExtraction},
        geometries::{cube_surface, sphere_surface},
        low_rank_matrices::KernelMatrix,
        norm_estimator::spectral_norm_estimator,
    },
};
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use num::NumCast;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rlst::{dense::tools::RandScalar, prelude::*};
use std::{
    env,
    sync::{Arc, Mutex},
};

type Real<T> = <T as rlst::RlstScalar>::Real;

type Errors<T> = (Real<T>, Real<T>);
type RelAbsErrors<T> = (Errors<T>, Errors<T>);
type LuErrors<T> = Vec<RelAbsErrors<T>>;
type IdErrors<T> = Vec<RelAbsErrors<T>>;

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
            res.fill_from_resize(exact_diag_box - diag_box.dbox.view());
            spectral_norm_estimator(res, 10).unwrap()
        })
        .collect();
    exact_boxes_errors
}

fn apply_lu_level_error<Item: RlstScalar + RandScalar + MatrixInverse + MatrixPseudoInverse>(
    rsrs_factors: &RsrsFactors<Item>,
    target_arr: &mut DynamicArray<Item, 2>,
    factor_options: &FactorOptions,
    level_it: usize,
) -> LuErrors<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let target_arr = Arc::new(Mutex::new(target_arr));
    let errors: Vec<_> = rsrs_factors.lu_factors[level_it]
        .iter()
        .map(|lu_batch| {
            let batch_errors: Vec<RelAbsErrors<Item>> = lu_batch
                .par_iter()
                .map(|lu_factor| {
                    let mut target_arr = target_arr.lock().unwrap();
                    let (arr_rt, arr_tr) = box_errors_lu(lu_factor, &mut target_arr);
                    lu_factor.mul(
                        &mut target_arr,
                        factor_options,
                        &FactorType::F,
                        &DecFactorOpType::Left,
                    );
                    lu_factor.mul(
                        &mut target_arr,
                        factor_options,
                        &FactorType::S,
                        &DecFactorOpType::Right,
                    );
                    let (arr_rt_ae, arr_tr_ae) = box_errors_lu(lu_factor, &mut target_arr);
                    let rel_errs: Errors<Item> = (arr_rt_ae / arr_rt, arr_tr_ae / arr_tr);
                    let abs_errs: Errors<Item> = (arr_rt_ae, arr_tr_ae);

                    println!("rel_errs lu, {:?}", rel_errs);

                    (rel_errs, abs_errs)
                })
                .collect();
            batch_errors
        })
        .collect();

    let errors: Vec<RelAbsErrors<Item>> = errors.into_iter().flatten().collect();

    errors
}

fn apply_id_level_error<Item: RlstScalar + RandScalar + MatrixInverse + MatrixId>(
    rsrs_factors: &RsrsFactors<Item>,
    target_arr: &mut DynamicArray<Item, 2>,
    factor_options: &FactorOptions,
    level_it: usize,
    blas_cores: &usize,
) -> IdErrors<Item>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let target_arr = Arc::new(Mutex::new(target_arr));
    let errors: Vec<RelAbsErrors<Item>> = rsrs_factors.id_factors[level_it]
        .par_iter()
        .map(|id_factor| {
            let mut target_arr = target_arr.lock().unwrap();
            let (arr_rf, arr_fr) = box_errors_id(id_factor, &mut target_arr);
            id_factor.mul(
                &mut target_arr,
                factor_options,
                &FactorType::F,
                &DecFactorOpType::Left,
            );
            id_factor.mul(
                &mut target_arr,
                factor_options,
                &FactorType::S,
                &DecFactorOpType::Right,
            );
            let (arr_rf_ae, arr_fr_ae) = box_errors_id(id_factor, &mut target_arr);
            let rel_errs: Errors<Item> = (arr_rf_ae / arr_rf, arr_fr_ae / arr_fr);
            let abs_errs: Errors<Item> = (arr_rf_ae, arr_fr_ae);
            println!("rel_errs id, {:?}", rel_errs);
            (rel_errs, abs_errs)
        })
        .collect();

    errors
}

fn el_factors_inv_mul_errors<
    Item: RlstScalar + RandScalar + MatrixInverse + MatrixId + MatrixPseudoInverse,
>(
    rsrs_factors: &RsrsFactors<Item>,
    target_arr: &mut DynamicArray<Item, 2>,
    blas_cores: &usize,
) -> Vec<(IdErrors<Item>, LuErrors<Item>)>
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let factor_options = FactorOptions {
        inv: true,
        trans: false,
    };
    let errors: Vec<(IdErrors<Item>, LuErrors<Item>)> = (0..rsrs_factors.num_levels)
        .map(|level_it| {
            let id_errors = apply_id_level_error(
                rsrs_factors,
                target_arr,
                &factor_options,
                level_it,
                blas_cores,
            );
            let lu_errors =
                apply_lu_level_error(rsrs_factors, target_arr, &factor_options, level_it);
            (id_errors, lu_errors)
        })
        .collect();

    errors
}

fn get_box_errors<Item: RlstScalar + RandScalar + MatrixInverse + MatrixPseudoInverse + MatrixId>(
    kernel_mat: &mut DynamicArray<Item, 2>,
    rsrs_factors: &mut RsrsFactors<Item>,
    blas_cores: &usize,
    _tol: Real<Item>,
    _path_str: &str,
) -> (Real<Item>, Real<Item>, Real<Item>)
where
    Real<Item>: for<'a> std::iter::Sum<&'a Real<Item>>,
    StandardNormal: Distribution<Real<Item>>,
    Standard: Distribution<Real<Item>>,
{
    let npoints = kernel_mat.shape()[0];
    let _errors = &el_factors_inv_mul_errors(rsrs_factors, kernel_mat, blas_cores);

    let diag_ae = get_diag_errors(rsrs_factors, kernel_mat);
    let diag_ae_r;
    let diag_ae_s;

    if diag_ae.len() > 1 {
        diag_ae_r = &diag_ae[0..diag_ae.len() - 2];
        diag_ae_s = diag_ae[diag_ae.len() - 1];
    } else {
        diag_ae_r = &[];
        diag_ae_s = diag_ae[0];
    }

    let diag_ae_r_sum = diag_ae_r.iter().sum::<<Item as rlst::RlstScalar>::Real>();
    let len: Real<Item> = NumCast::from(diag_ae_r.len()).unwrap();
    let diag_ae_r_mean = diag_ae_r_sum / len;

    let mut ident = rlst_dynamic_array2!(Item, [npoints, npoints]);
    ident.set_identity();

    let factor_options = FactorOptions {
        inv: true,
        trans: false,
    };

    rsrs_factors
        .diag_box_factor
        .left_mul(kernel_mat, &factor_options, blas_cores);

    let zero = ident - kernel_mat.view();
    let mut res = empty_array();
    res.fill_from_resize(zero.view());

    let norm = spectral_norm_estimator(res, 10).unwrap();

    println!("Norm: {}, Mean: {}, S: {}", norm, diag_ae_r_mean, diag_ae_s);

    (norm, diag_ae_r_mean, diag_ae_s)
}
//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.
pub trait TestFramework: RlstScalar {
    fn test_rsrs_geometry(
        geometry: &str,
        kernel: &str,
        geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point>,
        kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>,
        npoints: usize,
        kappa: <Self as RlstScalar>::Real,
        id_tols: &[<Self as RlstScalar>::Real],
        comm: &SimpleCommunicator,
    );
    fn run_test(
        geometry: &str,
        kernel: &str,
        kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>,
        npoints: &[usize],
        kappa: <Self as RlstScalar>::Real,
    );
}

macro_rules! implement_test_framework {
    ($scalar:ty) => {
        impl TestFramework for $scalar {
            fn test_rsrs_geometry(
                geometry: &str,
                kernel: &str,
                geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point>,
                kernel_fn: fn(
                    &[bempp_octree::Point],
                    <$scalar as RlstScalar>::Real,
                ) -> DynamicArray<$scalar, 2>,
                npoints: usize,
                kappa: <$scalar as RlstScalar>::Real,
                id_tols: &[<$scalar as RlstScalar>::Real],
                comm: &SimpleCommunicator,
            ) {
                let points: Vec<bempp_octree::Point> = geometry_fn(npoints, &comm);
                let max_level: usize = 16;
                let max_leaf_points: usize = 50;
                let tree: Octree<'_, SimpleCommunicator> =
                    Octree::new(&points, max_level, max_leaf_points, &comm);
                let global_number_of_points: usize = tree.global_number_of_points();
                let global_max_level: usize = tree.global_max_level();
                if comm.rank() == 0 {
                    println!(
                        "Setup octree with {} points and maximum level {}",
                        global_number_of_points, global_max_level
                    );
                }
                let mut app_inv = Vec::new();
                let mut diag_errs = Vec::new();
                let mut skel_errs = Vec::new();
                let mut tot_num_samples = Vec::new();
                let mut path_str = "examples/results/".to_string();
                let mut geometry_and_points = geometry.to_string();
                geometry_and_points.push('_');
                geometry_and_points.push_str(kernel);
                geometry_and_points.push('_');
                geometry_and_points.push_str(&npoints.to_string());
                path_str.push_str(&geometry_and_points);
                for &id_tol in id_tols.iter() {
                    println!("Test: {} points, tol:{}", npoints, id_tol);
                    let tols: Tols<$scalar> = Tols {
                        id: id_tol,
                        id_2: id_tol * 10.0,
                        null: num::Zero::zero(),
                        lstq: num::Zero::zero(),
                    };
                    let mut kernel_mat: DynamicArray<$scalar, 2> = kernel_fn(&points, kappa);
                    let mut rsrs_algo: RsrsData<$scalar> =
                        <RsrsData<$scalar> as Rsrs>::new(&kernel_mat, tols, &tree);
                    let options = RsrsOptions {
                        hermitian: true,
                        silent: true,
                        split: true,
                        termination: Termination::ReachRoot,
                        oversampling: 5,
                        adaptive_tol: true,
                        blas_cores: 3,
                    };
                    let mut rsrs_factors =
                        rsrs_algo.tree_cycle_and_diag_block_extraction(&kernel_mat, &options);

                    let (norm_app_inv, diag_ae_mean, skel_ae) = get_box_errors(
                        &mut kernel_mat,
                        &mut rsrs_factors,
                        &options.blas_cores,
                        id_tol,
                        &path_str,
                    );
                    app_inv.push(norm_app_inv);
                    if !diag_ae_mean.is_nan() {
                        diag_errs.push(diag_ae_mean);
                    }
                    skel_errs.push(skel_ae);
                    tot_num_samples.push(rsrs_algo.y_data.num_samples as f64);
                }
            }

            fn run_test(
                geometry: &str,
                kernel: &str,
                kernel_fn: fn(
                    &[bempp_octree::Point],
                    <Self as RlstScalar>::Real,
                ) -> DynamicArray<Self, 2>,
                npoints: &[usize],
                kappa: <Self as RlstScalar>::Real,
            ) {
                let universe: mpi::environment::Universe = mpi::initialize().unwrap();
                let comm: SimpleCommunicator = universe.world();
                for &n in npoints {
                    let id_tols = [1e-6]; //, 1e-6, 1e-8];
                    let mut geometry_fn: fn(
                        usize,
                        &SimpleCommunicator,
                    ) -> Vec<bempp_octree::Point> = sphere_surface;
                    if geometry == "cube" {
                        geometry_fn = cube_surface;
                    }
                    Self::test_rsrs_geometry(
                        geometry,
                        kernel,
                        geometry_fn,
                        kernel_fn,
                        n,
                        kappa,
                        &id_tols,
                        &comm,
                    );
                }
            }
        }
    };
}

implement_test_framework!(f64);
implement_test_framework!(c64);

pub fn main() {
    let geometry = "sphere";
    let kernel = "laplace";
    let npoints = [1000]; //[500, 1000, 3000, 5000, 10000, 20000];
    env::set_var("OPENBLAS_NUM_THREADS", "1");
    if kernel == "standard_real" {
        <f64 as TestFramework>::run_test(
            geometry,
            kernel,
            KernelMatrix::get_exp_real_kernel_matrix,
            &npoints,
            0.0,
        );
    } else if kernel == "standard_complex" {
        <c64 as TestFramework>::run_test(
            geometry,
            kernel,
            KernelMatrix::get_exp_complex_kernel_matrix,
            &npoints,
            0.0,
        );
    } else if kernel == "laplace" {
        <f64 as TestFramework>::run_test(
            geometry,
            kernel,
            KernelMatrix::get_laplace_matrix,
            &npoints,
            0.0,
        );
    } else {
        let pi = std::f64::consts::PI;
        <c64 as TestFramework>::run_test(
            geometry,
            kernel,
            KernelMatrix::get_helmholtz_matrix,
            &npoints,
            pi,
        );
    }
}
