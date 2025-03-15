use bempp_rsrs::{rsrs::{box_skeletonisation::Tols, rsrs_cycle::{Rsrs, RsrsData, RsrsOptions, Termination}, rsrs_factors::{RsrsFactors, RsrsFactorsOps}}, utils::{geometries::{cube_surface, sphere_surface}, low_rank_matrices::KernelMatrix}};
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use std::{error::Error, fs};
use bempp_octree::Octree;
use std::io::BufWriter;
use rlst::prelude::*;
use std::path::Path;
use std::io::Write;
use std::fs::File;
use num::NumCast;

fn write_vec_to_new_file_u128(path: impl AsRef<Path>, value: &[u128]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    for x in value {
        writeln!(writer, "{x}")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_vec_to_new_file<Item: RlstScalar>(path: impl AsRef<Path>, value: &[Item]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    for x in value {
        writeln!(writer, "{x}")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_multi_vec_to_new_file<Item: RlstScalar>(path: impl AsRef<Path>, values: &[Vec<Item>]){
    for (id, val) in values.iter().enumerate(){
        let mut new_path = path.as_ref().to_str().unwrap().to_string();
        new_path.push_str(&id.to_string());
        new_path.push_str(".json");
        let _ = write_vec_to_new_file(new_path,val);
    }
}

type Real<T> = <T as rlst::RlstScalar>::Real;

fn save_stats<Item: RlstScalar>(rsrs_data: &RsrsData<Item>, tol: Real<Item>, path_str: &str){
    let string_tol = format!("{:e}", tol);

    let mut times_path = path_str.to_string();
    times_path.push_str("/times_");
    times_path.push_str(&string_tol);
    times_path.push('/');

    fs::create_dir_all(Path::new(&times_path)).unwrap();

    let mut sampling_path = times_path.clone();
    sampling_path.push_str("sampling.json");

    let mut nullification_path = times_path.clone();
    nullification_path.push_str("nullification.json");

    let mut id_time_path = times_path.clone();
    id_time_path.push_str("id.json");

    let mut lu_time_path = times_path.clone();
    lu_time_path.push_str("lu.json");

    let mut update_id_time_path = times_path.clone();
    update_id_time_path.push_str("update_id.json");

    let mut update_lu_time_path = times_path.clone();
    update_lu_time_path.push_str("update_lu.json");

    let mut mixed_path = times_path.clone();
    mixed_path.push_str("mixed.json");

    let mixed_res = [rsrs_data.stats.total_elapsed_time as u128, rsrs_data.stats.extraction_time as u128, rsrs_data.stats.residual_size as u128];    

    /*let _ = write_vec_to_new_file_u128(sampling_path, &rsrs_data.stats.sampling_time);
    let _ = write_vec_to_new_file_u128(nullification_path, &rsrs_data.stats.nullification_time);
    let _ = write_vec_to_new_file_u128(id_time_path, &rsrs_data.stats.id_time);
    let _ = write_vec_to_new_file_u128(lu_time_path, &rsrs_data.stats.lu_time);
    let _ = write_vec_to_new_file_u128(update_id_time_path, &rsrs_data.stats.update_id_time);
    let _ = write_vec_to_new_file_u128(update_lu_time_path, &rsrs_data.stats.update_lu_time);
    let _ = write_vec_to_new_file_u128(mixed_path, &mixed_res);*/
    

}

fn get_box_errors<Item: RlstScalar + MatrixInverse + MatrixPseudoInverse + MatrixId>(kernel_mat: &mut DynamicArray<Item, 2>, rsrs_factors: &RsrsFactors<Item>, tol: Real<Item>, path_str: &str)->(Real<Item>, Real<Item>, Real<Item>)
where Real<Item>: for<'a> std::iter::Sum<&'a Real<Item>>
{
    let npoints = kernel_mat.shape()[0];
    let (rel_errs, abs_errs) = rsrs_factors.get_boxes_errors(kernel_mat, true);
    let diag_ae = rsrs_factors.get_diag_errors(kernel_mat);

    let string_tol = format!("{:e}", tol);
    let mut rel_errors_path = path_str.to_string();
    rel_errors_path.push_str("/relative_errors_");
    rel_errors_path.push_str(&string_tol);
    rel_errors_path.push('/');

    fs::create_dir_all(Path::new(&rel_errors_path)).unwrap();

    let mut abs_errors_path =  path_str.to_string();
    abs_errors_path.push_str("/absolute_errors_");
    abs_errors_path.push_str(&string_tol);
    abs_errors_path.push('/');

    fs::create_dir_all(Path::new(&abs_errors_path)).unwrap();

    write_multi_vec_to_new_file(&rel_errors_path, &rel_errs);
    write_multi_vec_to_new_file(&abs_errors_path, &abs_errs);

    let diag_ae_r;
    let diag_ae_s;
    
    if diag_ae.len() >1{
        diag_ae_r = &diag_ae[0..diag_ae.len()-2];
        diag_ae_s = diag_ae[diag_ae.len()-1];
    }
    else{
        diag_ae_r = &[];
        diag_ae_s = diag_ae[0];
    }

    let diag_ae_r_sum = diag_ae_r.iter().sum::<<Item as rlst::RlstScalar>::Real>();
    let len: Real<Item> = NumCast::from(diag_ae_r.len()).unwrap();
    let diag_ae_r_mean = diag_ae_r_sum/len;

    let mut ident = rlst_dynamic_array2!(Item, [npoints, npoints]);
    ident.set_identity();

    let zero = ident - kernel_mat.view();

    (zero.view().norm_1(), diag_ae_r_mean, diag_ae_s)

}
//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.
pub trait TestFramework: RlstScalar {
    fn test_rsrs_geometry(geometry: &str, kernel: &str, geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point> , kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real)-> DynamicArray<Self, 2>, npoints: usize, kappa: <Self as RlstScalar>::Real, id_tols: &[<Self as RlstScalar>::Real], comm: &SimpleCommunicator);
    fn run_test(geometry: &str, kernel: &str, kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>, npoints: &[usize], kappa: <Self as RlstScalar>::Real);
}


macro_rules! implement_test_framework{
    ($scalar:ty) => {
            impl TestFramework for $scalar {
                fn test_rsrs_geometry(geometry: &str, kernel: &str, geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point> , kernel_fn: fn(&[bempp_octree::Point], <$scalar as RlstScalar>::Real)-> DynamicArray<$scalar, 2>, npoints: usize, kappa: <$scalar as RlstScalar>::Real, id_tols: &[<$scalar as RlstScalar>::Real], comm: &SimpleCommunicator)
                {
                    let points: Vec<bempp_octree::Point> = geometry_fn(npoints, &comm);
                    let max_level: usize = 16;
                    let max_leaf_points: usize = 50;
                    let tree: Octree<'_, SimpleCommunicator>  = Octree::new(&points, max_level, max_leaf_points, &comm);
                    let global_number_of_points: usize = tree.global_number_of_points();
                    let global_max_level: usize = tree.global_max_level();
                    if comm.rank() == 0 {
                        println!(
                            "Setup octree with {} points and maximum level {}",
                            global_number_of_points,
                            global_max_level
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
                    for &id_tol in id_tols.iter(){
                        println!("Test: {} points, tol:{}", npoints, id_tol);
                        let tols : Tols<$scalar> = Tols{id: id_tol, null: num::Zero::zero(), lstq: num::Zero::zero()};
                        let mut kernel_mat: DynamicArray<$scalar, 2> = kernel_fn(&points, kappa);
                        let mut rsrs_algo: RsrsData<$scalar> = <RsrsData<$scalar> as Rsrs>::new(&kernel_mat, tols, &tree);
                        let options = RsrsOptions{ hermitian: false, silent: true, split: true, termination: Termination::ReachRoot, oversampling: 5};
                        let rsrs_factors = rsrs_algo.tree_cycle_and_diag_block_extraction(&kernel_mat, options);
                        save_stats(&rsrs_algo, id_tol, &path_str);
                        let (norm_app_inv, diag_ae_mean, skel_ae) = get_box_errors(&mut kernel_mat, &rsrs_factors,  id_tol, &path_str);
                        app_inv.push(norm_app_inv);
                        if !diag_ae_mean.is_nan(){
                            diag_errs.push(diag_ae_mean);
                        }
                        skel_errs.push(skel_ae);
                        tot_num_samples.push(rsrs_algo.y_data.num_samples as f64);
                    }

                    let mut app_id_path = path_str.clone();
                    let mut diag_errs_path = path_str.clone();
                    let mut skel_errs_path = path_str.clone();
                    let mut num_samples_path = path_str.clone();

                    app_id_path.push_str("/app_id.json");
                    diag_errs_path.push_str("/diag_errs.json");
                    skel_errs_path.push_str("/skel_errs.json");
                    num_samples_path.push_str("/num_samples.json");

                    let _= write_vec_to_new_file(&app_id_path, &app_inv);
                    let _= write_vec_to_new_file(&diag_errs_path, &diag_errs);
                    let _= write_vec_to_new_file(&skel_errs_path, &skel_errs);
                    let _= write_vec_to_new_file(&num_samples_path, &tot_num_samples);

                }
            
                fn run_test(geometry: &str, kernel: &str, kernel_fn: fn(&[bempp_octree::Point], <Self as RlstScalar>::Real) -> DynamicArray<Self, 2>, npoints: &[usize], kappa:<Self as RlstScalar>::Real){
                    let universe: mpi::environment::Universe = mpi::initialize().unwrap();
                    let comm: SimpleCommunicator = universe.world();
                    for &n in npoints{
                        let id_tols = [1e-2, 1e-4];//, 1e-6, 1e-8];
                        let mut geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point> = sphere_surface;
                        if geometry == "cube"{
                            geometry_fn = cube_surface;
                        }
                        Self::test_rsrs_geometry(geometry, kernel, geometry_fn, kernel_fn, n, kappa, &id_tols, &comm);
                    }
                }
            }
    }
}

implement_test_framework!(f64);
implement_test_framework!(c64);


pub fn main() {
    let geometry = "sphere";
    let kernel = "helmholtz";
    let npoints = [2000];//[500, 1000, 3000, 5000, 10000, 20000];

    
    if kernel == "standard_real"{
        <f64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_exp_real_kernel_matrix, &npoints, 0.0);
    }
    else if kernel == "standard_complex"{
        <c64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_exp_complex_kernel_matrix, &npoints, 0.0);
    }
    else if kernel == "laplace"{
        <f64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_laplace_matrix, &npoints, 0.0);
    }
    else{
        let pi =std::f64::consts::PI;
        <c64 as TestFramework>::run_test(geometry, kernel, KernelMatrix::get_helmholtz_matrix, &npoints, pi);
    }
    
}
