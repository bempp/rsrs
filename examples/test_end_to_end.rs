use bempp_rsrs::{rsrs::{rsrs_cycle::{Rsrs, RsrsData, RsrsOptions}, rsrs_factors::{RsrsFactors, RsrsFactorsOps}}, utils::{geometries::{cube_surface, sphere_surface}, low_rank_matrices::get_matrix}};
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use std::{error::Error, fs};
use bempp_octree::Octree;
use std::io::BufWriter;
use rlst::prelude::*;
use std::path::Path;
use std::io::Write;
use std::fs::File;


fn write_vec_to_new_file(path: impl AsRef<Path>, value: &[f64]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    for x in value {
        writeln!(writer, "{x}")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_multi_vec_to_new_file(path: impl AsRef<Path>, values: &[Vec<f64>]){
    for (id, val) in values.iter().enumerate(){
        let mut new_path = path.as_ref().to_str().unwrap().to_string();
        new_path.push_str(&id.to_string());
        new_path.push_str(".json");
        let _ = write_vec_to_new_file(new_path,val);
    }
}


fn get_box_errors(kernel_mat: &mut DynamicArray<f64, 2>, rsrs_factors: &RsrsFactors<f64>, tol: f64, path_str: &str)->(f64, f64, f64){
    let npoints = kernel_mat.shape()[0];
    let (rel_errs, abs_errs) = rsrs_factors.get_boxes_errors(kernel_mat, true);
    let diag_ae = rsrs_factors.get_diag_errors(kernel_mat);
    println!("{:?}", diag_ae);

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


    let diag_ae_r = &diag_ae[0..diag_ae.len()-2];
    let diag_ae_s = diag_ae[diag_ae.len()-1];

    let diag_ae_r_sum = diag_ae_r.iter().sum::<f64>();
    let diag_ae_r_mean = diag_ae_r_sum/ diag_ae_r.len() as f64;

    let mut ident = rlst_dynamic_array2!(f64, [npoints, npoints]);
    ident.set_identity();

    let zero = ident - kernel_mat.view();

    (zero.view().norm_1(), diag_ae_r_mean, diag_ae_s)

}
//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.

fn test_rsrs_geometry(geometry: &str, npoints: usize){

    let mut geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point> = sphere_surface;
    
    if geometry == "cube"{
        geometry_fn = cube_surface;
    }

    let universe: mpi::environment::Universe = mpi::initialize().unwrap();
    let comm: SimpleCommunicator = universe.world();
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

    let id_tols = [1e-3];//, 5e-3, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14];

    let mut app_inv = Vec::new();
    let mut diag_errs = Vec::new();
    let mut skel_errs = Vec::new();
    let mut tot_num_samples = Vec::new();

    let mut path_str = "examples/results/".to_string();
    let mut geometry_and_points = geometry.to_string();
    geometry_and_points.push('_');
    geometry_and_points.push_str(&npoints.to_string());
    path_str.push_str(&geometry_and_points);
    
    for id_tol in id_tols{
        let tols : bempp_rsrs::rsrs::box_skeletonisation::Tols<f64> = bempp_rsrs::rsrs::box_skeletonisation::Tols{id: id_tol, null: 1e-15, lstq: 1e-15};
        let mut kernel_mat: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> = get_matrix(&points);
        let mut rsrs_algo: RsrsData<f64> = <RsrsData<f64> as Rsrs>::new(&kernel_mat, tols, &tree);
        let options = RsrsOptions{ hermitian: false, silent: true };
        let rsrs_factors = rsrs_algo.tree_cycle_and_diag_block_extraction(&kernel_mat, options);
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
pub fn main() {

    let geometry = "sphere";
    let npoints = 3000;

    test_rsrs_geometry(geometry, npoints);
    
}
