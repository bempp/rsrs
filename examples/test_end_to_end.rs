use bempp_rsrs::{rsrs::{box_skeletonisation::Tols, rsrs_cycle::{Rsrs, RsrsData}}, utils::{data_ins_ext::{matrix_insertion, solve_left, ExtInsType, Extraction, MatrixExtraction}, elementary_matrix::{ElMatOptions, ElementaryOperations}, geometries::{cube_surface, sphere_surface}, low_rank_matrices::get_matrix}};
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use bempp_octree::Octree;
use rlst::prelude::*;


use std::{error::Error, fs};
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;


fn write_vec_to_new_file(path: impl AsRef<Path>, value: &[f64]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    for x in value {
        writeln!(writer, "{x}")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_multi_vec_to_new_file(path: impl AsRef<Path>, values: &Vec<Vec<f64>>){
    for (id, val) in values.iter().enumerate(){
        let mut new_path = path.as_ref().to_str().unwrap().to_string();
        new_path.push_str(&id.to_string());
        new_path.push_str(".json");
        let _ = write_vec_to_new_file(new_path,val);
    }
}

fn get_far_indices(n: usize, near_indices: Vec<usize>)->Vec<usize>{
    let mut domain: Vec<usize> = (0..n).collect();
    domain.retain(|x| !near_indices.contains(x));
    domain
}

fn compute_block_error(arr: &mut DynamicArray<f64, 2>, ind_r: &Vec<usize>, ind_t: &Vec<usize>, far_indices: &Vec<usize>)-> (f64, f64, f64, f64){
    let arr_rf: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_r.clone(), far_indices.clone())).unwrap().ext;
    let arr_fr: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(far_indices.clone(), ind_r.clone())).unwrap().ext;
    let arr_rt: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_r.clone(), ind_t.clone())).unwrap().ext;
    let arr_tr: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_t.clone(), ind_r.clone())).unwrap().ext;

    let arr_rf = arr_rf.norm_2_alloc().unwrap();
    let arr_fr = arr_fr.norm_2_alloc().unwrap();
    let arr_rt = arr_rt.norm_2_alloc().unwrap();
    let arr_tr = arr_tr.norm_2_alloc().unwrap();

    (arr_rf, arr_fr, arr_rt, arr_tr)
}

fn get_box_errors(arr: &mut DynamicArray<f64, 2>, rsrs_data: &mut RsrsData<f64>, npoints: usize, tol: f64, path_str: &String)-> (f64, f64, f64, usize){
    let mut rel_errs: Vec<Vec<f64>> = Vec::new();
    let mut abs_errs: Vec<Vec<f64>> = Vec::new();

    rel_errs.resize(4, Vec::new());
    abs_errs.resize(4, Vec::new());

    let decoupled = &rsrs_data.dec_boxes;

    for (box_id, d_box) in decoupled.iter().enumerate(){
        println!("Computing errors... box {} of {}, level iteration {}\n", box_id, decoupled.len(), d_box.level_it);

        let ind_r = &d_box.box_features.ind_r;
        let ind_t = &d_box.box_features.ind_t;
        let ind_s = &d_box.box_features.ind_s;
        let far_indices = get_far_indices(arr.shape()[0], d_box.box_features.near_field_inds.clone());

        println!("Relevant indices sizes: residual {}, sketch {}, near field {}, far field {}", ind_r.len(), ind_s.len(), d_box.box_features.near_field_inds.len(), far_indices.len());

        let (arr_rf, arr_fr, arr_rt, arr_tr) = compute_block_error(arr, ind_r, ind_t, &far_indices);

        d_box.operators.fact_id.left.mul(arr, ElMatOptions{inv: true, trans: false, left: true});
        d_box.operators.fact_id.right.mul(arr, ElMatOptions{inv: true, trans: false, left: false});
        d_box.operators.fact_lu.left.mul(arr, ElMatOptions{inv: true, trans: false, left: true});
        d_box.operators.fact_lu.right.mul(arr, ElMatOptions{inv: true, trans: false, left: false});

        let (arr_rf_ae, arr_fr_ae, arr_rt_ae, arr_tr_ae) = compute_block_error(arr, ind_r, ind_t, &far_indices);

        rel_errs[0].push(arr_rf_ae/arr_rf);
        rel_errs[1].push(arr_fr_ae/arr_fr);
        rel_errs[2].push(arr_rt_ae/arr_rt);
        rel_errs[3].push(arr_tr_ae/arr_tr);

        abs_errs[0].push(arr_rf_ae);
        abs_errs[1].push(arr_fr_ae);
        abs_errs[2].push(arr_rt_ae);
        abs_errs[3].push(arr_tr_ae);

        println!("Absolute errors: {}, {}, {}, {}", arr_rf_ae, arr_fr_ae, arr_rt_ae, arr_tr_ae);
        println!("Relative errors: {}, {}, {}, {}\n\n", arr_rf_ae/arr_rf, arr_fr_ae/arr_fr, arr_rt_ae/arr_rt, arr_tr_ae/arr_tr);
    }


    let string_tol = format!("{:e}", tol);
    let mut rel_errors_path = path_str.clone();
    rel_errors_path.push_str("/relative_errors_");
    rel_errors_path.push_str(&string_tol);
    rel_errors_path.push('/');

    fs::create_dir_all(Path::new(&rel_errors_path)).unwrap();

    let mut abs_errors_path =  path_str.clone();
    abs_errors_path.push_str("/absolute_errors_");
    abs_errors_path.push_str(&string_tol);
    abs_errors_path.push('/');

    fs::create_dir_all(Path::new(&abs_errors_path)).unwrap();

    write_multi_vec_to_new_file(&rel_errors_path, &rel_errs);
    write_multi_vec_to_new_file(&abs_errors_path, &abs_errs);

    rsrs_data.p_matrix.mul(arr, ElMatOptions{inv: false, trans: false, left: true});

    let mut trans = empty_array();
    trans.fill_from_resize(arr.view().conj().transpose());
    rsrs_data.p_matrix.mul(&mut trans, ElMatOptions{inv: false, trans: false, left: true});
    arr.fill_from_resize(trans.view().conj().transpose());

    let mut diag_ae = Vec::new();
    let mut skel_ae = 0.0;

    let mut count = 0;
    let num_factors = rsrs_data.diag_boxes.len();
    for diag_box_id in 0..num_factors{
        let diag_box = &mut rsrs_data.diag_boxes[diag_box_id];
        let len_box = diag_box.shape()[0];
        let inds: Vec<usize> = (count..len_box + count).collect();
        let e_diag_box: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(inds.clone(), inds.clone())).unwrap().ext;
        let res_rows: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Axis(inds.clone(), 0, false)).unwrap().ext;
        let mut res = empty_array(); 
        res.fill_from_resize(e_diag_box - diag_box.view());
        let ae = res.norm_2_alloc().unwrap(); 
        if diag_box_id < num_factors-1
        {
            diag_ae.push(ae);
        }
        else{
            skel_ae = ae;
        }
        //diag_box.view_mut().into_inverse_alloc().unwrap();
        //let mut id_block = empty_array();
        //id_block.view_mut().simple_mult_into_resize(diag_box.view(), res_rows.view());
        //matrix_insertion(arr, &mut id_block, ExtInsType::Axis(inds.clone(), 0, false));
        matrix_insertion(arr, &mut solve_left(diag_box, &res_rows, 0.0), ExtInsType::Axis(inds.clone(), 0, false));
        count += len_box;
    }

    let diag_ae_sum = diag_ae.iter().sum::<f64>();
    let diag_ae_mean = diag_ae_sum/ diag_ae.len() as f64;

    let mut ident = rlst_dynamic_array2!(f64, [npoints, npoints]);
    ident.set_identity();

    let zero = ident - arr.view();

    (zero.view().view_flat().norm_2(), diag_ae_mean, skel_ae, rsrs_data.y_data.num_samples)

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

    let id_tols = [1e-2, 5e-3, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14];

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
        let tols: Tols<f64> = Tols{id: id_tol, null: 1e-15, lstq: 1e-15};
        let mut kernel_mat: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> = get_matrix(&points);
        let mut rsrs_algo: RsrsData<f64> = <RsrsData<f64> as Rsrs>::new(&kernel_mat, tols, &tree);
        rsrs_algo.tree_cycle_and_diag_block_extraction(&kernel_mat);
        
        let (norm_app_inv, diag_ae_mean, skel_ae, num_samples) = get_box_errors(&mut kernel_mat, &mut rsrs_algo,  npoints, id_tol, &path_str);
        app_inv.push(norm_app_inv);
        
        if !diag_ae_mean.is_nan(){
            diag_errs.push(diag_ae_mean);
        }
        skel_errs.push(skel_ae);
        tot_num_samples.push(num_samples as f64);
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
    let npoints = 500;

    test_rsrs_geometry(geometry, npoints);
    
}
