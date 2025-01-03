use bempp_rsrs::box_skeletonisation::BoxFeatures;
use bempp_rsrs::box_skeletonisation::SkelBox;
use bempp_rsrs::box_skeletonisation::Tols;
use bempp_rsrs::elementary_matrix::ElMatOptions;
use bempp_rsrs::elementary_matrix::ElementaryOperations;
use bempp_rsrs::sketch::BoxesData;
use bempp_rsrs::sketch::SketchOps;
use bempp_rsrs::utils_linear_algebra::ExtInsType;
use bempp_rsrs::utils_linear_algebra::Extraction;
use bempp_rsrs::utils_linear_algebra::MatrixExtraction;
use rlst::dense::linalg::interpolative_decomposition::MatrixIdDecomposition;
use rlst::prelude::*;
use core::f64;
use std::fs::File;
use std::io::BufReader;
use std::io::BufRead;
use std::time::Instant;

fn load_matrix_from_file(file_path: &str, arr: &mut DynamicArray<f64, 2>) {
    let file = File::open(file_path).expect("file wasn't found.");
    let reader = BufReader::new(file);

    for (i, line) in  reader.lines().enumerate(){
        let aux_line = line.unwrap();
        for (j, el) in aux_line.split(' ').enumerate(){
            *arr.get_mut([i, j]).unwrap() = el.parse::<f64>().unwrap();
        }
    }
}

fn get_vector(file_path: &str, res_vec: &mut Vec<Vec<usize>>)
{
    let file = File::open(file_path).expect("file wasn't found.");
    let reader = BufReader::new(file);

    for line in  reader.lines(){
        let aux_line = line.unwrap();
        let aux_vec: Vec<usize> = aux_line.split(' ').map(|el| el.parse::<usize>().unwrap()).collect();
        res_vec.push(aux_vec);
    }

}

fn load_tree_indices(target_indices: &mut Vec<Vec<usize>>, near_indices: &mut Vec<Vec<usize>>, far_indices: &mut Vec<Vec<usize>>) {

    get_vector("examples/support_files/target.txt", target_indices);
    get_vector("examples/support_files/near.txt", near_indices);
    get_vector("examples/support_files/far.txt", far_indices);
}

fn compute_id_error(arr: DynamicArray<f64, 2>, id_sketch: &mut IdDecomposition<f64>, slice: usize){
    //We extract the residuals of the matrix
    let mut perm_mat: DynamicArray<f64, 2> = rlst_dynamic_array2!(f64, [slice, slice]);
    id_sketch.get_p(perm_mat.view_mut());
    let perm_arr: DynamicArray<f64, 2> = empty_array::<f64, 2>()
        .simple_mult_into_resize(perm_mat.view_mut(), arr.view());

    let dim = perm_mat.shape()[1];
    let mut a_rs: DynamicArray<f64, 2> = rlst_dynamic_array2!(f64, [slice-id_sketch.rank, dim]);
    a_rs.fill_from(perm_arr.into_subview([id_sketch.rank, 0], [slice-id_sketch.rank, dim]));
    //We compute an approximation of the residual columns of the matrix
    id_sketch.skel.pretty_print();
    let a_rs_app: DynamicArray<f64, 2> = empty_array().simple_mult_into_resize(id_sketch.id_mat.view(), id_sketch.skel.view());
    let error = a_rs.view().sub(a_rs_app.view());
    error.pretty_print();
    println!("Interpolative Decomposition L2 absolute error: {}", error.view_flat().norm_2());

}

pub fn main() {

    let n = 5000;

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
  
    let mut kernel_mat: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> = rlst_dynamic_array2!(f64, [n, n]);
    load_matrix_from_file("examples/support_files/kernel_prob.txt", &mut kernel_mat);
    let mut kernel_mat2 = empty_array();
    kernel_mat2.fill_from_resize(kernel_mat.view());

    let mut target_indices: Vec<Vec<usize>> = Vec::new();
    let mut near_indices: Vec<Vec<usize>> = Vec::new();
    let mut far_indices: Vec<Vec<usize>> = Vec::new();

    
    load_tree_indices(&mut target_indices, &mut near_indices, &mut far_indices);

    
    let tols: Tols<f64> = Tols{id: 1e-3, null: 1e-15, lstq: 1e-15};

    let min_num_samples = near_indices[0].len();
    let mut y_data: BoxesData<f64>= <BoxesData<f64> as SketchOps>::new(&kernel_mat,  min_num_samples, false, &comm);
    let mut z_data: BoxesData<f64>= <BoxesData<f64> as SketchOps>::new(&kernel_mat, min_num_samples, true, &comm);
    
    for i in 0..near_indices.len() {
        println!("{}, {}", i, near_indices.len());
        
        let min_num_samples = near_indices[i].len();

        println!("min num samples {}, current_samples {}", min_num_samples, y_data.num_samples);

       if min_num_samples > y_data.num_samples{
            let extra_num_samples = min_num_samples - y_data.num_samples;
            y_data.add_samples(extra_num_samples, &kernel_mat);
            z_data.add_samples(extra_num_samples, &kernel_mat);
        }

        let start = Instant::now();
        let mut points_box: BoxFeatures = <BoxFeatures as SkelBox<f64>>::new( target_indices[i].clone(), near_indices[i].clone());
        let _res: bempp_rsrs::box_skeletonisation::Rank<f64> = points_box.decouple(n, &mut y_data, &mut z_data, &tols);
        let duration = start.elapsed();

        println!(
            "Decoupling in {} ms",
            duration.as_millis()
        );

       match _res {
            bempp_rsrs::box_skeletonisation::Rank::Low(decoupled_box) => {

                let start = Instant::now();
                let arr_rf: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(&mut kernel_mat2, ExtInsType::Cross(points_box.ind_r.clone(), far_indices[i].clone())).unwrap().ext;
                let arr_fr: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(&mut kernel_mat2, ExtInsType::Cross(far_indices[i].clone(), points_box.ind_r.clone())).unwrap().ext;
                let arr_rt: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(&mut kernel_mat2, ExtInsType::Cross(points_box.ind_r.clone(), points_box.ind_t.clone())).unwrap().ext;
                let arr_tr: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(&mut kernel_mat2, ExtInsType::Cross(points_box.ind_t.clone(), points_box.ind_r.clone())).unwrap().ext;
                let duration = start.elapsed();

                println!(
                    "Extraction in {} ms",
                    duration.as_millis()
                );

                println!("{}", arr_rf.norm_2_alloc().unwrap());
                println!("{}", arr_fr.norm_2_alloc().unwrap());
                println!("{}", arr_rt.norm_2_alloc().unwrap());
                println!("{}", arr_tr.norm_2_alloc().unwrap());

                let start = Instant::now();
                decoupled_box.fact_id.left.mul(&mut kernel_mat2, ElMatOptions{inv: true, trans: false, left: true});
                decoupled_box.fact_id.right.mul(&mut kernel_mat2, ElMatOptions{inv: true, trans: false, left: false});
                decoupled_box.fact_lu.left.mul(&mut kernel_mat2, ElMatOptions{inv: true, trans: false, left: true});
                decoupled_box.fact_lu.right.mul(&mut kernel_mat2, ElMatOptions{inv: true, trans: false, left: false});
                let duration = start.elapsed();
                println!(
                    "Multiplication in {} ms",
                    duration.as_millis()
                );

                let start = Instant::now();
                let arr_rf: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(&mut kernel_mat2, ExtInsType::Cross(points_box.ind_r.clone(), far_indices[i].clone())).unwrap().ext;
                let arr_fr: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(&mut kernel_mat2, ExtInsType::Cross(far_indices[i].clone(), points_box.ind_r.clone())).unwrap().ext;
                let arr_rt: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(&mut kernel_mat2, ExtInsType::Cross(points_box.ind_r.clone(), points_box.ind_t.clone())).unwrap().ext;
                let arr_tr: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(&mut kernel_mat2, ExtInsType::Cross(points_box.ind_t.clone(), points_box.ind_r.clone())).unwrap().ext;
                let duration = start.elapsed();

                println!(
                    "Extraction in {} ms",
                    duration.as_millis()
                );
                
                println!("{}", arr_rf.norm_2_alloc().unwrap());
                println!("{}", arr_fr.norm_2_alloc().unwrap());
                println!("{}", arr_rt.norm_2_alloc().unwrap());
                println!("{}", arr_tr.norm_2_alloc().unwrap());
                
            },
            bempp_rsrs::box_skeletonisation::Rank::Full => {
                
            },
        }
    } 
      
}

