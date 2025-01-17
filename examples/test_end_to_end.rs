use bempp_rsrs::{rsrs::{box_skeletonisation::Tols, rsrs_cycle::{DecoupledBoxData, Rsrs, RsrsData}}, utils::{elementary_matrix::{ElMatOptions, ElementaryOperations}, geometries::{cube_surface, sphere_surface}, linear_algebra::{ExtInsType, Extraction, MatrixExtraction}, low_rank_matrices::get_matrix}};
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use bempp_octree::Octree;
use rlst::prelude::*;


fn get_far_indices(n: usize, near_indices: Vec<usize>)->Vec<usize>{
    let mut domain: Vec<usize> = (0..n).collect();
    domain.retain(|x| !near_indices.contains(x));
    domain
}


fn apply_decomposition(arr: &mut DynamicArray<f64, 2>, decoupled: Vec<DecoupledBoxData<f64>>){
    for (box_id, d_box) in decoupled.iter().enumerate(){
        println!("Box num {} of {}, level iteration {}", box_id, decoupled.len(), d_box.level_it);

        let ind_r = &d_box.box_features.ind_r;
        let ind_t = &d_box.box_features.ind_t;
        let far_indices = get_far_indices(arr.shape()[0], d_box.box_features.near_field_inds.clone());

        println!("Box sizes: {}, {}, {}", ind_r.len(), ind_t.len(), far_indices.len());

        let arr_rf: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_r.clone(), far_indices.clone())).unwrap().ext;
        let arr_fr: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(far_indices.clone(), ind_r.clone())).unwrap().ext;
        let arr_rt: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_r.clone(), ind_t.clone())).unwrap().ext;
        let arr_tr: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_t.clone(), ind_r.clone())).unwrap().ext;

        let arr_rf_n = arr_rf.norm_2_alloc().unwrap();
        let arr_fr_n = arr_fr.norm_2_alloc().unwrap();
        let arr_rt_n = arr_rt.norm_2_alloc().unwrap();
        let arr_tr_n = arr_tr.norm_2_alloc().unwrap();

        //println!("Before decoupling: {}, {}, {}, {}\n", arr_rf_n, arr_fr_n, arr_rt_n, arr_tr_n);

        d_box.operators.fact_id.left.mul(arr, ElMatOptions{inv: true, trans: false, left: true});
        d_box.operators.fact_id.right.mul(arr, ElMatOptions{inv: true, trans: false, left: false});
        d_box.operators.fact_lu.left.mul(arr, ElMatOptions{inv: true, trans: false, left: true});
        d_box.operators.fact_lu.right.mul(arr, ElMatOptions{inv: true, trans: false, left: false});

        let arr_rf: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_r.clone(), far_indices.clone())).unwrap().ext;
        let arr_fr: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(far_indices.clone(), ind_r.clone())).unwrap().ext;
        let arr_rt: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_r.clone(), ind_t.clone())).unwrap().ext;
        let arr_tr: DynamicArray<f64, 2> = <Extraction<f64> as MatrixExtraction>::new(arr, ExtInsType::Cross(ind_t.clone(), ind_r.clone())).unwrap().ext;

        let arr_rf_ae = arr_rf.norm_2_alloc().unwrap();
        let arr_fr_ae = arr_fr.norm_2_alloc().unwrap();
        let arr_rt_ae = arr_rt.norm_2_alloc().unwrap();
        let arr_tr_ae = arr_tr.norm_2_alloc().unwrap();

        //println!("After decoupling absolute errors: {}, {}, {}, {}", arr_rf_ae, arr_fr_ae, arr_rt_ae, arr_tr_ae);
        println!("Relative errors: {}, {}, {}, {}\n", arr_rf_ae/arr_rf_n, arr_fr_ae/arr_fr_n, arr_rt_ae/arr_rt_n, arr_tr_ae/arr_tr_n);
    
    }

}

//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.

fn get_example(geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point>){
    let universe: mpi::environment::Universe = mpi::initialize().unwrap();
    let comm: SimpleCommunicator = universe.world();
    let npoints:usize = 5000;
    let points: Vec<bempp_octree::Point> = geometry_fn(npoints, &comm);

    let max_level: usize = 16;
    let max_leaf_points: usize = 50;

    // The following code will create a complete octree with a maximum level of 16.
    let tree: Octree<'_, SimpleCommunicator>  = Octree::new(&points, max_level, max_leaf_points, &comm);
    let global_number_of_points: usize = tree.global_number_of_points();
    let global_max_level: usize = tree.global_max_level();
    
    // We now check that each node of the tree has all its neighbors available.

    if comm.rank() == 0 {
        println!(
            "Setup octree with {} points and maximum level {}",
            global_number_of_points,
            global_max_level
        );
    }

    let tols: Tols<f64> = Tols{id: 1e-2, null: 1e-15, lstq: 1e-15};
    //Create a low rank matrix
    let mut kernel_mat: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> = get_matrix(&points);
    let mut rsrs_algo: RsrsData<f64> = <RsrsData<f64> as Rsrs>::new(&kernel_mat, tols, tree);
    rsrs_algo.tree_cycle(&kernel_mat);

    apply_decomposition(&mut kernel_mat, rsrs_algo.dec_boxes);

}
pub fn main() {
    get_example(sphere_surface);
}
