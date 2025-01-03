use bempp_octree::{generate_random_points, MortonKey, Octree};
use rlst::prelude::*;
use bempp_rsrs::{box_skeletonisation::Tols, rsrs::{Rsrs, RsrsData}};
use mpi::{topology::SimpleCommunicator, traits::Communicator};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.
fn low_rank_matrix(n: usize, arr: &mut DynamicArray<f64, 2>, points_x: &[bempp_octree::Point]){
    //Obtain n equally distributed angles 0<phi<pi and n equally distributed angles 0<theta<2pi
    let pi: f64 = std::f64::consts::PI;

    let mut angles1 = rlst_dynamic_array2!(f64, [n, 1]);
    angles1.fill_from_seed_equally_distributed(0);
    angles1.scale_inplace(pi);

    let mut angles2 = rlst_dynamic_array2!(f64, [n, 1]);
    angles2.fill_from_seed_equally_distributed(1);
    angles2.scale_inplace(2.0*pi);

    for (i, point_x) in points_x.iter().enumerate(){
        for (j, point_y) in points_x.iter().enumerate(){
            if i!=j{
            let coords_x = point_x.coords();
            let coords_y = point_y.coords();
            let dist = ((coords_x[0]-coords_y[0]).pow(2.0) + (coords_x[1]-coords_y[1]).pow(2.0) + (coords_x[2]-coords_y[2]).pow(2.0)).sqrt();
            println!("{}, {}, {}, {}", dist, dist.pow(2.0), (-dist.pow(2.0)).exp(), (1.0/(points_x.len() as f64))*(-dist.pow(2.0)).exp());
            *arr.get_mut([i, j]).unwrap() = (1.0/(points_x.len() as f64))*(-dist.pow(2.0)).exp();
            }
            else{
                //If points are equal, set the value to 1
                *arr.get_mut([i, j]).unwrap() = 1.0;
            }
        }
    }

}
pub fn main() {
    type C = SimpleCommunicator;
    // Initialise MPI
    let universe: mpi::environment::Universe = mpi::initialize().unwrap();

    // Get the world communicator
    let comm: SimpleCommunicator = universe.world();

    // Initialise a seeded Rng.
    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(comm.rank() as u64);

    let npoints:usize = 10;

    let mut points: Vec<bempp_octree::Point> = generate_random_points(npoints, &mut rng, &comm);
    // Make sure that the points live on the unit sphere.
    for point in points.iter_mut() {
        println!("{:?}", point.coords());
        let len = point.coords()[0] * point.coords()[0]
            + point.coords()[1] * point.coords()[1]
            + point.coords()[2] * point.coords()[2];
        let len = len.sqrt();
        point.coords_mut()[0] /= len;
        point.coords_mut()[1] /= len;
        point.coords_mut()[2] /= len;
    }

    let max_level = 6;
    let max_leaf_points = 2;
    
    // The following code will create a complete octree with a maximum level of 16.
    let tree = Octree::new(&points, max_level, max_leaf_points, &comm);

    let global_number_of_points = tree.global_number_of_points();
    let global_max_level = tree.global_max_level();

    // We now check that each node of the tree has all its neighbors available.

    if comm.rank() == 0 {
        println!(
            "Setup octree with {} points and maximum level {}",
            global_number_of_points,
            global_max_level
        );
    }

    let tols: Tols<f64> = Tols{id: 1e-1, null: 1e-3, lstq: 1e-15};
    //Create a low rank matrix
    let mut arr: DynamicArray<f64, 2> = rlst_dynamic_array2!(f64, [npoints, npoints]);
    low_rank_matrix(npoints, &mut arr, &points);

    arr.pretty_print();
    let min_num_samples = 5;
    
    let mut rsrs_algo: RsrsData<f64>= <RsrsData<f64> as Rsrs<C>>::new(arr, tols, min_num_samples, tree, &comm);
    <bempp_rsrs::rsrs::RsrsData<f64> as bempp_rsrs::rsrs::Rsrs<'_, C>>::level_iteration(&mut rsrs_algo);
    
}
