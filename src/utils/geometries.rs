use mpi::traits::CommunicatorCollectives;
use bempp_octree::generate_random_points;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.


pub fn sphere_surface<C: CommunicatorCollectives>(npoints: usize, comm: &C)-> std::vec::Vec<bempp_octree::Point>{
    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(0);//ChaCha8Rng::seed_from_u64(comm.rank() as u64);
    let mut points: Vec<bempp_octree::Point> = generate_random_points(npoints, &mut rng, comm);

    // Find centre points
    let x: Vec<f64> = points.iter().map(|point| point.coords()[0]).collect();
    let y: Vec<f64> = points.iter().map(|point| point.coords()[0]).collect();
    let z: Vec<f64> = points.iter().map(|point| point.coords()[0]).collect();
    let x_min = x.clone().into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let y_min = y.clone().into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let z_min = z.clone().into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let x_max = x.into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let y_max = y.into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let z_max = z.into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let centre = bempp_octree::Point::new([(x_min + x_max)*0.5, (y_min + y_max)*0.5, (z_min + z_max)*0.5], 000);
    
    // Make sure that the points live on the unit sphere.
    for point in points.iter_mut() {
        let mut aux:[f64; 3]  = [point.coords()[0] - centre.coords()[0], point.coords()[1] - centre.coords()[1], point.coords()[2] - centre.coords()[2]];
        let len: f64 = (aux[0]*aux[0] + aux[1]*aux[1] + aux[2]*aux[2]).sqrt();
        aux[0] /= len;
        aux[1] /= len;
        aux[2] /= len;
        point.coords_mut()[0] = aux[0] + centre.coords()[0];
        point.coords_mut()[1] = aux[1] + centre.coords()[1];
        point.coords_mut()[2] = aux[2] + centre.coords()[2];
    }

    points
}


pub fn cube_surface<C: CommunicatorCollectives>(npoints: usize, comm: &C)-> std::vec::Vec<bempp_octree::Point>{
    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(comm.rank() as u64);
    let mut points: Vec<bempp_octree::Point> = generate_random_points(npoints, &mut rng, comm);
    
    // Make sure that the points live on the unit cube.
    for point in points.iter_mut() {
        let aux = point.coords();
        let max = aux.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        point.coords_mut()[0] /= max;
        point.coords_mut()[1] /= max;
        point.coords_mut()[2] /= max;
    }

    points
}