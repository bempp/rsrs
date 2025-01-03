use bempp_rsrs::utils_linear_algebra::{ExtInsType, Extraction, MPIMatrixExtraction, MatrixExtraction};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::Uniform;
use rlst::prelude::*;
use core::f64;
use std::time::Instant;


pub fn main() {


    let n = 1000;
    let mut arr: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> = rlst_dynamic_array2!(f64, [n, n]);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    arr.fill_from_standard_normal(&mut rng); 
    let range = Uniform::new(0, 1000);

    let rows: Vec<usize> = (0..100).map(|_| rng.sample(range)).collect();
    let cols: Vec<usize> = (0..100).map(|_| rng.sample(range)).collect();


    //let universe = mpi::initialize().unwrap();
    //let comm = universe.world();

    let start = Instant::now();
    let ext1 = <Extraction<f64> as MPIMatrixExtraction>::new(&mut arr, ExtInsType::Axis(rows.clone(), 0, false)).unwrap().ext;
    let duration = start.elapsed();

    println!(
        "Extraction in {} ms",
        duration.as_millis()
    );

    let start = Instant::now();
    let ext2 = <Extraction<f64> as MatrixExtraction>::new(&mut arr, ExtInsType::Axis(rows.clone(), 0, false)).unwrap().ext;
    let duration = start.elapsed();

    println!(
        "Extraction in {} ms",
        duration.as_millis()
    );

      
}

