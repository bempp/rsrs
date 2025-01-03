use bempp_rsrs::mpi_multiply::matrix_multiply;
use rlst::prelude::*;
use core::f64;
use std::time::Instant;


pub fn main() {


    let n = 100;
    let m = 100;
    let mut arr1: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> = rlst_dynamic_array2!(f64, [n, n]);
    let mut arr2: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> = rlst_dynamic_array2!(f64, [n, m]);
    let mut arr3: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> = rlst_dynamic_array2!(f64, [n, m]);
    
    for i in 0..n{
        for j in 0..n{
            *arr1.get_mut([i, j]).unwrap() = (i*n+ j + 1) as f64;
        }
    }

    for i in 0..n{
        for j in 0..m{
            if i == j{
                *arr2.get_mut([i, j]).unwrap() = 1.0;
            }
        }
    }

    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let start = Instant::now();
    matrix_multiply(TransMode::NoTrans, TransMode::NoTrans, num::One::one(), &arr1, &arr2, num::Zero::zero(), &mut arr3, &comm);
    let duration = start.elapsed();

    println!(
        "Testing in {} ms",
        duration.as_millis()
    );
      
}

