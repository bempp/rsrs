use rlst::{prelude::*, Array, VectorContainer, BaseArray};

//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.

fn exp_kernel(dist: f64, npoints: usize)-> f64{
    let n: f64 = npoints as f64;
    1.0/(n*(dist*dist).exp())
}

pub fn get_matrix(points_x: &[bempp_octree::Point])-> Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>{
    let n: usize = points_x.len();
    let mut arr: DynamicArray<f64, 2> = rlst_dynamic_array2!(f64, [n, n]);
    for (i, point_x) in points_x.iter().enumerate(){
        for (j, point_y) in points_x.iter().enumerate(){
            let coords_x: [f64; 3] = point_x.coords();
            let coords_y: [f64; 3] = point_y.coords();
            let dist = ((coords_x[0]-coords_y[0]).pow(2.0) + (coords_x[1]-coords_y[1]).pow(2.0) + (coords_x[2]-coords_y[2]).pow(2.0)).sqrt();
            if dist> 0.0{
                *arr.get_mut([i, j]).unwrap() = exp_kernel(dist, n);
            }
            else{
                //If points are equal, set the value to 1
                *arr.get_mut([i, j]).unwrap() = 1.0;
            }
        }
    }
    arr
}
