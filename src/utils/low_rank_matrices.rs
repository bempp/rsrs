use rlst::prelude::*;
use paste;
//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.

macro_rules! impl_real_exp_kernel{
    ($scalar:ty) => {
        paste::item! {
            fn [<exp_real_kernel_$scalar>](dist: f64, npoints: usize)-> $scalar{
                let d : $scalar = num::NumCast::from(dist).unwrap();
                let n: $scalar = npoints as $scalar;
                (1.0/(n*(d*d).exp())) as $scalar
            }
        }
    }
}

macro_rules! impl_laplace_kernel{
    ($scalar:ty) => {
        paste::item! {
            fn [<laplace_kernel_$scalar>](dist: f64, npoints: usize)-> $scalar{
                let pi: $scalar = std::$scalar::consts::PI;
                let d : $scalar = num::NumCast::from(dist).unwrap();
                let n: $scalar = npoints as $scalar;
                (1.0/(4.0*pi*n*d)) as $scalar
            }
        }
    }
}

macro_rules! impl_complex_exp_kernel{
    ($scalar:ty) => {
        paste::item! {
            fn [<exp_complex_kernel_$scalar>](dist: f64, npoints: usize)-> num::Complex<$scalar>{
                let d : $scalar = num::NumCast::from(dist).unwrap();
                let n: $scalar = npoints as $scalar;
                let i = num::Complex::<$scalar>::new(1.0, 0.0);
                (1.0/i*(n*(d*d).exp()))
            }
        }
    }
}

impl_laplace_kernel!(f32);
impl_laplace_kernel!(f64);
impl_real_exp_kernel!(f32);
impl_real_exp_kernel!(f64);
impl_complex_exp_kernel!(f32);
impl_complex_exp_kernel!(f64);



macro_rules! impl_get_complex_matrix {
    ($scalar:ty) => {
        paste::item! {
            pub fn [<get_complex_matrix_ $scalar>](points_x: &[bempp_octree::Point])-> DynamicArray< num::Complex<$scalar>, 2>{
                let n: usize = points_x.len();
                let mut arr: DynamicArray<num::Complex<$scalar>, 2> = rlst_dynamic_array2!(num::Complex<$scalar>, [n, n]);
                for (i, point_x) in points_x.iter().enumerate(){
                    for (j, point_y) in points_x.iter().enumerate(){
                        let coords_x: [f64; 3] = point_x.coords();
                        let coords_y: [f64; 3] = point_y.coords();
                        let dist: f64 = ((coords_x[0]-coords_y[0]).pow(2.0) + (coords_x[1]-coords_y[1]).pow(2.0) + (coords_x[2]-coords_y[2]).pow(2.0)).sqrt();
                        if dist > 0.0{
                            *arr.get_mut([i, j]).unwrap() = [<exp_complex_kernel_ $scalar>](dist, n).into();
                        }
                        else{
                            //If points are equal, set the value to 1
                            *arr.get_mut([i, j]).unwrap() = 1.0.into();
                        }
                    }
                }
                arr
            }
        }
    }
}

macro_rules! impl_get_real_matrix {
    ($scalar:ty, $kernel_name:ident) => {
        paste::item! {
            pub fn [<get_ $kernel_name _ $scalar>](points_x: &[bempp_octree::Point])-> DynamicArray<$scalar, 2>{
                let n: usize = points_x.len();
                let mut arr: DynamicArray<$scalar, 2> = rlst_dynamic_array2!($scalar, [n, n]);
                for (i, point_x) in points_x.iter().enumerate(){
                    for (j, point_y) in points_x.iter().enumerate(){
                        let coords_x: [f64; 3] = point_x.coords();
                        let coords_y: [f64; 3] = point_y.coords();
                        let dist: f64 = ((coords_x[0]-coords_y[0]).pow(2.0) + (coords_x[1]-coords_y[1]).pow(2.0) + (coords_x[2]-coords_y[2]).pow(2.0)).sqrt();
                        if dist > 0.0{
                            *arr.get_mut([i, j]).unwrap() = [<$kernel_name _kernel_ $scalar>](dist, n).into();
                        }
                        else{
                            //If points are equal, set the value to 1
                            *arr.get_mut([i, j]).unwrap() = 1.0;
                        }
                    }
                }
                arr
            }
        }
    }
}


impl_get_real_matrix!(f64, exp_real);
impl_get_real_matrix!(f32, exp_real);
impl_get_real_matrix!(f64, laplace);
impl_get_real_matrix!(f32, laplace);
impl_get_complex_matrix!(f64);
impl_get_complex_matrix!(f32);