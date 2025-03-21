use rlst::prelude::*;
//Function that creates a low rank matrix by calculating a kernel given a random point distribution on an unit sphere.

pub trait ExpKernel: RlstScalar {
    fn exp_real_kernel(dist: Self, npoints: usize, kappa: Self) -> Self;
    fn exp_complex_kernel(dist: Self, npoints: usize, kappa: Self) -> num::Complex<Self>;
    fn laplace_kernel(dist: Self, npoints: usize, kappa: Self) -> Self;
    fn helmholtz_kernel(dist: Self, npoints: usize, kappa: Self) -> num::Complex<Self>;
}

macro_rules! implement_exp_kernel {
    ($scalar:ty) => {
        impl ExpKernel for $scalar {
            fn exp_real_kernel(dist: Self, npoints: usize, _kappa: Self) -> Self {
                let d: Self = num::NumCast::from(dist).unwrap();
                let n: Self = num::NumCast::from(npoints).unwrap();
                (1.0 / (n * (d * d).exp()))
            }
            fn exp_complex_kernel(dist: Self, npoints: usize, _kappa: Self) -> num::Complex<Self> {
                let d: Self = num::NumCast::from(dist).unwrap();
                let n: Self = num::NumCast::from(npoints).unwrap();
                let i = num::Complex::<Self>::new(0.0, 1.0);
                (1.0 / (n * (i * d * d).exp()))
            }
            fn laplace_kernel(dist: Self, npoints: usize, _kappa: Self) -> Self {
                let pi: Self = if std::any::TypeId::of::<Self>() == std::any::TypeId::of::<f32>() {
                    std::f32::consts::PI as Self
                } else {
                    std::f64::consts::PI as Self
                };
                let d: Self = num::NumCast::from(dist).unwrap();
                let n: Self = num::NumCast::from(npoints).unwrap();
                (1.0 / (4.0 * pi * n * d))
            }
            fn helmholtz_kernel(dist: Self, npoints: usize, kappa: Self) -> num::Complex<Self> {
                let pi: Self = if std::any::TypeId::of::<Self>() == std::any::TypeId::of::<f32>() {
                    std::f32::consts::PI as Self
                } else {
                    std::f64::consts::PI as Self
                };
                let d: Self = num::NumCast::from(dist).unwrap();
                let n: Self = num::NumCast::from(npoints).unwrap();
                let i = num::Complex::<Self>::new(0.0, 1.0);
                ((i * kappa * d).exp() / (4.0 * pi * n * d))
            }
        }
    };
}

implement_exp_kernel!(f32);
implement_exp_kernel!(f64);

pub trait LowRankMatrix: RlstScalar {
    fn get_real_matrix(
        points_x: &[bempp_octree::Point],
        kernel_fn: fn(dist: Self, npoints: usize, kappa: Self) -> Self,
        kappa: Self,
    ) -> DynamicArray<Self, 2>;
    fn get_complex_matrix(
        points_x: &[bempp_octree::Point],
        kernel_fn: fn(
            dist: Self,
            npoints: usize,
            kappa: <Self as RlstScalar>::Real,
        ) -> num::Complex<Self>,
        kappa: <Self as RlstScalar>::Real,
    ) -> DynamicArray<num::Complex<Self>, 2>
    where
        Self: PartialOrd;
}

macro_rules! implement_low_rank_matrix {
    ($scalar:ty) => {
        impl LowRankMatrix for $scalar {
            fn get_real_matrix(
                points_x: &[bempp_octree::Point],
                kernel_fn: fn(dist: Self, npoints: usize, kappa: Self) -> Self,
                kappa: Self,
            ) -> DynamicArray<Self, 2> {
                let n: usize = points_x.len();
                let mut arr: DynamicArray<Self, 2> = rlst_dynamic_array2!(Self, [n, n]);
                for (i, point_x) in points_x.iter().enumerate() {
                    for (j, point_y) in points_x.iter().enumerate() {
                        let coords_x: [f64; 3] = point_x.coords();
                        let coords_y: [f64; 3] = point_y.coords();
                        let dist: <$scalar as RlstScalar>::Real = num::NumCast::from(
                            ((coords_x[0] - coords_y[0]).powi(2)
                                + (coords_x[1] - coords_y[1]).powi(2)
                                + (coords_x[2] - coords_y[2]).powi(2))
                            .sqrt(),
                        )
                        .unwrap();
                        if dist > 0.0 {
                            *arr.get_mut([i, j]).unwrap() = kernel_fn(dist, n, kappa);
                        } else {
                            //If points are equal, set the value to 1
                            *arr.get_mut([i, j]).unwrap() = 1.0.into();
                        }
                    }
                }
                arr
            }

            fn get_complex_matrix(
                points_x: &[bempp_octree::Point],
                kernel_fn: fn(
                    dist: Self,
                    npoints: usize,
                    kappa: <Self as RlstScalar>::Real,
                ) -> num::Complex<Self>,
                kappa: <Self as RlstScalar>::Real,
            ) -> DynamicArray<num::Complex<Self>, 2> {
                let n: usize = points_x.len();
                let mut arr: DynamicArray<num::Complex<Self>, 2> =
                    rlst_dynamic_array2!(num::Complex<Self>, [n, n]);
                for (i, point_x) in points_x.iter().enumerate() {
                    for (j, point_y) in points_x.iter().enumerate() {
                        let coords_x: [f64; 3] = point_x.coords();
                        let coords_y: [f64; 3] = point_y.coords();
                        let dist: <$scalar as RlstScalar>::Real = num::NumCast::from(
                            ((coords_x[0] - coords_y[0]).powi(2)
                                + (coords_x[1] - coords_y[1]).powi(2)
                                + (coords_x[2] - coords_y[2]).powi(2))
                            .sqrt(),
                        )
                        .unwrap();
                        if dist > 0.0 {
                            *arr.get_mut([i, j]).unwrap() = kernel_fn(dist, n, kappa);
                        } else {
                            //If points are equal, set the value to 1
                            *arr.get_mut([i, j]).unwrap() = 1.0.into();
                        }
                    }
                }
                arr
            }
        }
    };
}

implement_low_rank_matrix!(f32);
implement_low_rank_matrix!(f64);

pub trait KernelMatrix: RlstScalar {
    fn get_exp_real_kernel_matrix(
        points_x: &[bempp_octree::Point],
        kappa: Self,
    ) -> DynamicArray<Self, 2>;
    fn get_exp_complex_kernel_matrix(
        points_x: &[bempp_octree::Point],
        kappa: Self,
    ) -> DynamicArray<num::Complex<Self>, 2>
    where
        Self: PartialOrd;
    fn get_laplace_matrix(points_x: &[bempp_octree::Point], kappa: Self) -> DynamicArray<Self, 2>;
    fn get_helmholtz_matrix(
        points_x: &[bempp_octree::Point],
        kappa: Self,
    ) -> DynamicArray<num::Complex<Self>, 2>
    where
        Self: PartialOrd;
}

macro_rules! implement_kernel_matrix {
    ($scalar:ty) => {
        impl KernelMatrix for $scalar {
            fn get_exp_real_kernel_matrix(
                points_x: &[bempp_octree::Point],
                _kappa: Self,
            ) -> DynamicArray<Self, 2> {
                LowRankMatrix::get_real_matrix(points_x, Self::exp_real_kernel, 0.0)
            }

            fn get_exp_complex_kernel_matrix(
                points_x: &[bempp_octree::Point],
                kappa: Self,
            ) -> DynamicArray<num::Complex<Self>, 2> {
                LowRankMatrix::get_complex_matrix(points_x, Self::exp_complex_kernel, kappa)
            }

            fn get_laplace_matrix(
                points_x: &[bempp_octree::Point],
                _kappa: Self,
            ) -> DynamicArray<Self, 2> {
                LowRankMatrix::get_real_matrix(points_x, Self::laplace_kernel, 0.0)
            }

            fn get_helmholtz_matrix(
                points_x: &[bempp_octree::Point],
                kappa: Self,
            ) -> DynamicArray<num::Complex<Self>, 2> {
                LowRankMatrix::get_complex_matrix(points_x, Self::helmholtz_kernel, kappa)
            }
        }
    };
}

implement_kernel_matrix!(f32);
implement_kernel_matrix!(f64);

/*macro_rules! impl_real_exp_kernel{
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
macro_rules! impl_complex_exp_kernel{
    ($scalar:ty) => {
        paste::item! {
            fn [<exp_complex_kernel_$scalar>](dist: f64, npoints: usize)-> num::Complex<$scalar>{
                let d : $scalar = num::NumCast::from(dist).unwrap();
                let n: $scalar = npoints as $scalar;
                let i = num::Complex::<$scalar>::new(0.0, 1.0);
                (1.0/(n*(i*d*d).exp()))
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



impl_laplace_kernel!(f32);
impl_laplace_kernel!(f64);
impl_real_exp_kernel!(f32);
impl_real_exp_kernel!(f64);
impl_complex_exp_kernel!(f32);
impl_complex_exp_kernel!(f64);

macro_rules! impl_get_complex_matrix {
    ($scalar:ty, $kernel_name:ident) => {
        paste::item! {
            pub fn [<get_ $kernel_name _ $scalar>](points_x: &[bempp_octree::Point])-> DynamicArray< num::Complex<$scalar>, 2>{
                let n: usize = points_x.len();
                let mut arr: DynamicArray<num::Complex<$scalar>, 2> = rlst_dynamic_array2!(num::Complex<$scalar>, [n, n]);
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
impl_get_complex_matrix!(f64, exp_complex);
impl_get_complex_matrix!(f32, exp_complex);*/
