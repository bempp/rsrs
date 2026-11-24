//! Numerical implementation units for dense SRS and sampled RSRS.

pub mod composed_factor;
pub mod configuration;
pub mod dense_cycle;
pub mod dense_multilevel;
pub mod diagonal_block;
pub mod elementary_factor;
pub mod factorized_operator;
pub mod interpolative_decomposition;
pub mod regularized_least_squares;
pub mod reporting;
pub mod sampled_cycle;
pub mod sampled_id;
pub mod sampled_multilevel;
pub mod sampled_options;
pub mod scheduling;

#[cfg(test)]
extern crate blas_src;
#[cfg(test)]
extern crate lapack_src;
#[cfg(all(test, target_os = "linux"))]
extern crate openblas_src;
