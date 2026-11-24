//! Public interface for dense SRS and sampled RSRS.
//!
//! Numerical factorization lives in [`core`], geometry and octree scheduling
//! in [`octree`], and the portable sample-cache contract in [`samples`].

pub use rsrs_core as core;
pub use rsrs_octree as octree;
pub use rsrs_samples as samples;
