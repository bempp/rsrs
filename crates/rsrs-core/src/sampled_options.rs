//! Numerical choices for sample-driven RSRS.

use crate::configuration::PivotMethod;
use crate::regularized_least_squares::{Regularization, RegularizedLeastSquaresMethod};
use rlst::RlstScalar;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NullificationMethod<Real> {
    Projection {
        solver: RegularizedLeastSquaresMethod,
        regularization: Regularization<Real>,
    },
    Qr {
        tolerance: Real,
    },
    Svd {
        tolerance: Real,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtractionOptions<Real> {
    pub solver: RegularizedLeastSquaresMethod,
    pub regularization: Regularization<Real>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PivotOptions<Real> {
    pub method: PivotMethod,
    pub shift: Real,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampledFactorizationOptions<Item: RlstScalar> {
    pub nullification: NullificationMethod<Item::Real>,
    pub near_extraction: ExtractionOptions<Item::Real>,
    pub diagonal_extraction: ExtractionOptions<Item::Real>,
    pub near_pivot: PivotOptions<Item::Real>,
    pub diagonal_pivot: PivotOptions<Item::Real>,
}
