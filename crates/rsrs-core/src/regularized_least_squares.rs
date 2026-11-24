use num::{Float, One, Zero};
use rlst::{
    base_types::{RlstResult, TransMode, UpLo},
    dense::{array::DynArray, linalg::lapack::interpolative_decomposition::MatrixId},
    dot_blas, CholeskySolve, EvaluateArray, Gemm, Lu, SingularValueDecomposition,
};
use serde::{Deserialize, Serialize};

/// Solver for the regularized normal equations used in block extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegularizedLeastSquaresMethod {
    /// Historical `LuLstSq`: pivoted LU applied to the normal equations.
    Lu,
    Cholesky,
    /// Direct pseudoinverse of the original rectangular matrix. This does not
    /// square its condition number and uses `regularization.relative` as the
    /// relative singular-value cutoff.
    Svd,
}

/// Interpretation of the relative regularization parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegularizationMode {
    /// Add no diagonal regularization. This is an intentional user choice;
    /// factorization may fail when the normal matrix is singular.
    None,
    /// Scaled regularization: `relative * scale`, with an `eps * n * scale` floor.
    Scaled,
    /// Historical scaling: `max(eps, relative^2) * scale`.
    Historical,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Regularization<Real> {
    pub mode: RegularizationMode,
    /// Scale relative to the largest diagonal entry of `A^H A`.
    /// In historical mode this is the old least-squares tolerance and is
    /// squared before it is applied.
    pub relative: Real,
    /// Absolute lower bound supplied by the user.
    pub absolute_floor: Real,
}

impl<Real: Zero> Regularization<Real> {
    pub fn none() -> Self {
        Self {
            mode: RegularizationMode::None,
            relative: Real::zero(),
            absolute_floor: Real::zero(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegularizedLeastSquaresReport<Real> {
    pub requested_relative: Real,
    pub requested_absolute_floor: Real,
    pub numerical_floor: Real,
    pub effective_regularization: Real,
    pub normal_diagonal_scale: Real,
}

/// Solve `(A^H A + lambda I) X = A^H B`.
///
/// `lambda` follows the selected policy. In particular, [`RegularizationMode::None`]
/// leaves the normal equations completely unregularized.
pub fn solve_regularized_least_squares<Item>(
    a: &DynArray<Item, 2>,
    rhs: &DynArray<Item, 2>,
    regularization: Regularization<Item::Real>,
    method: RegularizedLeastSquaresMethod,
) -> RlstResult<(DynArray<Item, 2>, RegularizedLeastSquaresReport<Item::Real>)>
where
    Item: MatrixId + Gemm,
    Item::Real: Into<Item>,
{
    let [rows, columns] = a.shape();
    assert_eq!(rhs.shape()[0], rows, "A and RHS row counts must agree");
    let ah = a.r().conj().transpose().eval();
    let mut normal = dot_blas!(ah.r(), a.r());
    let projected_rhs = dot_blas!(ah.r(), rhs.r());

    let mut diagonal_scale = Item::Real::zero();
    for index in 0..columns {
        diagonal_scale = diagonal_scale.max(normal[[index, index]].abs());
    }
    // The scaled policy deliberately clamps the scale to one so that tiny
    // matrices still receive a useful numerical floor.  Historical LuLstSq
    // did not: every positive diagonal scale was used verbatim, with one only
    // as the fallback for an exactly zero normal matrix.
    let scaled_scale = diagonal_scale.max(Item::Real::one());
    let historical_scale = if diagonal_scale > Item::Real::zero() {
        diagonal_scale
    } else {
        Item::Real::one()
    };
    let (_requested_relative, numerical_floor, effective) = match regularization.mode {
        RegularizationMode::None => (Item::Real::zero(), Item::Real::zero(), Item::Real::zero()),
        RegularizationMode::Scaled => {
            let requested = regularization.relative * diagonal_scale;
            let floor = Item::Real::epsilon() * Item::real(columns.max(1)) * scaled_scale;
            (
                requested,
                floor,
                requested.max(regularization.absolute_floor).max(floor),
            )
        }
        RegularizationMode::Historical => {
            let requested =
                regularization.relative.abs() * regularization.relative.abs() * historical_scale;
            let floor = Item::Real::epsilon() * historical_scale;
            (
                requested,
                floor,
                requested.max(regularization.absolute_floor).max(floor),
            )
        }
    };
    for index in 0..columns {
        normal[[index, index]] = normal[[index, index]].add_real(effective);
    }

    let solution = match method {
        RegularizedLeastSquaresMethod::Lu => {
            normal.lu()?.solve(TransMode::NoTrans, &projected_rhs)?
        }
        RegularizedLeastSquaresMethod::Cholesky => {
            normal.cholesky_solve(UpLo::Upper, &projected_rhs)?
        }
        RegularizedLeastSquaresMethod::Svd => {
            let tolerance = regularization.relative.max(Item::Real::epsilon());
            a.pseudo_inverse(None, Some(tolerance))?.apply(rhs)
        }
    };
    Ok((
        solution,
        RegularizedLeastSquaresReport {
            requested_relative: regularization.relative,
            requested_absolute_floor: regularization.absolute_floor,
            numerical_floor,
            effective_regularization: effective,
            normal_diagonal_scale: diagonal_scale,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlst::{base_types::c64, dot_blas, DynArray};

    fn check_real(method: RegularizedLeastSquaresMethod) {
        let mut a = DynArray::<f64, 2>::from_shape([8, 3]);
        for row in 0..8 {
            let point = (row + 1) as f64 / 8.0;
            a[[row, 0]] = 1.0;
            a[[row, 1]] = point;
            a[[row, 2]] = point * point;
        }
        let mut expected = DynArray::<f64, 2>::from_shape([3, 2]);
        expected[[0, 0]] = 1.0;
        expected[[1, 0]] = -2.0;
        expected[[2, 0]] = 0.5;
        expected[[0, 1]] = -0.25;
        expected[[1, 1]] = 0.75;
        expected[[2, 1]] = 1.5;
        let rhs = dot_blas!(a.r(), expected.r());
        let (actual, report) = solve_regularized_least_squares(
            &a,
            &rhs,
            Regularization {
                mode: RegularizationMode::Scaled,
                relative: 1.0e-14,
                absolute_floor: 0.0,
            },
            method,
        )
        .unwrap();
        rlst::assert_array_relative_eq!(actual, expected, 2.0e-10);
        assert!(report.effective_regularization > 0.0);
    }

    fn check_complex(method: RegularizedLeastSquaresMethod) {
        let mut a = DynArray::<c64, 2>::from_shape([8, 3]);
        for row in 0..8 {
            let point = c64::new((row + 1) as f64 / 8.0, ((row * 2 + 1) % 5) as f64 / 20.0);
            a[[row, 0]] = c64::new(1.0, 0.0);
            a[[row, 1]] = point;
            a[[row, 2]] = point * point;
        }
        let mut expected = DynArray::<c64, 2>::from_shape([3, 2]);
        expected[[0, 0]] = c64::new(1.0, -0.5);
        expected[[1, 0]] = c64::new(-2.0, 0.25);
        expected[[2, 0]] = c64::new(0.5, 1.0);
        expected[[0, 1]] = c64::new(-0.25, 0.2);
        expected[[1, 1]] = c64::new(0.75, -1.0);
        expected[[2, 1]] = c64::new(1.5, 0.4);
        let rhs = dot_blas!(a.r(), expected.r());
        let (actual, report) = solve_regularized_least_squares(
            &a,
            &rhs,
            Regularization {
                mode: RegularizationMode::Scaled,
                relative: 1.0e-14,
                absolute_floor: 0.0,
            },
            method,
        )
        .unwrap();
        rlst::assert_array_relative_eq!(actual, expected, 2.0e-10);
        assert!(report.effective_regularization > 0.0);
    }

    #[test]
    fn cholesky_real_and_complex() {
        check_real(RegularizedLeastSquaresMethod::Cholesky);
        check_complex(RegularizedLeastSquaresMethod::Cholesky);
    }

    #[test]
    fn lu_real_and_complex() {
        check_real(RegularizedLeastSquaresMethod::Lu);
        check_complex(RegularizedLeastSquaresMethod::Lu);
    }

    #[test]
    fn zero_regularization_is_preserved() {
        let mut a = DynArray::<f64, 2>::from_shape([3, 2]);
        a[[0, 0]] = 1.0;
        a[[1, 1]] = 1.0;
        a[[2, 0]] = 1.0;
        a[[2, 1]] = 1.0;
        let rhs = DynArray::<f64, 2>::new_from(&a);
        let (_, report) = solve_regularized_least_squares(
            &a,
            &rhs,
            Regularization::none(),
            RegularizedLeastSquaresMethod::Lu,
        )
        .unwrap();
        assert_eq!(report.effective_regularization, 0.0);
        assert_eq!(report.numerical_floor, 0.0);
    }

    #[test]
    fn historical_regularization_does_not_clamp_a_positive_scale_to_one() {
        let mut a = DynArray::<f64, 2>::from_shape([2, 2]);
        a[[0, 0]] = 0.1;
        a[[1, 1]] = 0.1;
        let rhs = DynArray::<f64, 2>::new_from(&a);
        let (_, report) = solve_regularized_least_squares(
            &a,
            &rhs,
            Regularization {
                mode: RegularizationMode::Historical,
                relative: 0.1,
                absolute_floor: 0.0,
            },
            RegularizedLeastSquaresMethod::Lu,
        )
        .unwrap();
        assert!((report.normal_diagonal_scale - 0.01).abs() < 1.0e-15);
        assert!((report.effective_regularization - 1.0e-4).abs() < 1.0e-15);
    }

    #[test]
    fn svd_real_and_complex() {
        check_real(RegularizedLeastSquaresMethod::Svd);
        check_complex(RegularizedLeastSquaresMethod::Svd);
    }
}
