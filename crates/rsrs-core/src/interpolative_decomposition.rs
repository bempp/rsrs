use rlst::{
    base_types::{RlstResult, TransMode},
    dense::{
        array::DynArray,
        linalg::lapack::interpolative_decomposition::{
            IdDecomposition, MatrixId, RankRevealingQrType, RankSelection,
        },
    },
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IdRankSelection<Real> {
    Tolerance(Real),
    Fixed(usize),
    ToleranceWithMinimumRank {
        tolerance: Real,
        minimum_rank: usize,
    },
    ToleranceWithMaximumRank {
        tolerance: Real,
        maximum_rank: usize,
    },
}

impl<Real> From<IdRankSelection<Real>> for RankSelection<Real> {
    fn from(value: IdRankSelection<Real>) -> Self {
        match value {
            IdRankSelection::Tolerance(tolerance) => Self::Tolerance { tolerance },
            IdRankSelection::Fixed(rank) => Self::Fixed { rank },
            IdRankSelection::ToleranceWithMinimumRank {
                tolerance,
                minimum_rank,
            } => Self::ToleranceWithMinimumRank {
                tolerance,
                minimum_rank,
            },
            IdRankSelection::ToleranceWithMaximumRank {
                tolerance,
                maximum_rank,
            } => Self::ToleranceWithMaximumRank {
                tolerance,
                maximum_rank,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IdMethod<Real> {
    Rrqr,
    Srrqr { coefficient_bound: Real },
}

/// Compute an ID of the candidate columns using the convention expected by
/// RSRS factors. `id_mat` contains conjugate-transposed interpolation weights.
pub fn compute_candidate_id<Item>(
    matrix: &DynArray<Item, 2>,
    rank: IdRankSelection<Item::Real>,
    method: IdMethod<Item::Real>,
    materialize_skeletons: bool,
) -> RlstResult<IdDecomposition<Item>>
where
    Item: MatrixId,
{
    let method = match method {
        IdMethod::Rrqr => RankRevealingQrType::RRQR,
        IdMethod::Srrqr { coefficient_bound } => RankRevealingQrType::SRRQR(coefficient_bound),
    };
    if materialize_skeletons {
        matrix.interpolative_decomposition(rank.into(), method, TransMode::Trans)
    } else {
        matrix.interpolative_decomposition_without_skeletons(rank.into(), method, TransMode::Trans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlst::{
        base_types::{c32, c64},
        RlstScalar,
    };

    fn check<Item, Make>(method: IdMethod<Item::Real>, make: Make, tolerance: Item::Real)
    where
        Item: MatrixId,
        Make: Fn(f64, f64) -> Item,
    {
        let features = 5;
        let candidates = 6;
        let rank = 2;
        let mut matrix = DynArray::<Item, 2>::from_shape([features, candidates]);
        let basis = [
            [make(1.0, 0.5), make(-0.3, 0.2)],
            [make(0.2, -0.7), make(1.1, 0.1)],
            [make(-0.8, 0.4), make(0.6, -0.9)],
            [make(0.5, 0.3), make(-1.2, 0.2)],
            [make(1.3, -0.2), make(0.4, 0.8)],
        ];
        let coefficients = [
            [make(1.0, 0.0), make(0.0, 0.0)],
            [make(0.0, 0.0), make(1.0, 0.0)],
            [make(0.4, 0.3), make(-0.2, 0.1)],
            [make(-0.6, 0.2), make(0.7, -0.4)],
            [make(0.1, -0.5), make(0.8, 0.2)],
            [make(0.9, 0.1), make(0.3, -0.6)],
        ];
        for feature in 0..features {
            for candidate in 0..candidates {
                matrix[[feature, candidate]] = basis[feature][0] * coefficients[candidate][0]
                    + basis[feature][1] * coefficients[candidate][1];
            }
        }
        let id = compute_candidate_id(&matrix, IdRankSelection::Fixed(rank), method, true).unwrap();
        assert_eq!(id.rank, rank);
        assert_eq!(id.skel.shape(), [rank, features]);
        for ordered in 0..candidates {
            let original = id.perm[ordered];
            for feature in 0..features {
                let reconstructed = if ordered < rank {
                    id.skel[[ordered, feature]]
                } else {
                    (0..rank)
                        .map(|skeleton| {
                            id.skel[[skeleton, feature]]
                                * id.id_mat[[ordered - rank, skeleton]].conj()
                        })
                        .fold(Item::zero(), |sum, value| sum + value)
                };
                assert!((reconstructed - matrix[[feature, original]]).abs() <= tolerance);
            }
        }
    }

    macro_rules! scalar_tests {
        ($rrqr:ident, $srrqr:ident, $item:ty, $make:expr, $tol:expr) => {
            #[test]
            fn $rrqr() {
                check::<$item, _>(IdMethod::Rrqr, $make, $tol);
            }

            #[test]
            fn $srrqr() {
                check::<$item, _>(
                    IdMethod::Srrqr {
                        coefficient_bound: <$item>::real(2.0),
                    },
                    $make,
                    $tol,
                );
            }
        };
    }

    scalar_tests!(rrqr_f32, srrqr_f32, f32, |re, _| re as f32, 2.0e-5);
    scalar_tests!(rrqr_f64, srrqr_f64, f64, |re, _| re, 2.0e-12);
    scalar_tests!(
        rrqr_c32,
        srrqr_c32,
        c32,
        |re, im| c32::new(re as f32, im as f32),
        5.0e-5
    );
    scalar_tests!(rrqr_c64, srrqr_c64, c64, c64::new, 5.0e-12);
}
