//! Typed configuration for the dense-SRS and sampled-RSRS drivers.

use serde::{Deserialize, Serialize};

use crate::{
    regularized_least_squares::RegularizationMode,
    scheduling::{BatchingStrategy, FixedRankSamplingMode},
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RankSelection {
    Fixed { rank: usize },
    Tolerance { tolerance: f64 },
    ToleranceWithMinimumRank { tolerance: f64, minimum_rank: usize },
    ToleranceWithMaximumRank { tolerance: f64, maximum_rank: usize },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParentRankPolicy {
    #[default]
    Minimum,
    DoubleMinimum,
    Maximum,
    Average,
    Midpoint,
    Tolerance,
}

impl Default for RankSelection {
    fn default() -> Self {
        Self::Tolerance { tolerance: 1.0e-6 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdAlgorithm {
    Rrqr,
    Srrqr,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeastSquaresAlgorithm {
    Lu,
    Cholesky,
    Svd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullSpaceAlgorithm {
    Projection,
    Qr,
    Svd,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RegularizationConfiguration {
    pub mode: RegularizationMode,
    pub relative: f64,
    pub absolute_floor: f64,
}

impl Default for RegularizationConfiguration {
    fn default() -> Self {
        Self {
            mode: RegularizationMode::Scaled,
            relative: 1.0e-16,
            absolute_floor: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NullSpaceConfiguration {
    pub algorithm: NullSpaceAlgorithm,
    pub tolerance: f64,
    /// Used only by the projection method.
    pub projection_solver: LeastSquaresAlgorithm,
    /// Used only by the projection method.
    pub regularization: RegularizationConfiguration,
}

impl Default for NullSpaceConfiguration {
    fn default() -> Self {
        Self {
            algorithm: NullSpaceAlgorithm::Projection,
            tolerance: 1.0e-16,
            projection_solver: LeastSquaresAlgorithm::Lu,
            regularization: RegularizationConfiguration {
                mode: RegularizationMode::Historical,
                relative: 1.0e-16,
                absolute_floor: 0.0,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractionConfiguration {
    pub solver: LeastSquaresAlgorithm,
    pub regularization: RegularizationConfiguration,
}

impl Default for ExtractionConfiguration {
    fn default() -> Self {
        Self {
            solver: LeastSquaresAlgorithm::Lu,
            regularization: RegularizationConfiguration {
                mode: RegularizationMode::Historical,
                relative: 1.0e-16,
                absolute_floor: 0.0,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PivotConfiguration {
    pub method: PivotMethod,
    /// Diagonal shift added before pivoted LU. Zero means no regularization.
    pub shift: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PivotMethod {
    DirectInverse,
    Lu,
    LuHybrid,
}

impl Default for PivotConfiguration {
    fn default() -> Self {
        Self {
            method: PivotMethod::LuHybrid,
            shift: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatrixSymmetry {
    General,
    Symmetric,
    Hermitian,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TreeConfiguration {
    pub maximum_depth: usize,
    /// `None` preserves the historical default of 50 points per leaf.
    pub leaf_size: Option<usize>,
    pub minimum_level: usize,
}

impl Default for TreeConfiguration {
    fn default() -> Self {
        Self {
            maximum_depth: 16,
            leaf_size: None,
            minimum_level: 1,
        }
    }
}

impl TreeConfiguration {
    pub fn effective_leaf_size(&self) -> usize {
        self.leaf_size.unwrap_or(50)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingConfiguration {
    pub rows: Option<usize>,
    /// Global floor for the fixed-rank sample budget. If it exceeds the
    /// predicted per-level budget, every cycle uses the same full prefix.
    pub minimum_rows: usize,
    pub oversampling: usize,
    pub diagonal_oversampling: usize,
    pub validate_before_factorization: bool,
    pub fail_if_insufficient: bool,
    /// Historical fixed-rank runs use the smallest safe prefix for each box.
    pub fixed_rank_mode: FixedRankSamplingMode,
}

impl Default for SamplingConfiguration {
    fn default() -> Self {
        Self {
            rows: None,
            minimum_rows: 0,
            oversampling: 8,
            diagonal_oversampling: 16,
            validate_before_factorization: true,
            fail_if_insufficient: true,
            fixed_rank_mode: FixedRankSamplingMode::PerLevel,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThreadConfiguration {
    pub rayon_threads: usize,
    pub blas_threads: usize,
}

impl Default for ThreadConfiguration {
    fn default() -> Self {
        Self {
            rayon_threads: 1,
            blas_threads: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RsrsConfiguration {
    pub tree: TreeConfiguration,
    pub sampling: SamplingConfiguration,
    pub threads: ThreadConfiguration,
    pub rank: RankSelection,
    pub parent_rank_policy: ParentRankPolicy,
    pub id_algorithm: IdAlgorithm,
    pub srrqr_coefficient_bound: f64,
    pub symmetry: MatrixSymmetry,
    /// How independent joint cycles are grouped. The default reproduces the
    /// historical active-box near-field coloring.
    pub batching_strategy: BatchingStrategy,
    pub null_space: NullSpaceConfiguration,
    pub near_extraction: ExtractionConfiguration,
    pub diagonal_extraction: ExtractionConfiguration,
    pub near_pivot: PivotConfiguration,
    pub diagonal_pivot: PivotConfiguration,
}

impl Default for RsrsConfiguration {
    fn default() -> Self {
        Self {
            tree: TreeConfiguration::default(),
            sampling: SamplingConfiguration::default(),
            threads: ThreadConfiguration::default(),
            rank: RankSelection::default(),
            parent_rank_policy: ParentRankPolicy::default(),
            id_algorithm: IdAlgorithm::Rrqr,
            srrqr_coefficient_bound: 2.0,
            symmetry: MatrixSymmetry::General,
            batching_strategy: BatchingStrategy::HistoricalActiveNearFields,
            null_space: NullSpaceConfiguration::default(),
            near_extraction: ExtractionConfiguration::default(),
            diagonal_extraction: ExtractionConfiguration::default(),
            near_pivot: PivotConfiguration::default(),
            diagonal_pivot: PivotConfiguration::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_historical_leaf_size_and_regularization() {
        let options = RsrsConfiguration::default();
        assert_eq!(options.tree.effective_leaf_size(), 50);
        assert_eq!(options.near_extraction.regularization.relative, 1.0e-16);
        assert_eq!(options.near_extraction.solver, LeastSquaresAlgorithm::Lu);
        assert!(options.sampling.fail_if_insufficient);
        assert_eq!(
            options.sampling.fixed_rank_mode,
            FixedRankSamplingMode::PerLevel
        );
        assert_eq!(
            options.batching_strategy,
            BatchingStrategy::HistoricalActiveNearFields
        );
    }
}
