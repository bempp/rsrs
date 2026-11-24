//! Stable serializable reports shared by Rust and Python drivers.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CycleReport {
    pub level: usize,
    pub batch: usize,
    pub target_size: usize,
    pub near_size: usize,
    pub skeleton_size: usize,
    pub redundant_size: usize,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LevelReport {
    pub level: usize,
    pub boxes: usize,
    pub batches: usize,
    pub active_dofs_before: usize,
    pub active_dofs_after: usize,
    pub elapsed_seconds: f64,
    #[serde(default)]
    pub maximum_sample_rows_used: usize,
    #[serde(default)]
    pub minimum_rank: usize,
    #[serde(default)]
    pub maximum_rank: usize,
    #[serde(default)]
    pub mean_rank: f64,
    /// Exact retained factor storage after this level (allocator overhead is
    /// intentionally excluded).
    #[serde(default)]
    pub retained_factor_bytes: u64,
    #[serde(default)]
    pub retained_id_bytes: u64,
    #[serde(default)]
    pub retained_elimination_bytes: u64,
    #[serde(default)]
    pub retained_diagonal_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchReport {
    pub level: usize,
    pub batch: usize,
    pub cycles: usize,
    pub active_dofs_before: usize,
    pub active_dofs_after: usize,
    pub eliminated_dofs: usize,
    pub max_sample_rows: usize,
    pub id_seconds: f64,
    pub id_update_seconds: f64,
    pub near_extraction_seconds: f64,
    pub lu_update_seconds: f64,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactorizationReport {
    pub dimension: usize,
    pub available_sample_rows: usize,
    pub used_sample_rows: usize,
    /// Largest prefix actually consumed by any cycle. This differs from
    /// `used_sample_rows`, which records the prefix loaded into memory.
    #[serde(default)]
    pub maximum_sample_rows_used: usize,
    pub required_sample_rows: usize,
    pub loading_seconds: f64,
    pub factorization_seconds: f64,
    pub final_active_dofs: usize,
    #[serde(default)]
    pub sample_buffer_bytes: u64,
    #[serde(default)]
    pub factor_memory: FactorMemoryReport,
    pub cycles: Vec<CycleReport>,
    #[serde(default)]
    pub batches: Vec<BatchReport>,
    pub levels: Vec<LevelReport>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactorMemoryReport {
    pub total_bytes: u64,
    pub id_bytes: u64,
    pub elimination_bytes: u64,
    pub diagonal_bytes: u64,
    pub id_count: usize,
    pub elimination_count: usize,
    pub diagonal_count: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ApplyReport {
    pub mode: String,
    pub inverse: bool,
    pub right_hand_sides: usize,
    pub elapsed_seconds: f64,
}
