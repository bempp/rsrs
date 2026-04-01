use serde::Serialize;

#[derive(Debug, Serialize, Clone)]
pub struct LuTimes {
    pub extraction: u128,
    pub lu: u128,
}

#[derive(Debug, Serialize, Clone)]
pub struct LevelEffort {
    pub time: u128,
    pub num_boxes: usize,
    pub num_batches: usize,
    pub effective_dofs: usize,
    pub sketch_len: usize,
    pub residual_len: usize,
}

#[derive(Debug, Serialize, Clone)]
pub struct IdTimes {
    pub nullification: u128,
    pub id: u128,
}

pub enum Times {
    Lu(LuTimes),
    Id(IdTimes),
}

#[derive(Debug, Serialize, Clone)]
pub struct UpdateTimes {
    pub id: u128,
    pub lu: u128,
}

macro_rules! impl_times_operations {
    ($struct_name:ident, $trait_name:ident, $arg_1:ident, $arg_2:ident) => {
        pub trait $trait_name {
            fn new() -> Self;
            fn sum(&mut self, $arg_1: u128, $arg_2: u128);
        }

        impl $trait_name for $struct_name {
            fn new() -> Self {
                Self {
                    $arg_1: 0_u128,
                    $arg_2: 0_u128,
                }
            }

            fn sum(&mut self, $arg_1: u128, $arg_2: u128) {
                self.$arg_1 += $arg_1;
                self.$arg_2 += $arg_2;
            }
        }
    };
}

impl_times_operations!(IdTimes, IdTimesOperations, nullification, id);
impl_times_operations!(LuTimes, LuTimesOperations, extraction, lu);
impl_times_operations!(UpdateTimes, UpdateTimesOperations, id, lu);

#[derive(Debug)]
pub struct LimitingLevel {
    pub level: usize,
    pub num_boxes: usize,
    pub active_points: usize,
    pub elapsed_time: u128,
}

#[derive(Debug)]
pub struct LimitingFactors {
    pub min_samples: usize,
    pub max_level: usize,
    pub limiting_level: LimitingLevel,
    pub leaf_count: usize,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct FactorMemoryStats {
    pub total_bytes: u64,
    pub id_bytes: u64,
    pub lu_bytes: u64,
    pub diag_bytes: u64,
    pub perm_bytes: u64,
    pub id_count: usize,
    pub lu_count: usize,
    pub diag_count: usize,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct MemorySnapshot {
    pub label: String,
    pub rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub baseline_rss_bytes: Option<u64>,
    pub sample_buffer_bytes: u64,
    pub factor_memory: FactorMemoryStats,
    pub accounted_factorization_bytes: u64,
    pub estimated_temporary_runtime_bytes: Option<u64>,
}

#[derive(Debug)]
pub struct Stats {
    pub sampling_time: Vec<u128>,
    pub sampling_extraction_time: u128,
    pub id_times: Vec<IdTimes>,
    pub tot_id_time: u128,
    pub lu_times: Vec<LuTimes>,
    pub tot_lu_time: u128,
    pub update_times: Vec<UpdateTimes>,
    pub total_elapsed_time: u128,
    pub total_elapsed_time_wo_sampling: u128,
    pub dim: usize,
    pub extraction_time: u128,
    pub residual_size: usize,
    pub ranks: Vec<usize>,
    pub box_sizes: Vec<usize>,
    pub near_field_sizes: Vec<usize>,
    pub dec_boxes_per_level: Vec<usize>,
    pub index_calculation: u128,
    pub sorting_near_field: u128,
    pub residual_calculation: u128,
    pub limiting_factors: LimitingFactors,
    pub level_effort: Vec<LevelEffort>,
    pub mv_avg_time: Vec<u128>,
    pub memory_snapshots: Vec<MemorySnapshot>,
    pub run_start_rss_bytes: Option<u64>,
    pub max_sample_buffer_bytes: u64,
    pub max_factor_bytes: u64,
    pub max_accounted_factorization_bytes: u64,
    pub max_estimated_temporary_runtime_bytes: Option<u64>,
}
