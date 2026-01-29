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
