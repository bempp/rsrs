use std::{rc::Rc, time::Instant};

use crate::rsrs::{
    rsrs_factors::{
        base_factors::{BaseFactorOptions, CondType},
        commutative_factors::{
            CommutativeFactorsOperations, DiagBoxFactors, FactorType, LevelIdFactors, MulOptions,
            PermFactor, RsrsFactors,
        },
    },
    sketch::SamplingSpace,
};
use mpi::{
    topology::SimpleCommunicator,
    traits::{Communicator, Equivalence},
};
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::{
    dense::{linalg::lu::MatrixLu, tools::RandScalar},
    prelude::*,
};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub enum FactType {
    Joint,
    Split,
}

#[derive(PartialEq)]
pub enum RsrsApply {
    Sandwich,
    Left(FactorType),
    Right(FactorType),
}

impl RsrsApply {
    fn get_level_factors_mult(&self, base_options: BaseFactorOptions) -> MulOptions {
        match self {
            RsrsApply::Sandwich => {
                MulOptions {
                    side: Side::Left,
                    factor_type: FactorType::F,
                    base_options, // Originally trans_target = false
                }
            }
            RsrsApply::Left(factor_type) => MulOptions {
                side: Side::Left,
                factor_type: factor_type.clone(),
                base_options: base_options,
            },
            RsrsApply::Right(factor_type) => MulOptions {
                side: Side::Right,
                factor_type: factor_type.clone(),
                base_options,
            },
        }
    }
}

pub struct LevelFactorsMult {
    pub side: Side,
    pub factor_type: FactorType,
    pub trans_target: bool,
}
pub trait RsrsFactorsImpl<Item: RlstScalar>: Sized {
    fn new(
        num_levels: usize,
        dim: usize,
        factorisation_type: &FactType,
        num_threads: usize,
    ) -> Self;

    fn apply_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        level_options: &MulOptions,
        dec: bool,
        level_it: usize,
    ) -> (u128, u128);

    fn apply_id_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        factor_options: &MulOptions,
        level_it: usize,
    );

    fn apply_lu_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        factor_options: &MulOptions,
        dec: bool,
        level_it: usize,
    );

    fn el_factors_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        mul_type: RsrsApply,
        base_options: &BaseFactorOptions,
        dec: bool,
    );

    fn matmul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &mut self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        side: Side,
        base_options: &BaseFactorOptions, //inv: bool,
                                          //trans_target: bool,
    );

    fn matvec(&self, x: &[Item], y: &mut [Item], side: Side, base_options: &BaseFactorOptions);

    fn perm_target_array<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
    );

    fn dim(&self) -> usize;

    #[allow(clippy::type_complexity)]
    fn get_condition_numbers(
        &self,
    ) -> (
        Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
        Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
        Vec<(CondType<Item>, Option<CondType<Item>>)>,
    );

    fn get_factors(&self) -> &RsrsFactors<Item>;
}

impl<
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
    > RsrsFactorsImpl<Item> for RsrsFactors<Item>
where
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    fn new(num_levels: usize, dim: usize, fact_type: &FactType, num_threads: usize) -> Self {
        let id_factors = match fact_type {
            FactType::Joint => {
                let mut factors: LevelIdFactors<Item> = LevelIdFactors::Batched(Vec::new());
                if let LevelIdFactors::Batched(ref mut v) = factors {
                    v.resize_with(num_levels, Vec::new);
                }
                factors
            }
            FactType::Split => {
                let mut factors: LevelIdFactors<Item> = LevelIdFactors::Single(Vec::new());
                if let LevelIdFactors::Single(ref mut v) = factors {
                    v.resize_with(num_levels, Vec::new);
                }
                factors
            }
        };

        let mut lu_factors = Vec::new();
        lu_factors.resize_with(num_levels, Vec::new);
        let mut near_field_inds = Vec::new();
        near_field_inds.resize_with(num_levels, Vec::new);
        let orig_indices = Vec::new();
        let perm_indices = Vec::new();
        let perm_factor = PermFactor::new(orig_indices, perm_indices).unwrap();
        let diag_box_factors = DiagBoxFactors::new();
        Self {
            num_levels,
            near_field_inds,
            id_factors,
            lu_factors,
            perm_factor,
            diag_box_factors,
            dim,
            fact_type: fact_type.clone(),
            num_threads,
        }
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn apply_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        level_options: &MulOptions,
        dec: bool,
        level_it: usize,
    ) -> (u128, u128) {
        let mut id_time = 0;
        let mut lu_time = 0;
        match &self.id_factors {
            LevelIdFactors::Single(_id_batches) => panic!("Apply level is only for joint steps"),
            LevelIdFactors::Batched(batched_factors) => {
                if let Some(id_batches) = batched_factors.get(level_it) {
                    let num_id_batches = id_batches.len();
                    if dec {
                        (0..num_id_batches).rev().for_each(|batch_ind| {
                            let start = Instant::now();
                            let lu_batch = &self.lu_factors[level_it][batch_ind];
                            lu_batch.mul(target_arr, self.num_threads, level_options);
                            lu_time += start.elapsed().as_millis();

                            let start = Instant::now();
                            let id_batch = &id_batches[batch_ind];
                            id_batch.mul(target_arr, self.num_threads, level_options);
                            id_time += start.elapsed().as_millis();
                        });
                    } else {
                        (0..num_id_batches).for_each(|batch_ind| {
                            let start = Instant::now();
                            let id_batch = &id_batches[batch_ind];
                            id_batch.mul(target_arr, self.num_threads, level_options);
                            id_time += start.elapsed().as_millis();

                            let start = Instant::now();
                            let lu_batch = &self.lu_factors[level_it][batch_ind];
                            lu_batch.mul(target_arr, self.num_threads, level_options);
                            lu_time += start.elapsed().as_millis();
                        });
                    }
                }
            }
        }
        (id_time, lu_time)
    }

    fn apply_id_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        factor_options: &MulOptions,
        level_it: usize,
    ) {
        match &self.id_factors {
            LevelIdFactors::Single(id_batches) => {
                if let Some(id_batch) = id_batches.get(level_it) {
                    id_batch.mul(target_arr, self.num_threads, factor_options);
                }
            }
            LevelIdFactors::Batched(_batched_factors) => {
                panic!("Apply ID level is only for split steps")
            }
        }
    }

    fn apply_lu_level<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        factor_options: &MulOptions,
        dec: bool,
        level_it: usize,
    ) {
        let num_lu_batches = self.lu_factors[level_it].len();

        if dec {
            (0..num_lu_batches).rev().for_each(|batch_ind| {
                let lu_batch = &self.lu_factors[level_it][batch_ind];
                lu_batch.mul(target_arr, self.num_threads, factor_options);
            });
        } else {
            (0..num_lu_batches).for_each(|batch_ind| {
                let lu_batch = &self.lu_factors[level_it][batch_ind];
                lu_batch.mul(target_arr, self.num_threads, factor_options);
            });
        }
    }

    fn el_factors_mul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        mul_type: RsrsApply,
        base_options: &BaseFactorOptions,
        dec: bool,
    ) {
        let levels = (0..self.num_levels).collect::<Vec<_>>();
        if matches!(mul_type, RsrsApply::Sandwich) {
            let left_options = MulOptions {
                base_options: base_options.clone(),
                side: Side::Left,
                factor_type: FactorType::F,
            };
            let right_options = MulOptions {
                base_options: base_options.clone(),
                side: Side::Right,
                factor_type: FactorType::S,
            };

            match self.fact_type {
                FactType::Joint => levels.iter().for_each(|&level_it| {
                    self.apply_level(target_arr, &left_options, dec, level_it);
                    self.apply_level(target_arr, &right_options, dec, level_it);
                }),
                FactType::Split => levels.iter().for_each(|&level_it| {
                    self.apply_id_level(target_arr, &left_options, level_it);
                    self.apply_id_level(target_arr, &right_options, level_it);
                    self.apply_lu_level(target_arr, &left_options, dec, level_it);
                    self.apply_lu_level(target_arr, &right_options, dec, level_it);
                }),
            }
        } else {
            let mul_options = mul_type.get_level_factors_mult(base_options.clone());
            match self.fact_type {
                FactType::Joint => {
                    if dec {
                        levels.iter().rev().for_each(|&level_it| {
                            self.apply_level(target_arr, &mul_options, dec, level_it);
                        });
                    } else {
                        levels.iter().for_each(|&level_it| {
                            self.apply_level(target_arr, &mul_options, dec, level_it);
                        });
                    }
                }
                FactType::Split => {
                    if dec {
                        levels.iter().rev().for_each(|&level_it| {
                            self.apply_lu_level(target_arr, &mul_options, dec, level_it);
                            self.apply_id_level(target_arr, &mul_options, level_it);
                        });
                    } else {
                        levels.iter().for_each(|&level_it| {
                            self.apply_id_level(target_arr, &mul_options, level_it);
                            self.apply_lu_level(target_arr, &mul_options, dec, level_it);
                        });
                    }
                }
            }
        }
    }

    fn matmul<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>
            + std::marker::Send
            + std::marker::Sync,
    >(
        &mut self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
        side: Side,
        base_options: &BaseFactorOptions,
        //inv: bool,
        //trans_target: bool,
        //factor_options: &MulOptions,
    ) {
        let diag_mul = MulOptions {
            base_options: base_options.clone(),
            side,
            factor_type: FactorType::F, //TODO: CHECK IF CORRECT
        };

        match side {
            Side::Left => {
                let (mul_type_1, mul_type_2) = if !base_options.inv {
                    (
                        RsrsApply::Left(FactorType::S), //, trans_target),
                        RsrsApply::Left(FactorType::F), //, trans_target),
                    )
                } else {
                    (
                        RsrsApply::Left(FactorType::F), //, trans_target),
                        RsrsApply::Left(FactorType::S), //, trans_target),
                    )
                };
                self.el_factors_mul(target_arr, mul_type_1, base_options, false);
                self.diag_box_factors
                    .mul(target_arr, self.num_threads, &diag_mul);
                self.el_factors_mul(target_arr, mul_type_2, base_options, true);
            }
            Side::Right => {
                let (mul_type_1, mul_type_2) = if !base_options.inv {
                    (
                        RsrsApply::Right(FactorType::F), //, trans_target),
                        RsrsApply::Right(FactorType::S), //, trans_target),
                    )
                } else {
                    (
                        RsrsApply::Right(FactorType::S), //, trans_target),
                        RsrsApply::Right(FactorType::F), //, trans_target),
                    )
                };

                self.el_factors_mul(target_arr, mul_type_1, base_options, false);
                self.diag_box_factors
                    .mul(target_arr, self.num_threads, &diag_mul);
                self.el_factors_mul(target_arr, mul_type_2, base_options, true);
            }
        }
    }

    fn matvec(&self, x: &[Item], y: &mut [Item], side: Side, base_options: &BaseFactorOptions) {
        //side: Side, inv: bool, trans_target: bool) {
        /*let diag_mul = LevelFactorsMult {
            side,
            factor_type: FactorType::F,
            trans_target: false, //TODO: CHECK IF CORRECT
        };*/

        let diag_mul = MulOptions {
            base_options: base_options.clone(),
            side,
            factor_type: FactorType::F, //TODO: CHECK IF CORRECT
        };

        let target_arr = match side {
            Side::Left => {
                let mut target_arr = rlst_dynamic_array2!(Item, [x.len(), 1]);
                for (i, val) in x.iter().enumerate() {
                    target_arr.r_mut()[[i, 0]] = *val;
                }

                let (mul_type_1, mul_type_2) = if !base_options.inv {
                    (
                        RsrsApply::Left(FactorType::S), //, false),
                        RsrsApply::Left(FactorType::F), //, false),
                    )
                } else {
                    (
                        RsrsApply::Left(FactorType::F), //, false),
                        RsrsApply::Left(FactorType::S), //, false),
                    )
                };

                self.el_factors_mul(&mut target_arr, mul_type_1, base_options, false);
                self.diag_box_factors
                    .mul(&mut target_arr, self.num_threads, &diag_mul);
                self.el_factors_mul(&mut target_arr, mul_type_2, base_options, true);
                target_arr
            }
            Side::Right => {
                let mut target_arr = rlst_dynamic_array2!(Item, [1, x.len()]);

                for (i, val) in x.iter().enumerate() {
                    target_arr.r_mut()[[0, i]] = *val;
                }

                let (mul_type_1, mul_type_2) = if !base_options.inv {
                    (
                        RsrsApply::Right(FactorType::F), //, false),
                        RsrsApply::Right(FactorType::S), //, false),
                    )
                } else {
                    (
                        RsrsApply::Right(FactorType::S), //, false),
                        RsrsApply::Right(FactorType::F), //, false),
                    )
                };

                self.el_factors_mul(&mut target_arr, mul_type_1, base_options, false);
                self.diag_box_factors
                    .mul(&mut target_arr, self.num_threads, &diag_mul);
                self.el_factors_mul(&mut target_arr, mul_type_2, base_options, true);
                target_arr
            }
        };

        for (i, val) in target_arr.r().iter().enumerate() {
            y[i] = val;
        }
    }

    fn perm_target_array<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + UnsafeRandomAccessByRef<2, Item = Item>,
    >(
        &self,
        target_arr: &mut Array<Item, ArrayImplMut, 2>,
    ) {
        let mul_options = BaseFactorOptions {
            inv: false,
            trans: TransMode::NoTrans,
            trans_target: false,
        };
        self.perm_factor.left_mul(target_arr, &mul_options);
        let mut aux_arr = empty_array();
        aux_arr.r_mut().fill_from_resize(target_arr.r().transpose());
        self.perm_factor.left_mul(&mut aux_arr, &mul_options);
        target_arr.r_mut().fill_from(aux_arr.r().transpose());
    }

    fn get_condition_numbers(
        &self,
    ) -> (
        Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
        Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
        Vec<(CondType<Item>, Option<CondType<Item>>)>,
    ) {
        let mut id_condition_numbers = Vec::new();
        let mut lu_condition_numbers = Vec::new();

        match &self.id_factors {
            LevelIdFactors::Single(id_batches) => {
                for id_batch in id_batches.iter() {
                    id_condition_numbers.push(id_batch.get_condition_numbers());
                }
            }
            LevelIdFactors::Batched(batched_factors) => {
                for batch in batched_factors.iter() {
                    for id_batch in batch.iter() {
                        id_condition_numbers.push(id_batch.get_condition_numbers());
                    }
                }
            }
        }

        for lu_level_batches in self.lu_factors.iter() {
            let mut lu_level_condition_numbers = Vec::new();
            for lu_batch in lu_level_batches.iter() {
                lu_level_condition_numbers.extend_from_slice(&lu_batch.get_condition_numbers());
            }
            lu_condition_numbers.push(lu_level_condition_numbers);
        }

        let diag_condition_numbers = self.diag_box_factors.get_condition_numbers();

        (
            id_condition_numbers,
            lu_condition_numbers,
            diag_condition_numbers,
        )
    }

    fn get_factors(&self) -> &Self {
        self
    }
}

impl<Item: RlstScalar> Shape<2> for RsrsFactors<Item> {
    fn shape(&self) -> [usize; 2] {
        [self.dim, self.dim]
    }
}

pub struct RsrsOperator<
    'a,
    Item: RlstScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr,
    Space: SamplingSpace<F = Item>,
    Op: RsrsFactorsImpl<Item> + Shape<2>,
> {
    pub op: &'a Op,
    domain: Rc<Space>,
    range: Rc<Space>,
    inv: bool,
}

impl<
        'a,
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Space: SamplingSpace<F = Item> + LinearSpace,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > RsrsOperator<'a, Item, Space, Op>
{
    pub fn get_factors(&self) -> &RsrsFactors<Item> {
        self.op.get_factors()
    }

    #[allow(clippy::type_complexity)]
    pub fn get_condition_numbers(
        &self,
    ) -> (
        Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
        Vec<Vec<(CondType<Item>, Option<CondType<Item>>)>>,
        Vec<(CondType<Item>, Option<CondType<Item>>)>,
    ) {
        self.op.get_condition_numbers()
    }
}

// Implement OperatorBase for RsrsOperator so it can be used with rlst::Operator
impl<
        'a,
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Space: SamplingSpace<F = Item> + LinearSpace,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > OperatorBase for RsrsOperator<'a, Item, Space, Op>
{
    type Domain = Space;
    type Range = Space;

    fn domain(&self) -> Rc<Self::Domain> {
        self.domain.clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.range.clone()
    }
}

impl<
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Space: SamplingSpace<F = Item>,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > std::fmt::Debug for RsrsOperator<'_, Item, Space, Op>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.op.shape();
        write!(f, "RsrsOperator: [{}x{}]", shape[0], shape[1]).unwrap();
        Ok(())
    }
}

pub trait LocalFromSpaces<
    'a,
    Item: RlstScalar + MatrixInverse + MatrixId + MatrixPseudoInverse + MatrixLu + RandScalar + MatrixQr,
    Space,
    Op,
>: Sized
{
    fn from_local_spaces(op: &'a Op, domain: Rc<Space>, range: Rc<Space>) -> Self;
}

pub trait Inv {
    fn inv(&mut self, inv: bool);
}

impl<
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Space: SamplingSpace<F = Item>,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > Inv for RsrsOperator<'_, Item, Space, Op>
where
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    fn inv(&mut self, inv: bool) {
        self.inv = inv;
    }
}

impl<
        'a,
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > LocalFromSpaces<'a, Item, ArrayVectorSpace<Item>, Op>
    for RsrsOperator<'a, Item, ArrayVectorSpace<Item>, Op>
where
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    fn from_local_spaces(
        op: &'a Op,
        domain: Rc<ArrayVectorSpace<Item>>,
        range: Rc<ArrayVectorSpace<Item>>,
    ) -> Self {
        RsrsOperator {
            op,
            domain: domain.clone(),
            range: range.clone(),
            inv: false,
        }
    }
}

impl<
        'a,
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr
            + Equivalence,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > LocalFromSpaces<'a, Item, DistributedArrayVectorSpace<'a, SimpleCommunicator, Item>, Op>
    for RsrsOperator<'a, Item, DistributedArrayVectorSpace<'a, SimpleCommunicator, Item>, Op>
where
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
    <Item as rlst::RlstScalar>::Real: RandScalar,
{
    fn from_local_spaces(
        op: &'a Op,
        domain: Rc<DistributedArrayVectorSpace<'a, SimpleCommunicator, Item>>,
        range: Rc<DistributedArrayVectorSpace<'a, SimpleCommunicator, Item>>,
    ) -> Self {
        RsrsOperator {
            op,
            domain: domain.clone(),
            range: range.clone(),
            inv: false,
        }
    }
}

impl<
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > AsApply for RsrsOperator<'_, Item, ArrayVectorSpace<Item>, Op>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
{
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        _alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        _beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
        trans_mode: TransMode,
    ) {
        let base_options = BaseFactorOptions {
            inv: self.inv,
            trans: TransMode::NoTrans,
            trans_target: false,
        };
        match trans_mode {
            TransMode::NoTrans => {
                // Reshape y to a 2D array before passing to mul
                self.op.matvec(
                    x.imp().view().data(),
                    y.imp_mut().view_mut().data_mut(),
                    Side::Left,
                    &base_options,
                );
            }
            TransMode::ConjNoTrans => {
                panic!("TransMode::ConjNoTrans not supported for multiplication.")
            }
            TransMode::Trans => {
                self.op.matvec(
                    x.imp().view().data(),
                    y.imp_mut().view_mut().data_mut(),
                    Side::Right,
                    &base_options,
                );
            }
            TransMode::ConjTrans => {
                panic!("TransMode::ConjTrans not supported for multiplication.")
            }
        }
    }

    fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
        &self,
        x: Element<ContainerIn>,
        trans_mode: rlst::TransMode,
    ) -> rlst::operator::ElementType<<Self::Range as LinearSpace>::E> {
        let mut y = zero_element(self.range());
        self.apply_extended(
            <<Self::Range as LinearSpace>::F as num::One>::one(),
            x,
            <<Self::Range as LinearSpace>::F as num::Zero>::zero(),
            y.r_mut(),
            trans_mode,
        );
        y
    }
}

impl<
        C: Communicator,
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr
            + Equivalence,
        Op: RsrsFactorsImpl<Item> + Shape<2>,
    > AsApply for RsrsOperator<'_, Item, DistributedArrayVectorSpace<'_, C, Item>, Op>
where
    <Item as rlst::RlstScalar>::Real: RandScalar,
    StandardNormal: Distribution<<Item as rlst::RlstScalar>::Real>,
    Standard: Distribution<<Item as rlst::RlstScalar>::Real>,
{
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        _alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        _beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
        trans_mode: TransMode,
    ) {
        let base_options = BaseFactorOptions {
            inv: self.inv,
            trans: TransMode::NoTrans,
            trans_target: false,
        };
        match trans_mode {
            TransMode::NoTrans => {
                // Reshape y to a 2D array before passing to mul
                self.op.matvec(
                    x.imp().view().local().data(),
                    y.imp_mut().view_mut().local_mut().data_mut(),
                    Side::Left,
                    &base_options,
                );
            }
            TransMode::ConjNoTrans => {
                panic!("TransMode::ConjNoTrans not supported for multiplication.")
            }
            TransMode::Trans => {
                self.op.matvec(
                    x.imp().view().local().data(),
                    y.imp_mut().view_mut().local_mut().data_mut(),
                    Side::Right,
                    &base_options,
                );
            }
            TransMode::ConjTrans => {
                panic!("TransMode::ConjTrans not supported for multiplication.")
            }
        }
    }

    fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
        &self,
        x: Element<ContainerIn>,
        trans_mode: rlst::TransMode,
    ) -> rlst::operator::ElementType<<Self::Range as LinearSpace>::E> {
        let mut y = zero_element(self.range());
        self.apply_extended(
            <<Self::Range as LinearSpace>::F as num::One>::one(),
            x,
            <<Self::Range as LinearSpace>::F as num::Zero>::zero(),
            y.r_mut(),
            trans_mode,
        );
        y
    }
}
