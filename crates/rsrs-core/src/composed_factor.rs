use crate::{
    diagonal_block::SquareBlock,
    elementary_factor::{
        commit_left_update, commit_right_update, FactorMode, IndexedLeftUpdate, IndexedRightUpdate,
    },
};
use rlst::{base_types::RlstResult, dense::array::DynArray, Lapack};
use std::collections::HashSet;

/// An indexed elementary factor whose update is retained as `P^-1 R`.
///
/// Keeping `P` factorized avoids explicitly forming its inverse. The global
/// factor is still `I + P_target (P^-1 R) P_source^T`.
pub struct ComposedElementaryFactor<Item: Lapack> {
    dimension: usize,
    source_indices: Vec<usize>,
    target_indices: Vec<usize>,
    pivot: Option<SquareBlock<Item>>,
    rectangular: DynArray<Item, 2>,
}

impl<Item: Lapack> ComposedElementaryFactor<Item> {
    pub fn new(
        dimension: usize,
        source_indices: Vec<usize>,
        target_indices: Vec<usize>,
        pivot: SquareBlock<Item>,
        rectangular: DynArray<Item, 2>,
    ) -> Result<Self, String> {
        if pivot.dimension() != target_indices.len() {
            return Err("pivot dimension must equal the target-index count".into());
        }
        if rectangular.shape() != [target_indices.len(), source_indices.len()] {
            return Err("rectangular block shape must be [targets, sources]".into());
        }
        if source_indices
            .iter()
            .chain(&target_indices)
            .any(|&i| i >= dimension)
        {
            return Err("factor index is outside the ambient dimension".into());
        }
        let source: HashSet<_> = source_indices.iter().copied().collect();
        let target: HashSet<_> = target_indices.iter().copied().collect();
        if source.len() != source_indices.len() || target.len() != target_indices.len() {
            return Err("factor indices must not contain duplicates".into());
        }
        if !source.is_disjoint(&target) {
            return Err("source and target indices must be disjoint".into());
        }
        Ok(Self {
            dimension,
            source_indices,
            target_indices,
            pivot: Some(pivot),
            rectangular,
        })
    }

    /// Store only the already-solved rectangular update. This is the
    /// historical LuHybrid representation: the temporary pivot is discarded
    /// after `pivot^-1 * rectangular` has been formed.
    pub fn from_materialized_update(
        dimension: usize,
        source_indices: Vec<usize>,
        target_indices: Vec<usize>,
        update: DynArray<Item, 2>,
    ) -> Result<Self, String> {
        if update.shape() != [target_indices.len(), source_indices.len()] {
            return Err("materialized update shape must be [targets, sources]".into());
        }
        if source_indices
            .iter()
            .chain(&target_indices)
            .any(|&i| i >= dimension)
        {
            return Err("factor index is outside the ambient dimension".into());
        }
        let source: HashSet<_> = source_indices.iter().copied().collect();
        let target: HashSet<_> = target_indices.iter().copied().collect();
        if source.len() != source_indices.len() || target.len() != target_indices.len() {
            return Err("factor indices must not contain duplicates".into());
        }
        if !source.is_disjoint(&target) {
            return Err("source and target indices must be disjoint".into());
        }
        Ok(Self {
            dimension,
            source_indices,
            target_indices,
            pivot: None,
            rectangular: update,
        })
    }

    fn rectangular_left(&self, input: &DynArray<Item, 2>, mode: FactorMode) -> DynArray<Item, 2> {
        let output_rows = match mode {
            FactorMode::NoTrans | FactorMode::Conjugate => self.rectangular.shape()[0],
            FactorMode::Transpose | FactorMode::Adjoint => self.rectangular.shape()[1],
        };
        let inner = match mode {
            FactorMode::NoTrans | FactorMode::Conjugate => self.rectangular.shape()[1],
            FactorMode::Transpose | FactorMode::Adjoint => self.rectangular.shape()[0],
        };
        assert_eq!(input.shape()[0], inner);
        let mut result = DynArray::<Item, 2>::from_shape([output_rows, input.shape()[1]]);
        for row in 0..output_rows {
            for column in 0..input.shape()[1] {
                let mut value = Item::zero();
                for index in 0..inner {
                    let coefficient = match mode {
                        FactorMode::NoTrans => self.rectangular[[row, index]],
                        FactorMode::Transpose => self.rectangular[[index, row]],
                        FactorMode::Adjoint => self.rectangular[[index, row]].conj(),
                        FactorMode::Conjugate => self.rectangular[[row, index]].conj(),
                    };
                    value += coefficient * input[[index, column]];
                }
                result[[row, column]] = value;
            }
        }
        result
    }

    fn left_update(
        &self,
        local: &mut DynArray<Item, 2>,
        mode: FactorMode,
    ) -> RlstResult<DynArray<Item, 2>> {
        if self.pivot.is_none() {
            return Ok(self.rectangular_left(local, mode));
        }
        let pivot = self.pivot.as_ref().expect("checked above");
        match mode {
            FactorMode::NoTrans => {
                let mut result = self.rectangular_left(local, FactorMode::NoTrans);
                pivot.apply_left_in_place(&mut result, FactorMode::NoTrans, true)?;
                Ok(result)
            }
            FactorMode::Transpose => {
                pivot.apply_left_in_place(local, FactorMode::Transpose, true)?;
                Ok(self.rectangular_left(local, FactorMode::Transpose))
            }
            FactorMode::Adjoint => {
                pivot.apply_left_in_place(local, FactorMode::Adjoint, true)?;
                Ok(self.rectangular_left(local, FactorMode::Adjoint))
            }
            FactorMode::Conjugate => {
                let mut result = self.rectangular_left(local, FactorMode::Conjugate);
                pivot.apply_left_in_place(&mut result, FactorMode::Conjugate, true)?;
                Ok(result)
            }
        }
    }

    fn source_and_destination(&self, mode: FactorMode) -> (&[usize], &[usize]) {
        match mode {
            FactorMode::NoTrans | FactorMode::Conjugate => {
                (&self.source_indices, &self.target_indices)
            }
            FactorMode::Transpose | FactorMode::Adjoint => {
                (&self.target_indices, &self.source_indices)
            }
        }
    }

    pub fn apply_left_in_place(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<()> {
        let update = self.indexed_left_update(target, mode, inverse)?;
        commit_left_update(target, update);
        Ok(())
    }

    pub(crate) fn indexed_left_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<IndexedLeftUpdate<Item>> {
        assert_eq!(target.shape()[0], self.dimension);
        let (source, destination) = self.source_and_destination(mode);
        let mut local = DynArray::<Item, 2>::from_shape([source.len(), target.shape()[1]]);
        for (local_row, &global_row) in source.iter().enumerate() {
            for column in 0..target.shape()[1] {
                local[[local_row, column]] = target[[global_row, column]];
            }
        }

        let delta = self.left_update(&mut local, mode)?;

        let mut delta = delta;
        if inverse {
            for value in delta
                .data_mut()
                .expect("composed update is contiguous")
                .iter_mut()
            {
                *value = -*value;
            }
        }
        Ok(IndexedLeftUpdate {
            indices: destination.to_vec(),
            values: delta,
        })
    }

    pub fn apply_right_in_place(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<()> {
        let update = self.indexed_right_update(target, mode, inverse)?;
        commit_right_update(target, update);
        Ok(())
    }

    pub(crate) fn indexed_right_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<IndexedRightUpdate<Item>> {
        assert_eq!(target.shape()[1], self.dimension);
        let rows = target.shape()[0];
        let left_mode = match mode {
            FactorMode::NoTrans => FactorMode::Transpose,
            FactorMode::Transpose => FactorMode::NoTrans,
            FactorMode::Adjoint => FactorMode::Conjugate,
            FactorMode::Conjugate => FactorMode::Adjoint,
        };
        let (source, destination) = self.source_and_destination(left_mode);

        // Applying from the right is equivalent to applying the corresponding
        // transposed mode from the left to target^T.  Materialising target^T
        // used to copy all s*N sample entries for every local elimination.
        // Gather only the indexed rows that the left application would read,
        // then scatter only the indexed rows that it changes.  This is the
        // same local update used by historical joint RSRS and makes eliminated
        // ambient DOFs free in all subsequent sample updates.
        let mut local = DynArray::<Item, 2>::from_shape([source.len(), rows]);
        for (local_row, &global_column) in source.iter().enumerate() {
            for row in 0..rows {
                local[[local_row, row]] = target[[row, global_column]];
            }
        }
        let delta = self.left_update(&mut local, left_mode)?;
        let mut values = DynArray::<Item, 2>::from_shape([rows, destination.len()]);
        for row in 0..rows {
            for local_column in 0..destination.len() {
                values[[row, local_column]] = if inverse {
                    -delta[[local_column, row]]
                } else {
                    delta[[local_column, row]]
                };
            }
        }
        Ok(IndexedRightUpdate {
            indices: destination.to_vec(),
            values,
        })
    }

    /// Materialize only the small effective update, primarily for diagnostics.
    pub fn materialize_update(&self) -> RlstResult<DynArray<Item, 2>> {
        let mut update = DynArray::<Item, 2>::from_shape(self.rectangular.shape());
        update.fill_from(&self.rectangular);
        if let Some(pivot) = &self.pivot {
            pivot.apply_left_in_place(&mut update, FactorMode::NoTrans, true)?;
        }
        Ok(update)
    }

    pub(crate) fn storage_bytes(&self) -> u64 {
        let rectangular =
            self.rectangular.shape()[0] * self.rectangular.shape()[1] * std::mem::size_of::<Item>();
        let indices =
            (self.source_indices.len() + self.target_indices.len()) * std::mem::size_of::<usize>();
        (rectangular + indices) as u64 + self.pivot.as_ref().map_or(0, SquareBlock::storage_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elementary_factor::ElementaryFactor;
    use rlst::{base_types::c64, RlstScalar};

    fn copy<Item: RlstScalar>(source: &DynArray<Item, 2>) -> DynArray<Item, 2> {
        let mut result = DynArray::<Item, 2>::from_shape(source.shape());
        result.fill_from(source);
        result
    }

    fn check<Item, Make>(make: Make, tolerance: Item::Real)
    where
        Item: Lapack,
        Make: Fn(f64, f64) -> Item,
    {
        let mut pivot = DynArray::<Item, 2>::from_shape([2, 2]);
        pivot[[0, 0]] = make(2.0, 0.2);
        pivot[[0, 1]] = make(-0.4, 0.1);
        pivot[[1, 0]] = make(0.3, -0.5);
        pivot[[1, 1]] = make(1.7, 0.3);
        let mut rectangular = DynArray::<Item, 2>::from_shape([2, 2]);
        rectangular[[0, 0]] = make(0.5, 0.2);
        rectangular[[0, 1]] = make(-0.6, 0.1);
        rectangular[[1, 0]] = make(0.2, -0.7);
        rectangular[[1, 1]] = make(0.8, 0.4);
        let composed = ComposedElementaryFactor::new(
            5,
            vec![0, 3],
            vec![1, 4],
            SquareBlock::lu(&pivot).unwrap(),
            rectangular,
        )
        .unwrap();
        let update = composed.materialize_update().unwrap();
        let materialized = ComposedElementaryFactor::from_materialized_update(
            5,
            vec![0, 3],
            vec![1, 4],
            copy(&update),
        )
        .unwrap();
        let explicit = ElementaryFactor::new(5, vec![0, 3], vec![1, 4], update).unwrap();

        let mut left = DynArray::<Item, 2>::from_shape([5, 3]);
        let mut right = DynArray::<Item, 2>::from_shape([3, 5]);
        for i in 0..5 {
            for j in 0..3 {
                left[[i, j]] = make((i + 2 * j + 1) as f64 / 8.0, (2 * i + j) as f64 / 13.0);
                right[[j, i]] = make((2 * i + j + 2) as f64 / 9.0, (i + 3 * j) as f64 / 17.0);
            }
        }
        for mode in [
            FactorMode::NoTrans,
            FactorMode::Transpose,
            FactorMode::Adjoint,
            FactorMode::Conjugate,
        ] {
            for inverse in [false, true] {
                let mut expected_left = copy(&left);
                explicit.apply_left_in_place(&mut expected_left, mode, inverse);
                let mut actual_left = copy(&left);
                composed
                    .apply_left_in_place(&mut actual_left, mode, inverse)
                    .unwrap();
                let mut materialized_left = copy(&left);
                materialized
                    .apply_left_in_place(&mut materialized_left, mode, inverse)
                    .unwrap();
                for i in 0..5 {
                    for j in 0..3 {
                        assert!(
                            RlstScalar::abs(actual_left[[i, j]] - expected_left[[i, j]])
                                <= tolerance
                        );
                        assert!(
                            RlstScalar::abs(materialized_left[[i, j]] - expected_left[[i, j]])
                                <= tolerance
                        );
                    }
                }

                let mut expected_right = copy(&right);
                explicit.apply_right_in_place(&mut expected_right, mode, inverse);
                let mut actual_right = copy(&right);
                composed
                    .apply_right_in_place(&mut actual_right, mode, inverse)
                    .unwrap();
                let mut materialized_right = copy(&right);
                materialized
                    .apply_right_in_place(&mut materialized_right, mode, inverse)
                    .unwrap();
                for i in 0..3 {
                    for j in 0..5 {
                        assert!(
                            RlstScalar::abs(actual_right[[i, j]] - expected_right[[i, j]])
                                <= tolerance
                        );
                        assert!(
                            RlstScalar::abs(materialized_right[[i, j]] - expected_right[[i, j]])
                                <= tolerance
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn real_composed_factor_matches_materialized_update() {
        check::<f64, _>(|real, _| real, 3.0e-12);
    }

    #[test]
    fn complex_composed_factor_matches_materialized_update() {
        check::<c64, _>(c64::new, 4.0e-12);
    }
}
