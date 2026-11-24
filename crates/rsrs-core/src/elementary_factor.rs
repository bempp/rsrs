use rlst::{dense::array::DynArray, RlstScalar};
use std::collections::HashSet;

/// A sparse row update computed from an immutable factor-application input.
/// Conflict-free cycles may compute these concurrently and commit them after
/// the whole joint batch has finished reading the input state.
pub(crate) struct IndexedLeftUpdate<Item: RlstScalar> {
    pub(crate) indices: Vec<usize>,
    pub(crate) values: DynArray<Item, 2>,
}

/// Column-oriented counterpart of [`IndexedLeftUpdate`].
pub(crate) struct IndexedRightUpdate<Item: RlstScalar> {
    pub(crate) indices: Vec<usize>,
    pub(crate) values: DynArray<Item, 2>,
}

pub(crate) fn commit_left_update<Item: RlstScalar>(
    target: &mut DynArray<Item, 2>,
    update: IndexedLeftUpdate<Item>,
) {
    for (local_row, global_row) in update.indices.into_iter().enumerate() {
        for column in 0..target.shape()[1] {
            target[[global_row, column]] += update.values[[local_row, column]];
        }
    }
}

pub(crate) fn commit_right_update<Item: RlstScalar>(
    target: &mut DynArray<Item, 2>,
    update: IndexedRightUpdate<Item>,
) {
    for row in 0..target.shape()[0] {
        for (local_column, &global_column) in update.indices.iter().enumerate() {
            target[[row, global_column]] += update.values[[row, local_column]];
        }
    }
}

/// Orientation in which an RSRS elementary factor is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorMode {
    NoTrans,
    Transpose,
    Adjoint,
    Conjugate,
}

/// An indexed RSRS elimination factor
///
/// ```text
/// E = I + P_target update P_source^T.
/// ```
///
/// Source and target indices must be disjoint. Consequently the embedded
/// update is nilpotent and `E^-1` is obtained exactly by changing `+` to `-`.
#[derive(Debug)]
pub struct ElementaryFactor<Item: RlstScalar> {
    dimension: usize,
    source_indices: Vec<usize>,
    target_indices: Vec<usize>,
    update: DynArray<Item, 2>,
}

impl<Item: RlstScalar> ElementaryFactor<Item> {
    pub fn new(
        dimension: usize,
        source_indices: Vec<usize>,
        target_indices: Vec<usize>,
        update: DynArray<Item, 2>,
    ) -> Result<Self, String> {
        if update.shape() != [target_indices.len(), source_indices.len()] {
            return Err(format!(
                "update shape {:?} does not match target/source sizes [{}, {}]",
                update.shape(),
                target_indices.len(),
                source_indices.len()
            ));
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
            update,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    pub fn target_indices(&self) -> &[usize] {
        &self.target_indices
    }

    pub fn update(&self) -> &DynArray<Item, 2> {
        &self.update
    }

    pub(crate) fn storage_bytes(&self) -> u64 {
        ((self.update.shape()[0] * self.update.shape()[1] * std::mem::size_of::<Item>())
            + (self.source_indices.len() + self.target_indices.len())
                * std::mem::size_of::<usize>()) as u64
    }

    /// Apply `op(E)` or `op(E^-1)` from the left to every RHS column.
    pub fn apply_left_in_place(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) {
        let update = self.left_update(target, mode, inverse);
        commit_left_update(target, update);
    }

    pub(crate) fn left_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> IndexedLeftUpdate<Item> {
        assert_eq!(target.shape()[0], self.dimension);
        let columns = target.shape()[1];
        let (source, destination) = match mode {
            FactorMode::NoTrans | FactorMode::Conjugate => {
                (&self.source_indices, &self.target_indices)
            }
            FactorMode::Transpose | FactorMode::Adjoint => {
                (&self.target_indices, &self.source_indices)
            }
        };
        let mut delta = DynArray::<Item, 2>::from_shape([destination.len(), columns]);
        for output in 0..destination.len() {
            for column in 0..columns {
                let mut value = Item::zero();
                for input in 0..source.len() {
                    let coefficient = match mode {
                        FactorMode::NoTrans => self.update[[output, input]],
                        FactorMode::Conjugate => self.update[[output, input]].conj(),
                        FactorMode::Transpose => self.update[[input, output]],
                        FactorMode::Adjoint => self.update[[input, output]].conj(),
                    };
                    value += coefficient * target[[source[input], column]];
                }
                delta[[output, column]] = value;
            }
        }
        if inverse {
            for value in delta
                .data_mut()
                .expect("elementary update is contiguous")
                .iter_mut()
            {
                *value = -*value;
            }
        }
        IndexedLeftUpdate {
            indices: destination.clone(),
            values: delta,
        }
    }

    /// Apply `op(E)` or `op(E^-1)` from the right to every matrix row.
    pub fn apply_right_in_place(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) {
        let update = self.right_update(target, mode, inverse);
        commit_right_update(target, update);
    }

    pub(crate) fn right_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> IndexedRightUpdate<Item> {
        assert_eq!(target.shape()[1], self.dimension);
        let rows = target.shape()[0];
        let (source, destination) = match mode {
            FactorMode::NoTrans | FactorMode::Conjugate => {
                (&self.target_indices, &self.source_indices)
            }
            FactorMode::Transpose | FactorMode::Adjoint => {
                (&self.source_indices, &self.target_indices)
            }
        };
        let mut delta = DynArray::<Item, 2>::from_shape([rows, destination.len()]);
        for row in 0..rows {
            for output in 0..destination.len() {
                let mut value = Item::zero();
                for input in 0..source.len() {
                    let coefficient = match mode {
                        FactorMode::NoTrans => self.update[[input, output]],
                        FactorMode::Conjugate => self.update[[input, output]].conj(),
                        FactorMode::Transpose => self.update[[output, input]],
                        FactorMode::Adjoint => self.update[[output, input]].conj(),
                    };
                    value += target[[row, source[input]]] * coefficient;
                }
                delta[[row, output]] = value;
            }
        }
        if inverse {
            for value in delta
                .data_mut()
                .expect("elementary update is contiguous")
                .iter_mut()
            {
                *value = -*value;
            }
        }
        IndexedRightUpdate {
            indices: destination.clone(),
            values: delta,
        }
    }
}

/// A mathematical product of elementary factors in stored order:
/// `factors[0] * factors[1] * ... * factors[n-1]`.
#[derive(Debug)]
pub struct FactorSequence<Item: RlstScalar> {
    dimension: usize,
    factors: Vec<ElementaryFactor<Item>>,
}

impl<Item: RlstScalar> FactorSequence<Item> {
    pub fn new(dimension: usize, factors: Vec<ElementaryFactor<Item>>) -> Result<Self, String> {
        if factors.iter().any(|factor| factor.dimension() != dimension) {
            return Err("all factors must have the sequence dimension".into());
        }
        Ok(Self { dimension, factors })
    }

    pub fn factors(&self) -> &[ElementaryFactor<Item>] {
        &self.factors
    }

    pub fn apply_left_in_place(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) {
        assert_eq!(target.shape()[0], self.dimension);
        // Left multiplication applies the rightmost mathematical factor first.
        // Transposition and inversion each reverse the product order.
        let forward = matches!(mode, FactorMode::Transpose | FactorMode::Adjoint) ^ inverse;
        if forward {
            for factor in &self.factors {
                factor.apply_left_in_place(target, mode, inverse);
            }
        } else {
            for factor in self.factors.iter().rev() {
                factor.apply_left_in_place(target, mode, inverse);
            }
        }
    }

    pub fn apply_right_in_place(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) {
        assert_eq!(target.shape()[1], self.dimension);
        // Right multiplication applies the leftmost mathematical factor first.
        let forward = !(matches!(mode, FactorMode::Transpose | FactorMode::Adjoint) ^ inverse);
        if forward {
            for factor in &self.factors {
                factor.apply_right_in_place(target, mode, inverse);
            }
        } else {
            for factor in self.factors.iter().rev() {
                factor.apply_right_in_place(target, mode, inverse);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlst::base_types::c64;

    fn dense_factor<Item: RlstScalar>(
        factor: &ElementaryFactor<Item>,
        mode: FactorMode,
        inverse: bool,
    ) -> DynArray<Item, 2> {
        let n = factor.dimension();
        let mut dense = DynArray::<Item, 2>::from_shape([n, n]);
        for index in 0..n {
            dense[[index, index]] = Item::one();
        }
        let sign = if inverse { -Item::one() } else { Item::one() };
        for row in 0..factor.target_indices.len() {
            for column in 0..factor.source_indices.len() {
                let (i, j, value) = match mode {
                    FactorMode::NoTrans => (
                        factor.target_indices[row],
                        factor.source_indices[column],
                        factor.update[[row, column]],
                    ),
                    FactorMode::Conjugate => (
                        factor.target_indices[row],
                        factor.source_indices[column],
                        factor.update[[row, column]].conj(),
                    ),
                    FactorMode::Transpose => (
                        factor.source_indices[column],
                        factor.target_indices[row],
                        factor.update[[row, column]],
                    ),
                    FactorMode::Adjoint => (
                        factor.source_indices[column],
                        factor.target_indices[row],
                        factor.update[[row, column]].conj(),
                    ),
                };
                dense[[i, j]] += sign * value;
            }
        }
        dense
    }

    fn multiply<Item: RlstScalar>(
        left: &DynArray<Item, 2>,
        right: &DynArray<Item, 2>,
    ) -> DynArray<Item, 2> {
        assert_eq!(left.shape()[1], right.shape()[0]);
        let mut result = DynArray::<Item, 2>::from_shape([left.shape()[0], right.shape()[1]]);
        for i in 0..left.shape()[0] {
            for j in 0..right.shape()[1] {
                let mut value = Item::zero();
                for k in 0..left.shape()[1] {
                    value += left[[i, k]] * right[[k, j]];
                }
                result[[i, j]] = value;
            }
        }
        result
    }

    fn copy_array<Item: RlstScalar>(source: &DynArray<Item, 2>) -> DynArray<Item, 2> {
        let mut result = DynArray::<Item, 2>::from_shape(source.shape());
        for i in 0..source.shape()[0] {
            for j in 0..source.shape()[1] {
                result[[i, j]] = source[[i, j]];
            }
        }
        result
    }

    fn check<Item, Make>(make: Make, tolerance: Item::Real)
    where
        Item: RlstScalar,
        Make: Fn(f64, f64) -> Item,
    {
        let mut update = DynArray::<Item, 2>::from_shape([2, 2]);
        update[[0, 0]] = make(0.3, 0.2);
        update[[0, 1]] = make(-0.7, 0.1);
        update[[1, 0]] = make(0.4, -0.6);
        update[[1, 1]] = make(0.2, 0.5);
        let factor = ElementaryFactor::new(5, vec![0, 3], vec![1, 4], update).unwrap();

        let mut rhs = DynArray::<Item, 2>::from_shape([5, 3]);
        for i in 0..5 {
            for j in 0..3 {
                rhs[[i, j]] = make((i * 3 + j + 1) as f64 / 7.0, (i + 2 * j) as f64 / 11.0);
            }
        }
        let mut left_input = DynArray::<Item, 2>::from_shape([3, 5]);
        for i in 0..3 {
            for j in 0..5 {
                left_input[[i, j]] = make((i * 5 + j + 2) as f64 / 9.0, (2 * i + j) as f64 / 13.0);
            }
        }

        for mode in [
            FactorMode::NoTrans,
            FactorMode::Transpose,
            FactorMode::Adjoint,
            FactorMode::Conjugate,
        ] {
            for inverse in [false, true] {
                let dense = dense_factor(&factor, mode, inverse);

                let expected_left = multiply(&dense, &rhs);
                let mut actual_left = copy_array(&rhs);
                factor.apply_left_in_place(&mut actual_left, mode, inverse);
                for i in 0..actual_left.shape()[0] {
                    for j in 0..actual_left.shape()[1] {
                        assert!(
                            RlstScalar::abs(actual_left[[i, j]] - expected_left[[i, j]])
                                <= tolerance
                        );
                    }
                }

                let expected_right = multiply(&left_input, &dense);
                let mut actual_right = copy_array(&left_input);
                factor.apply_right_in_place(&mut actual_right, mode, inverse);
                for i in 0..actual_right.shape()[0] {
                    for j in 0..actual_right.shape()[1] {
                        assert!(
                            RlstScalar::abs(actual_right[[i, j]] - expected_right[[i, j]])
                                <= tolerance
                        );
                    }
                }
            }
        }

        let mut restored = copy_array(&rhs);
        factor.apply_left_in_place(&mut restored, FactorMode::Adjoint, false);
        factor.apply_left_in_place(&mut restored, FactorMode::Adjoint, true);
        for i in 0..restored.shape()[0] {
            for j in 0..restored.shape()[1] {
                assert!(RlstScalar::abs(restored[[i, j]] - rhs[[i, j]]) <= tolerance);
            }
        }
    }

    fn check_sequence<Item, Make>(make: Make, tolerance: Item::Real)
    where
        Item: RlstScalar,
        Make: Fn(f64, f64) -> Item,
    {
        let mut update_0 = DynArray::<Item, 2>::from_shape([1, 2]);
        update_0[[0, 0]] = make(0.3, 0.2);
        update_0[[0, 1]] = make(-0.4, 0.1);
        let mut update_1 = DynArray::<Item, 2>::from_shape([2, 1]);
        update_1[[0, 0]] = make(0.2, -0.5);
        update_1[[1, 0]] = make(0.6, 0.3);
        let factor_0 = ElementaryFactor::new(5, vec![0, 2], vec![1], update_0).unwrap();
        let factor_1 = ElementaryFactor::new(5, vec![1], vec![3, 4], update_1).unwrap();

        // Build the untransformed product independently as E0 E1.
        let dense_0 = dense_factor(&factor_0, FactorMode::NoTrans, false);
        let dense_1 = dense_factor(&factor_1, FactorMode::NoTrans, false);
        let product = multiply(&dense_0, &dense_1);
        let sequence = FactorSequence::new(5, vec![factor_0, factor_1]).unwrap();

        let mut rhs = DynArray::<Item, 2>::from_shape([5, 2]);
        let mut lhs = DynArray::<Item, 2>::from_shape([2, 5]);
        for i in 0..5 {
            for j in 0..2 {
                rhs[[i, j]] = make((i + 2 * j + 1) as f64 / 8.0, (2 * i + j) as f64 / 17.0);
                lhs[[j, i]] = make((2 * i + j + 2) as f64 / 10.0, (i + 3 * j) as f64 / 19.0);
            }
        }

        let expected_left = multiply(&product, &rhs);
        let mut actual_left = copy_array(&rhs);
        sequence.apply_left_in_place(&mut actual_left, FactorMode::NoTrans, false);
        let expected_right = multiply(&lhs, &product);
        let mut actual_right = copy_array(&lhs);
        sequence.apply_right_in_place(&mut actual_right, FactorMode::NoTrans, false);
        for i in 0..5 {
            for j in 0..2 {
                assert!(RlstScalar::abs(actual_left[[i, j]] - expected_left[[i, j]]) <= tolerance);
                assert!(
                    RlstScalar::abs(actual_right[[j, i]] - expected_right[[j, i]]) <= tolerance
                );
            }
        }

        for mode in [
            FactorMode::NoTrans,
            FactorMode::Transpose,
            FactorMode::Adjoint,
            FactorMode::Conjugate,
        ] {
            let mut restored = copy_array(&rhs);
            sequence.apply_left_in_place(&mut restored, mode, false);
            sequence.apply_left_in_place(&mut restored, mode, true);
            for i in 0..5 {
                for j in 0..2 {
                    assert!(RlstScalar::abs(restored[[i, j]] - rhs[[i, j]]) <= tolerance);
                }
            }
        }
    }

    #[test]
    fn real_factor_matches_dense_reference() {
        check::<f64, _>(|re, _| re, 2.0e-13);
    }

    #[test]
    fn complex_factor_matches_dense_reference() {
        check::<c64, _>(c64::new, 2.0e-13);
    }

    #[test]
    fn real_sequence_preserves_product_order() {
        check_sequence::<f64, _>(|re, _| re, 2.0e-13);
    }

    #[test]
    fn complex_sequence_preserves_product_order() {
        check_sequence::<c64, _>(c64::new, 2.0e-13);
    }

    #[test]
    fn rejects_overlapping_indices() {
        let update = DynArray::<f64, 2>::from_shape([1, 1]);
        assert!(ElementaryFactor::new(3, vec![1], vec![1], update).is_err());
    }
}
