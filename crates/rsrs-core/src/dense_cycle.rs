use crate::{
    composed_factor::ComposedElementaryFactor,
    diagonal_block::SquareBlock,
    elementary_factor::{ElementaryFactor, FactorMode},
    interpolative_decomposition::{compute_candidate_id, IdMethod, IdRankSelection},
};
use rlst::{
    base_types::RlstResult,
    dense::{array::DynArray, linalg::lapack::interpolative_decomposition::MatrixId},
    Lapack,
};
use std::collections::HashSet;

/// A serial dense reference implementation of one nonsymmetric RSRS cycle.
///
/// This is intentionally independent of the tree and sampling scheduler. It
/// establishes the factor signs, ID permutation, and reconstruction ordering
/// that the multilevel implementation will reuse.
pub struct DenseSkeletonizationCycle<Item: MatrixId + Lapack> {
    dimension: usize,
    skeleton_indices: Vec<usize>,
    redundant_indices: Vec<usize>,
    remaining_indices: Vec<usize>,
    id_factor: ElementaryFactor<Item>,
    lower_factor: ElementaryFactor<Item>,
    upper_factor: ComposedElementaryFactor<Item>,
}

impl<Item: MatrixId + Lapack> DenseSkeletonizationCycle<Item> {
    pub fn factorize(
        matrix: &DynArray<Item, 2>,
        target_indices: &[usize],
        near_indices: &[usize],
        far_indices: &[usize],
        rank: usize,
        method: IdMethod<Item::Real>,
    ) -> RlstResult<(Self, DynArray<Item, 2>)> {
        let [rows, columns] = matrix.shape();
        assert_eq!(rows, columns);
        assert!(rank > 0 && rank < target_indices.len());
        let dimension = rows;
        let mut combined =
            DynArray::<Item, 2>::from_shape([2 * far_indices.len(), target_indices.len()]);
        for (candidate, &target) in target_indices.iter().enumerate() {
            for (feature, &far) in far_indices.iter().enumerate() {
                combined[[feature, candidate]] = matrix[[far, target]];
                // Plain transpose is deliberate: complex nonsymmetric RSRS
                // approximates row and column interactions with one selection.
                combined[[far_indices.len() + feature, candidate]] = matrix[[target, far]];
            }
        }
        let id = compute_candidate_id(&combined, IdRankSelection::Fixed(rank), method, false)?;
        let ordered = id
            .perm
            .iter()
            .map(|&local| target_indices[local])
            .collect::<Vec<_>>();
        let skeleton_indices = ordered[..rank].to_vec();
        let redundant_indices = ordered[rank..].to_vec();

        // Current RLST stores conjugate-transposed interpolation coefficients.
        // The elementary factor stores the ordinary row update, hence conjugate.
        let mut id_update =
            DynArray::<Item, 2>::from_shape([redundant_indices.len(), skeleton_indices.len()]);
        for redundant in 0..redundant_indices.len() {
            for skeleton in 0..skeleton_indices.len() {
                id_update[[redundant, skeleton]] = id.id_mat[[redundant, skeleton]].conj();
            }
        }
        let id_factor = ElementaryFactor::new(
            dimension,
            skeleton_indices.clone(),
            redundant_indices.clone(),
            id_update,
        )
        .expect("ID indices and update dimensions are internally consistent");

        let mut after_id = copy_matrix(matrix);
        id_factor.apply_left_in_place(&mut after_id, FactorMode::NoTrans, true);
        id_factor.apply_right_in_place(&mut after_id, FactorMode::Transpose, true);

        let redundant: HashSet<_> = redundant_indices.iter().copied().collect();
        let mut remaining_indices = near_indices
            .iter()
            .copied()
            .filter(|index| !redundant.contains(index))
            .collect::<Vec<_>>();
        remaining_indices.sort_unstable();
        remaining_indices.dedup();
        let pivot = extract_block(&after_id, &redundant_indices, &redundant_indices);
        let right_coupling = extract_block(&after_id, &redundant_indices, &remaining_indices);
        let left_coupling = extract_block(&after_id, &remaining_indices, &redundant_indices);

        // U = A_rr^-1 A_rt is retained in composed form.
        let upper_factor = ComposedElementaryFactor::new(
            dimension,
            remaining_indices.clone(),
            redundant_indices.clone(),
            SquareBlock::lu(&pivot)?,
            right_coupling,
        )
        .expect("upper factor dimensions are internally consistent");

        // L = A_tr A_rr^-1. Compute L^T through one transposed pivot solve.
        let mut lower_transpose = transpose_copy(&left_coupling, false);
        SquareBlock::lu(&pivot)?.apply_left_in_place(
            &mut lower_transpose,
            FactorMode::Transpose,
            true,
        )?;
        let lower_update = transpose_copy(&lower_transpose, false);
        let lower_factor = ElementaryFactor::new(
            dimension,
            redundant_indices.clone(),
            remaining_indices.clone(),
            lower_update,
        )
        .expect("lower factor dimensions are internally consistent");

        let mut transformed = after_id;
        lower_factor.apply_left_in_place(&mut transformed, FactorMode::NoTrans, true);
        upper_factor.apply_right_in_place(&mut transformed, FactorMode::NoTrans, true)?;

        Ok((
            Self {
                dimension,
                skeleton_indices,
                redundant_indices,
                remaining_indices,
                id_factor,
                lower_factor,
                upper_factor,
            },
            transformed,
        ))
    }

    pub fn skeleton_indices(&self) -> &[usize] {
        &self.skeleton_indices
    }
    pub fn redundant_indices(&self) -> &[usize] {
        &self.redundant_indices
    }
    pub fn remaining_indices(&self) -> &[usize] {
        &self.remaining_indices
    }

    pub(crate) fn id_left(&self, target: &mut DynArray<Item, 2>, mode: FactorMode, inverse: bool) {
        self.id_factor.apply_left_in_place(target, mode, inverse);
    }
    pub(crate) fn id_right(&self, target: &mut DynArray<Item, 2>, mode: FactorMode, inverse: bool) {
        self.id_factor.apply_right_in_place(target, mode, inverse);
    }
    pub(crate) fn lower_left(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) {
        self.lower_factor.apply_left_in_place(target, mode, inverse);
    }
    pub(crate) fn lower_right(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) {
        self.lower_factor
            .apply_right_in_place(target, mode, inverse);
    }
    pub(crate) fn upper_left(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<()> {
        self.upper_factor.apply_left_in_place(target, mode, inverse)
    }
    pub(crate) fn upper_right(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<()> {
        self.upper_factor
            .apply_right_in_place(target, mode, inverse)
    }

    /// Reconstruct the matrix before this cycle from its transformed form.
    pub fn reconstruct(&self, transformed: &DynArray<Item, 2>) -> RlstResult<DynArray<Item, 2>> {
        assert_eq!(transformed.shape(), [self.dimension, self.dimension]);
        let mut result = copy_matrix(transformed);
        // B = L D U.
        self.lower_factor
            .apply_left_in_place(&mut result, FactorMode::NoTrans, false);
        self.upper_factor
            .apply_right_in_place(&mut result, FactorMode::NoTrans, false)?;
        // A = E_ID B E_ID^T.
        self.id_factor
            .apply_left_in_place(&mut result, FactorMode::NoTrans, false);
        self.id_factor
            .apply_right_in_place(&mut result, FactorMode::Transpose, false);
        Ok(result)
    }
}

fn copy_matrix<Item: Lapack>(source: &DynArray<Item, 2>) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape(source.shape());
    result.fill_from(source);
    result
}

fn transpose_copy<Item: Lapack>(source: &DynArray<Item, 2>, conjugate: bool) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape([source.shape()[1], source.shape()[0]]);
    for row in 0..source.shape()[0] {
        for column in 0..source.shape()[1] {
            let value = source[[row, column]];
            result[[column, row]] = if conjugate { value.conj() } else { value };
        }
    }
    result
}

fn extract_block<Item: Lapack>(
    matrix: &DynArray<Item, 2>,
    rows: &[usize],
    columns: &[usize],
) -> DynArray<Item, 2> {
    let mut block = DynArray::<Item, 2>::from_shape([rows.len(), columns.len()]);
    for (local_row, &row) in rows.iter().enumerate() {
        for (local_column, &column) in columns.iter().enumerate() {
            block[[local_row, local_column]] = matrix[[row, column]];
        }
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlst::{base_types::c64, RlstScalar};

    fn check<Item, Make>(make: Make, tolerance: Item::Real)
    where
        Item: MatrixId + Lapack,
        Make: Fn(f64, f64) -> Item,
    {
        let n = 8;
        let target = vec![0, 1, 2, 3];
        let far = vec![4, 5, 6, 7];
        let mut matrix = DynArray::<Item, 2>::from_shape([n, n]);
        // A well-conditioned nonsymmetric local/near part.
        for row in 0..n {
            for column in 0..n {
                let diagonal = if row == column { 5.0 } else { 0.0 };
                matrix[[row, column]] = make(
                    diagonal + ((3 * row + 2 * column + 1) % 11) as f64 / 19.0,
                    ((5 * row + column + 2) % 7) as f64 / 23.0,
                );
            }
        }
        // Make both far interaction panels share exact rank two with the same
        // interpolation coefficients, while retaining nonsymmetric values.
        let coefficients = [
            [make(1.0, 0.0), make(0.0, 0.0)],
            [make(0.0, 0.0), make(1.0, 0.0)],
            [make(0.4, 0.2), make(-0.3, 0.1)],
            [make(-0.2, 0.3), make(0.7, -0.4)],
        ];
        for (far_local, &far_index) in far.iter().enumerate() {
            let column_basis = [
                make((far_local + 1) as f64 / 5.0, (far_local + 2) as f64 / 17.0),
                make((2 * far_local + 3) as f64 / 7.0, -(far_local as f64) / 13.0),
            ];
            let row_basis = [
                make((far_local + 2) as f64 / 6.0, -(far_local as f64) / 11.0),
                make(
                    (3 * far_local + 1) as f64 / 8.0,
                    (far_local + 1) as f64 / 14.0,
                ),
            ];
            for candidate in 0..4 {
                matrix[[far_index, target[candidate]]] = column_basis[0]
                    * coefficients[candidate][0]
                    + column_basis[1] * coefficients[candidate][1];
                matrix[[target[candidate], far_index]] = row_basis[0] * coefficients[candidate][0]
                    + row_basis[1] * coefficients[candidate][1];
            }
        }

        let original = copy_matrix(&matrix);
        let (cycle, transformed) = DenseSkeletonizationCycle::factorize(
            &matrix,
            &target,
            &(0..n).collect::<Vec<_>>(),
            &far,
            2,
            IdMethod::Rrqr,
        )
        .unwrap();

        // Redundant variables are decoupled from every remaining variable.
        for &redundant in cycle.redundant_indices() {
            for &remaining in cycle.remaining_indices() {
                assert!(RlstScalar::abs(transformed[[redundant, remaining]]) <= tolerance);
                assert!(RlstScalar::abs(transformed[[remaining, redundant]]) <= tolerance);
            }
        }
        let reconstructed = cycle.reconstruct(&transformed).unwrap();
        for row in 0..n {
            for column in 0..n {
                assert!(
                    RlstScalar::abs(reconstructed[[row, column]] - original[[row, column]])
                        <= tolerance
                );
            }
        }
    }

    #[test]
    fn real_dense_cycle_decouples_and_reconstructs() {
        check::<f64, _>(|real, _| real, 2.0e-10);
    }

    #[test]
    fn complex_dense_cycle_decouples_and_reconstructs() {
        check::<c64, _>(c64::new, 3.0e-10);
    }
}
