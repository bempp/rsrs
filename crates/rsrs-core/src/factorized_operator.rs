use crate::{
    dense_multilevel::DenseMultilevelFactorization,
    diagonal_block::{IndexedDiagonalBlock, SquareBlock},
    elementary_factor::FactorMode,
};
use rlst::{
    base_types::RlstResult,
    dense::{array::DynArray, linalg::lapack::interpolative_decomposition::MatrixId},
    Lapack,
};

/// Apply and solve with a multilevel RSRS approximation without materializing
/// either the approximation or its inverse.
pub struct DenseFactorizedOperator<Item: MatrixId + Lapack> {
    factors: DenseMultilevelFactorization<Item>,
    diagonal: Vec<IndexedDiagonalBlock<Item>>,
}

impl<Item: MatrixId + Lapack> DenseFactorizedOperator<Item> {
    pub fn new(
        factors: DenseMultilevelFactorization<Item>,
        transformed: &DynArray<Item, 2>,
    ) -> RlstResult<Self> {
        let dimension = factors.dimension();
        assert_eq!(transformed.shape(), [dimension, dimension]);
        let mut diagonal = Vec::new();
        for executed in factors.cycles() {
            let indices = executed.cycle.redundant_indices().to_vec();
            if !indices.is_empty() {
                let block = extract_block(transformed, &indices);
                diagonal.push(
                    IndexedDiagonalBlock::new(dimension, indices, SquareBlock::lu(&block)?)
                        .expect("cycle diagonal indices are valid"),
                );
            }
        }
        if !factors.active_indices().is_empty() {
            let indices = factors.active_indices().to_vec();
            let block = extract_block(transformed, &indices);
            diagonal.push(
                IndexedDiagonalBlock::new(dimension, indices, SquareBlock::lu(&block)?)
                    .expect("active diagonal indices are valid"),
            );
        }
        Ok(Self { factors, diagonal })
    }

    pub fn dimension(&self) -> usize {
        self.factors.dimension()
    }

    /// Apply `op(A)` or solve `op(A) x = b` when `inverse` is true.
    pub fn apply(
        &self,
        input: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<DynArray<Item, 2>> {
        assert_eq!(input.shape()[0], self.dimension());
        match mode {
            FactorMode::NoTrans => self.apply_left_no_trans(input, inverse),
            FactorMode::Transpose | FactorMode::Adjoint => {
                let conjugate = mode == FactorMode::Adjoint;
                let mut row_input = transpose_copy(input, conjugate);
                self.apply_right_no_trans_in_place(&mut row_input, inverse)?;
                Ok(transpose_copy(&row_input, conjugate))
            }
            FactorMode::Conjugate => {
                let mut conjugated = copy_matrix(input);
                conjugate_in_place(&mut conjugated);
                let mut result = self.apply_left_no_trans(&conjugated, inverse)?;
                conjugate_in_place(&mut result);
                Ok(result)
            }
        }
    }

    fn apply_left_no_trans(
        &self,
        input: &DynArray<Item, 2>,
        inverse: bool,
    ) -> RlstResult<DynArray<Item, 2>> {
        let mut result = copy_matrix(input);
        if !inverse {
            // Q x, where A = P D Q.
            for executed in self.factors.cycles() {
                executed
                    .cycle
                    .id_left(&mut result, FactorMode::Transpose, false);
                executed
                    .cycle
                    .upper_left(&mut result, FactorMode::NoTrans, false)?;
            }
            for block in &self.diagonal {
                block.apply_left_in_place(&mut result, FactorMode::NoTrans, false)?;
            }
            // P (D Q x).
            for executed in self.factors.cycles().iter().rev() {
                executed
                    .cycle
                    .lower_left(&mut result, FactorMode::NoTrans, false);
                executed
                    .cycle
                    .id_left(&mut result, FactorMode::NoTrans, false);
            }
        } else {
            // P^-1 b.
            for executed in self.factors.cycles() {
                executed
                    .cycle
                    .id_left(&mut result, FactorMode::NoTrans, true);
                executed
                    .cycle
                    .lower_left(&mut result, FactorMode::NoTrans, true);
            }
            for block in &self.diagonal {
                block.apply_left_in_place(&mut result, FactorMode::NoTrans, true)?;
            }
            // Q^-1 D^-1 P^-1 b.
            for executed in self.factors.cycles().iter().rev() {
                executed
                    .cycle
                    .upper_left(&mut result, FactorMode::NoTrans, true)?;
                executed
                    .cycle
                    .id_left(&mut result, FactorMode::Transpose, true);
            }
        }
        Ok(result)
    }

    fn apply_right_no_trans_in_place(
        &self,
        result: &mut DynArray<Item, 2>,
        inverse: bool,
    ) -> RlstResult<()> {
        assert_eq!(result.shape()[1], self.dimension());
        if !inverse {
            // X P.
            for executed in self.factors.cycles() {
                executed.cycle.id_right(result, FactorMode::NoTrans, false);
                executed
                    .cycle
                    .lower_right(result, FactorMode::NoTrans, false);
            }
            for block in &self.diagonal {
                block.apply_right_in_place(result, FactorMode::NoTrans, false)?;
            }
            // X P D Q.
            for executed in self.factors.cycles().iter().rev() {
                executed
                    .cycle
                    .upper_right(result, FactorMode::NoTrans, false)?;
                executed
                    .cycle
                    .id_right(result, FactorMode::Transpose, false);
            }
        } else {
            // X Q^-1.
            for executed in self.factors.cycles() {
                executed.cycle.id_right(result, FactorMode::Transpose, true);
                executed
                    .cycle
                    .upper_right(result, FactorMode::NoTrans, true)?;
            }
            for block in &self.diagonal {
                block.apply_right_in_place(result, FactorMode::NoTrans, true)?;
            }
            // X Q^-1 D^-1 P^-1.
            for executed in self.factors.cycles().iter().rev() {
                executed
                    .cycle
                    .lower_right(result, FactorMode::NoTrans, true);
                executed.cycle.id_right(result, FactorMode::NoTrans, true);
            }
        }
        Ok(())
    }
}

fn extract_block<Item: Lapack>(matrix: &DynArray<Item, 2>, indices: &[usize]) -> DynArray<Item, 2> {
    let mut block = DynArray::<Item, 2>::from_shape([indices.len(), indices.len()]);
    for (local_row, &row) in indices.iter().enumerate() {
        for (local_column, &column) in indices.iter().enumerate() {
            block[[local_row, local_column]] = matrix[[row, column]];
        }
    }
    block
}

fn copy_matrix<Item: Lapack>(source: &DynArray<Item, 2>) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape(source.shape());
    result.fill_from(source);
    result
}
fn conjugate_in_place<Item: Lapack>(matrix: &mut DynArray<Item, 2>) {
    for value in matrix.data_mut().expect("owned arrays are contiguous") {
        *value = value.conj();
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dense_multilevel::DenseCyclePlan, interpolative_decomposition::IdMethod};
    use rlst::{base_types::c64, RlstScalar};

    fn dense_apply<Item: Lapack>(
        matrix: &DynArray<Item, 2>,
        input: &DynArray<Item, 2>,
        mode: FactorMode,
    ) -> DynArray<Item, 2> {
        let n = matrix.shape()[0];
        let mut result = DynArray::<Item, 2>::from_shape([n, input.shape()[1]]);
        for row in 0..n {
            for column in 0..input.shape()[1] {
                let mut value = Item::zero();
                for inner in 0..n {
                    let coefficient = match mode {
                        FactorMode::NoTrans => matrix[[row, inner]],
                        FactorMode::Transpose => matrix[[inner, row]],
                        FactorMode::Adjoint => matrix[[inner, row]].conj(),
                        FactorMode::Conjugate => matrix[[row, inner]].conj(),
                    };
                    value += coefficient * input[[inner, column]];
                }
                result[[row, column]] = value;
            }
        }
        result
    }

    fn check<Item, Make>(make: Make, tolerance: Item::Real)
    where
        Item: MatrixId + Lapack,
        Make: Fn(f64, f64) -> Item,
    {
        let n = 9;
        let mut matrix = DynArray::<Item, 2>::from_shape([n, n]);
        for row in 0..n {
            for column in 0..n {
                matrix[[row, column]] = make(
                    if row == column { 8.0 } else { 0.0 }
                        + ((3 * row + 5 * column + 1) % 17) as f64 / 31.0,
                    ((7 * row + 2 * column + 3) % 13) as f64 / 37.0,
                );
            }
        }
        let all = (0..n).collect::<Vec<_>>();
        let schedule = vec![vec![DenseCyclePlan {
            target_indices: vec![0, 1, 2, 3],
            near_indices: all,
            far_indices: vec![5, 6, 7, 8],
            rank: 2,
        }]];
        let (factors, transformed) =
            DenseMultilevelFactorization::factorize(&matrix, &schedule, IdMethod::Rrqr).unwrap();
        let approximation = factors.approximate_matrix(&transformed).unwrap();
        let operator = DenseFactorizedOperator::new(factors, &transformed).unwrap();
        let mut rhs = DynArray::<Item, 2>::from_shape([n, 2]);
        for row in 0..n {
            for column in 0..2 {
                rhs[[row, column]] = make(
                    (row + 2 * column + 1) as f64 / 11.0,
                    (2 * row + column) as f64 / 23.0,
                );
            }
        }
        for mode in [
            FactorMode::NoTrans,
            FactorMode::Transpose,
            FactorMode::Adjoint,
            FactorMode::Conjugate,
        ] {
            let expected = dense_apply(&approximation, &rhs, mode);
            let actual = operator.apply(&rhs, mode, false).unwrap();
            for row in 0..n {
                for column in 0..2 {
                    assert!(
                        RlstScalar::abs(actual[[row, column]] - expected[[row, column]])
                            <= tolerance
                    );
                }
            }
            let solution = operator.apply(&rhs, mode, true).unwrap();
            let recovered = operator.apply(&solution, mode, false).unwrap();
            for row in 0..n {
                for column in 0..2 {
                    assert!(
                        RlstScalar::abs(recovered[[row, column]] - rhs[[row, column]]) <= tolerance
                    );
                }
            }
        }
    }

    #[test]
    fn real_factorized_apply_and_solve() {
        check::<f64, _>(|real, _| real, 5.0e-10);
    }
    #[test]
    fn complex_factorized_apply_and_solve() {
        check::<c64, _>(c64::new, 8.0e-10);
    }
}
