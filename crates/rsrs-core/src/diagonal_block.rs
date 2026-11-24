use crate::{
    configuration::PivotMethod,
    elementary_factor::{
        commit_left_update, commit_right_update, FactorMode, IndexedLeftUpdate, IndexedRightUpdate,
    },
};
use num::Zero;
use rlst::{
    base_types::{RlstResult, TransMode},
    dense::{array::DynArray, linalg::lapack::lu::SquareLuFactors},
    Inverse, Lapack, Lu,
};

/// Storage used for a nonsingular square RSRS diagonal block.
pub enum SquareBlock<Item: Lapack> {
    /// Keep the block and its explicitly computed inverse.
    Dense {
        matrix: DynArray<Item, 2>,
        inverse: DynArray<Item, 2>,
    },
    /// Keep reusable pivoted LU factors, avoiding a second dense matrix.
    Lu(SquareLuFactors<Item>),
    /// Historical `LuHybrid`: retain the shifted block and an inverse formed
    /// by solving the LU factors against the identity.
    LuHybrid {
        matrix: DynArray<Item, 2>,
        inverse: DynArray<Item, 2>,
    },
}

impl<Item: Lapack> SquareBlock<Item> {
    pub fn from_pivot_method(
        matrix: &DynArray<Item, 2>,
        method: PivotMethod,
        shift: Item::Real,
    ) -> RlstResult<Self> {
        match method {
            PivotMethod::DirectInverse => Self::explicit_inverse_shifted(matrix, shift),
            PivotMethod::Lu => Self::lu_shifted(matrix, shift),
            PivotMethod::LuHybrid => Self::lu_hybrid_shifted(matrix, shift),
        }
    }

    pub fn dense(matrix: DynArray<Item, 2>, inverse: DynArray<Item, 2>) -> Result<Self, String> {
        let shape = matrix.shape();
        if shape[0] != shape[1] || inverse.shape() != shape {
            return Err("dense square block and inverse must have the same square shape".into());
        }
        Ok(Self::Dense { matrix, inverse })
    }

    pub fn lu(matrix: &DynArray<Item, 2>) -> RlstResult<Self> {
        Self::lu_shifted(matrix, Item::Real::zero())
    }

    pub fn lu_shifted(matrix: &DynArray<Item, 2>, shift: Item::Real) -> RlstResult<Self> {
        assert_eq!(matrix.shape()[0], matrix.shape()[1]);
        let mut shifted = DynArray::<Item, 2>::new_from(matrix);
        if shift != Item::Real::zero() {
            for index in 0..shifted.shape()[0] {
                shifted[[index, index]] = shifted[[index, index]].add_real(shift);
            }
        }
        Ok(Self::Lu(shifted.lu()?.to_owned_square_factors()?))
    }

    pub fn lu_hybrid_shifted(matrix: &DynArray<Item, 2>, shift: Item::Real) -> RlstResult<Self> {
        assert_eq!(matrix.shape()[0], matrix.shape()[1]);
        let mut shifted = DynArray::<Item, 2>::new_from(matrix);
        if shift != Item::Real::zero() {
            for index in 0..shifted.shape()[0] {
                shifted[[index, index]] = shifted[[index, index]].add_real(shift);
            }
        }
        let factors = shifted.r().lu()?.to_owned_square_factors()?;
        let mut inverse = DynArray::<Item, 2>::from_shape(shifted.shape());
        for index in 0..shifted.shape()[0] {
            inverse[[index, index]] = Item::one();
        }
        factors.solve_mat(TransMode::NoTrans, inverse.r_mut())?;
        Ok(Self::LuHybrid {
            matrix: shifted,
            inverse,
        })
    }

    /// Materialize a shifted block and its inverse through RLST's explicit
    /// inverse path. This remains distinct from `LuHybrid`, which forms its
    /// stored inverse by solving reusable LU factors against the identity.
    pub fn explicit_inverse_shifted(
        matrix: &DynArray<Item, 2>,
        shift: Item::Real,
    ) -> RlstResult<Self> {
        assert_eq!(matrix.shape()[0], matrix.shape()[1]);
        let mut shifted = DynArray::<Item, 2>::new_from(matrix);
        if shift != Item::Real::zero() {
            for index in 0..shifted.shape()[0] {
                shifted[[index, index]] = shifted[[index, index]].add_real(shift);
            }
        }
        let inverse = shifted.r().inverse()?;
        Ok(Self::Dense {
            matrix: shifted,
            inverse,
        })
    }

    pub fn dimension(&self) -> usize {
        match self {
            Self::Dense { matrix, .. } => matrix.shape()[0],
            Self::Lu(factors) => factors.perm().len(),
            Self::LuHybrid { matrix, .. } => matrix.shape()[0],
        }
    }

    pub(crate) fn storage_bytes(&self) -> u64 {
        let dimension = self.dimension();
        match self {
            Self::Dense { .. } | Self::LuHybrid { .. } => {
                (2 * dimension * dimension * std::mem::size_of::<Item>()) as u64
            }
            Self::Lu(_) => {
                (dimension * dimension * std::mem::size_of::<Item>()
                    + dimension * std::mem::size_of::<i32>()) as u64
            }
        }
    }

    pub fn apply_left_in_place(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<()> {
        assert_eq!(target.shape()[0], self.dimension());
        match self {
            Self::Dense {
                matrix,
                inverse: inv,
            }
            | Self::LuHybrid {
                matrix,
                inverse: inv,
            } => {
                let block = if inverse { inv } else { matrix };
                let [rows, columns] = target.shape();
                let mut result = DynArray::<Item, 2>::from_shape([rows, columns]);
                for row in 0..rows {
                    for column in 0..columns {
                        let mut value = Item::zero();
                        for inner in 0..rows {
                            let coefficient = match mode {
                                FactorMode::NoTrans => block[[row, inner]],
                                FactorMode::Transpose => block[[inner, row]],
                                FactorMode::Adjoint => block[[inner, row]].conj(),
                                FactorMode::Conjugate => block[[row, inner]].conj(),
                            };
                            value += coefficient * target[[inner, column]];
                        }
                        result[[row, column]] = value;
                    }
                }
                target.fill_from(&result);
                Ok(())
            }
            Self::Lu(factors) => {
                if mode == FactorMode::Conjugate {
                    conjugate_in_place(target);
                    let result = if inverse {
                        factors.solve_mat(TransMode::NoTrans, target.r_mut())
                    } else {
                        factors.mul_mat(TransMode::NoTrans, target.r_mut())
                    };
                    conjugate_in_place(target);
                    return result;
                }
                let trans = match mode {
                    FactorMode::NoTrans => TransMode::NoTrans,
                    FactorMode::Transpose => TransMode::Trans,
                    FactorMode::Adjoint => TransMode::ConjTrans,
                    FactorMode::Conjugate => unreachable!(),
                };
                if inverse {
                    factors.solve_mat(trans, target.r_mut())
                } else {
                    factors.mul_mat(trans, target.r_mut())
                }
            }
        }
    }

    pub fn apply_right_in_place(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<()> {
        assert_eq!(target.shape()[1], self.dimension());
        let [rows, columns] = target.shape();
        let left_mode = match mode {
            FactorMode::NoTrans => FactorMode::Transpose,
            FactorMode::Transpose => FactorMode::NoTrans,
            FactorMode::Adjoint => FactorMode::Conjugate,
            FactorMode::Conjugate => FactorMode::Adjoint,
        };
        let mut work = DynArray::<Item, 2>::from_shape([columns, rows]);
        for row in 0..rows {
            for column in 0..columns {
                work[[column, row]] = target[[row, column]];
            }
        }
        self.apply_left_in_place(&mut work, left_mode, inverse)?;
        for row in 0..rows {
            for column in 0..columns {
                target[[row, column]] = work[[column, row]];
            }
        }
        Ok(())
    }
}

fn conjugate_in_place<Item: Lapack>(matrix: &mut DynArray<Item, 2>) {
    for value in matrix.data_mut().expect("owned arrays are contiguous") {
        *value = value.conj();
    }
}

/// A square block acting only on the selected global degrees of freedom.
pub struct IndexedDiagonalBlock<Item: Lapack> {
    ambient_dimension: usize,
    indices: Vec<usize>,
    block: SquareBlock<Item>,
}

impl<Item: Lapack> IndexedDiagonalBlock<Item> {
    pub fn new(
        ambient_dimension: usize,
        indices: Vec<usize>,
        block: SquareBlock<Item>,
    ) -> Result<Self, String> {
        if block.dimension() != indices.len() {
            return Err("block dimension does not match its index count".into());
        }
        if indices.iter().any(|&index| index >= ambient_dimension) {
            return Err("diagonal-block index is outside the ambient dimension".into());
        }
        let mut unique = indices.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != indices.len() {
            return Err("diagonal-block indices must be unique".into());
        }
        Ok(Self {
            ambient_dimension,
            indices,
            block,
        })
    }

    pub fn apply_left_in_place(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<()> {
        let update = self.left_update(target, mode, inverse)?;
        commit_left_update(target, update);
        Ok(())
    }

    pub(crate) fn left_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<IndexedLeftUpdate<Item>> {
        assert_eq!(target.shape()[0], self.ambient_dimension);
        let columns = target.shape()[1];
        let mut local = DynArray::<Item, 2>::from_shape([self.indices.len(), columns]);
        for (local_row, &global_row) in self.indices.iter().enumerate() {
            for column in 0..columns {
                local[[local_row, column]] = target[[global_row, column]];
            }
        }
        self.block.apply_left_in_place(&mut local, mode, inverse)?;
        for (local_row, &global_row) in self.indices.iter().enumerate() {
            for column in 0..columns {
                local[[local_row, column]] -= target[[global_row, column]];
            }
        }
        Ok(IndexedLeftUpdate {
            indices: self.indices.clone(),
            values: local,
        })
    }

    pub fn apply_right_in_place(
        &self,
        target: &mut DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<()> {
        let update = self.right_update(target, mode, inverse)?;
        commit_right_update(target, update);
        Ok(())
    }

    pub(crate) fn right_update(
        &self,
        target: &DynArray<Item, 2>,
        mode: FactorMode,
        inverse: bool,
    ) -> RlstResult<IndexedRightUpdate<Item>> {
        assert_eq!(target.shape()[1], self.ambient_dimension);
        let rows = target.shape()[0];
        let mut local = DynArray::<Item, 2>::from_shape([rows, self.indices.len()]);
        for row in 0..rows {
            for (local_column, &global_column) in self.indices.iter().enumerate() {
                local[[row, local_column]] = target[[row, global_column]];
            }
        }
        self.block.apply_right_in_place(&mut local, mode, inverse)?;
        for row in 0..rows {
            for (local_column, &global_column) in self.indices.iter().enumerate() {
                local[[row, local_column]] -= target[[row, global_column]];
            }
        }
        Ok(IndexedRightUpdate {
            indices: self.indices.clone(),
            values: local,
        })
    }

    pub(crate) fn storage_bytes(&self) -> u64 {
        self.block.storage_bytes() + (self.indices.len() * std::mem::size_of::<usize>()) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlst::{base_types::c64, RlstScalar};

    fn inverse_2x2<Item: RlstScalar>(matrix: &DynArray<Item, 2>) -> DynArray<Item, 2> {
        let determinant = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];
        let mut inverse = DynArray::<Item, 2>::from_shape([2, 2]);
        inverse[[0, 0]] = matrix[[1, 1]] / determinant;
        inverse[[0, 1]] = -matrix[[0, 1]] / determinant;
        inverse[[1, 0]] = -matrix[[1, 0]] / determinant;
        inverse[[1, 1]] = matrix[[0, 0]] / determinant;
        inverse
    }

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
        let mut matrix = DynArray::<Item, 2>::from_shape([2, 2]);
        matrix[[0, 0]] = make(2.0, 0.4);
        matrix[[0, 1]] = make(-0.3, 0.2);
        matrix[[1, 0]] = make(0.7, -0.5);
        matrix[[1, 1]] = make(1.6, 0.1);
        let dense = SquareBlock::dense(copy(&matrix), inverse_2x2(&matrix)).unwrap();
        let lu = SquareBlock::lu(&matrix).unwrap();
        let direct =
            SquareBlock::from_pivot_method(&matrix, PivotMethod::DirectInverse, Item::Real::zero())
                .unwrap();
        let hybrid =
            SquareBlock::from_pivot_method(&matrix, PivotMethod::LuHybrid, Item::Real::zero())
                .unwrap();
        assert!(matches!(direct, SquareBlock::Dense { .. }));
        assert!(matches!(hybrid, SquareBlock::LuHybrid { .. }));
        let materialized_bytes = (2 * 2 * 2 * std::mem::size_of::<Item>()) as u64;
        assert_eq!(direct.storage_bytes(), materialized_bytes);
        assert_eq!(hybrid.storage_bytes(), materialized_bytes);
        assert!(lu.storage_bytes() < materialized_bytes);

        let mut input = DynArray::<Item, 2>::from_shape([2, 3]);
        for row in 0..2 {
            for column in 0..3 {
                input[[row, column]] = make(
                    (row + 2 * column + 1) as f64 / 5.0,
                    (2 * row + column) as f64 / 9.0,
                );
            }
        }
        for mode in [
            FactorMode::NoTrans,
            FactorMode::Transpose,
            FactorMode::Adjoint,
            FactorMode::Conjugate,
        ] {
            let mut reference = copy(&input);
            lu.apply_left_in_place(&mut reference, mode, false).unwrap();
            for materialized in [&dense, &direct, &hybrid] {
                let mut actual = copy(&input);
                materialized
                    .apply_left_in_place(&mut actual, mode, false)
                    .unwrap();
                for row in 0..2 {
                    for column in 0..3 {
                        assert!(
                            RlstScalar::abs(actual[[row, column]] - reference[[row, column]])
                                <= tolerance
                        );
                    }
                }
                materialized
                    .apply_left_in_place(&mut actual, mode, true)
                    .unwrap();
                for row in 0..2 {
                    for column in 0..3 {
                        assert!(
                            RlstScalar::abs(actual[[row, column]] - input[[row, column]])
                                <= tolerance
                        );
                    }
                }
            }
            lu.apply_left_in_place(&mut reference, mode, true).unwrap();
            for row in 0..2 {
                for column in 0..3 {
                    assert!(
                        RlstScalar::abs(reference[[row, column]] - input[[row, column]])
                            <= tolerance
                    );
                }
            }
        }

        let mut right_input = DynArray::<Item, 2>::from_shape([3, 2]);
        for row in 0..3 {
            for column in 0..2 {
                right_input[[row, column]] = make(
                    (2 * row + column + 1) as f64 / 7.0,
                    (row + column) as f64 / 8.0,
                );
            }
        }
        for mode in [
            FactorMode::NoTrans,
            FactorMode::Transpose,
            FactorMode::Adjoint,
        ] {
            let mut actual = copy(&right_input);
            lu.apply_right_in_place(&mut actual, mode, false).unwrap();
            lu.apply_right_in_place(&mut actual, mode, true).unwrap();
            for row in 0..3 {
                for column in 0..2 {
                    assert!(
                        RlstScalar::abs(actual[[row, column]] - right_input[[row, column]])
                            <= tolerance
                    );
                }
            }
        }
    }

    #[test]
    fn real_dense_and_lu_blocks_agree() {
        check::<f64, _>(|real, _| real, 2.0e-12);
    }

    #[test]
    fn complex_dense_and_lu_blocks_agree() {
        check::<c64, _>(c64::new, 3.0e-12);
    }

    #[test]
    fn indexed_block_only_changes_selected_rows() {
        let mut matrix = DynArray::<f64, 2>::from_shape([2, 2]);
        matrix[[0, 0]] = 2.0;
        matrix[[1, 1]] = 3.0;
        let block =
            IndexedDiagonalBlock::new(5, vec![1, 4], SquareBlock::lu(&matrix).unwrap()).unwrap();
        let mut target = DynArray::<f64, 2>::from_shape([5, 1]);
        for row in 0..5 {
            target[[row, 0]] = row as f64 + 1.0;
        }
        block
            .apply_left_in_place(&mut target, FactorMode::NoTrans, false)
            .unwrap();
        assert_eq!(target[[0, 0]], 1.0);
        assert_eq!(target[[1, 0]], 4.0);
        assert_eq!(target[[2, 0]], 3.0);
        assert_eq!(target[[3, 0]], 4.0);
        assert_eq!(target[[4, 0]], 15.0);
    }
}
