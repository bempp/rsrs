extern crate blas_src;
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;

use rlst::{base_types::c64, dense::array::DynArray, RlstScalar};
use rsrs_core::{
    dense_multilevel::{DenseCyclePlan, DenseMultilevelFactorization},
    elementary_factor::FactorMode,
    factorized_operator::DenseFactorizedOperator,
    interpolative_decomposition::IdMethod,
};
use std::{fs, path::PathBuf};

const MAGIC: &[u8] = b"RSRS_BIEGRID_FIXTURE_V1";

struct Reader {
    bytes: Vec<u8>,
    offset: usize,
}
impl Reader {
    fn new(bytes: Vec<u8>) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, count: usize) -> &[u8] {
        let start = self.offset;
        self.offset += count;
        &self.bytes[start..self.offset]
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take(8).try_into().unwrap())
    }
}

struct Fixture {
    points: Vec<[f64; 3]>,
    real_symmetric: DynArray<f64, 2>,
    real_nonsymmetric: DynArray<f64, 2>,
    complex_symmetric: DynArray<c64, 2>,
    complex_nonsymmetric: DynArray<c64, 2>,
}

fn fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/biegrid_perturbed/cells22_scale1e-2_seed12345.bin");
    let mut reader = Reader::new(fs::read(path).unwrap());
    assert_eq!(reader.take(MAGIC.len()), MAGIC);
    let n = reader.u64() as usize;
    let _cells = reader.u64();
    let _scale = reader.f64();
    let _seed = reader.u64();
    let points = (0..n)
        .map(|_| [reader.f64(), reader.f64(), reader.f64()])
        .collect();
    let mut real = || {
        let mut matrix = DynArray::<f64, 2>::from_shape([n, n]);
        for row in 0..n {
            for column in 0..n {
                matrix[[row, column]] = reader.f64();
            }
        }
        matrix
    };
    let real_symmetric = real();
    let real_nonsymmetric = real();
    let mut complex = || {
        let mut matrix = DynArray::<c64, 2>::from_shape([n, n]);
        for row in 0..n {
            for column in 0..n {
                matrix[[row, column]] = c64::new(reader.f64(), reader.f64());
            }
        }
        matrix
    };
    let complex_symmetric = complex();
    let complex_nonsymmetric = complex();
    Fixture {
        points,
        real_symmetric,
        real_nonsymmetric,
        complex_symmetric,
        complex_nonsymmetric,
    }
}

fn plans(points: &[[f64; 3]], boxes_per_axis: usize, rank: usize) -> Vec<Vec<DenseCyclePlan>> {
    let min_x = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
    let max_x = points
        .iter()
        .map(|p| p[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let max_y = points
        .iter()
        .map(|p| p[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let mut boxes = vec![Vec::new(); boxes_per_axis * boxes_per_axis];
    for (index, point) in points.iter().enumerate() {
        let x = (((point[0] - min_x) / (max_x - min_x) * boxes_per_axis as f64).floor() as usize)
            .min(boxes_per_axis - 1);
        let y = (((point[1] - min_y) / (max_y - min_y) * boxes_per_axis as f64).floor() as usize)
            .min(boxes_per_axis - 1);
        boxes[y * boxes_per_axis + x].push(index);
    }
    let all = (0..points.len()).collect::<Vec<_>>();
    let mut level = Vec::new();
    for by in 0..boxes_per_axis {
        for bx in 0..boxes_per_axis {
            let target = boxes[by * boxes_per_axis + bx].clone();
            if target.len() <= rank {
                continue;
            }
            let mut near = Vec::new();
            for ny in by.saturating_sub(1)..=(by + 1).min(boxes_per_axis - 1) {
                for nx in bx.saturating_sub(1)..=(bx + 1).min(boxes_per_axis - 1) {
                    near.extend_from_slice(&boxes[ny * boxes_per_axis + nx]);
                }
            }
            near.sort_unstable();
            near.dedup();
            let near_mask = near
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>();
            let far = all
                .iter()
                .copied()
                .filter(|i| !near_mask.contains(i))
                .collect();
            level.push(DenseCyclePlan {
                target_indices: target,
                near_indices: near,
                far_indices: far,
                rank,
            });
        }
    }
    vec![level]
}

fn relative_frobenius<Item: RlstScalar>(
    actual: &DynArray<Item, 2>,
    expected: &DynArray<Item, 2>,
) -> f64
where
    Item::Real: Into<f64>,
{
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for row in 0..actual.shape()[0] {
        for column in 0..actual.shape()[1] {
            let error: f64 =
                RlstScalar::abs(actual[[row, column]] - expected[[row, column]]).into();
            let reference: f64 = RlstScalar::abs(expected[[row, column]]).into();
            numerator += error * error;
            denominator += reference * reference;
        }
    }
    (numerator / denominator).sqrt()
}

fn run<Item>(matrix: &DynArray<Item, 2>, points: &[[f64; 3]], label: &str)
where
    Item: rlst::dense::linalg::lapack::interpolative_decomposition::MatrixId + rlst::Lapack,
    Item::Real: Into<f64>,
{
    let schedule = plans(points, 4, 16);
    let (factors, transformed) =
        DenseMultilevelFactorization::factorize(matrix, &schedule, IdMethod::Rrqr).unwrap();
    let reconstructed = factors.reconstruct(&transformed).unwrap();
    let reconstruction_error = relative_frobenius(&reconstructed, matrix);
    let approximation = factors.approximate_matrix(&transformed).unwrap();
    let approximation_error = relative_frobenius(&approximation, matrix);
    let operator = DenseFactorizedOperator::new(factors, &transformed).unwrap();
    let mut rhs = DynArray::<Item, 2>::from_shape([matrix.shape()[0], 1]);
    for row in 0..matrix.shape()[0] {
        rhs[[row, 0]] = Item::from_real(Item::real(
            ((row + 1) as f64).sin() + 0.25 * ((row + 1) as f64).cos(),
        ));
    }
    let dense_product = dense_apply(matrix, &rhs);
    let factor_product = operator.apply(&rhs, FactorMode::NoTrans, false).unwrap();
    let forward_error = relative_frobenius(&factor_product, &dense_product);
    let solution = operator.apply(&rhs, FactorMode::NoTrans, true).unwrap();
    let recovered = dense_apply(matrix, &solution);
    let inverse_residual = relative_frobenius(&recovered, &rhs);
    println!(
        "biegrid label={label} n={} reconstruction_error={reconstruction_error:.6e} approximation_error={approximation_error:.6e} forward_error={forward_error:.6e} inverse_residual={inverse_residual:.6e}",
        matrix.shape()[0],
    );
    assert!(reconstruction_error < 2.0e-10);
    assert!(approximation_error.is_finite());
    assert!(approximation_error < 0.25);
    assert!(forward_error < 0.25);
    assert!(inverse_residual < 0.25);
}

fn dense_apply<Item: RlstScalar>(
    matrix: &DynArray<Item, 2>,
    input: &DynArray<Item, 2>,
) -> DynArray<Item, 2> {
    let mut result = DynArray::<Item, 2>::from_shape([matrix.shape()[0], input.shape()[1]]);
    for row in 0..matrix.shape()[0] {
        for column in 0..input.shape()[1] {
            let mut value = Item::zero();
            for inner in 0..matrix.shape()[1] {
                value += matrix[[row, inner]] * input[[inner, column]];
            }
            result[[row, column]] = value;
        }
    }
    result
}

#[test]
fn perturbed_biegrid_four_scalar_structures() {
    std::thread::Builder::new()
        .name("rsrs_biegrid_factorized".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(perturbed_biegrid_four_scalar_structures_inner)
        .unwrap()
        .join()
        .unwrap();
}

fn perturbed_biegrid_four_scalar_structures_inner() {
    let fixture = fixture();
    run(&fixture.real_symmetric, &fixture.points, "real-symmetric");
    run(
        &fixture.real_nonsymmetric,
        &fixture.points,
        "real-nonsymmetric",
    );
    run(
        &fixture.complex_symmetric,
        &fixture.points,
        "complex-symmetric",
    );
    run(
        &fixture.complex_nonsymmetric,
        &fixture.points,
        "complex-nonsymmetric",
    );
}
