//! Diagnostic regression tests for RSRS matrix-vector products.
//!
//! These cases are especially useful for separating plain transpose routing
//! from conjugate-transpose routing.
//!
//! The structure of the test file is:
//! - load a deterministic perturbed BIEGrid point cloud and reference matrices,
//! - compare RSRS operator application against both direct factor matvecs and
//!   dense references,
//! - cover symmetric real, nonsymmetric real, complex symmetric,
//!   complex nonsymmetric, and Hermitian complex cases.

use bempp_octree::{generate_random_points, Octree, Point};
use bempp_rsrs::{
    rsrs::{
        args::{RankPicking, RsrsArgs, RsrsOptions, Symmetry},
        rsrs_cycle::Rsrs,
        rsrs_factors::{
            base_factors::BaseFactorOptions,
            commutative_factors::{
                Factor, FactorOperations, FactorType, MulOptions, MultiLevelIdFactors, RsrsFactors,
            },
            null_and_extract::PivotMethod,
            rsrs_operator::{FactType, Inv, RsrsFactorsImpl},
        },
        sketch::Shift,
    },
    utils::linear_algebra::{BlockExtractionMethod, NullMethod},
};
use mpi::{topology::SimpleCommunicator, traits::CommunicatorCollectives};
use num::Complex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Standard, StandardNormal};
use rlst::{
    dense::{
        linalg::{interpolative_decomposition::MatrixIdNoSkel, lu::MatrixLu},
        tools::RandScalar,
    },
    prelude::*,
};

/// Fixed rank used so the regression stays close to the rsrs-exps BIEGrid
/// diagnostic while remaining deterministic across runs.
const TEST_FIXED_RANK: f64 = 16.0;
/// Small tree depth that still produces both off-diagonal and diagonal work.
const TEST_MAX_LEVEL: usize = 3;
/// Leaf occupancy chosen so the test reaches compression quickly.
const TEST_MAX_LEAF_POINTS: usize = 8;
const BIEGRID_FIXTURE_MAGIC: &[u8] = b"RSRS_BIEGRID_FIXTURE_V1";
const RUN_DENSE_DIAGNOSTICS_ENV: &str = "RSRS_RUN_DENSE_DIAGNOSTICS";
// The test embeds a precomputed dense BIEGrid fixture. It does not call the
// Python/BIEGrid generator at test runtime; `generate.py` only regenerates this
// byte blob when the fixture should be changed.
const BIEGRID_FIXTURE_BYTES: &[u8] =
    include_bytes!("fixtures/biegrid_perturbed/cells22_scale1e-2_seed12345.bin");

struct BiegridPerturbedFixture {
    points: Vec<Point>,
    cells_per_axis: usize,
    perturbation_scale: f64,
    perturbation_seed: u64,
    real_symmetric: DynamicArray<f64, 2>,
    real_nonsymmetric: DynamicArray<f64, 2>,
    complex_symmetric: DynamicArray<Complex<f64>, 2>,
    complex_nonsymmetric: DynamicArray<Complex<f64>, 2>,
}

struct FixtureReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> FixtureReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact(&mut self, len: usize) -> &'a [u8] {
        let end = self
            .offset
            .checked_add(len)
            .expect("BIEGrid fixture offset overflow");
        let out = self
            .bytes
            .get(self.offset..end)
            .expect("BIEGrid fixture ended unexpectedly");
        self.offset = end;
        out
    }

    fn read_u64(&mut self) -> u64 {
        u64::from_le_bytes(
            self.read_exact(8)
                .try_into()
                .expect("BIEGrid fixture u64 has wrong width"),
        )
    }

    fn read_f64(&mut self) -> f64 {
        f64::from_le_bytes(
            self.read_exact(8)
                .try_into()
                .expect("BIEGrid fixture f64 has wrong width"),
        )
    }

    fn finish(self) {
        assert_eq!(
            self.offset,
            self.bytes.len(),
            "BIEGrid fixture has trailing bytes"
        );
    }
}

fn load_real_fixture_matrix(reader: &mut FixtureReader<'_>, dim: usize) -> DynamicArray<f64, 2> {
    let mut matrix = rlst_dynamic_array2!(f64, [dim, dim]);
    for row in 0..dim {
        for col in 0..dim {
            matrix[[row, col]] = reader.read_f64();
        }
    }
    matrix
}

fn load_complex_fixture_matrix(
    reader: &mut FixtureReader<'_>,
    dim: usize,
) -> DynamicArray<Complex<f64>, 2> {
    let mut matrix = rlst_dynamic_array2!(Complex<f64>, [dim, dim]);
    for row in 0..dim {
        for col in 0..dim {
            matrix[[row, col]] = Complex::new(reader.read_f64(), reader.read_f64());
        }
    }
    matrix
}

fn load_biegrid_perturbed_fixture() -> BiegridPerturbedFixture {
    let mut reader = FixtureReader::new(BIEGRID_FIXTURE_BYTES);
    assert_eq!(
        reader.read_exact(BIEGRID_FIXTURE_MAGIC.len()),
        BIEGRID_FIXTURE_MAGIC,
        "BIEGrid fixture has an unexpected magic header"
    );

    let dim = reader.read_u64() as usize;
    let cells_per_axis = reader.read_u64() as usize;
    let perturbation_scale = reader.read_f64();
    let perturbation_seed = reader.read_u64();
    let mut points = Vec::with_capacity(dim);
    for _ in 0..dim {
        points.push(Point::new(
            [reader.read_f64(), reader.read_f64(), reader.read_f64()],
            0,
        ));
    }

    let real_symmetric = load_real_fixture_matrix(&mut reader, dim);
    let real_nonsymmetric = load_real_fixture_matrix(&mut reader, dim);
    let complex_symmetric = load_complex_fixture_matrix(&mut reader, dim);
    let complex_nonsymmetric = load_complex_fixture_matrix(&mut reader, dim);
    reader.finish();

    BiegridPerturbedFixture {
        points,
        cells_per_axis,
        perturbation_scale,
        perturbation_seed,
        real_symmetric,
        real_nonsymmetric,
        complex_symmetric,
        complex_nonsymmetric,
    }
}

fn run_dense_diagnostics() -> bool {
    std::env::var(RUN_DENSE_DIAGNOSTICS_ENV)
        .map(|value| !matches!(value.as_str(), "" | "0" | "false" | "False" | "FALSE"))
        .unwrap_or(false)
}

/// Generates deterministic points and projects them back onto an approximate
/// sphere to keep the geometry smooth and reproducible.
#[allow(dead_code)]
fn sphere_surface<C: CommunicatorCollectives>(npoints: usize, comm: &C) -> Vec<Point> {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut points = generate_random_points(npoints, &mut rng, comm);

    let x: Vec<f64> = points.iter().map(|point| point.coords()[0]).collect();
    let y: Vec<f64> = points.iter().map(|point| point.coords()[1]).collect();
    let z: Vec<f64> = points.iter().map(|point| point.coords()[2]).collect();
    let centre = Point::new(
        [
            (x.iter().copied().fold(f64::INFINITY, f64::min)
                + x.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                * 0.5,
            (y.iter().copied().fold(f64::INFINITY, f64::min)
                + y.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                * 0.5,
            (z.iter().copied().fold(f64::INFINITY, f64::min)
                + z.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                * 0.5,
        ],
        0,
    );

    for point in &mut points {
        let mut aux = [
            point.coords()[0] - centre.coords()[0],
            point.coords()[1] - centre.coords()[1],
            point.coords()[2] - centre.coords()[2],
        ];
        let len = (aux[0] * aux[0] + aux[1] * aux[1] + aux[2] * aux[2]).sqrt();
        aux[0] /= len;
        aux[1] /= len;
        aux[2] /= len;
        point.coords_mut()[0] = aux[0] + centre.coords()[0];
        point.coords_mut()[1] = aux[1] + centre.coords()[1];
        point.coords_mut()[2] = aux[2] + centre.coords()[2];
    }

    points
}

/// Scalar Laplace kernel used for the dense reference matrices.
#[allow(dead_code)]
fn laplace_kernel(dist: f64, npoints: usize) -> f64 {
    let pi = std::f64::consts::PI;
    let n = npoints as f64;
    1.0 / (4.0 * pi * n * dist)
}

/// Dense symmetric reference matrix based on the Laplace kernel.
#[allow(dead_code)]
fn laplace_matrix(points: &[Point]) -> DynamicArray<f64, 2> {
    let n = points.len();
    let mut arr = rlst_dynamic_array2!(f64, [n, n]);
    let mut view = arr.r_mut();
    for (i, point_x) in points.iter().enumerate() {
        for (j, point_y) in points.iter().enumerate() {
            let coords_x = point_x.coords();
            let coords_y = point_y.coords();
            let dist = ((coords_x[0] - coords_y[0]).powi(2)
                + (coords_x[1] - coords_y[1]).powi(2)
                + (coords_x[2] - coords_y[2]).powi(2))
            .sqrt();
            view[[i, j]] = if dist > 0.0 {
                laplace_kernel(dist, n)
            } else {
                1.0
            };
        }
    }
    arr
}

/// Real-valued perturbation of the Laplace matrix that deliberately breaks
/// symmetry while staying close to the same scaling.
#[allow(dead_code)]
fn nonsymmetric_real_matrix(points: &[Point]) -> DynamicArray<f64, 2> {
    let base = laplace_matrix(points);
    let n = points.len();
    let mut arr = rlst_dynamic_array2!(f64, [n, n]);
    let mut view = arr.r_mut();
    for i in 0..n {
        for j in 0..n {
            let skew = 0.20 * (points[i].coords()[0] - points[j].coords()[0]);
            view[[i, j]] = if i == j {
                1.0
            } else {
                base[[i, j]] * (1.0 + skew)
            };
        }
    }
    arr
}

/// Complex Hermitian matrix used to exercise transpose and conjugation logic.
///
/// The imaginary part is antisymmetric, so the full matrix satisfies
/// `A^H = A` while still being genuinely complex.
#[allow(dead_code)]
fn hermitian_complex_matrix(points: &[Point]) -> DynamicArray<Complex<f64>, 2> {
    let base = laplace_matrix(points);
    let n = points.len();
    let mut arr = rlst_dynamic_array2!(Complex<f64>, [n, n]);
    let mut view = arr.r_mut();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                view[[i, j]] = Complex::new(1.0, 0.0);
            } else {
                let imag = 0.20 * (points[i].coords()[0] - points[j].coords()[0]);
                view[[i, j]] = Complex::new(base[[i, j]], base[[i, j]] * imag);
            }
        }
    }
    arr
}

/// Complex symmetric matrix used to compare the `Symmetric` and `NoSymm`
/// construction paths on the same operator.
///
/// The imaginary part is symmetric, so `A^T = A`, but the matrix is not
/// Hermitian because conjugation changes the sign of that component.
#[allow(dead_code)]
fn symmetric_complex_matrix(points: &[Point]) -> DynamicArray<Complex<f64>, 2> {
    let base = laplace_matrix(points);
    let n = points.len();
    let mut arr = rlst_dynamic_array2!(Complex<f64>, [n, n]);
    let mut view = arr.r_mut();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                view[[i, j]] = Complex::new(1.0, 0.0);
            } else {
                let imag = 0.15 * (points[i].coords()[0] + points[j].coords()[0])
                    + 0.05 * points[i].coords()[1] * points[j].coords()[1];
                view[[i, j]] = Complex::new(base[[i, j]], base[[i, j]] * imag);
            }
        }
    }
    arr
}

/// Complex matrix with no transpose or Hermitian symmetry.
#[allow(dead_code)]
fn nonsymmetric_complex_matrix(points: &[Point]) -> DynamicArray<Complex<f64>, 2> {
    let base = laplace_matrix(points);
    let n = points.len();
    let mut arr = rlst_dynamic_array2!(Complex<f64>, [n, n]);
    let mut view = arr.r_mut();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                view[[i, j]] = Complex::new(1.0, 0.10);
            } else {
                let real_skew = 0.18 * (points[i].coords()[0] - points[j].coords()[1]);
                let imag_skew = 0.14 * points[i].coords()[2] + 0.07 * points[j].coords()[0]
                    - 0.05 * points[i].coords()[1] * points[j].coords()[2];
                view[[i, j]] =
                    Complex::new(base[[i, j]] * (1.0 + real_skew), base[[i, j]] * imag_skew);
            }
        }
    }
    arr
}

/// Deterministic real input vector used across the real-valued cases.
fn deterministic_real_vector(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = (i + 1) as f64;
            t.sin() + 0.25 * t.cos()
        })
        .collect()
}

/// Deterministic complex input vector used across the complex-valued cases.
fn deterministic_complex_vector(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| {
            let t = (i + 1) as f64;
            Complex::new(t.sin() + 0.25 * t.cos(), 0.5 * t.cos() - 0.15 * t.sin())
        })
        .collect()
}

/// Relative `l2` error with a small denominator floor for near-zero references.
fn rel_l2_error<Item: RlstScalar>(actual: &[Item], expected: &[Item]) -> f64 {
    let mut actual_arr = rlst_dynamic_array1!(Item, [actual.len()]);
    let mut expected_arr = rlst_dynamic_array1!(Item, [expected.len()]);
    actual_arr.fill_from_raw_data(actual);
    expected_arr.fill_from_raw_data(expected);

    let mut diff = empty_array();
    diff.fill_from_resize(actual_arr.r() - expected_arr.r());

    let num = diff.norm_2();
    let den = expected_arr.norm_2();
    let num_f64: f64 = num::NumCast::from(num).unwrap();
    let den_f64: f64 = num::NumCast::from(den).unwrap();
    num_f64 / den_f64.max(1.0e-14)
}

// Glossary for the vector relative-l2 diagnostics printed in this test:
// - `op_vs_factor_left`: public operator `NoTrans` versus direct left factor apply.
// - `op_vs_factor_trans_identity`: public operator `Trans` versus direct right
//   factor apply, the identity used to implement transpose-vector products.
// - `left_dense`: public operator `NoTrans` versus dense `A x`.
// - `trans_dense`: public operator `Trans` versus dense `A^T x`.
// - `left_trans_dense`: direct left factor `Trans` apply versus dense `A^T x`.
// - `right_dense`: direct right factor `NoTrans` apply versus dense `x A`.
// - `right_trans_dense`: direct right factor `Trans` apply versus dense `x A^T`.
// - `conj_no_trans`: public operator `ConjNoTrans` versus dense `conj(A) x`.
// - `conj_trans`: public operator `ConjTrans` versus dense `A^H x`.
// - `no_trans_vs_trans`: public operator `A x` versus public operator `A^T x`.
// - `dense_no_trans_vs_trans`: dense `A x` versus dense `A^T x`; this is the
//   reference signal that a nonsymmetric matrix has not collapsed to symmetric.

/// Dense reference apply used to validate both the operator wrapper and the
/// direct factor-level matvec path.
///
/// `use_adjoint_reference` exists for cases where a caller wants the dense
/// `Trans` reference to behave like an adjoint instead of a plain transpose.
fn dense_apply<Item: RlstScalar>(
    matrix: &DynamicArray<Item, 2>,
    input: &[Item],
    side: Side,
    trans_mode: TransMode,
    use_adjoint_reference: bool,
) -> Vec<Item> {
    let mut input_arr = match side {
        Side::Left => rlst_dynamic_array2!(Item, [input.len(), 1]),
        Side::Right => rlst_dynamic_array2!(Item, [1, input.len()]),
    };
    input_arr.fill_from_raw_data(input);

    let mut out = empty_array();
    if use_adjoint_reference && matches!(trans_mode, TransMode::Trans) {
        let mut adjoint = empty_array();
        adjoint.fill_from_resize(matrix.r().transpose().conj());
        match side {
            Side::Left => out
                .r_mut()
                .simple_mult_into_resize(adjoint.r(), input_arr.r()),
            Side::Right => out
                .r_mut()
                .simple_mult_into_resize(input_arr.r(), adjoint.r()),
        };
    } else {
        match side {
            Side::Left => out.r_mut().mult_into_resize(
                trans_mode,
                TransMode::NoTrans,
                Item::from_real(Item::real(1.0)),
                matrix.r(),
                input_arr.r(),
                Item::from_real(Item::real(0.0)),
            ),
            Side::Right => out.r_mut().mult_into_resize(
                TransMode::NoTrans,
                trans_mode,
                Item::from_real(Item::real(1.0)),
                input_arr.r(),
                matrix.r(),
                Item::from_real(Item::real(0.0)),
            ),
        };
    }

    out.r().iter().collect()
}

/// Canonical basis vector used to materialize full operator action matrices.
fn basis_vector<Item: RlstScalar>(dim: usize, index: usize) -> Vec<Item> {
    let mut basis = vec![Item::from_real(Item::real(0.0)); dim];
    basis[index] = Item::from_real(Item::real(1.0));
    basis
}

/// Flattens a dense matrix in column-major order so transposed actions can be
/// compared with basis-vector operator assembly.
fn flatten_column_major<Item: RlstScalar + Copy>(matrix: &DynamicArray<Item, 2>) -> Vec<Item> {
    let shape = matrix.shape();
    let view = matrix.r();
    let mut values = Vec::with_capacity(shape[0] * shape[1]);

    for col in 0..shape[1] {
        for row in 0..shape[0] {
            values.push(view[[row, col]]);
        }
    }

    values
}

/// Transposes a square column-major matrix while preserving the same flattened
/// column-major layout in the output.
fn transpose_column_major<Item: Copy>(matrix: &[Item], dim: usize) -> Vec<Item> {
    let mut transposed = Vec::with_capacity(matrix.len());

    for col in 0..dim {
        for row in 0..dim {
            transposed.push(matrix[row * dim + col]);
        }
    }

    transposed
}

fn conjugate_column_major<Item: RlstScalar>(matrix: &[Item]) -> Vec<Item> {
    matrix.iter().map(|value| value.conj()).collect()
}

fn adjoint_column_major<Item: RlstScalar>(matrix: &[Item], dim: usize) -> Vec<Item> {
    conjugate_column_major(&transpose_column_major(matrix, dim))
}

/// Materializes the full action matrix of the public RSRS operator for one
/// transpose mode by applying it to each basis vector.
fn assemble_operator_matrix<Item, Op>(op: &Op, dim: usize, trans_mode: TransMode) -> Vec<Item>
where
    Item: RlstScalar,
    Op: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
{
    let mut matrix = Vec::with_capacity(dim * dim);

    for col in 0..dim {
        let basis = basis_vector(dim, col);
        matrix.extend(apply_operator(op, &basis, trans_mode));
    }

    matrix
}

/// Materializes the left- or right-application matrix exposed by the factor
/// container, again by probing the canonical basis.
fn assemble_factor_matrix<Item, Factors>(
    factors: &Factors,
    dim: usize,
    side: Side,
    base_options: &BaseFactorOptions,
) -> Vec<Item>
where
    Item: RlstScalar,
    Factors: RsrsFactorsImpl<Item>,
{
    let mut matrix = Vec::with_capacity(dim * dim);

    for col in 0..dim {
        let basis = basis_vector(dim, col);
        let mut output = vec![Item::from_real(Item::real(0.0)); dim];
        factors.matvec(&basis, &mut output, side, base_options);
        matrix.extend(output);
    }

    matrix
}

/// Applies the high-level `rlst` operator interface and materializes the output
/// as a plain vector for easy comparison.
fn apply_operator<Item, Op>(op: &Op, input: &[Item], trans_mode: TransMode) -> Vec<Item>
where
    Item: RlstScalar,
    Op: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
{
    let mut x = zero_element(op.domain());
    x.imp_mut().fill_inplace_raw(input);
    let y = op.apply(x.r(), trans_mode);
    y.view().iter().collect()
}

fn frobenius_norm_flat<Item: RlstScalar>(matrix: &[Item]) -> f64 {
    matrix
        .iter()
        .map(|value| {
            let norm_sqr: f64 = num::NumCast::from((*value).square()).unwrap();
            norm_sqr
        })
        .sum::<f64>()
        .sqrt()
}

fn rel_fro_error_flat<Item: RlstScalar>(actual: &[Item], expected: &[Item]) -> f64 {
    let diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| {
            let norm_sqr: f64 = num::NumCast::from((*actual - *expected).square()).unwrap();
            norm_sqr
        })
        .sum::<f64>()
        .sqrt();

    diff / frobenius_norm_flat(expected).max(1.0e-14)
}

fn multiply_column_major<Item: RlstScalar>(left: &[Item], right: &[Item], dim: usize) -> Vec<Item> {
    let mut product = vec![Item::from_real(Item::real(0.0)); dim * dim];
    for col in 0..dim {
        for row in 0..dim {
            let mut value = Item::from_real(Item::real(0.0));
            for k in 0..dim {
                value = value + left[row + k * dim] * right[k + col * dim];
            }
            product[row + col * dim] = value;
        }
    }

    product
}

fn rel_identity_fro_error_flat<Item: RlstScalar>(matrix: &[Item], dim: usize) -> f64 {
    let diff = matrix
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let row = index % dim;
            let col = index / dim;
            let expected = if row == col {
                Item::from_real(Item::real(1.0))
            } else {
                Item::from_real(Item::real(0.0))
            };
            let norm_sqr: f64 = num::NumCast::from((*value - expected).square()).unwrap();
            norm_sqr
        })
        .sum::<f64>()
        .sqrt();

    diff / (dim as f64).sqrt().max(1.0e-14)
}

fn run_inverse_dense_diagnostic<Item, Op>(
    label: &str,
    matrix: &DynamicArray<Item, 2>,
    rsrs_op: &mut Op,
) where
    Item: RlstScalar,
    Op: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>> + Inv,
{
    let dim = matrix.shape()[0];
    let dense_matrix = flatten_column_major(matrix);

    rsrs_op.inv(true);
    let inverse_matrix = assemble_operator_matrix(rsrs_op, dim, TransMode::NoTrans);
    rsrs_op.inv(false);

    // Full left inverse check: ||RSRS^{-1} A - I||_F / ||I||_F.
    let left_identity = multiply_column_major(&inverse_matrix, &dense_matrix, dim);
    // Full right inverse check: ||A RSRS^{-1} - I||_F / ||I||_F.
    let right_identity = multiply_column_major(&dense_matrix, &inverse_matrix, dim);
    // Full sandwich check: ||RSRS^{-1} A RSRS^{-1} - RSRS^{-1}||_F / ||RSRS^{-1}||_F.
    let sandwich = multiply_column_major(&left_identity, &inverse_matrix, dim);

    println!(
        "{label}: dense_inverse full_left_identity={:.3e}, full_right_identity={:.3e}, full_sandwich_vs_inverse={:.3e}, inverse_norm={:.3e}",
        rel_identity_fro_error_flat(&left_identity, dim),
        rel_identity_fro_error_flat(&right_identity, dim),
        rel_fro_error_flat(&sandwich, &inverse_matrix),
        frobenius_norm_flat(&inverse_matrix)
    );
}

fn clone_matrix<Item: RlstScalar>(matrix: &DynamicArray<Item, 2>) -> DynamicArray<Item, 2> {
    let mut out = empty_array();
    out.r_mut().fill_from_resize(matrix.r());
    out
}

fn block_fro_norm<Item: RlstScalar>(
    matrix: &DynamicArray<Item, 2>,
    rows: &[usize],
    cols: &[usize],
) -> f64 {
    rows.iter()
        .flat_map(|&row| cols.iter().map(move |&col| matrix[[row, col]]))
        .map(|value| {
            let norm_sqr: f64 = num::NumCast::from(value.square()).unwrap();
            norm_sqr
        })
        .sum::<f64>()
        .sqrt()
}

fn apply_dense_factor<Item>(
    factor: &Factor<Item>,
    target: &mut DynamicArray<Item, 2>,
    options: &MulOptions,
) where
    Item: RlstScalar
        + RandScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    StandardNormal: rand_distr::Distribution<Item::Real>,
    Standard: rand_distr::Distribution<Item::Real>,
    <Item as RlstScalar>::Real: RandScalar,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    match factor {
        Factor::Id(id_factor) => id_factor.mul(target, options),
        Factor::Lu(lu_factor) => lu_factor.mul(target, options),
        Factor::Diag(diag_factor) => diag_factor.mul(target, options),
    }
}

#[derive(Clone, Copy)]
enum DenseBoxApplyMode {
    LeftOnly,
    RightOnly,
    Sandwich,
}

impl DenseBoxApplyMode {
    fn label(self) -> &'static str {
        match self {
            Self::LeftOnly => "left_only",
            Self::RightOnly => "right_only",
            Self::Sandwich => "sandwich",
        }
    }
}

#[derive(Clone, Copy)]
struct DenseBoxNorms {
    first: f64,
    second: f64,
}

struct DenseBoxKind {
    name: &'static str,
    first_label: &'static str,
    second_label: &'static str,
}

fn dense_box_rel_norms(after: DenseBoxNorms, before: DenseBoxNorms) -> DenseBoxNorms {
    DenseBoxNorms {
        first: after.first / before.first.max(1.0e-14),
        second: after.second / before.second.max(1.0e-14),
    }
}

fn dense_box_norms<Item: RlstScalar>(
    factor: &Factor<Item>,
    target: &DynamicArray<Item, 2>,
) -> Option<(DenseBoxKind, DenseBoxNorms)> {
    match factor {
        Factor::Id(id_factor) => Some((
            DenseBoxKind {
                name: "id",
                first_label: "rf",
                second_label: "fr",
            },
            DenseBoxNorms {
                // ID left-zero target: residual rows against far-field columns.
                first: block_fro_norm(target, &id_factor.ind_r, &id_factor.ind_f),
                // ID right-zero target: far-field rows against residual columns.
                second: block_fro_norm(target, &id_factor.ind_f, &id_factor.ind_r),
            },
        )),
        Factor::Lu(lu_factor) => Some((
            DenseBoxKind {
                name: "lu",
                first_label: "rt",
                second_label: "tr",
            },
            DenseBoxNorms {
                // LU left-zero target: residual rows against target columns.
                first: block_fro_norm(target, &lu_factor.ind_r, &lu_factor.ind_t),
                // LU right-zero target: target rows against residual columns.
                second: block_fro_norm(target, &lu_factor.ind_t, &lu_factor.ind_r),
            },
        )),
        Factor::Diag(_) => None,
    }
}

fn apply_dense_box_factor_mode<Item>(
    factor: &Factor<Item>,
    target: &mut DynamicArray<Item, 2>,
    mode: DenseBoxApplyMode,
    left_options: &MulOptions,
    right_options: &MulOptions,
) where
    Item: RlstScalar
        + RandScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    StandardNormal: rand_distr::Distribution<Item::Real>,
    Standard: rand_distr::Distribution<Item::Real>,
    <Item as RlstScalar>::Real: RandScalar,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    match mode {
        DenseBoxApplyMode::LeftOnly => apply_dense_factor(factor, target, left_options),
        DenseBoxApplyMode::RightOnly => apply_dense_factor(factor, target, right_options),
        DenseBoxApplyMode::Sandwich => {
            apply_dense_factor(factor, target, left_options);
            apply_dense_factor(factor, target, right_options);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_dense_box_error_factor<Item>(
    label: &str,
    mode: DenseBoxApplyMode,
    level: usize,
    batch: usize,
    index: usize,
    factor: &Factor<Item>,
    target: &mut DynamicArray<Item, 2>,
    left_options: &MulOptions,
    right_options: &MulOptions,
) -> Option<DenseBoxNorms>
where
    Item: RlstScalar
        + RandScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    StandardNormal: rand_distr::Distribution<Item::Real>,
    Standard: rand_distr::Distribution<Item::Real>,
    <Item as RlstScalar>::Real: RandScalar,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    let (kind, before) = dense_box_norms(factor, target)?;

    let mut conj_target = empty_array();
    conj_target.r_mut().fill_from_resize(target.r().conj());

    apply_dense_box_factor_mode(factor, target, mode, left_options, right_options);
    apply_dense_box_factor_mode(factor, &mut conj_target, mode, left_options, right_options);

    let (_, after) = dense_box_norms(factor, target).expect("ID/LU factor kind changed");
    let (_, conj_after) = dense_box_norms(factor, &conj_target).expect("ID/LU factor kind changed");
    let rel = dense_box_rel_norms(after, before);
    let conj_rel = dense_box_rel_norms(conj_after, before);

    println!(
        "{label}: dense_boxes mode={} family={} level={} batch={} index={} target=({}={:.3e}, {}={:.3e}) conj_target=({}={:.3e}, {}={:.3e}) before=({}={:.3e}, {}={:.3e}) after=({}={:.3e}, {}={:.3e}) conj_after=({}={:.3e}, {}={:.3e})",
        mode.label(),
        kind.name,
        level,
        batch,
        index,
        kind.first_label,
        rel.first,
        kind.second_label,
        rel.second,
        kind.first_label,
        conj_rel.first,
        kind.second_label,
        conj_rel.second,
        kind.first_label,
        before.first,
        kind.second_label,
        before.second,
        kind.first_label,
        after.first,
        kind.second_label,
        after.second,
        kind.first_label,
        conj_after.first,
        kind.second_label,
        conj_after.second
    );

    Some(rel)
}

fn run_dense_box_error_batch<Item>(
    label: &str,
    mode: DenseBoxApplyMode,
    level: usize,
    batch: usize,
    factors: &[Factor<Item>],
    target: &mut DynamicArray<Item, 2>,
    left_options: &MulOptions,
    right_options: &MulOptions,
) where
    Item: RlstScalar
        + RandScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    StandardNormal: rand_distr::Distribution<Item::Real>,
    Standard: rand_distr::Distribution<Item::Real>,
    <Item as RlstScalar>::Real: RandScalar,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    for (index, factor) in factors.iter().enumerate() {
        run_dense_box_error_factor(
            label,
            mode,
            level,
            batch,
            index,
            factor,
            target,
            left_options,
            right_options,
        );
    }
}

fn run_dense_box_errors_mode<Item>(
    label: &str,
    mode: DenseBoxApplyMode,
    matrix: &DynamicArray<Item, 2>,
    factors: &RsrsFactors<Item>,
) where
    Item: RlstScalar
        + RandScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    StandardNormal: rand_distr::Distribution<Item::Real>,
    Standard: rand_distr::Distribution<Item::Real>,
    <Item as RlstScalar>::Real: RandScalar,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    println!(
        "{label}: dense_boxes mode={} copied from get_boxes_errors; target matrix is mutated step by step",
        mode.label()
    );

    let mut target = clone_matrix(matrix);
    let base_options = BaseFactorOptions {
        inv: true,
        trans: TransMode::NoTrans,
        trans_target: false,
    };
    let left_options = MulOptions {
        base_options: base_options.clone(),
        side: Side::Left,
        factor_type: FactorType::F,
    };
    let right_options = MulOptions {
        base_options,
        side: Side::Right,
        factor_type: FactorType::S,
    };

    match &factors.id_factors {
        MultiLevelIdFactors::Batched(levels) => {
            for level in 0..factors.num_levels {
                for (batch, id_batch) in levels[level].iter().enumerate() {
                    run_dense_box_error_batch(
                        label,
                        mode,
                        level,
                        batch,
                        id_batch,
                        &mut target,
                        &left_options,
                        &right_options,
                    );
                    run_dense_box_error_batch(
                        label,
                        mode,
                        level,
                        batch,
                        &factors.lu_factors[level][batch],
                        &mut target,
                        &left_options,
                        &right_options,
                    );
                }
            }
        }
        MultiLevelIdFactors::Single(levels) => {
            for level in 0..factors.num_levels {
                run_dense_box_error_batch(
                    label,
                    mode,
                    level,
                    0,
                    &levels[level],
                    &mut target,
                    &left_options,
                    &right_options,
                );
                for (batch, lu_batch) in factors.lu_factors[level].iter().enumerate() {
                    run_dense_box_error_batch(
                        label,
                        mode,
                        level,
                        batch,
                        lu_batch,
                        &mut target,
                        &left_options,
                        &right_options,
                    );
                }
            }
        }
    }
}

fn run_dense_box_errors_left_application<Item>(
    label: &str,
    matrix: &DynamicArray<Item, 2>,
    factors: &RsrsFactors<Item>,
) where
    Item: RlstScalar
        + RandScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    StandardNormal: rand_distr::Distribution<Item::Real>,
    Standard: rand_distr::Distribution<Item::Real>,
    <Item as RlstScalar>::Real: RandScalar,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    run_dense_box_errors_mode(label, DenseBoxApplyMode::LeftOnly, matrix, factors);
}

fn run_dense_box_errors_right_application<Item>(
    label: &str,
    matrix: &DynamicArray<Item, 2>,
    factors: &RsrsFactors<Item>,
) where
    Item: RlstScalar
        + RandScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    StandardNormal: rand_distr::Distribution<Item::Real>,
    Standard: rand_distr::Distribution<Item::Real>,
    <Item as RlstScalar>::Real: RandScalar,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    run_dense_box_errors_mode(label, DenseBoxApplyMode::RightOnly, matrix, factors);
}

fn run_dense_box_errors_sandwich_application<Item>(
    label: &str,
    matrix: &DynamicArray<Item, 2>,
    factors: &RsrsFactors<Item>,
) where
    Item: RlstScalar
        + RandScalar
        + MatrixId
        + MatrixIdNoSkel
        + MatrixInverse
        + MatrixPseudoInverse
        + MatrixLu
        + MatrixQr,
    StandardNormal: rand_distr::Distribution<Item::Real>,
    Standard: rand_distr::Distribution<Item::Real>,
    <Item as RlstScalar>::Real: RandScalar,
    LuDecomposition<Item, BaseArray<Item, VectorContainer<Item>, 2>>:
        MatrixLuDecomposition<Item = Item>,
    TriangularMatrix<Item>: TriangularOperations<Item = Item>,
{
    run_dense_box_errors_mode(label, DenseBoxApplyMode::Sandwich, matrix, factors);
}

struct FrobeniusMetrics {
    // ||A||_F for the dense no-transpose reference.
    reference_norm: f64,
    // ||A^T||_F for the dense transpose reference.
    transpose_reference_norm: f64,
    // rsrs-exps `rel_fro`/`apply_fro`: ||RSRS - A||_F / ||A||_F.
    op_no_trans_dense_err: f64,
    // rsrs-exps `rel_fro_T`: ||RSRS^T_route - A^T||_F / ||A^T||_F.
    op_trans_dense_err: f64,
    // ||factor_right - A^T||_F / ||A^T||_F, checking the right-apply route.
    factor_right_dense_err: f64,
    // ||operator_no_trans - factor_left||_F / ||factor_left||_F.
    op_no_trans_factor_err: f64,
    // ||operator_trans - factor_right||_F / ||factor_right||_F.
    op_trans_factor_err: f64,
    // Plain transpose-route check: ||operator_trans - operator_no_trans^T||_F.
    op_trans_vs_no_trans_transpose: f64,
    // rsrs-exps `adj(T)`: ||operator_trans - operator_no_trans^H||_F.
    // For real matrices this is the same as the plain transpose-route check.
    op_trans_vs_no_trans_adjoint: f64,
    // Plain factor transpose-route check: ||factor_right - factor_left^T||_F.
    factor_right_vs_left_transpose: f64,
    // Symmetry collapse check: ||operator_no_trans - operator_trans||_F.
    no_trans_vs_trans: f64,
    // Dense-reference symmetry check: ||A - A^T||_F.
    dense_no_trans_vs_trans: f64,
}

fn run_frobenius_matrix_check<Item, Op, Factors>(
    label: &str,
    matrix: &DynamicArray<Item, 2>,
    rsrs_op: &Op,
    factors: &Factors,
    base_no_trans: &BaseFactorOptions,
) -> FrobeniusMetrics
where
    Item: RlstScalar,
    Op: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
    Factors: RsrsFactorsImpl<Item>,
{
    let dim = matrix.shape()[0];
    let op_no_trans_matrix = assemble_operator_matrix(rsrs_op, dim, TransMode::NoTrans);
    let op_trans_matrix = assemble_operator_matrix(rsrs_op, dim, TransMode::Trans);
    let factor_left_matrix = assemble_factor_matrix(factors, dim, Side::Left, base_no_trans);
    let factor_right_matrix = assemble_factor_matrix(factors, dim, Side::Right, base_no_trans);
    let dense_no_trans_matrix = flatten_column_major(matrix);
    let dense_trans_matrix = transpose_column_major(&dense_no_trans_matrix, dim);

    let metrics = FrobeniusMetrics {
        reference_norm: frobenius_norm_flat(&dense_no_trans_matrix),
        transpose_reference_norm: frobenius_norm_flat(&dense_trans_matrix),
        op_no_trans_dense_err: rel_fro_error_flat(&op_no_trans_matrix, &dense_no_trans_matrix),
        op_trans_dense_err: rel_fro_error_flat(&op_trans_matrix, &dense_trans_matrix),
        factor_right_dense_err: rel_fro_error_flat(&factor_right_matrix, &dense_trans_matrix),
        op_no_trans_factor_err: rel_fro_error_flat(&op_no_trans_matrix, &factor_left_matrix),
        op_trans_factor_err: rel_fro_error_flat(&op_trans_matrix, &factor_right_matrix),
        op_trans_vs_no_trans_transpose: rel_fro_error_flat(
            &op_trans_matrix,
            &transpose_column_major(&op_no_trans_matrix, dim),
        ),
        op_trans_vs_no_trans_adjoint: rel_fro_error_flat(
            &op_trans_matrix,
            &adjoint_column_major(&op_no_trans_matrix, dim),
        ),
        factor_right_vs_left_transpose: rel_fro_error_flat(
            &factor_right_matrix,
            &transpose_column_major(&factor_left_matrix, dim),
        ),
        no_trans_vs_trans: rel_fro_error_flat(&op_no_trans_matrix, &op_trans_matrix),
        dense_no_trans_vs_trans: rel_fro_error_flat(&dense_no_trans_matrix, &dense_trans_matrix),
    };

    println!(
        "{label}: frob_ref={:.3e}, frob_ref_trans={:.3e}, frob_no_trans_dense={:.3e}, frob_trans_dense={:.3e}, frob_factor_right_dense={:.3e}, frob_no_trans_factor={:.3e}, frob_trans_factor={:.3e}, frob_trans_vs_no_trans_transpose={:.3e}, frob_trans_vs_no_trans_adjoint={:.3e}, frob_factor_right_vs_left_transpose={:.3e}, frob_no_trans_vs_trans={:.3e}, frob_dense_no_trans_vs_trans={:.3e}",
        metrics.reference_norm,
        metrics.transpose_reference_norm,
        metrics.op_no_trans_dense_err,
        metrics.op_trans_dense_err,
        metrics.factor_right_dense_err,
        metrics.op_no_trans_factor_err,
        metrics.op_trans_factor_err,
        metrics.op_trans_vs_no_trans_transpose,
        metrics.op_trans_vs_no_trans_adjoint,
        metrics.factor_right_vs_left_transpose,
        metrics.no_trans_vs_trans,
        metrics.dense_no_trans_vs_trans
    );
    // Frobenius diagnostics mirrored from rsrs-exps. We deliberately do not
    // add the rsrs-exps l2/spectral/solve checks here; the existing single
    // vector relative-l2 diagnostics remain printed separately above.
    println!(
        "{label}: rsrs_exps_frob rel_fro={:.3e}, rel_fro_T={:.3e}, apply_fro={:.3e}, adj(T)={:.3e}, transpose_route={:.3e}",
        metrics.op_no_trans_dense_err,
        metrics.op_trans_dense_err,
        metrics.op_no_trans_dense_err,
        metrics.op_trans_vs_no_trans_adjoint,
        metrics.op_trans_vs_no_trans_transpose
    );

    metrics
}

struct ConjugateFrobeniusMetrics {
    // ||ConjNoTrans(RSRS) - conj(A)||_F / ||conj(A)||_F.
    conj_no_trans_dense_err: f64,
    // ||ConjTrans(RSRS) - A^H||_F / ||A^H||_F.
    conj_trans_dense_err: f64,
    // rsrs-exps `adj(H)`: ||ConjTrans(RSRS) - NoTrans(RSRS)^H||_F.
    conj_trans_vs_no_trans_adjoint: f64,
    // Exact dense identity error for direct ConjTrans vs conj(Trans(conj(x))).
    exact_manual_conjtrans: f64,
    // RSRS identity error for direct ConjTrans vs conj(Trans(conj(x))).
    rsrs_manual_conjtrans: f64,
}

fn run_conjugate_frobenius_matrix_check<Item, Op>(
    label: &str,
    matrix: &DynamicArray<Item, 2>,
    rsrs_op: &Op,
) -> ConjugateFrobeniusMetrics
where
    Item: RlstScalar,
    Op: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
{
    let dim = matrix.shape()[0];
    let dense_no_trans_matrix = flatten_column_major(matrix);
    let dense_trans_matrix = transpose_column_major(&dense_no_trans_matrix, dim);
    let dense_conj_no_trans_matrix = conjugate_column_major(&dense_no_trans_matrix);
    let dense_conj_trans_matrix = adjoint_column_major(&dense_no_trans_matrix, dim);
    let op_no_trans_matrix = assemble_operator_matrix(rsrs_op, dim, TransMode::NoTrans);
    let op_trans_matrix = assemble_operator_matrix(rsrs_op, dim, TransMode::Trans);
    let op_conj_no_trans_matrix = assemble_operator_matrix(rsrs_op, dim, TransMode::ConjNoTrans);
    let op_conj_trans_matrix = assemble_operator_matrix(rsrs_op, dim, TransMode::ConjTrans);

    let metrics = ConjugateFrobeniusMetrics {
        conj_no_trans_dense_err: rel_fro_error_flat(
            &op_conj_no_trans_matrix,
            &dense_conj_no_trans_matrix,
        ),
        conj_trans_dense_err: rel_fro_error_flat(&op_conj_trans_matrix, &dense_conj_trans_matrix),
        conj_trans_vs_no_trans_adjoint: rel_fro_error_flat(
            &op_conj_trans_matrix,
            &adjoint_column_major(&op_no_trans_matrix, dim),
        ),
        exact_manual_conjtrans: rel_fro_error_flat(
            &dense_conj_trans_matrix,
            &conjugate_column_major(&dense_trans_matrix),
        ),
        rsrs_manual_conjtrans: rel_fro_error_flat(
            &op_conj_trans_matrix,
            &conjugate_column_major(&op_trans_matrix),
        ),
    };

    println!(
        "{label}: frob_conj_no_trans_dense={:.3e}, frob_conj_trans_dense={:.3e}, frob_conj_trans_vs_no_trans_adjoint={:.3e}, frob_exact_manual_H={:.3e}, frob_rsrs_manual_H={:.3e}",
        metrics.conj_no_trans_dense_err,
        metrics.conj_trans_dense_err,
        metrics.conj_trans_vs_no_trans_adjoint,
        metrics.exact_manual_conjtrans,
        metrics.rsrs_manual_conjtrans
    );
    println!(
        "{label}: rsrs_exps_complex_frob adj(H)={:.3e}, exact_manual_H={:.3e}, rsrs_manual_H={:.3e}",
        metrics.conj_trans_vs_no_trans_adjoint,
        metrics.exact_manual_conjtrans,
        metrics.rsrs_manual_conjtrans
    );

    metrics
}

/// Convenience helper for the explicit conjugation identities used in the
/// Hermitian test.
fn conjugated<Item: RlstScalar>(values: &[Item]) -> Vec<Item> {
    values.iter().map(|value| value.conj()).collect()
}

/// Runs the common real-valued regression checks.
///
/// Each case compares three views of the same operation:
/// - the public operator wrapper,
/// - direct factor matvec application,
/// - dense multiplication against the original matrix.
fn run_real_case(
    points: &[Point],
    comm: &SimpleCommunicator,
    matrix: DynamicArray<f64, 2>,
    symmetry: Symmetry,
    label: &str,
    transpose_reference_is_adjoint: bool,
    dense_tolerance: f64,
) {
    let n = matrix.shape()[0];
    let tree = Octree::new(points, TEST_MAX_LEVEL, TEST_MAX_LEAF_POINTS, comm);
    let args = RsrsArgs::new(
        8,
        16,
        0,
        120,
        Shift::False,
        NullMethod::Projection,
        RankRevealingQrType::SRRQR(1.01),
        BlockExtractionMethod::LuLstSq,
        BlockExtractionMethod::LuLstSq,
        PivotMethod::Lu(1e-10),
        PivotMethod::Lu(0.0),
        1e-10,
        TEST_FIXED_RANK,
        1e-10,
        1e-10,
        4,
        1,
        symmetry,
        RankPicking::Min,
        FactType::Joint,
        false,
        1,
        false,
        true,
    );
    let options = RsrsOptions::<f64>::new(Some(args));
    let operator = Operator::from(&matrix);
    let mut rsrs = Rsrs::new(&tree, options, operator.domain().dimension());
    let rsrs_op = rsrs.get_rsrs_operator(operator);
    let factors = rsrs_op.get_factors();
    let x = deterministic_real_vector(n);
    let base_no_trans = BaseFactorOptions {
        inv: false,
        trans: TransMode::NoTrans,
        trans_target: false,
    };
    let base_trans = BaseFactorOptions {
        inv: false,
        trans: TransMode::Trans,
        trans_target: false,
    };

    let op_left = apply_operator(&rsrs_op, &x, TransMode::NoTrans);
    let op_trans = apply_operator(&rsrs_op, &x, TransMode::Trans);
    let mut factor_left = vec![0.0; n];
    let mut factor_left_trans = vec![0.0; n];
    let mut factor_right = vec![0.0; n];
    let mut factor_right_trans = vec![0.0; n];
    factors.matvec(&x, &mut factor_left, Side::Left, &base_no_trans);
    factors.matvec(&x, &mut factor_left_trans, Side::Left, &base_trans);
    factors.matvec(&x, &mut factor_right, Side::Right, &base_no_trans);
    factors.matvec(&x, &mut factor_right_trans, Side::Right, &base_trans);

    let dense_left = dense_apply(
        &matrix,
        &x,
        Side::Left,
        TransMode::NoTrans,
        transpose_reference_is_adjoint,
    );
    let dense_left_trans = dense_apply(
        &matrix,
        &x,
        Side::Left,
        TransMode::Trans,
        transpose_reference_is_adjoint,
    );
    let dense_right = dense_apply(
        &matrix,
        &x,
        Side::Right,
        TransMode::NoTrans,
        transpose_reference_is_adjoint,
    );
    let dense_right_trans = dense_apply(
        &matrix,
        &x,
        Side::Right,
        TransMode::Trans,
        transpose_reference_is_adjoint,
    );

    // `apply_vec_mode` implements transpose-vector products through a right
    // application identity, so the operator `Trans` result is expected to match
    // the factor-level right-apply path.
    let op_vs_factor_left = rel_l2_error(&op_left, &factor_left);
    let op_vs_factor_trans = rel_l2_error(&op_trans, &factor_right);
    let left_dense_err = rel_l2_error(&op_left, &dense_left);
    let trans_dense_err = rel_l2_error(&op_trans, &dense_left_trans);
    let left_trans_dense_err = rel_l2_error(&factor_left_trans, &dense_left_trans);
    let right_dense_err = rel_l2_error(&factor_right, &dense_right);
    let right_trans_dense_err = rel_l2_error(&factor_right_trans, &dense_right_trans);
    let no_trans_vs_trans = rel_l2_error(&op_left, &op_trans);

    println!(
        "{label}: op_vs_factor_left={op_vs_factor_left:.3e}, op_vs_factor_trans_identity={op_vs_factor_trans:.3e}, left_dense={left_dense_err:.3e}, trans_dense={trans_dense_err:.3e}, left_trans_dense={left_trans_dense_err:.3e}, right_dense={right_dense_err:.3e}, right_trans_dense={right_trans_dense_err:.3e}, no_trans_vs_trans={no_trans_vs_trans:.3e}"
    );

    let frob_metrics =
        run_frobenius_matrix_check(label, &matrix, &rsrs_op, factors, &base_no_trans);

    assert!(
        op_vs_factor_left <= 1.0e-12,
        "{label}: operator NoTrans diverges from factor matvec (rel l2 = {op_vs_factor_left})"
    );
    assert!(
        op_vs_factor_trans <= 1.0e-12,
        "{label}: operator Trans diverges from factor matvec (rel l2 = {op_vs_factor_trans})"
    );
    assert!(
        left_dense_err <= dense_tolerance,
        "{label}: NoTrans RSRS-vs-dense error too large ({left_dense_err} > {dense_tolerance})"
    );
    assert!(
        trans_dense_err <= dense_tolerance,
        "{label}: Trans RSRS-vs-dense error too large ({trans_dense_err} > {dense_tolerance})"
    );
    assert!(
        left_trans_dense_err <= dense_tolerance,
        "{label}: left Trans factor matvec error too large ({left_trans_dense_err} > {dense_tolerance})"
    );
    assert!(
        right_dense_err <= dense_tolerance,
        "{label}: right NoTrans factor matvec error too large ({right_dense_err} > {dense_tolerance})"
    );
    assert!(
        right_trans_dense_err <= dense_tolerance,
        "{label}: right Trans factor matvec error too large ({right_trans_dense_err} > {dense_tolerance})"
    );
    assert!(
        frob_metrics.op_no_trans_factor_err <= 1.0e-12,
        "{label}: Frobenius operator NoTrans diverges from factor matrix ({})",
        frob_metrics.op_no_trans_factor_err
    );
    assert!(
        frob_metrics.op_trans_factor_err <= 1.0e-12,
        "{label}: Frobenius operator Trans diverges from factor right-apply matrix ({})",
        frob_metrics.op_trans_factor_err
    );
    assert!(
        frob_metrics.op_no_trans_dense_err <= dense_tolerance,
        "{label}: Frobenius NoTrans RSRS-vs-dense error too large ({} > {dense_tolerance})",
        frob_metrics.op_no_trans_dense_err
    );
    assert!(
        frob_metrics.op_trans_dense_err <= dense_tolerance,
        "{label}: Frobenius Trans RSRS-vs-dense error too large ({} > {dense_tolerance})",
        frob_metrics.op_trans_dense_err
    );
    assert!(
        frob_metrics.op_trans_vs_no_trans_transpose <= 1.0e-11,
        "{label}: Frobenius operator Trans is not the transpose of NoTrans ({})",
        frob_metrics.op_trans_vs_no_trans_transpose
    );
    assert!(
        frob_metrics.factor_right_vs_left_transpose <= 1.0e-11,
        "{label}: Frobenius factor right-apply is not the transpose of left-apply ({})",
        frob_metrics.factor_right_vs_left_transpose
    );
    if matches!(label, "symmetric-real") {
        // For a symmetric real matrix, `A x` and `A^T x` should agree up to the
        // RSRS approximation error.
        assert!(
            no_trans_vs_trans <= 2.5e-3,
            "{label}: NoTrans and Trans should match for symmetric matrices ({no_trans_vs_trans})"
        );
        assert!(
            frob_metrics.no_trans_vs_trans <= 2.5e-3,
            "{label}: Frobenius NoTrans and Trans should match for symmetric matrices ({})",
            frob_metrics.no_trans_vs_trans
        );
    } else if matches!(label, "nonsymmetric-real") {
        assert!(
            frob_metrics.dense_no_trans_vs_trans >= 1.0e-4,
            "{label}: dense Frobenius NoTrans and Trans references are too similar ({})",
            frob_metrics.dense_no_trans_vs_trans
        );
        assert!(
            frob_metrics.no_trans_vs_trans >= 1.0e-4,
            "{label}: RSRS Frobenius NoTrans and Trans collapsed unexpectedly ({})",
            frob_metrics.no_trans_vs_trans
        );
    }
}

/// Runs the complex Hermitian regression checks.
///
/// Besides the usual dense-vs-RSRS comparisons, this case verifies the
/// conjugation identities that the operator wrapper uses to implement
/// `ConjNoTrans` and `ConjTrans`.
#[allow(dead_code)]
fn run_complex_hermitian_case(points: &[Point], comm: &SimpleCommunicator) {
    // This case stresses the operator wrapper rather than just approximation
    // quality: Hermitian structure means `ConjTrans` should line up with the
    // ordinary `NoTrans` action, while direct factor-level transpose diagnostics
    // still go through the lower-level transposed apply path.
    let n = points.len();
    let matrix = hermitian_complex_matrix(points);
    let tree = Octree::new(points, TEST_MAX_LEVEL, TEST_MAX_LEAF_POINTS, comm);
    let args = RsrsArgs::new(
        8,
        16,
        0,
        120,
        Shift::False,
        NullMethod::Projection,
        RankRevealingQrType::RRQR,
        BlockExtractionMethod::LuLstSq,
        BlockExtractionMethod::LuLstSq,
        PivotMethod::Lu(1e-10),
        PivotMethod::Lu(0.0),
        1e-10,
        TEST_FIXED_RANK,
        1e-10,
        1e-10,
        4,
        1,
        Symmetry::Hermitian,
        RankPicking::Min,
        FactType::Joint,
        false,
        1,
        false,
        true,
    );
    let options = RsrsOptions::<Complex<f64>>::new(Some(args));
    let operator = Operator::from(&matrix);
    let mut rsrs = Rsrs::new(&tree, options, operator.domain().dimension());
    let rsrs_op = rsrs.get_rsrs_operator(operator);
    let factors = rsrs_op.get_factors();
    let x = deterministic_complex_vector(n);
    let base_no_trans = BaseFactorOptions {
        inv: false,
        trans: TransMode::NoTrans,
        trans_target: false,
    };
    let base_trans = BaseFactorOptions {
        inv: false,
        trans: TransMode::Trans,
        trans_target: false,
    };

    let op_left = apply_operator(&rsrs_op, &x, TransMode::NoTrans);
    let op_trans = apply_operator(&rsrs_op, &x, TransMode::Trans);
    let op_conj_no_trans = apply_operator(&rsrs_op, &x, TransMode::ConjNoTrans);
    let op_conj_trans = apply_operator(&rsrs_op, &x, TransMode::ConjTrans);
    let mut factor_left = vec![Complex::new(0.0, 0.0); n];
    let mut factor_left_trans = vec![Complex::new(0.0, 0.0); n];
    let mut factor_right = vec![Complex::new(0.0, 0.0); n];
    let mut factor_right_trans = vec![Complex::new(0.0, 0.0); n];
    factors.matvec(&x, &mut factor_left, Side::Left, &base_no_trans);
    factors.matvec(&x, &mut factor_left_trans, Side::Left, &base_trans);
    factors.matvec(&x, &mut factor_right, Side::Right, &base_no_trans);
    factors.matvec(&x, &mut factor_right_trans, Side::Right, &base_trans);

    let x_conj = conjugated(&x);
    let dense_left = dense_apply(&matrix, &x, Side::Left, TransMode::NoTrans, false);
    let dense_left_trans = dense_apply(&matrix, &x, Side::Left, TransMode::Trans, false);
    let dense_right = dense_apply(&matrix, &x, Side::Right, TransMode::NoTrans, false);
    let dense_right_trans = dense_apply(&matrix, &x, Side::Right, TransMode::Trans, false);
    let dense_left_conj_no_trans = conjugated(&dense_apply(
        &matrix,
        &x_conj,
        Side::Left,
        TransMode::NoTrans,
        false,
    ));
    let dense_left_conj_trans = conjugated(&dense_apply(
        &matrix,
        &x_conj,
        Side::Left,
        TransMode::Trans,
        false,
    ));

    // As in the real cases, transpose-vector products are normalized through
    // right-application before they reach the factor kernels.
    let op_vs_factor_left = rel_l2_error(&op_left, &factor_left);
    let op_vs_factor_trans = rel_l2_error(&op_trans, &factor_right);
    let conj_no_trans_dense_err = rel_l2_error(&op_conj_no_trans, &dense_left_conj_no_trans);
    let conj_trans_dense_err = rel_l2_error(&op_conj_trans, &dense_left_conj_trans);
    let left_dense_err = rel_l2_error(&op_left, &dense_left);
    let trans_dense_err = rel_l2_error(&op_trans, &dense_left_trans);
    let left_trans_dense_err = rel_l2_error(&factor_left_trans, &dense_left_trans);
    let right_dense_err = rel_l2_error(&factor_right, &dense_right);
    let right_trans_dense_err = rel_l2_error(&factor_right_trans, &dense_right_trans);
    let no_trans_vs_trans = rel_l2_error(&op_left, &op_trans);
    let conj_trans_vs_no_trans = rel_l2_error(&op_conj_trans, &op_left);

    println!(
        "hermitian-complex: op_vs_factor_left={op_vs_factor_left:.3e}, op_vs_factor_trans_identity={op_vs_factor_trans:.3e}, left_dense={left_dense_err:.3e}, trans_dense={trans_dense_err:.3e}, conj_no_trans={conj_no_trans_dense_err:.3e}, conj_trans={conj_trans_dense_err:.3e}, left_trans_dense={left_trans_dense_err:.3e}, right_dense={right_dense_err:.3e}, right_trans_dense={right_trans_dense_err:.3e}, no_trans_vs_trans={no_trans_vs_trans:.3e}, conj_trans_vs_no_trans={conj_trans_vs_no_trans:.3e}"
    );

    let frob_metrics = run_frobenius_matrix_check(
        "hermitian-complex",
        &matrix,
        &rsrs_op,
        factors,
        &base_no_trans,
    );
    let conj_frob_metrics =
        run_conjugate_frobenius_matrix_check("hermitian-complex", &matrix, &rsrs_op);

    assert!(
        op_vs_factor_left <= 1.0e-11,
        "hermitian-complex: operator NoTrans diverges from factor matvec (rel l2 = {op_vs_factor_left})"
    );
    assert!(
        op_vs_factor_trans <= 1.0e-11,
        "hermitian-complex: operator Trans diverges from factor matvec (rel l2 = {op_vs_factor_trans})"
    );
    assert!(
        left_dense_err <= 1.0e-2,
        "hermitian-complex: NoTrans RSRS-vs-dense error too large ({left_dense_err})"
    );
    assert!(
        trans_dense_err <= 1.0e-2,
        "hermitian-complex: Trans RSRS-vs-dense error too large ({trans_dense_err})"
    );
    assert!(
        right_dense_err <= 1.0e-2,
        "hermitian-complex: right NoTrans factor matvec error too large ({right_dense_err})"
    );
    assert!(
        conj_no_trans_dense_err <= 1.0e-2,
        "hermitian-complex: ConjNoTrans RSRS-vs-dense error too large ({conj_no_trans_dense_err})"
    );
    assert!(
        conj_trans_dense_err <= 1.0e-2,
        "hermitian-complex: ConjTrans RSRS-vs-dense error too large ({conj_trans_dense_err})"
    );
    assert!(
        conj_trans_vs_no_trans <= 1.0e-2,
        "hermitian-complex: ConjTrans should match NoTrans for Hermitian matrices ({conj_trans_vs_no_trans})"
    );
    // The operator-level transpose path intentionally reuses right-application.
    // Depending on the low-level storage view, either direct factor-level
    // transpose diagnostic can be the sharper signal, so we require at least
    // one of them to stay accurate.
    assert!(
        left_trans_dense_err <= 2.0e-2 || right_trans_dense_err <= 2.0e-2,
        "hermitian-complex: both factor-level Trans diagnostics are unexpectedly poor (left={left_trans_dense_err}, right={right_trans_dense_err})"
    );
    assert!(
        frob_metrics.op_no_trans_factor_err <= 1.0e-11,
        "hermitian-complex: Frobenius operator NoTrans diverges from factor matrix ({})",
        frob_metrics.op_no_trans_factor_err
    );
    assert!(
        frob_metrics.op_trans_factor_err <= 1.0e-11,
        "hermitian-complex: Frobenius operator Trans diverges from factor right-apply matrix ({})",
        frob_metrics.op_trans_factor_err
    );
    assert!(
        frob_metrics.op_no_trans_dense_err <= 1.0e-2,
        "hermitian-complex: Frobenius NoTrans RSRS-vs-dense error too large ({})",
        frob_metrics.op_no_trans_dense_err
    );
    assert!(
        frob_metrics.op_trans_dense_err <= 1.0e-2,
        "hermitian-complex: Frobenius Trans RSRS-vs-dense error too large ({})",
        frob_metrics.op_trans_dense_err
    );
    assert!(
        conj_frob_metrics.conj_no_trans_dense_err <= 1.0e-2,
        "hermitian-complex: Frobenius ConjNoTrans RSRS-vs-dense error too large ({})",
        conj_frob_metrics.conj_no_trans_dense_err
    );
    assert!(
        conj_frob_metrics.conj_trans_dense_err <= 1.0e-2,
        "hermitian-complex: Frobenius ConjTrans RSRS-vs-dense error too large ({})",
        conj_frob_metrics.conj_trans_dense_err
    );
}

struct ComplexSymmetricMetrics {
    // Public operator `NoTrans` versus direct left factor apply.
    op_vs_factor_left: f64,
    // Public operator `Trans` versus direct right factor apply.
    op_vs_factor_trans: f64,
    // Public operator `NoTrans` versus dense `A x`.
    left_dense_err: f64,
    // Public operator `Trans` versus dense `A^T x`.
    trans_dense_err: f64,
    // Public operator `A x` versus public operator `A^T x`.
    no_trans_vs_trans: f64,
}

/// Runs a complex symmetric case and reports the main operator-level
/// diagnostics.
fn run_complex_symmetric_mode(
    points: &[Point],
    comm: &SimpleCommunicator,
    matrix: &DynamicArray<Complex<f64>, 2>,
    symmetry: Symmetry,
    label: &str,
) -> ComplexSymmetricMetrics {
    let n = matrix.shape()[0];
    let tree = Octree::new(points, TEST_MAX_LEVEL, TEST_MAX_LEAF_POINTS, comm);
    let args = RsrsArgs::new(
        8,
        16,
        0,
        120,
        Shift::False,
        NullMethod::Projection,
        RankRevealingQrType::RRQR,
        BlockExtractionMethod::LuLstSq,
        BlockExtractionMethod::LuLstSq,
        PivotMethod::Lu(1e-10),
        PivotMethod::Lu(0.0),
        1e-10,
        TEST_FIXED_RANK,
        1e-10,
        1e-10,
        4,
        1,
        symmetry,
        RankPicking::Min,
        FactType::Joint,
        false,
        1,
        false,
        true,
    );
    let options = RsrsOptions::<Complex<f64>>::new(Some(args));
    let operator = Operator::from(matrix);
    let mut rsrs = Rsrs::new(&tree, options, operator.domain().dimension());
    let rsrs_op = rsrs.get_rsrs_operator(operator);
    let factors = rsrs_op.get_factors();
    let x = deterministic_complex_vector(n);
    let base_no_trans = BaseFactorOptions {
        inv: false,
        trans: TransMode::NoTrans,
        trans_target: false,
    };

    let op_left = apply_operator(&rsrs_op, &x, TransMode::NoTrans);
    let op_trans = apply_operator(&rsrs_op, &x, TransMode::Trans);
    let mut factor_left = vec![Complex::new(0.0, 0.0); n];
    let mut factor_right = vec![Complex::new(0.0, 0.0); n];
    factors.matvec(&x, &mut factor_left, Side::Left, &base_no_trans);
    factors.matvec(&x, &mut factor_right, Side::Right, &base_no_trans);

    let dense_left = dense_apply(matrix, &x, Side::Left, TransMode::NoTrans, false);
    let dense_left_trans = dense_apply(matrix, &x, Side::Left, TransMode::Trans, false);

    let metrics = ComplexSymmetricMetrics {
        op_vs_factor_left: rel_l2_error(&op_left, &factor_left),
        op_vs_factor_trans: rel_l2_error(&op_trans, &factor_right),
        left_dense_err: rel_l2_error(&op_left, &dense_left),
        trans_dense_err: rel_l2_error(&op_trans, &dense_left_trans),
        no_trans_vs_trans: rel_l2_error(&op_left, &op_trans),
    };

    println!(
        "{label}: op_vs_factor_left={:.3e}, op_vs_factor_trans_identity={:.3e}, left_dense={:.3e}, trans_dense={:.3e}, no_trans_vs_trans={:.3e}",
        metrics.op_vs_factor_left,
        metrics.op_vs_factor_trans,
        metrics.left_dense_err,
        metrics.trans_dense_err,
        metrics.no_trans_vs_trans
    );

    let frob_metrics = run_frobenius_matrix_check(label, matrix, &rsrs_op, factors, &base_no_trans);
    let conj_frob_metrics = run_conjugate_frobenius_matrix_check(label, matrix, &rsrs_op);
    assert!(
        frob_metrics.op_no_trans_factor_err <= 1.0e-11,
        "{label}: Frobenius operator NoTrans diverges from factor matrix ({})",
        frob_metrics.op_no_trans_factor_err
    );
    assert!(
        frob_metrics.op_trans_factor_err <= 1.0e-11,
        "{label}: Frobenius operator Trans diverges from factor right-apply matrix ({})",
        frob_metrics.op_trans_factor_err
    );
    assert!(
        frob_metrics.op_no_trans_dense_err <= 2.0e-2,
        "{label}: Frobenius NoTrans RSRS-vs-dense error too large ({})",
        frob_metrics.op_no_trans_dense_err
    );
    assert!(
        frob_metrics.op_trans_dense_err <= 2.0e-2,
        "{label}: Frobenius Trans RSRS-vs-dense error too large ({})",
        frob_metrics.op_trans_dense_err
    );
    assert!(
        conj_frob_metrics.conj_no_trans_dense_err <= 2.0e-2,
        "{label}: Frobenius ConjNoTrans RSRS-vs-dense error too large ({})",
        conj_frob_metrics.conj_no_trans_dense_err
    );
    assert!(
        conj_frob_metrics.conj_trans_dense_err <= 2.0e-2,
        "{label}: Frobenius ConjTrans RSRS-vs-dense error too large ({})",
        conj_frob_metrics.conj_trans_dense_err
    );

    metrics
}

/// Runs the complex symmetric fixture through the complex-symmetric path.
fn run_complex_symmetric_case(
    points: &[Point],
    comm: &SimpleCommunicator,
    matrix: &DynamicArray<Complex<f64>, 2>,
) {
    let symmetric_metrics = run_complex_symmetric_mode(
        points,
        comm,
        matrix,
        Symmetry::Symmetric,
        "symmetric-complex",
    );

    assert!(
        symmetric_metrics.op_vs_factor_left <= 1.0e-11,
        "symmetric-complex: operator NoTrans diverges from factor matvec (rel l2 = {})",
        symmetric_metrics.op_vs_factor_left
    );
    assert!(
        symmetric_metrics.op_vs_factor_trans <= 1.0e-11,
        "symmetric-complex: operator Trans diverges from factor matvec (rel l2 = {})",
        symmetric_metrics.op_vs_factor_trans
    );
    assert!(
        symmetric_metrics.left_dense_err <= 2.0e-2,
        "symmetric-complex: NoTrans RSRS-vs-dense error too large ({})",
        symmetric_metrics.left_dense_err
    );
    assert!(
        symmetric_metrics.trans_dense_err <= 2.0e-2,
        "symmetric-complex: Trans RSRS-vs-dense error too large ({})",
        symmetric_metrics.trans_dense_err
    );
    assert!(
        symmetric_metrics.no_trans_vs_trans <= 2.0e-2,
        "symmetric-complex: NoTrans and Trans should match for complex symmetric matrices ({})",
        symmetric_metrics.no_trans_vs_trans
    );
}

/// Runs a genuinely complex nonsymmetric case through the `NoSymm` path.
fn run_complex_nonsymmetric_case(
    points: &[Point],
    comm: &SimpleCommunicator,
    matrix: &DynamicArray<Complex<f64>, 2>,
) {
    let n = matrix.shape()[0];
    let tree = Octree::new(points, TEST_MAX_LEVEL, TEST_MAX_LEAF_POINTS, comm);
    let args = RsrsArgs::new(
        8,
        16,
        0,
        120,
        Shift::False,
        NullMethod::Projection,
        RankRevealingQrType::RRQR,
        BlockExtractionMethod::LuLstSq,
        BlockExtractionMethod::LuLstSq,
        PivotMethod::Lu(1e-10),
        PivotMethod::Lu(0.0),
        1e-10,
        TEST_FIXED_RANK,
        1e-10,
        1e-10,
        4,
        1,
        Symmetry::NoSymm,
        RankPicking::Min,
        FactType::Joint,
        false,
        1,
        false,
        true,
    );
    let options = RsrsOptions::<Complex<f64>>::new(Some(args));
    let operator = Operator::from(matrix);
    let mut rsrs = Rsrs::new(&tree, options, operator.domain().dimension());
    let mut rsrs_op = rsrs.get_rsrs_operator(operator);
    let factors = rsrs_op.get_factors();
    let x = deterministic_complex_vector(n);
    let base_no_trans = BaseFactorOptions {
        inv: false,
        trans: TransMode::NoTrans,
        trans_target: false,
    };
    let base_trans = BaseFactorOptions {
        inv: false,
        trans: TransMode::Trans,
        trans_target: false,
    };

    let op_left = apply_operator(&rsrs_op, &x, TransMode::NoTrans);
    let op_trans = apply_operator(&rsrs_op, &x, TransMode::Trans);
    let op_conj_no_trans = apply_operator(&rsrs_op, &x, TransMode::ConjNoTrans);
    let op_conj_trans = apply_operator(&rsrs_op, &x, TransMode::ConjTrans);
    let mut factor_left = vec![Complex::new(0.0, 0.0); n];
    let mut factor_left_trans = vec![Complex::new(0.0, 0.0); n];
    let mut factor_right = vec![Complex::new(0.0, 0.0); n];
    let mut factor_right_trans = vec![Complex::new(0.0, 0.0); n];
    factors.matvec(&x, &mut factor_left, Side::Left, &base_no_trans);
    factors.matvec(&x, &mut factor_left_trans, Side::Left, &base_trans);
    factors.matvec(&x, &mut factor_right, Side::Right, &base_no_trans);
    factors.matvec(&x, &mut factor_right_trans, Side::Right, &base_trans);

    let x_conj = conjugated(&x);
    let dense_left = dense_apply(matrix, &x, Side::Left, TransMode::NoTrans, false);
    let dense_left_trans = dense_apply(matrix, &x, Side::Left, TransMode::Trans, false);
    let dense_right = dense_apply(matrix, &x, Side::Right, TransMode::NoTrans, false);
    let dense_right_trans = dense_apply(matrix, &x, Side::Right, TransMode::Trans, false);
    let dense_left_conj_no_trans = conjugated(&dense_apply(
        matrix,
        &x_conj,
        Side::Left,
        TransMode::NoTrans,
        false,
    ));
    let dense_left_conj_trans = conjugated(&dense_apply(
        matrix,
        &x_conj,
        Side::Left,
        TransMode::Trans,
        false,
    ));

    let op_vs_factor_left = rel_l2_error(&op_left, &factor_left);
    let op_vs_factor_trans = rel_l2_error(&op_trans, &factor_right);
    let left_dense_err = rel_l2_error(&op_left, &dense_left);
    let trans_dense_err = rel_l2_error(&op_trans, &dense_left_trans);
    let left_trans_dense_err = rel_l2_error(&factor_left_trans, &dense_left_trans);
    let right_dense_err = rel_l2_error(&factor_right, &dense_right);
    let right_trans_dense_err = rel_l2_error(&factor_right_trans, &dense_right_trans);
    let conj_no_trans_dense_err = rel_l2_error(&op_conj_no_trans, &dense_left_conj_no_trans);
    let conj_trans_dense_err = rel_l2_error(&op_conj_trans, &dense_left_conj_trans);
    let no_trans_vs_trans = rel_l2_error(&op_left, &op_trans);
    let dense_no_trans_vs_trans = rel_l2_error(&dense_left, &dense_left_trans);

    println!(
        "nonsymmetric-complex-general: op_vs_factor_left={op_vs_factor_left:.3e}, op_vs_factor_trans_identity={op_vs_factor_trans:.3e}, left_dense={left_dense_err:.3e}, trans_dense={trans_dense_err:.3e}, left_trans_dense={left_trans_dense_err:.3e}, right_dense={right_dense_err:.3e}, right_trans_dense={right_trans_dense_err:.3e}, conj_no_trans={conj_no_trans_dense_err:.3e}, conj_trans={conj_trans_dense_err:.3e}, no_trans_vs_trans={no_trans_vs_trans:.3e}, dense_no_trans_vs_trans={dense_no_trans_vs_trans:.3e}"
    );

    let frob_metrics = run_frobenius_matrix_check(
        "nonsymmetric-complex-general",
        matrix,
        &rsrs_op,
        factors,
        &base_no_trans,
    );
    let conj_frob_metrics =
        run_conjugate_frobenius_matrix_check("nonsymmetric-complex-general", matrix, &rsrs_op);
    if run_dense_diagnostics() {
        run_dense_box_errors_left_application("nonsymmetric-complex-general", matrix, factors);
        run_dense_box_errors_right_application("nonsymmetric-complex-general", matrix, factors);
        run_dense_box_errors_sandwich_application("nonsymmetric-complex-general", matrix, factors);
        run_inverse_dense_diagnostic("nonsymmetric-complex-general", matrix, &mut rsrs_op);
    }

    assert!(
        op_vs_factor_left <= 1.0e-11,
        "nonsymmetric-complex-general: operator NoTrans diverges from factor matvec (rel l2 = {op_vs_factor_left})"
    );
    assert!(
        op_vs_factor_trans <= 1.0e-11,
        "nonsymmetric-complex-general: operator Trans diverges from factor matvec (rel l2 = {op_vs_factor_trans})"
    );
    assert!(
        left_dense_err <= 2.0e-2,
        "nonsymmetric-complex-general: NoTrans RSRS-vs-dense error too large ({left_dense_err})"
    );
    assert!(
        trans_dense_err <= 2.0e-2,
        "nonsymmetric-complex-general: Trans RSRS-vs-dense error too large ({trans_dense_err})"
    );
    assert!(
        left_trans_dense_err <= 2.0e-2,
        "nonsymmetric-complex-general: left Trans factor matvec error too large ({left_trans_dense_err})"
    );
    assert!(
        right_dense_err <= 2.0e-2,
        "nonsymmetric-complex-general: right NoTrans factor matvec error too large ({right_dense_err})"
    );
    assert!(
        right_trans_dense_err <= 2.0e-2,
        "nonsymmetric-complex-general: right Trans factor matvec error too large ({right_trans_dense_err})"
    );
    assert!(
        conj_no_trans_dense_err <= 2.0e-2,
        "nonsymmetric-complex-general: ConjNoTrans RSRS-vs-dense error too large ({conj_no_trans_dense_err})"
    );
    assert!(
        conj_trans_dense_err <= 2.0e-2,
        "nonsymmetric-complex-general: ConjTrans RSRS-vs-dense error too large ({conj_trans_dense_err})"
    );
    assert!(
        dense_no_trans_vs_trans >= 1.0e-4,
        "nonsymmetric-complex-general: dense NoTrans and Trans references are too similar ({dense_no_trans_vs_trans})"
    );
    assert!(
        no_trans_vs_trans >= 1.0e-4,
        "nonsymmetric-complex-general: RSRS NoTrans and Trans collapsed unexpectedly ({no_trans_vs_trans})"
    );
    assert!(
        frob_metrics.op_no_trans_factor_err <= 1.0e-11,
        "nonsymmetric-complex-general: Frobenius operator NoTrans diverges from factor matrix ({})",
        frob_metrics.op_no_trans_factor_err
    );
    assert!(
        frob_metrics.op_trans_factor_err <= 1.0e-11,
        "nonsymmetric-complex-general: Frobenius operator Trans diverges from factor right-apply matrix ({})",
        frob_metrics.op_trans_factor_err
    );
    assert!(
        frob_metrics.op_no_trans_dense_err <= 2.0e-2,
        "nonsymmetric-complex-general: Frobenius NoTrans RSRS-vs-dense error too large ({})",
        frob_metrics.op_no_trans_dense_err
    );
    assert!(
        frob_metrics.op_trans_dense_err <= 2.0e-2,
        "nonsymmetric-complex-general: Frobenius Trans RSRS-vs-dense error too large ({})",
        frob_metrics.op_trans_dense_err
    );
    assert!(
        conj_frob_metrics.conj_no_trans_dense_err <= 2.0e-2,
        "nonsymmetric-complex-general: Frobenius ConjNoTrans RSRS-vs-dense error too large ({})",
        conj_frob_metrics.conj_no_trans_dense_err
    );
    assert!(
        conj_frob_metrics.conj_trans_dense_err <= 2.0e-2,
        "nonsymmetric-complex-general: Frobenius ConjTrans RSRS-vs-dense error too large ({})",
        conj_frob_metrics.conj_trans_dense_err
    );
    assert!(
        frob_metrics.dense_no_trans_vs_trans >= 1.0e-4,
        "nonsymmetric-complex-general: dense Frobenius NoTrans and Trans references are too similar ({})",
        frob_metrics.dense_no_trans_vs_trans
    );
    assert!(
        frob_metrics.no_trans_vs_trans >= 1.0e-4,
        "nonsymmetric-complex-general: RSRS Frobenius NoTrans and Trans collapsed unexpectedly ({})",
        frob_metrics.no_trans_vs_trans
    );
}

#[test]
fn rsrs_operator_matvec_diagnostic() {
    std::thread::Builder::new()
        .name("rsrs_operator_matvec_diagnostic_worker".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(rsrs_operator_matvec_diagnostic_worker)
        .unwrap()
        .join()
        .unwrap();
}

fn rsrs_operator_matvec_diagnostic_worker() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");

    let universe = mpi::initialize().unwrap();
    let comm: SimpleCommunicator = universe.world();
    let fixture = load_biegrid_perturbed_fixture();
    let points = &fixture.points;
    println!(
        "loaded perturbed BIEGrid fixture: n={}, cells_per_axis={}, perturbation_scale={:.3e}, perturbation_seed={}",
        points.len(),
        fixture.cells_per_axis,
        fixture.perturbation_scale,
        fixture.perturbation_seed
    );

    // These cases together cover the main routing logic:
    // - symmetric real: transpose should collapse to the same action,
    // - nonsymmetric real: transpose must remain distinct,
    // - symmetric complex: transpose should still collapse without Hermitian
    //   conjugation,
    // - nonsymmetric complex: transpose must remain distinct.
    run_real_case(
        points,
        &comm,
        fixture.real_symmetric,
        Symmetry::Symmetric,
        "symmetric-real",
        false,
        1.0e-2,
    );
    run_real_case(
        points,
        &comm,
        fixture.real_nonsymmetric,
        Symmetry::NoSymm,
        "nonsymmetric-real",
        false,
        1.5e-2,
    );
    run_complex_symmetric_case(points, &comm, &fixture.complex_symmetric);
    run_complex_nonsymmetric_case(points, &comm, &fixture.complex_nonsymmetric);
    // The analytic Hermitian diagnostic is kept available, but this fixture
    // set intentionally covers the four perturbed BIEGrid cases above.
    // run_complex_hermitian_case(points, &comm);
}
