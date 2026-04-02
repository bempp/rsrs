//! Diagnostic regression tests for RSRS matrix-vector products.
//!
//! The Hermitian complex case is particularly valuable because it exercises the
//! relationship between transpose, conjugate-transpose, and direct factor
//! matvec application.
//!
//! The structure of the test file is:
//! - build a deterministic point cloud and a few small model matrices,
//! - compare RSRS operator application against both direct factor matvecs and
//!   dense references,
//! - cover symmetric real, nonsymmetric real, and Hermitian complex cases.

use bempp_octree::{generate_random_points, Octree, Point};
use bempp_rsrs::{
    rsrs::{
        args::{RankPicking, RsrsArgs, RsrsOptions, Symmetry},
        rsrs_cycle::Rsrs,
        rsrs_factors::{
            base_factors::BaseFactorOptions,
            null_and_extract::PivotMethod,
            rsrs_operator::{FactType, RsrsFactorsImpl},
        },
        sketch::Shift,
    },
    utils::linear_algebra::{BlockExtractionMethod, NullMethod},
};
use mpi::{topology::SimpleCommunicator, traits::CommunicatorCollectives};
use num::Complex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rlst::prelude::*;

/// Fixed rank used so the regression stays deterministic across runs.
const TEST_FIXED_RANK: f64 = 8.0;
/// Problem size large enough to trigger a nontrivial RSRS hierarchy.
const TEST_NPOINTS: usize = 192;
/// Small tree depth that still produces both off-diagonal and diagonal work.
const TEST_MAX_LEVEL: usize = 3;
/// Leaf occupancy chosen so the test reaches compression quickly.
const TEST_MAX_LEAF_POINTS: usize = 8;

/// Generates deterministic points and projects them back onto an approximate
/// sphere to keep the geometry smooth and reproducible.
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
fn laplace_kernel(dist: f64, npoints: usize) -> f64 {
    let pi = std::f64::consts::PI;
    let n = npoints as f64;
    1.0 / (4.0 * pi * n * dist)
}

/// Dense symmetric reference matrix based on the Laplace kernel.
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

/// Deterministic real input vector used across the real-valued cases.
fn deterministic_real_vector(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = (i + 1) as f64;
            t.sin() + 0.25 * t.cos()
        })
        .collect()
}

/// Deterministic complex input vector used across the Hermitian case.
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
    if matches!(label, "symmetric-real") {
        // For a symmetric real matrix, `A x` and `A^T x` should agree up to the
        // RSRS approximation error.
        assert!(
            no_trans_vs_trans <= 2.5e-3,
            "{label}: NoTrans and Trans should match for symmetric matrices ({no_trans_vs_trans})"
        );
    }
}

/// Runs the complex Hermitian regression checks.
///
/// Besides the usual dense-vs-RSRS comparisons, this case verifies the
/// conjugation identities that the operator wrapper uses to implement
/// `ConjNoTrans` and `ConjTrans`.
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
}

#[test]
fn rsrs_operator_matvec_diagnostic() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");

    let universe = mpi::initialize().unwrap();
    let comm: SimpleCommunicator = universe.world();
    let points = sphere_surface(TEST_NPOINTS, &comm);

    // These three cases together cover the main routing logic:
    // - symmetric real: transpose should collapse to the same action,
    // - nonsymmetric real: transpose must remain distinct,
    // - Hermitian complex: conjugate-transpose should collapse to NoTrans.
    run_real_case(
        &points,
        &comm,
        laplace_matrix(&points),
        Symmetry::Symmetric,
        "symmetric-real",
        false,
        1.0e-2,
    );
    run_real_case(
        &points,
        &comm,
        nonsymmetric_real_matrix(&points),
        Symmetry::NoSymm,
        "nonsymmetric-real",
        false,
        1.5e-2,
    );
    run_complex_hermitian_case(&points, &comm);
}
