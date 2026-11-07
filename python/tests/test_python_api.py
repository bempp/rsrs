import numpy as np
import pytest

from bempp_rsrs import (
    InsufficientSamplesError,
    RsrsFactors,
    RsrsOptions,
    RsrsSession,
    SampleBundle,
)


def test_legacy_threads_and_tree_options_are_losslessly_mapped():
    options = RsrsOptions.from_mapping(
        {
            "num_threads": 56,
            "factorization_threads": 1,
            "apply_threads": 32,
            "leaf_size": 400,
        },
        scenario={"max_tree_depth": 16},
    )
    assert options.threads.factorization_threads == 1
    assert options.threads.apply_threads == 32
    assert options.tree.max_tree_depth == 16
    assert options.tree.leaf_size == 400


def test_complex_nonsymmetric_sample_bundle_preserves_transpose_rows():
    matrix = np.array([[1 + 2j, 3], [4j, 5 - 1j]])
    tests = np.eye(2, dtype=np.complex128)
    bundle = SampleBundle(
        y_test=tests,
        y_sketch=(matrix @ tests.T).T,
        z_test=tests,
        z_sketch=(matrix.T @ tests.T).T,
    )
    bundle.validate(require_z=True)
    # Samples are stored by row. Row i is (A^T e_i)^T = e_i^T A,
    # therefore the stacked row-oriented sketch equals A, not A^T.
    np.testing.assert_allclose(bundle.z_sketch, matrix)


def test_partial_z_family_is_rejected():
    with pytest.raises(ValueError, match="both be present"):
        SampleBundle(np.eye(2), np.eye(2), z_test=np.eye(2))


def test_insufficient_cache_requires_a_fresh_factorization_attempt():
    class FakeNative:
        dimension = 2
        requires_z = True

        def add_samples(self, *_arrays):
            pass

        def advance(self):
            return {
                "status": "needs_samples",
                "y_rows": 3,
                "z_rows": 3,
                "total_required": 5,
                "level": 4,
                "limiting_box": 7,
            }

    session = RsrsSession.__new__(RsrsSession)
    session._native = FakeNative()
    session._started = False
    samples = SampleBundle(
        np.eye(2),
        np.eye(2),
        z_test=np.eye(2),
        z_sketch=np.eye(2),
    )

    with pytest.raises(InsufficientSamplesError) as error:
        session.factorize(samples)
    assert error.value.request.total_required == 5
    assert error.value.request.y_rows == 3
    with pytest.raises(RuntimeError, match="already used"):
        session.factorize(samples)


def test_factor_object_exposes_forward_transpose_and_inverse_products():
    class NativeFactors:
        dimension = 2
        dtype = "float64"

        @staticmethod
        def apply_vector(values, *, transpose, inverse):
            multiplier = 1 + 10 * transpose + 100 * inverse
            return multiplier * values

        @staticmethod
        def apply_matrix(values, *, transpose, inverse):
            multiplier = 1 + 10 * transpose + 100 * inverse
            return multiplier * values

    factors = RsrsFactors(NativeFactors())
    assert factors.report == {}
    assert RsrsFactors(NativeFactors(), report={"ranks": [3]}).report == {"ranks": [3]}
    vector = np.array([1.0, 2.0])
    matrix = np.column_stack((vector, 2 * vector))

    np.testing.assert_allclose(factors.matvec(vector), vector)
    np.testing.assert_allclose(factors.transpose_matvec(vector), 11 * vector)
    np.testing.assert_allclose(factors.inverse_matvec(vector), 101 * vector)
    np.testing.assert_allclose(
        factors.inverse_transpose_matvec(vector), 111 * vector
    )
    np.testing.assert_allclose(factors.inverse_transpose_matmat(matrix), 111 * matrix)


def test_complex_adjoint_conjugates_around_algebraic_transpose():
    dense = np.array([[1 + 2j, 3 - 1j], [4 + 5j, 6 - 2j]])

    class NativeFactors:
        dimension = 2
        dtype = "complex128"

        @staticmethod
        def apply_vector(values, *, transpose, inverse):
            matrix = dense.T if transpose else dense
            if inverse:
                matrix = np.linalg.inv(matrix)
            return matrix @ values

        @staticmethod
        def apply_matrix(values, *, transpose, inverse):
            matrix = dense.T if transpose else dense
            if inverse:
                matrix = np.linalg.inv(matrix)
            return matrix @ values

    factors = RsrsFactors(NativeFactors())
    vector = np.array([2 - 1j, 1 + 3j])
    np.testing.assert_allclose(
        factors.aslinearoperator().rmatvec(vector), dense.conj().T @ vector
    )
