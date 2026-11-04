# Identification and rejection of outliers using Dixon's r statistics.
#
# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dixonstat._quadrature import alpha0
from dixonstat._quadrature import cost_and_jac
from dixonstat._quadrature import half_hermgauss
from scripts.reference_quadrature import half_hermite_moment
from scripts.reference_quadrature import half_hermite_rule
import numpy as np
import pytest
import sympy

# The number of leading moments checked. Higher moments are dominated by the
# outermost weights, whose relative accuracy is limited by the eigensolver.
LOW_MOMENTS = 8
EPS = np.finfo(float).eps

# The tolerances are a small multiple of the deviation measured against the
# reference: 10 eps for the moments, 2 eps of the largest node for the nodes and
# 55 eps of the largest weight for the weights. Eigensolvers bound the absolute
# error of eigenvalues and eigenvector components, hence the absolute scale.
MOMENT_REL_TOL = 32 * EPS
NODE_ABS_TOL = 8 * EPS
WEIGHT_ABS_TOL = 128 * EPS

# The number of Newton iterations for 34 nodes starting from the asymptotic
# estimate. A linearly converging iteration needs more than twice as many.
NEWTON_ITERATIONS = 6


def test_cost_and_jac_matches_symbolic_derivative():
    # Cross-check against a symbolically differentiated residual.
    y, g_prev, g_cur, g_next, gamma = sympy.symbols('y g_prev g_cur g_next gamma')

    # Eq. 3.14
    residual_expr = ((y + 1) / 3 - g_next - g_cur) * ((y - 1) / 3 - g_cur - g_prev) * (
        y / 12 + g_cur
    ) ** 2 - ((y / 6 - g_cur) ** 2 - gamma**2 / 16) ** 2

    symbols = (y, g_prev, g_cur, g_next, gamma)
    residual_fn = sympy.lambdify(symbols, residual_expr, modules='numpy')
    jac_prev_fn = sympy.lambdify(
        symbols, sympy.diff(residual_expr, g_prev), modules='numpy'
    )
    jac_cur_fn = sympy.lambdify(
        symbols, sympy.diff(residual_expr, g_cur), modules='numpy'
    )
    jac_next_fn = sympy.lambdify(
        symbols, sympy.diff(residual_expr, g_next), modules='numpy'
    )

    rng = np.random.default_rng(0)
    n = 64
    idxs = np.arange(n)
    gamma_val = 1.7
    g = np.pad(rng.uniform(-0.05, 0.05, n), 1)

    residual, jac_prev_val, jac_cur_val, jac_next_val = cost_and_jac(idxs, gamma_val, g)

    y_val = 2.0 * idxs + gamma_val
    g_prev_val = g[idxs - 1]
    g_cur_val = g[idxs]
    g_next_val = g[idxs + 1]
    args = (y_val, g_prev_val, g_cur_val, g_next_val, gamma_val)

    np.testing.assert_allclose(residual, residual_fn(*args))
    np.testing.assert_allclose(jac_prev_val, jac_prev_fn(*args))
    np.testing.assert_allclose(jac_cur_val, jac_cur_fn(*args))
    np.testing.assert_allclose(jac_next_val, jac_next_fn(*args))


def test_alpha0_matches_gamma_zero_case():
    np.testing.assert_allclose(alpha0(0.0), 1.0 / np.sqrt(np.pi))


def test_alpha0_stays_finite_for_large_gamma():
    # Individually exponentiating each gammaln term overflows float64
    # (around gammaln(172)) long before their ratio does.
    value = alpha0(342.0)

    assert np.isfinite(value)
    np.testing.assert_allclose(value, 13.08625930053484)


@pytest.mark.parametrize('n_iter', [100, 12])
def test_exp_minus_10x(n_iter):
    '''Evaluate the integral

        ∞
        ⌠
        ⎮           2
        ⎮  -10⋅x  -x
        ⎮ ℯ     ⋅ℯ    dx
        ⌡
        0

    numerically using half-range Gauss-Hermite quadrature and compare against
    the symbolic evaluation.
    '''

    x, w = half_hermgauss(50, gamma=0, n_iter=n_iter)
    inter = np.inner(w, np.exp(-10 * x))
    np.testing.assert_almost_equal(inter, 0.0981094307315388, 7)


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('n', [8, 17, 34, 68])
def test_half_hermgauss_integrates_low_moments(n):
    x, w = half_hermgauss(n)

    exact = [float(half_hermite_moment(k)) for k in range(LOW_MOMENTS)]
    computed = [w @ x**k for k in range(LOW_MOMENTS)]

    assert computed == pytest.approx(exact, rel=MOMENT_REL_TOL, abs=0.0)


@pytest.mark.parametrize('n', [34, 68])
def test_half_hermgauss_matches_reference_rule(n):
    x, w = half_hermgauss(n)

    reference_x, reference_w = half_hermite_rule(n)
    expected_x = np.array([float(v) for v in reference_x])
    expected_w = np.array([float(v) for v in reference_w])

    node_tol = NODE_ABS_TOL * expected_x.max()
    weight_tol = WEIGHT_ABS_TOL * expected_w.max()
    assert x == pytest.approx(expected_x, rel=0.0, abs=node_tol)
    assert w == pytest.approx(expected_w, rel=0.0, abs=weight_tol)


def test_half_hermgauss_warns_at_the_caller_if_newton_does_not_converge():
    with pytest.warns(
        RuntimeWarning, match='did not converge in 5 iterations'
    ) as record:
        half_hermgauss(34, n_iter=NEWTON_ITERATIONS - 1)

    assert record[0].filename == __file__


@pytest.mark.filterwarnings('error')
def test_half_hermgauss_converges_quadratically():
    half_hermgauss(34, n_iter=NEWTON_ITERATIONS)
