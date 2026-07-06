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

from dixonstat._quadrature import cost_and_jac
from dixonstat._quadrature import half_hermgauss
import numpy as np
import pytest
import sympy


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
