# Identification and rejection of outliers using Dixon's r statistics.
#
# Copyright 2024 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

from scipy.linalg import eigh_tridiagonal
from scipy.linalg import solve_banded
from scipy.special import gammaln
import numpy as np


def estimate_g(i, gamma):
    gamma2 = np.square(gamma)

    # Use Horner's rule for efficient polynomial evaluation of Eqs. 3.10-3.13.
    # NOTE the sign changes are due to introduced parentheses.
    C0 = 1.0 / 36.0 - gamma2 / 8.0
    C1 = 23.0 / 432.0 - (11.0 / 48.0 - 3.0 / 32.0 * gamma2) * gamma2
    C2 = (
        1189.0 / 2592.0
        - (409.0 / 192.0 - (75.0 / 64.0 - 9.0 / 64.0 * gamma2) * gamma2) * gamma2
    )
    C3 = (
        196057.0 / 20736.0
        - (
            153559.0 / 3456.0
            - (7111.0 / 256.0 - (639.0 / 128.0 - 135.0 / 512.0 * gamma2) * gamma2)
            * gamma2
        )
        * gamma2
    )

    y = 2.0 * (i + 1) + gamma
    y2 = np.square(y)

    # Eq. 3.9
    g = (C0 + (C1 + (C2 + C3 / y2) / y2) / y2) / y

    # Pad with 0 to allow for indexing g_{n-1} and g_{n+1}
    return np.pad(g, 1)


def cost_F(i, gamma, g):
    y = 2.0 * i + gamma

    g_prev = g[i - 1]
    g_cur = g[i]
    g_next = g[i + 1]

    # Eq. 3.14
    return ((y + 1) / 3 - g_next - g_cur) * ((y - 1) / 3 - g_cur - g_prev) * (
        y / 12 + g_cur
    ) ** 2 - ((y / 6 - g_cur) ** 2 - gamma**2 / 16) ** 2


def jac_dF_dg_prev(i, gamma, g):
    y = 2.0 * i + gamma

    g_cur = g[i]
    g_next = g[i + 1]

    return -((y + 1.0) / 3.0 - g_next - g_cur) * (y / 12.0 + g_cur) ** 2


def jac_dF_dg_next(i, gamma, g):
    y = 2.0 * i + gamma

    g_prev = g[i - 1]
    g_cur = g[i]

    return -((y - 1.0) / 3.0 - g_cur - g_prev) * (y / 12.0 + g_cur) ** 2


def jac_dF_dg_cur(i, gamma, g):
    y = 2.0 * i + gamma

    g_prev = g[i - 1]
    g_cur = g[i]
    g_next = g[i + 1]

    return (
        ((y + 1) / 3 - g_next - g_cur)
        * ((y - 1) / 3 - g_cur - g_prev)
        * 2
        * (y / 12 + g_cur)
        - ((y + 1) / 3 - g_next - g_cur) * (y / 12 + g_cur) ** 2
        - ((y - 1) / 3 - g_cur - g_prev) * (y / 12 + g_cur) ** 2
        + 2 * ((y / 6 - g_cur) ** 2 - gamma * gamma / 16) * 2 * (y / 6 - g_cur)
    )


def half_hermgauss(n, gamma=0.0, eps=1e-14, n_iter=100):
    idxs = np.arange(n)

    g = estimate_g(idxs, gamma)

    it = 0

    # Refine g using Newton's method.
    while it < n_iter:
        residual = cost_F(idxs, gamma, g)

        if np.dot(residual, residual) < eps:
            break

        a_coeffs = jac_dF_dg_prev(idxs, gamma, g)
        b_coeffs = jac_dF_dg_cur(idxs, gamma, g)
        c_coeffs = jac_dF_dg_next(idxs, gamma, g)

        a_coeffs[:2] = 0
        # Tridiagonal Jacobian w.r.t g_{n-1}, g_n, and g_{n+1}.
        J = np.vstack((a_coeffs[1:], b_coeffs[1:], c_coeffs[1:]))

        delta = solve_banded((1, 1), J, residual[1:])
        g[1:-2] -= delta

        it = it + 1

    # Evaluate the recurrence formula
    alpha = np.empty_like(a_coeffs)

    # Eq. 3.1
    alpha[0] = np.exp(gammaln(1 + gamma / 2)) / np.exp(gammaln((1 + gamma) / 2))

    idxs1 = idxs[1:]
    # Eq. 3.2
    alpha[1:] = np.sqrt((2 * idxs1 + gamma + 1.0) / 3.0 - g[idxs1 + 1] - g[idxs1])
    # Eq. 3.3
    beta = (idxs1 + gamma / 2.0) / 6.0 + g[idxs1]

    # Coefficients on the subdiagonal
    s = np.sqrt(beta)

    # Compute the eigenvalues and the eigenvectors. Eigenvectors are already
    # sorted and normalized.
    eval, evec = eigh_tridiagonal(alpha, s)

    # Gaussian weights
    weights = np.sqrt(np.pi) / 2.0 * np.square(evec[0, :])

    return eval, weights
