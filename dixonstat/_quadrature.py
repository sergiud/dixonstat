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

from scipy.linalg import eigvalsh_tridiagonal
from scipy.linalg import solve_banded
from scipy.special import gammaln
import numpy as np
import warnings

# Coefficients solved beyond those a rule uses, which keeps the error of the
# asymptotic boundary value below rounding error for every order.
NEWTON_GUARD_COEFFICIENTS = 8


def alpha0(gamma):
    # Eq. 3.1, evaluated as a single exponential of a log-ratio instead of
    # a ratio of two exponentials, since the individual gamma function
    # values overflow float64 far earlier than their ratio does.
    return np.exp(gammaln(1 + gamma / 2) - gammaln((1 + gamma) / 2))


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


def cost_and_jac(i, gamma, g):
    y = 2.0 * i + gamma

    g_prev = g[i - 1]
    g_cur = g[i]
    g_next = g[i + 1]

    a = (y + 1) / 3 - g_next - g_cur
    b = (y - 1) / 3 - g_cur - g_prev
    c = y / 12 + g_cur
    e = y / 6 - g_cur
    d = e**2 - gamma**2 / 16
    c2 = c**2

    # Eq. 3.14
    residual = a * b * c2 - d**2

    jac_prev = -a * c2
    jac_next = -b * c2
    jac_cur = 2 * a * b * c - a * c2 - b * c2 + 4 * d * e

    return residual, jac_prev, jac_cur, jac_next


def _christoffel_weights(x, alpha, s, zeroth_moment):
    # The weights from the first components of the eigenvectors are accurate
    # only relative to the largest weight, so the tiny outermost weights consist
    # of rounding errors. The Christoffel function 1 / sum_k p_k(x)^2 of the
    # orthonormal polynomials retains their relative accuracy.
    previous = np.zeros_like(x)
    current = np.ones_like(x)
    total = np.ones_like(x)

    # The polynomials overflow at the outermost nodes of very high orders. Such
    # a node has a weight below the smallest float64 and ends up with an infinite
    # or undefined sum.
    with np.errstate(over='ignore', invalid='ignore'):
        for k, s_k in enumerate(s):
            s_prev = s[k - 1] if k > 0 else 0.0
            previous, current = (
                current,
                ((x - alpha[k]) * current - s_prev * previous) / s_k,
            )
            total += np.square(current)

    return np.where(np.isnan(total), 0.0, zeroth_moment / total)


def half_hermgauss(n, gamma=0.0, rtol=1e-8, n_iter=100):
    # g_m at the boundary keeps its asymptotic estimate, which is poor for small
    # m. Its error decays by more than a factor of 10 per index away from the
    # boundary, so solving for additional coefficients removes it from those the
    # rule uses.
    m = n + NEWTON_GUARD_COEFFICIENTS
    idxs = np.arange(m)

    g = estimate_g(idxs, gamma)

    # Refine g_1, ..., g_{m-1} using Newton's method on Eq. 3.14 for
    # k = 1, ..., m-1. g_0 and the asymptotic estimate of g_m are boundary
    # values. The Jacobian is tridiagonal and solved via solve_banded in O(n)
    # per iteration. scipy.optimize.root has no banded-Jacobian option, so it
    # would fall back to a dense solve here and slow this down rather than
    # help.
    #
    # solve_banded expects the diagonal ordered form ab[u + r - c, c] = J[r, c].
    # Row 0 holds the superdiagonal shifted right by one, row 2 the subdiagonal
    # shifted left by one.
    ab = np.zeros((3, m - 1))

    for _ in range(n_iter):
        residual, jac_prev, jac_cur, jac_next = cost_and_jac(idxs, gamma, g)

        ab[0, 1:] = jac_next[1:-1]
        ab[1, :] = jac_cur[1:]
        ab[2, :-1] = jac_prev[2:]

        delta = solve_banded((1, 1), ab, residual[1:])
        g[1:-2] -= delta

        # Newton converges quadratically, so an updated iterate is accurate to
        # rounding error once the relative step drops below rtol. The residual
        # stalls at a rounding floor due to cancellation in Eq. 3.14, which
        # makes it unsuitable as a stopping criterion.
        if np.all(np.abs(delta) <= rtol * np.abs(g[1:-2])):
            break
    else:
        warnings.warn(
            f'half-range Hermite recurrence did not converge in {n_iter} iterations',
            RuntimeWarning,
            stacklevel=2,
        )

    # Evaluate the recurrence formula
    alpha = np.empty(n)

    alpha[0] = alpha0(gamma)

    idxs1 = idxs[1:n]
    # Eq. 3.2
    alpha[1:] = np.sqrt((2 * idxs1 + gamma + 1.0) / 3.0 - g[idxs1 + 1] - g[idxs1])
    # Eq. 3.3
    beta = (idxs1 + gamma / 2.0) / 6.0 + g[idxs1]

    # Coefficients on the subdiagonal
    s = np.sqrt(beta)

    eval = eigvalsh_tridiagonal(alpha, s)

    # Gaussian weights
    weights = _christoffel_weights(eval, alpha, s, np.sqrt(np.pi) / 2.0)

    return eval, weights
