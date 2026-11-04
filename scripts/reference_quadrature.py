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

"""High-precision reference for half-range Gauss-Hermite quadrature.

The moments of the weight :math:`\\exp(-x^2)` on :math:`[0,\\infty)` are known
in closed form. The recurrence coefficients follow from them through the
Chebyshev algorithm. Mapping moments to coefficients is severely
ill-conditioned, which is why the computation needs far more digits than the
result keeps.
"""

import mpmath as mp

# Comparing 400 with 800 working digits for n = 68 shows a loss of 74 digits, so
# this default leaves more than 70 correct digits up to that order.
DEFAULT_DPS = 150


def half_hermite_moment(k):
    r"""Return :math:`\int_0^\infty x^k \exp(-x^2)\,\mathrm{d}x` as an mpf."""
    return mp.gamma(mp.mpf(k + 1) / 2) / 2


def half_hermite_recurrence(n, dps=DEFAULT_DPS):
    r"""Compute the recurrence coefficients of the monic orthogonal polynomials.

    Parameters
    ----------
    n : int
        The number of coefficient pairs.
    dps : int
        The working precision in decimal digits.

    Returns
    -------
    alpha, beta : list of mpmath.mpf
        The diagonal and the squared subdiagonal of the Jacobi matrix.
        ``beta[0]`` is the zeroth moment.
    """
    with mp.workdps(dps):
        size = 2 * n
        moments = [half_hermite_moment(k) for k in range(size)]
        alpha = [mp.mpf(0)] * n
        beta = [mp.mpf(0)] * n
        alpha[0] = moments[1] / moments[0]
        beta[0] = moments[0]
        previous = [mp.mpf(0)] * size
        current = moments
        for k in range(1, n):
            following = [mp.mpf(0)] * size
            for m in range(k, size - k):
                following[m] = (
                    current[m + 1]
                    - alpha[k - 1] * current[m]
                    - beta[k - 1] * previous[m]
                )
            alpha[k] = following[k + 1] / following[k] - current[k] / current[k - 1]
            beta[k] = following[k] / current[k - 1]
            previous = current
            current = following
        return alpha, beta


def half_hermite_rule(n, dps=DEFAULT_DPS):
    """Compute the nodes and the weights of the n-point half-range Gauss rule.

    Parameters
    ----------
    n : int
        The number of nodes.
    dps : int
        The working precision in decimal digits.

    Returns
    -------
    nodes, weights : list of mpmath.mpf
        The nodes in ascending order and their weights.
    """
    alpha, beta = half_hermite_recurrence(n, dps)
    with mp.workdps(dps):
        jacobi = mp.matrix(n, n)
        for k in range(n):
            jacobi[k, k] = alpha[k]
        for k in range(1, n):
            jacobi[k, k - 1] = jacobi[k - 1, k] = mp.sqrt(beta[k])
        eigenvalues, eigenvectors = mp.eigsy(jacobi)
        nodes = [eigenvalues[k] for k in range(n)]
        weights = [beta[0] * eigenvectors[0, k] ** 2 for k in range(n)]
        return nodes, weights
