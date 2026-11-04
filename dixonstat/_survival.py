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

from scipy.integrate import cubature
from scipy.special import gammaln
from scipy.special import log_ndtr
from scipy.special import ndtr
from scipy.special import xlogy
import numpy as np
import warnings

# Relative tolerance of the integral. The error estimate of the rule exceeds the
# actual error by more than an order of magnitude across the golden table.
DEFAULT_RTOL = 1e-11
DEFAULT_MAX_SUBDIVISIONS = 10000


def _normal_mass(lower, upper):
    # Evaluate in the tail closer to the arguments, where the distribution
    # function keeps its relative accuracy instead of cancelling against 1.
    return np.where(lower > 0.0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower))


def survival(
    R, size, j, i, rtol=DEFAULT_RTOL, max_subdivisions=DEFAULT_MAX_SUBDIVISIONS
):
    r"""Compute the upper tail of Dixon's :math:`r_{j,i-1}` statistic.

    Conditioning on :math:`a=x_i` and :math:`b=x_n` leaves :math:`m=n-i-1`
    independent observations on :math:`(a,b)`. The statistic exceeds :math:`R`
    exactly if fewer than :math:`j` of them lie above :math:`c=b-R(b-a)`, which
    reduces the distribution to a two-dimensional integral over a finite sum:

    .. math::

        1-G(R)
        =
        \frac{n!}{(i-1)!\,m!}
        \iint \Phi(a)^{i-1} \phi(a)\phi(b)
        \sum_{k=0}^{j-1} \binom{m}{k}
        \left[\Phi(b)-\Phi(c)\right]^k
        \left[\Phi(c)-\Phi(a)\right]^{m-k}
        \mathrm{d}a\,\mathrm{d}b

    The integral is evaluated by adaptive Gauss-Kronrod cubature in the
    variables :math:`b` and :math:`b-a`.

    Parameters
    ----------
    R : float
        The ratio.
    size : int
        The number of observations :math:`n`.
    j, i : int
        The indices of the statistic.
    rtol : float
        The relative tolerance of the integral.
    max_subdivisions : int
        The maximum number of subdivisions of the integration domain.

    Returns
    -------
    value : float
        The probability that the statistic exceeds `R`.
    error : float
        An estimate of the absolute error of `value`.

    Warns
    -----
    RuntimeWarning
        If the integral does not reach `rtol` within `max_subdivisions`.
    """
    if R <= 0.0:
        return 1.0, 0.0

    if R >= 1.0:
        return 0.0, 0.0

    m = size - i - 1
    k = np.arange(j)[:, np.newaxis]
    log_binomial = gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1)
    log_constant = gammaln(size + 1) - gammaln(i) - gammaln(m + 1) - np.log(2.0 * np.pi)

    def integrand(points):
        b = points[:, 0]
        s = points[:, 1]
        a = b - s
        c = b - R * s
        tail = np.exp(
            log_binomial
            + xlogy(k, _normal_mass(c, b))
            + xlogy(m - k, _normal_mass(a, c))
        ).sum(axis=0)
        return tail * np.exp(
            log_constant + (i - 1) * log_ndtr(a) - 0.5 * (a * a + b * b)
        )

    result = cubature(
        integrand,
        [-np.inf, 0.0],
        [np.inf, np.inf],
        rule='gk21',
        rtol=rtol,
        atol=0.0,
        max_subdivisions=max_subdivisions,
    )

    if result.status != 'converged':
        warnings.warn(
            f'survival function at R={R} did not reach the relative tolerance '
            f'{rtol} in {max_subdivisions} subdivisions, estimated error '
            f'{result.error} of the value {result.estimate}',
            RuntimeWarning,
            stacklevel=2,
        )

    return float(result.estimate), float(result.error)
