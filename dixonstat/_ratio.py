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

from ._quadrature import half_hermgauss
from ._survival import survival
from functools import partial
from scipy.optimize import brentq
from scipy.special import gammaln
from scipy.stats import norm
import numpy as np

# The density concentrates with increasing sample size, so a fixed order cannot
# serve every size. Up to 300 observations, this rule keeps the relative error
# of the probability between the 0.95 and 0.995 percentiles below 1e-11 and that
# of the tail above the 0.995 percentile below 1e-10, at the lowest cost among
# the rules compared against golden values.
BASE_HALF_RANGE_ORDER = 30
HALF_RANGE_ORDER_SLOPE = 6
FULL_RANGE_ORDER_FACTOR = 2

# The highest degree for which numpy.polynomial.hermite.hermgauss returns finite
# weights, determined with NumPy 2.5.
MAX_FULL_RANGE_ORDER = 370


def _default_half_range_order(size):
    return BASE_HALF_RANGE_ORDER + round(HALF_RANGE_ORDER_SLOPE * np.sqrt(size))


# Absolute tolerance of a critical value. The default of brentq is 2e-12.
PPF_XTOL = 1e-14

# Step of the central difference for the density, relative to the distance of a
# critical value from the ends of the unit interval.
DENSITY_RELATIVE_STEP = 1e-3

METHODS = ('gauss-kronrod', 'quadrature')


def _validate_method(method):
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}', expected one of {METHODS}")


def _apply1d(func, data):
    arr = np.asarray(data)
    result = np.apply_along_axis(func, 0, np.atleast_2d(arr))

    return result.item() if arr.size == 1 else result.reshape(arr.shape)


class RangeRatio:
    R"""Dixon's :math:`r_{j,i}` range ratio statistic.

    The statistic can be used to detect outliers in data sets containing samples
    assumed to be normally distributed.

    Parameters
    ----------
    size : int
        The number of observations in the data set.
    j : int
        The number of possible suspected outliers on the same end of the data as
        the value being tested.
    i : int
        The number of possible outliers on the opposite end of the data from the
        suspected value.
    hgh_order : int, optional
        The number of quadrature points for the half-range :math:`[0,\infty)`
        Gauss-Hermite quadrature used to compute probability density. Defaults
        to :math:`30+6\sqrt{n}` for :math:`n` observations. Quadrature orders
        affect :meth:`pdf` and ``method='quadrature'`` only.
    fgh_order : int, optional
        The number of quadrature points for the full-range
        :math:`[-\infty,\infty)` Gauss-Hermite quadrature used to compute
        probability density. Defaults to twice the default of `hgh_order` up to
        an order of 370, the largest NumPy supports, which limits the accuracy
        beyond about 660 observations.
    gl_order : int
        The number of quadrature points for the Gauss-Legendre quadrature used
        to compute the cumulative distribution.

    Raises
    ------
    ValueError
        Thrown if `size` is smaller than 3.
    """

    def __init__(self, size, j, i, hgh_order=None, fgh_order=None, gl_order=16):
        if size < 3:
            raise ValueError("Dixon's Q test requires at least 3 samples")

        self.Phi = norm().cdf
        self.sqrt2 = np.sqrt(2.0)
        self.sqrt4_3 = np.sqrt(4.0 / 3.0)
        self.sqrt1_3 = np.sqrt(1.0 / 3.0)
        self.sqrt2_3 = np.sqrt(2.0 / 3.0)

        self.size = size
        self.i = i
        self.j = j

        default_order = _default_half_range_order(size)
        nhh = default_order if hgh_order is None else hgh_order
        nfh = (
            min(FULL_RANGE_ORDER_FACTOR * default_order, MAX_FULL_RANGE_ORDER)
            if fgh_order is None
            else fgh_order
        )
        ngl = gl_order

        nvec = nhh * nfh

        xhh, whh = half_hermgauss(nhh)
        xfh, wfh = np.polynomial.hermite.hermgauss(nfh)
        xgl, wgl = np.polynomial.legendre.leggauss(ngl)

        # Composite index m = l * nfh + k combines the half-range index l
        # and the full-range index k into a single flat grid.
        t = np.repeat(xhh, nfh)
        u = np.tile(xfh, nhh)
        x = u * self.sqrt2_3
        w = np.outer(whh, wfh).ravel()  # combined weight
        with np.errstate(divide='ignore'):
            log_w = np.log(w)
        z = t * u
        c2 = self.Phi(x)

        if self.size - j - i - 1 < 0:
            raise ValueError(
                f'too few samples ({self.size}); at least {j + i + 1} are required'
            )

        # Compute normalization factor (term in Dixon's eqn containing
        # factorials, plus three 1/sqrt(2*pi) terms from normal distributions)
        # in log-space using gammaln, since the individual factorials
        # overflow float64 for sample sizes as small as 171 even though
        # their ratio remains representable.
        log_den = gammaln(self.i) + gammaln(self.size - j - i) + gammaln(self.j)
        log_factor = -1.5 * np.log(2.0 * np.pi) + gammaln(self.size + 1) - log_den
        factor = np.exp(log_factor)

        self.nvec = nvec
        self.ngl = ngl
        self.t = t
        self.u = u
        self.x = x
        self.w = w
        self.log_w = log_w
        self.z = z
        self.c2 = c2

        self.xhh = xhh
        self.whh = whh
        self.xfh = xfh
        self.wfh = wfh
        self.xgl = xgl
        self.wgl = wgl

        self.factor = factor

    def pdf(self, r):
        R"""Computes the probability density function (PDF) of the distribution.

        .. math::

            P(x_i,x_{n-j},x_n)
            =
            \frac{n!}{(i-1)!(n-j-i-1)!(j-1)!}
            \left[ \int_{-\infty}^{x_i} \phi(t)\,\mathrm{d}t \right]^{i-1}
            \left[ \int_{x_i}^{x_{n-j}} \phi(t)\,\mathrm{d}t \right]^{n-j-i-1}
            \left[ \int_{x_{n-j}}^{x_n} \phi(t)\,\mathrm{d}t \right]^{j-1}
            \phi(x_i)\phi(x_{n-j})\phi(x_n)

        where :math:`\phi(x)` is the probability density function of the
        standard normal distribution defined as

        .. math::

            \phi(x)= \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{x^2}{2} \right)

        Parameters
        ----------
        r : array_like
            The :math:`r` value.

        Returns
        -------
        numpy.ndarray
            The PDF at the specified value.
        """
        factor = np.reciprocal(np.sqrt(1.0 + np.square(r)))
        f = (1.0 + r) * 2.0 * factor * self.sqrt1_3

        v = self.t[..., np.newaxis, np.newaxis] * factor * self.sqrt2
        c1 = self.Phi(self.x[..., np.newaxis, np.newaxis] - v)
        c3 = self.Phi(self.x[..., np.newaxis, np.newaxis] - r * v)
        g = (
            c1 ** (self.i - 1)
            * (c3 - c1) ** (self.size - self.j - self.i - 1)
            * (self.c2[..., np.newaxis, np.newaxis] - c3) ** (self.j - 1)
        )
        # Combine the weight with the exponential in log space. The exponential
        # overflows at high orders even though the weight compensates for it. A
        # weight that underflowed to zero contributes an exponent of -inf.
        exponent = self.z[..., np.newaxis, np.newaxis] * f
        exponent = exponent + self.log_w[..., np.newaxis, np.newaxis]
        # Weighted sum over the quadrature nodes (axis 0).
        density = np.sum(v * np.exp(exponent) * g, axis=0)

        return density * factor * self.factor * self.sqrt4_3

    def sf(self, R, return_error=False):
        R"""Computes the survival function :math:`1-G(R)` directly.

        The tail is integrated by adaptive Gauss-Kronrod cubature without
        subtracting from 1, so it keeps its relative accuracy where critical
        values lie. See :func:`dixonstat._survival.survival` for the formula.

        Parameters
        ----------
        R : array_like
            The ratio.
        return_error : bool
            Whether to return an estimate of the absolute error as well.

        Returns
        -------
        values : numpy.ndarray
            The probability that the statistic exceeds each ratio.
        errors : numpy.ndarray
            The estimated absolute errors, only if `return_error` is set.
        """
        results = [survival(r, self.size, self.j, self.i) for r in np.ravel(R)]
        values = np.reshape([value for value, _ in results], np.shape(R))

        if not return_error:
            return values

        return values, np.reshape([error for _, error in results], np.shape(R))

    def cdf(self, R, method='gauss-kronrod'):
        R"""The cumulative distribution function.

        The monotonically increasing cumulative distribution function (CDF)
        :math:`G(R)` is given by

        .. math::

            G(R) = \int_0^R P(r)\,\mathrm{d}r,
            \quad
            \text{for}~0\leq r\leq 1

        Parameters
        ----------
        R : array_like
            The ratio.
        method : {'gauss-kronrod', 'quadrature'}
            ``gauss-kronrod`` subtracts the error controlled :meth:`sf` from 1.
            ``quadrature`` integrates :meth:`pdf` with the orders of
            :class:`RangeRatio`. It is much faster and provides no error
            control.

        Raises
        ------
        ValueError
            If `method` is unknown.
        """
        _validate_method(method)

        if method == 'gauss-kronrod':
            return 1.0 - self.sf(R)

        # Save the shape such that we can return a tensor of CDF values with
        # the same shape.
        shape = np.shape(R)
        R = np.atleast_2d(R)

        # For each ratio, we have multiple weights.
        half_R = R * 0.5
        # Quadrature nodes
        prob = self.pdf(half_R[..., np.newaxis] * (self.xgl + 1))
        # Summing over the weighted probabilities results in the integral from
        # 0 to R (i.e., each ratio).
        cdf = np.einsum('...k,k->...', prob, self.wgl) * half_R

        # The quadrature can overshoot slightly past the theoretical bounds
        # near the tails; clip to enforce the CDF invariant.
        cdf = np.clip(cdf, 0.0, 1.0)

        # Restore the shape of input parameter R.
        return np.reshape(cdf, shape)

    def __ppf_err(self, R, q, method):
        return q - self.cdf(R, method=method)

    def __single_ppf(self, q, method):
        return brentq(
            partial(self.__ppf_err, q=q.item(), method=method),
            a=0.0,
            b=1.0,
            xtol=PPF_XTOL,
        )

    def __ppf_error(self, R):
        # An error of the tail moves its root by about error / density. The
        # density follows from a central difference of the tail itself, so that
        # the estimate does not depend on the quadrature orders.
        step = DENSITY_RELATIVE_STEP * min(R, 1.0 - R)

        if step <= 0.0:
            return PPF_XTOL

        upper = survival(R + step, self.size, self.j, self.i)[0]
        lower = survival(R - step, self.size, self.j, self.i)[0]
        density = (lower - upper) / (2.0 * step)

        return survival(R, self.size, self.j, self.i)[1] / density + PPF_XTOL

    def ppf(self, q, method='gauss-kronrod', return_error=False):
        R"""Computes percent point function value at the percentile ``q``.

        Percent point function (inverse of cdf). The critical values correspond
        to the roots of

        .. math::

            q-G(R)=0

        Parameters
        ----------
        q
            The percentile.
        method : {'gauss-kronrod', 'quadrature'}
            The integration method. See :meth:`cdf`.
        return_error : bool
            Whether to return an estimate of the absolute error as well. The
            estimate propagates the error of the tail through the density as
            :math:`\delta R\approx\delta G/P(R)` and adds the tolerance of the
            root finding. It costs three additional integrals per percentile
            and is not a rigorous bound.

        Returns
        -------
        R : float or numpy.ndarray
            The critical value for each percentile.
        error : float or numpy.ndarray
            The estimated absolute error, only if `return_error` is set.

        Raises
        ------
        ValueError
            If `q` lies outside :math:`[0,1)`, `method` is unknown, or
            `return_error` is set for a method without an error estimate.
        """
        if np.any(q < 0.0):
            raise ValueError('percentile cannot be negative')

        if np.any(q >= 1.0):
            raise ValueError('percentile cannot be larger than or equal to 1.0')

        if return_error and method != 'gauss-kronrod':
            raise ValueError(f"no error estimate for method '{method}'")

        roots = _apply1d(partial(self.__single_ppf, method=method), q)

        if not return_error:
            return roots

        if np.ndim(roots) == 0:
            return roots, self.__ppf_error(roots)

        errors = [self.__ppf_error(root) for root in np.ravel(roots)]

        return roots, np.reshape(errors, np.shape(roots))


def r10(size, **kwargs):
    """Returns the :math:`r_{10}` range ratio statistic."""
    return RangeRatio(size, 1, 1, **kwargs)


def r11(size, **kwargs):
    """Returns the :math:`r_{11}` range ratio statistic."""
    return RangeRatio(size, 1, 2, **kwargs)


def r12(size, **kwargs):
    """Returns the :math:`r_{12}` range ratio statistic."""
    return RangeRatio(size, 1, 3, **kwargs)


def r20(size, **kwargs):
    """Returns the :math:`r_{20}` range ratio statistic."""
    return RangeRatio(size, 2, 1, **kwargs)


def r21(size, **kwargs):
    """Returns the :math:`r_{21}` range ratio statistic."""
    return RangeRatio(size, 2, 2, **kwargs)


def r22(size, **kwargs):
    """Returns the :math:`r_{22}` range ratio statistic."""
    return RangeRatio(size, 2, 3, **kwargs)


Q = r10
"""Returns Dixon's :math:`Q` statistic.

This is a convenience alias for :meth:`dixonstat.r10`.

Parameters
----------
size
    The number of observations in the data set being analysed.
kwargs
    The number of quadrature points used for initializing :class:`RangeRatio`.
"""


def ratiotest(ratio, rvs, alpha=0.05, alternative='one-sided', **kwargs):
    """Perform Dixon's ratio test.

    If the statistic :math:`Q > p`, :math:`x_1` is considered an outlier.

    Parameters
    ----------
    ratio : collections.abc.Callable, RangeRatio
        Dixon's ratio statistic.
    rvs : array
        1-D array of observations of random variables.
    alternative : `{'one-sided', 'two-sided'}`, str
        ``one-sided`` for one-tailed test, ``two-sided`` for two-tailed test.

    Returns
    -------
    statistic : float
        The ratio statistic.
    p-value: float
    """

    s = ratio(size=len(rvs), **kwargs)
    sorted_rvs = np.sort(rvs)

    xi = sorted_rvs[s.i - 1]
    xn = sorted_rvs[-1]
    xnj = sorted_rvs[-s.j - 1]

    # rij
    # gap = (xn - xi) / (xnj - xi)

    # rji
    gap = (xn - xnj) / (xn - xi)

    if alternative == 'one-sided':
        q = 1 - alpha
    elif alternative == 'two-sided':
        q = 1 - alpha / 2.0
    else:
        raise ValueError('unknown parameter value \'{}\''.format(alternative))

    return gap, s.ppf(q)


def main():
    # r11 = RangeRatio(30, 1, 2)

    # print(r11.ppf(1 - 0.005))

    # import matplotlib.pyplot as plt

    # plt.figure()

    # s = r21(30)

    # x = np.linspace(0, 0.9999, 500)
    # y = s.ppf(x)

    # plt.plot(x, y)
    # plt.show()
    pass
