# Identification and rejection of outliers using Dixon's r statistics.
#
# Copyright 2017 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

from functools import partial
from scipy.optimize import brentq
from scipy.special import gammaln
from scipy.stats import norm
import numpy as np

def _apply1d(func, data):
    arr = np.asarray(data)
    result = np.apply_along_axis(func, 0, np.atleast_2d(arr))

    return result.item() if arr.size == 1 else result.reshape(arr.shape)

def estimate_g(i, gamma):
    gamma2 = np.square(gamma)

    # Use Horner's rule for efficient polynomial evaluation of Eqs. 3.10-3.13.
    # NOTE the sign changes are due to introduced parentheses.
    C0 = 1.0 / 36.0 - gamma2 / 8.0
    C1 = 23.0 / 432.0 - (11.0 / 48.0 - 3.0 / 32.0 * gamma2) * gamma2
    C2 = 1189.0 / 2592.0 - (409.0 / 192.0 - (75.0 / 64.0 - 9.0 / 64.0 * gamma2) * gamma2) * gamma2
    C3 = 196057.0 / 20736.0 - (153559.0 / 3456.0 - (7111.0 / 256.0 - (639.0 / 128.0 - 135.0 / 512.0 * gamma2) * gamma2) * gamma2) * gamma2

    y = 2.0 * (i + 1) + gamma
    y2 = np.square(y)

    # Eq. 3.9
    g = (C0 + (C1 + (C2 + C3 / y2) / y2) / y2) / y

    return np.concatenate(([0], g, [0]))


def cost_F(i, gamma, g):
    y = 2.0 * i + gamma

    g_prev = g[i - 1]
    g_cur = g[i]
    g_next = g[i + 1]

    F = ((y + 1) / 3 - g_next - g_cur) * \
        ((y - 1) / 3 - g_cur - g_prev) * (y / 12 + g_cur) ** 2 - \
        ((y / 6 - g_cur) ** 2 - gamma ** 2 / 16) ** 2

    FF = np.concatenate((F, [0]))

    return FF, np.dot(FF, FF)


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

    return ((y + 1) / 3 - g_next - g_cur) * \
        ((y - 1) / 3 - g_cur - g_prev) * 2 * (y / 12 + g_cur) - \
        ((y + 1) / 3 - g_next - g_cur) * (y / 12 + g_cur) ** 2 - \
        ((y - 1) / 3 - g_cur - g_prev) * (y / 12 + g_cur) ** 2 + \
        2 * ((y / 6 - g_cur) ** 2 - gamma * gamma / 16) * 2 * (y / 6 - g_cur)

def tdma(a, b, c, d):
    c[0] = c[0] / b[0]
    d[0] = d[0] / b[0]

    n = len(a)

    for i in range(1, n):
        den = (b[i] - c[i - 1] * a[i])
        c[i] = c[i] / den
        d[i] = (d[i] - d[i - 1] * a[i]) / den

    x = np.zeros_like(d)

    # Back substitution
    for i in reversed(range(n)):
        if i == n - 1:
            x[i] = d[i]
        else:
            x[i] = d[i] - c[i] * x[i + 1]

    return x


def tqli(d, sub, return_eigenvectors=True, eps=1e-14):
    n = len(d)

    if len(sub) != n - 1:
        raise ValueError('incorrect subdiagonal vector size')

    e = np.concatenate((sub, [0.0]))

    if return_eigenvectors:
        z = np.eye(n)

    for l in range(n):
        iter = 0
        idxs = np.arange(l, n - 1)

        while True:
            dd = np.abs(d[idxs]) + np.abs(d[idxs + 1])
            mask = ~(np.abs(e[l:-1]) > eps * dd)
            a, = np.where(mask)

            m = n - 1 if a.size == 0 else idxs[a[0]]

            if m != l:
                iter = iter + 1

                g = (d[l + 1] - d[l]) / (2.0 * e[l])
                r = np.hypot(g, 1.0)
                g = d[m] - d[l] + e[l] / (g + r * np.sign(g))
                s = 1.0
                c = 1.0
                p = 0.0

                for i in reversed(range(l, m)):
                    f = s * e[i]
                    b = c * e[i]
                    r = np.hypot(f, g)
                    e[i + 1] = r

                    if r == 0.0:
                        d[i + 1] -= p
                        e[m] = 0
                        break

                    s = f / r
                    c = g / r
                    g = d[i + 1] - p
                    r = (d[i] - g) * s + 2.0 * c * b
                    p = s * r
                    d[i + 1] = g + p
                    g = c * r - b

                    if return_eigenvectors:
                        f = np.copy(z[:, i + 1])
                        z[:, i + 1] = s * z[:, i] + c * f
                        z[:, i] = c * z[:, i] - s * f

                if r == 0.0 and i >= 1:
                    continue

                d[l] -= p
                e[l] = g
                e[m] = 0
            else:
                break

    if return_eigenvectors:
        return d, z

    return d


def half_hermgauss(n, gamma=0.0, eps=1e-14, n_iter=100):
    idxs = np.arange(n)

    g = estimate_g(idxs, gamma)
    f, f_error = cost_F(idxs, gamma, g)

    it = 0

    while f_error > eps and it < n_iter:
        it = it + 1

        a_coeffs = jac_dF_dg_prev(idxs, gamma, g)
        b_coeffs = jac_dF_dg_cur(idxs, gamma, g)
        c_coeffs = jac_dF_dg_next(idxs, gamma, g)

        a_coeffs[:2] = 0

        delta = tdma(a_coeffs[1:], b_coeffs[1:], c_coeffs[1:], f[1:])
        g[1:-1] -= delta

        f, f_error = cost_F(idxs, gamma, g)

    alpha = np.zeros_like(a_coeffs)
    beta = np.zeros_like(alpha)

    alpha[0] = np.exp(gammaln(1 + gamma / 2)) / \
        np.exp(gammaln((1 + gamma) / 2))
    beta[0] = np.sqrt(np.pi) / 2.0

    idxs1 = idxs[1:]
    alpha[1:] = np.sqrt((2 * idxs1 + gamma + 1.0) / 3.0 -
                        g[idxs1 + 1] - g[idxs1])
    beta[1:] = (idxs1 + gamma / 2.0) / 6.0 + g[idxs1]

    # Coefficients on the subdiagonal
    s = np.sqrt(beta[1:])

    # Compute the eigenvalues and the eigenvectors
    eval, evec = tqli(alpha, s)

    # Sort the eigenvalues and the corresponding eigenvectors in ascending
    # order.
    sorted_idxs = np.argsort(eval)
    eval = eval[sorted_idxs]
    evec = evec[..., sorted_idxs]

    # Rowwise sum of squared eigenvector matrix coefficients
    norm = np.sum(np.square(evec), axis=0)

    evec /= np.tile(np.atleast_2d(norm).T, (1, evec.shape[1]))

    # Gaussian weights
    weights = beta[0] * np.square(evec[0, :])

    return eval, weights


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
    hgh_order : int
        The number of quadrature points for the half-range :math:`[0,\infty)`
        Gauss-Hermite quadrature used to compute probability density.
    fgh_order : int
        The number of quadrature points for the full-range
        :math:`[-\infty,\infty)` Gauss-Hermite quadrature used to compute
        probability density.
    gl_order : int
        The number of quadrature points for the Gauss-Legendre quadrature used
        to compute the cumulative distribution.

    Raises
    ------
    ValueError
        Thrown if `size` is smaller than 3.
    """

    def __init__(self, size, j, i, hgh_order=17, fgh_order=31, gl_order=16):
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

        ngl = gl_order
        nfh = fgh_order
        nhh = hgh_order

        nvec = nhh * nfh
        t = np.zeros(nvec)
        u = np.zeros(nvec)
        x = np.zeros(nvec)
        w = np.zeros(nvec)
        z = np.zeros(nvec)
        c2 = np.zeros(nvec)

        xhh, whh = half_hermgauss(nhh)
        xfh, wfh = np.polynomial.hermite.hermgauss(nfh)
        xgl, wgl = np.polynomial.legendre.leggauss(ngl)

        m = 0

        for l in range(nhh):     # half-range index
            for k in range(nfh):  # full-range index
                t[m] = xhh[l]
                u[m] = xfh[k]
                x[m] = u[m] * self.sqrt2_3
                w[m] = whh[l] * wfh[k]  # combined weight
                z[m] = t[m] * u[m]
                c2[m] = self.Phi(x[m])

                m = m + 1          # composite index

        # Compute normalization factor (term in Dixon's eqn containing
        # factorials, plus three 1/sqrt(2*pi) terms from normal distributions)
        factor = np.reciprocal(np.sqrt((2.0 * np.pi)**3))

        for k in reversed(np.arange(self.size) + 1):
            pf = np.astype(k, float)

            if k <= self.i - 1:
                pf = pf / k
            if k <= self.size - self.j - self.i - 1:
                pf = pf / k
            if k <= self.j - 1:
                pf = pf / k

            factor = factor * pf

        self.nvec = nvec
        self.ngl = ngl
        self.t = t
        self.u = u
        self.x = x
        self.w = w
        self.z = z
        self.c2 = c2

        self.xhh = xhh
        self.whh = whh
        self.xfh = xfh
        self.wfh = wfh
        self.xgl = xgl
        self.wgl = wgl

        self.factor = factor

    def __single_pdf(self, r):
        factor = np.reciprocal(np.sqrt(1.0 + np.square(r)))
        f = (1.0 + r) * 2.0 * factor * self.sqrt1_3

        v = self.t * factor * self.sqrt2
        c1 = self.Phi(self.x - v)
        c3 = self.Phi(self.x - r * v)
        g = c1**(self.i - 1) * (c3 - c1)**(self.size -
                                           self.j - self.i - 1) * (self.c2 - c3)**(self.j - 1)
        density = self.w * v * np.exp(f * self.z) * g

        return np.sum(density) * factor * self.factor * self.sqrt4_3

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
        return _apply1d(self.__single_pdf, r)

    def __single_cdf(self, R):
        pr = self.pdf(0.5 * R * (self.xgl + 1))
        cdf = self.wgl * pr

        return np.sum(cdf) * R * 0.5

    def cdf(self, R):
        R"""The cumulative distribution function.

        The monotonically increasing cumulative distribution function (CDF)
        :math:`G(R)` is given by

        .. math::

            G(R) = \int_0^R P(r)\,\mathrm{d}r,
            \quad
            \text{for}~0\leq r\leq 1
        """

        return _apply1d(self.__single_cdf, R)

    def __ppf_err(self, R, q):
        return q - self.cdf(R)

    def __single_ppf(self, q):
        return brentq(partial(self.__ppf_err, q=q.item()), a=0.0, b=1.0)

    def ppf(self, q):
        R"""Computes percent point function value at the percentile ``q``.

        Percent point function (inverse of cdf). The critical values correspond
        to the roots of

        .. math::

            q-G(R)=0

        Parameters
        ----------
        q
            The percentile.
        """
        if np.any(q < 0.0):
            raise ValueError('percentile cannot be negative')

        if not np.any(q < 1.0):
            raise ValueError('percentile cannot be larger than or equal to 1.0')

        return _apply1d(self.__single_ppf, q)


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
    ratio : callable
        Dixon's ratio statistic.
    rvs : array
        1-D array of observations of random variables.
    alternative : str
        `{'one-sided', 'two-sided'}` for one-tailed test, ``two-sided`` for
        two-tailed test.

    Returns
    -------
    tuple
        statistic: `float` p-value: `float`
    """

    s = ratio(size=len(rvs), **kwargs)
    sorted_rvs = np.sort(rvs)

    xi = sorted_rvs[s.i - 1]
    xn = sorted_rvs[-1]
    xnj = sorted_rvs[-s.j]

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


if __name__ == '__main__':
    #r11 = RangeRatio(7, 1, 2)

    #print(r11.cdf(0.49))

    #r11 = RangeRatio(30, 1, 2)

    #print(r11.ppf(1 - 0.005))

    #import matplotlib.pyplot as plt

    #plt.figure()

    #s = r21(30)

    #x = np.linspace(0, 0.9999, 500)
    #y = s.ppf(x)

    #plt.plot(x, y)
    #plt.show()
    pass