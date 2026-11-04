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

r"""Arbitrary-precision survival function of Dixon's range ratios.

Conditioning on :math:`a=x_{(i)}` and :math:`b=x_{(n)}` leaves
:math:`m=n-i-1` independent points on :math:`(a,b)`. The statistic exceeds
:math:`R` exactly when fewer than :math:`j` of them lie above
:math:`c=b-R(b-a)`, which is a binomial probability. Only the integral over
:math:`(a,b)` remains:

.. math::

    1-G(R)
    =
    \frac{n!}{(i-1)!\,m!}
    \iint \Phi(a)^{i-1} \phi(a)\phi(b)
    \sum_{k=0}^{j-1} \binom{m}{k}
    (\Phi(b)-\Phi(c))^k (\Phi(c)-\Phi(a))^{m-k}
    \,\mathrm{d}a\,\mathrm{d}b

This formulation shares no code or numerical method with the density based
quadrature of the package, which makes it suitable as an independent oracle.
Nested tanh-sinh quadrature is orders of magnitude too slow for interactive
use, so values are stored by running this module as a script.
"""

from scipy.integrate import dblquad
from scipy.optimize import brentq
from scipy.special import gammaln
from scipy.special import ndtr
import argparse
import json
import mpmath as mp
import multiprocessing
import numpy as np
import os
import pathlib
import warnings

# The number of significant digits of the stored golden values, which exceeds
# the 17 digits needed to round float64 values correctly.
DEFAULT_DPS = 30

# The golden values have to round to float64 correctly. Fewer digits also lose
# the fixed offset of the secant iteration in rounding noise, which makes its
# first step divide by zero.
MIN_DPS = 17

# Guard digits of the outer and the inner integral. The inner integral needs
# more digits than the outer one, since otherwise its rounding noise keeps the
# outer rule from meeting its tolerance.
OUTER_GUARD_DIGITS = 5
INNER_GUARD_DIGITS = 15

# The float64 seed of the secant iteration only has to be close enough for the
# first step to reach the region of quadratic convergence.
SEED_BRACKET = (1e-3, 1.0 - 1e-6)
SEED_XTOL = 1e-11
SEED_EPSREL = 1e-10
SECANT_RELATIVE_OFFSET = '1e-9'
SECANT_GUARD_DIGITS = 2
MAX_SECANT_STEPS = 8

GENERATOR_COMMAND = 'python -m scripts.reference_dixon tests/data/critical_values.json'


def survival(R, size, j, i, dps=DEFAULT_DPS):
    r"""Compute :math:`1-G(R)` of the statistic :math:`r_{j,i-1}`.

    Parameters
    ----------
    R : str or mpmath.mpf
        The ratio. Pass a decimal string to avoid rounding to float64.
    size : int
        The number of observations :math:`n`.
    j, i : int
        The indices of the statistic. See :class:`dixonstat.RangeRatio`.
    dps : int
        The number of significant decimal digits of the result.

    Returns
    -------
    mpmath.mpf
        The probability that the statistic exceeds `R`.
    """
    m = size - i - 1
    outer_dps = dps + OUTER_GUARD_DIGITS
    inner_dps = dps + INNER_GUARD_DIGITS

    with mp.workdps(inner_dps):
        R = mp.mpf(R)
        constant = (
            mp.factorial(size) / (mp.factorial(i - 1) * mp.factorial(m)) / (2 * mp.pi)
        )
        binomials = [mp.binomial(m, k) for k in range(j)]
        # x_(n) concentrates near sqrt(2 ln n) and the sample range near twice
        # that value. The integrand narrows around both with increasing size,
        # and tanh-sinh quadrature misses such a peak without an error estimate
        # showing it unless a panel boundary lies next to it.
        mode = mp.sqrt(2 * mp.log(size))
        spread = 2 * mode
        inner_panels = [
            0,
            spread / 2,
            spread - 1,
            spread,
            spread + 1,
            spread + 3,
            mp.inf,
        ]
        outer_panels = [-mp.inf, 0, mode - 1, mode, mode + 1, mode + 3, mp.inf]

    def inner(b):
        with mp.workdps(inner_dps):
            b = mp.mpf(b)
            phi_b = mp.ncdf(b)

            def integrand(s):
                a = b - s
                phi_a = mp.ncdf(a)
                phi_c = mp.ncdf(b - R * s)
                # Fewer than j points above c as a finite sum. The differences
                # are taken directly so that none of them cancels against 1.
                above = phi_b - phi_c
                below = phi_c - phi_a
                tail = mp.fsum(
                    binomial * above**k * below ** (m - k)
                    for k, binomial in enumerate(binomials)
                )
                return phi_a ** (i - 1) * tail * mp.exp(-(a * a + b * b) / 2)

            return mp.quad(integrand, inner_panels)

    with mp.workdps(outer_dps):
        value = constant * mp.quad(inner, outer_panels)

    with mp.workdps(dps):
        return +value


def _float_survival(R, size, j, i):
    """Evaluate the survival function in float64 to seed the secant iteration."""
    m = size - i - 1
    log_constant = gammaln(size + 1) - gammaln(i) - gammaln(m + 1) - np.log(2.0 * np.pi)
    log_binomials = [
        gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1) for k in range(j)
    ]

    def integrand(a, b):
        phi_a = ndtr(a)
        phi_c = ndtr(b - R * (b - a))
        above = ndtr(b) - phi_c
        below = phi_c - phi_a
        if above + below <= 0.0:
            return 0.0
        tail = sum(
            np.exp(log_binomial) * above**k * below ** (m - k)
            for k, log_binomial in enumerate(log_binomials)
        )
        return phi_a ** (i - 1) * tail * np.exp(log_constant - (a * a + b * b) / 2)

    # An absolute tolerance would let the inner integrals stop early, whose
    # error is not part of the estimate of the outer integral.
    return dblquad(
        integrand, -np.inf, np.inf, -np.inf, lambda b: b, epsabs=0.0, epsrel=SEED_EPSREL
    )[0]


def _seed(q, size, j, i):
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        return brentq(
            lambda R: _float_survival(R, size, j, i) - (1.0 - q),
            SEED_BRACKET[0],
            SEED_BRACKET[1],
            xtol=SEED_XTOL,
        )


def critical_value(q, size, j, i, dps=DEFAULT_DPS, start=None):
    r"""Solve :math:`G(R)=q` with the secant method.

    Parameters
    ----------
    q : str
        The percentile as a decimal string, so that it is not rounded to
        float64.
    size : int
        The number of observations :math:`n`.
    j, i : int
        The indices of the statistic. See :class:`dixonstat.RangeRatio`.
    dps : int
        The number of significant decimal digits of the result.
    start : float, optional
        An approximation of the root. Computed in float64 if omitted.

    Returns
    -------
    R : mpmath.mpf
        The critical value.
    residual : mpmath.mpf
        :math:`(1-G(R))-(1-q)` evaluated at `R`.

    Raises
    ------
    RuntimeError
        If the iteration does not converge in `MAX_SECANT_STEPS` steps.
    ValueError
        If `dps` is below `MIN_DPS`.
    """
    if dps < MIN_DPS:
        raise ValueError(f'precision of {dps} digits is below the minimum of {MIN_DPS}')

    if start is None:
        start = _seed(float(q), size, j, i)

    with mp.workdps(dps + OUTER_GUARD_DIGITS):
        target = 1 - mp.mpf(q)
        previous = mp.mpf(start)
        current = previous * (1 + mp.mpf(SECANT_RELATIVE_OFFSET))
        f_previous = survival(previous, size, j, i, dps=dps) - target
        f_current = survival(current, size, j, i, dps=dps) - target
        # The error after a secant step is about the product of the last two
        # errors, so a step below the square root of the target accuracy already
        # leaves an iterate that reaches the target.
        tolerance = mp.mpf(10) ** (-dps / 2 - SECANT_GUARD_DIGITS)

        for _ in range(MAX_SECANT_STEPS):
            step = f_current * (current - previous) / (f_current - f_previous)
            previous, f_previous = current, f_current
            current = current - step
            f_current = survival(current, size, j, i, dps=dps) - target
            if abs(step) <= tolerance * abs(current):
                return +current, f_current

    raise RuntimeError(
        f'secant iteration for n={size}, j={j}, i={i}, q={q} did not converge '
        f'in {MAX_SECANT_STEPS} steps, last step {mp.nstr(step, 3)} exceeds '
        f'tolerance {mp.nstr(tolerance * abs(current), 3)}'
    )


def default_cases():
    """Return the (size, j, i, q) tuples of the stored golden table."""
    cases = [(size, 1, 1, '0.95') for size in (5, 10, 15, 20, 25, 30, 40, 50, 60, 70)]
    cases.append((7, 1, 2, '0.9'))
    cases += [(3, 1, 1, q) for q in ('0.5', '0.95', '0.995')]
    cases.append((6, 2, 3, '0.95'))
    for j, i in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)):
        for size in (10, 30, 100):
            for q in ('0.5', '0.95', '0.995'):
                cases.append((size, j, i, q))
    for j, i in ((1, 1), (2, 3)):
        for q in ('0.95', '0.995'):
            cases.append((300, j, i, q))
    # Negating a sample of size 4 maps r20 to 1 - r10, which ties these two
    # entries together exactly.
    cases += [(4, 2, 1, '0.95'), (4, 1, 1, '0.05')]
    # Drop repetitions while keeping the order.
    return list(dict.fromkeys(cases))


def _compute(arguments):
    (size, j, i, q), dps = arguments
    R, residual = critical_value(q, size, j, i, dps=dps)
    return {
        'size': size,
        'j': j,
        'i': i,
        'q': q,
        'critical_value': mp.nstr(R, dps, strip_zeros=False),
        'residual': mp.nstr(residual, 3),
    }


def _key(entry):
    return (entry['size'], entry['j'], entry['i'], entry['q'])


def main(argv=None):
    """Regenerate the golden table."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('output', type=pathlib.Path, help='JSON file to write')
    parser.add_argument('--dps', type=int, default=DEFAULT_DPS)
    parser.add_argument('--jobs', type=int, default=os.cpu_count())
    parser.add_argument(
        '--checkpoint',
        type=pathlib.Path,
        help='JSON lines file holding finished entries so that a run can resume',
    )
    args = parser.parse_args(argv)

    done = {}
    if args.checkpoint and args.checkpoint.exists():
        entries = map(json.loads, args.checkpoint.read_text().splitlines())
        done = {_key(entry): entry for entry in entries}

    cases = default_cases()
    pending = [(case, args.dps) for case in cases if case not in done]

    with multiprocessing.Pool(args.jobs) as pool:
        for entry in pool.imap_unordered(_compute, pending):
            done[_key(entry)] = entry
            print(json.dumps(entry), flush=True)
            if args.checkpoint:
                with args.checkpoint.open('a') as stream:
                    stream.write(json.dumps(entry) + '\n')

    table = {
        'generator': GENERATOR_COMMAND,
        'mpmath': mp.__version__,
        'dps': args.dps,
        'entries': [done[case] for case in cases],
    }
    args.output.write_text(json.dumps(table, indent=1) + '\n')


if __name__ == '__main__':
    main()
