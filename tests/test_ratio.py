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

import dixonstat
import numpy as np
import pytest
import scipy.stats

# A small multiple of the deviation from the closed form, measured as 5e-15.
CLOSED_FORM_REL_TOL = 64 * np.finfo(float).eps

# The density is smooth between the critical values, so this order integrates it
# to rounding error. The tolerance is a small multiple of the largest deviation
# from the golden probability, 7e-12 for r22 and 100 samples.
MASS_QUADRATURE_ORDER = 64
MASS_REL_TOL = 2e-11

# Small multiples of the largest deviation from the golden values, 2.7e-13 for
# the critical values and 3.4e-13 for the tail probabilities, both at 100
# samples.
PPF_REL_TOL = 1e-12
SF_REL_TOL = 1e-12


@pytest.mark.parametrize(
    'hgh_order,fgh_order',
    [(48, 124), (68, 62), (68, 124)],
)
def test_pdf_of_three_samples_converges_at_high_orders(hgh_order, fgh_order):
    # The density grows like exp(c t u) along t = u, which amplifies the
    # outermost quadrature weights as the orders increase.
    R = 0.97
    expected = 3.0 / np.pi * (2.0 / np.sqrt(3.0)) / (1.0 + (2.0 * R - 1.0) ** 2 / 3.0)
    s = dixonstat.RangeRatio(3, 1, 1, hgh_order=hgh_order, fgh_order=fgh_order)

    assert s.pdf(R) == pytest.approx(expected, rel=CLOSED_FORM_REL_TOL, abs=0.0)


DEFAULT_ORDER_CASES = [
    (10, 1, 1),
    (10, 2, 3),
    (30, 1, 1),
    (30, 2, 3),
    (100, 1, 1),
    (100, 2, 3),
    (300, 1, 1),
    (300, 2, 3),
]


@pytest.mark.parametrize('size,j,i', DEFAULT_ORDER_CASES)
def test_default_orders_integrate_density_to_golden_probability(
    golden_value, size, j, i
):
    # The probability between two golden critical values is known exactly.
    lower = float(golden_value(size, j, i, '0.95'))
    upper = float(golden_value(size, j, i, '0.995'))
    x, w = np.polynomial.legendre.leggauss(MASS_QUADRATURE_ORDER)
    half_width = 0.5 * (upper - lower)
    nodes = lower + half_width * (x + 1.0)
    s = dixonstat.RangeRatio(size, j, i)

    mass = half_width * (w @ np.squeeze(s.pdf(nodes)))

    assert mass == pytest.approx(0.995 - 0.95, rel=MASS_REL_TOL, abs=0.0)


@pytest.mark.filterwarnings('error')
def test_default_orders_stay_finite_for_very_large_samples():
    s = dixonstat.RangeRatio(1000, 1, 1)

    # The exponential in the density peaks at a ratio of 1. Its overflow must not
    # reach the result, since the weights compensate for it.
    assert np.isfinite(s.pdf(np.array([0.1, 1.0]))).all()


@pytest.mark.parametrize('size,j,i', DEFAULT_ORDER_CASES)
def test_default_orders_give_golden_probability_from_cdf(golden_value, size, j, i):
    lower = float(golden_value(size, j, i, '0.95'))
    upper = float(golden_value(size, j, i, '0.995'))
    s = dixonstat.RangeRatio(size, j, i)

    mass = s.cdf(upper, method='quadrature') - s.cdf(lower, method='quadrature')

    assert mass == pytest.approx(0.995 - 0.95, rel=MASS_REL_TOL, abs=0.0)


@pytest.mark.parametrize(
    'keyword,nodes',
    [('hgh_order', 'xhh'), ('fgh_order', 'xfh'), ('gl_order', 'xgl')],
)
def test_explicit_order_overrides_only_that_default(keyword, nodes):
    default = dixonstat.RangeRatio(30, 1, 1)
    explicit = dixonstat.RangeRatio(30, 1, 1, **{keyword: 7})

    assert getattr(explicit, nodes).size == 7
    for other in {'xhh', 'xfh', 'xgl'} - {nodes}:
        assert getattr(explicit, other).size == getattr(default, other).size


def test_pdf_matches_reference_computation():
    # pdf fuses the quadrature-weighted sum via tensordot instead of
    # materializing a weighted array and summing it. Reimplement the
    # unfused formula independently and check the two agree.
    r = dixonstat.RangeRatio(10, 1, 1)
    R = np.linspace(0.01, 0.99, 25)

    factor = np.reciprocal(np.sqrt(1.0 + np.square(R)))
    f = (1.0 + R) * 2.0 * factor * r.sqrt1_3
    v = r.t[..., np.newaxis, np.newaxis] * factor * r.sqrt2
    c1 = r.Phi(r.x[..., np.newaxis, np.newaxis] - v)
    c3 = r.Phi(r.x[..., np.newaxis, np.newaxis] - R * v)
    g = (
        c1 ** (r.i - 1)
        * (c3 - c1) ** (r.size - r.j - r.i - 1)
        * (r.c2[..., np.newaxis, np.newaxis] - c3) ** (r.j - 1)
    )
    density = (
        r.w[..., np.newaxis, np.newaxis]
        * v
        * np.exp(r.z[..., np.newaxis, np.newaxis] * f)
        * g
    )
    expected = np.sum(density, axis=0) * factor * r.factor * r.sqrt4_3

    np.testing.assert_array_almost_equal_nulp(r.pdf(R), expected, nulp=64)


def test_cdf_matches_reference_computation():
    # cdf builds its quadrature nodes and reduces the weighted
    # probabilities via broadcasting and einsum instead of the
    # atleast_3d/squeeze/transpose sequence it used to. Reimplement the
    # original sequence independently and check the two agree.
    r = dixonstat.RangeRatio(10, 1, 1)
    R = np.linspace(0.01, 0.99, 25)

    shape = np.shape(R)
    half_R = np.atleast_2d(R) * 0.5
    arg = np.atleast_3d(half_R * (np.atleast_2d(r.xgl + 1).T)).T
    prob = r.pdf(arg)
    weighted_prob = np.squeeze(r.wgl * prob, axis=0).T
    expected = np.sum(weighted_prob, axis=0) * half_R
    expected = np.clip(expected, 0.0, 1.0)
    expected = np.reshape(expected, shape)

    np.testing.assert_array_almost_equal_nulp(
        r.cdf(R, method='quadrature'), expected, nulp=64
    )


@pytest.mark.parametrize(
    'ratio',
    [
        dixonstat.r10,
        dixonstat.r11,
        dixonstat.r12,
        dixonstat.r20,
        dixonstat.r21,
        dixonstat.r22,
    ],
)
@pytest.mark.parametrize('size', [5, 10, 20, 30])
def test_quadrature_cdf_is_bounded_by_one(ratio, size):
    try:
        s = ratio(size)
    except ValueError:
        pytest.skip('not enough samples for this combination')

    r = np.linspace(0.0, 1.0, 201)
    assert np.max(s.cdf(r, method='quadrature')) <= 1.0


@pytest.mark.parametrize(
    'ratio',
    [
        dixonstat.r10,
        dixonstat.r11,
        dixonstat.r12,
        dixonstat.r20,
        dixonstat.r21,
        dixonstat.r22,
    ],
)
def test_failures(ratio):
    with pytest.raises(ValueError, match='at least 3 samples'):
        ratio(2)

    with pytest.raises(ValueError, match='cannot be negative'):
        ratio(10).ppf(-1e-16)

    with pytest.raises(ValueError, match='cannot be larger'):
        ratio(15).ppf(1.0)

    with pytest.raises(ValueError, match='cannot be larger'):
        ratio(15).ppf(np.array([0.5, 1.0]))


def test_ppf_matches_golden_value(golden_entry):
    s = dixonstat.RangeRatio(golden_entry['size'], golden_entry['j'], golden_entry['i'])
    expected = float(golden_entry['critical_value'])

    actual = s.ppf(float(golden_entry['q']))

    assert actual == pytest.approx(expected, rel=PPF_REL_TOL, abs=0.0)


def test_sf_matches_golden_value(golden_entry):
    s = dixonstat.RangeRatio(golden_entry['size'], golden_entry['j'], golden_entry['i'])
    expected = 1.0 - float(golden_entry['q'])

    actual = s.sf(float(golden_entry['critical_value']))

    assert actual == pytest.approx(expected, rel=SF_REL_TOL, abs=0.0)


def test_cdf_is_the_complement_of_sf():
    s = dixonstat.r22(30)
    R = np.array([0.2, 0.48])

    np.testing.assert_array_equal(s.cdf(R), 1.0 - s.sf(R))


def test_sf_keeps_the_shape():
    R = np.array([[0.2, 0.3], [0.4, 0.5]])

    assert dixonstat.r10(10).sf(R).shape == R.shape


@pytest.mark.parametrize('method', ['gauss-kronrod', 'quadrature'])
def test_ppf_of_array_keeps_the_shape(method):
    q = np.array([0.5, 0.95])
    s = dixonstat.r10(10)

    actual = s.ppf(q, method=method)

    assert actual.shape == q.shape
    assert actual[0] < actual[1]


def test_quadrature_cdf_uses_the_tensor_rule():
    s = dixonstat.r22(100)

    assert s.cdf(0.34, method='quadrature') != s.cdf(0.34)


def test_quadrature_ppf_uses_the_tensor_rule():
    s = dixonstat.r22(100)

    assert s.ppf(0.995, method='quadrature') != s.ppf(0.995)


def test_ppf_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method 'foo', expected one of"):
        dixonstat.r10(10).ppf(0.5, method='foo')


def test_cdf_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method 'foo', expected one of"):
        dixonstat.r10(10).cdf(0.5, method='foo')


def test_too_few_samples():
    with pytest.raises(ValueError, match='too few samples'):
        dixonstat.r22(5)


def test_construction_scales_with_quadrature_order(benchmark):
    # Building the quadrature grid used to be a pure-Python nested loop,
    # which made construction take the better part of a second for
    # moderately large quadrature orders. A generous budget is used to
    # avoid flakiness while still catching a regression back to the loop.
    benchmark(dixonstat.RangeRatio, 30, 1, 1, hgh_order=150, fgh_order=150)

    assert benchmark.stats.stats.mean < 0.2


@pytest.mark.parametrize(
    'ratio',
    [
        dixonstat.r10,
        dixonstat.r11,
        dixonstat.r12,
        dixonstat.r20,
        dixonstat.r21,
        dixonstat.r22,
    ],
)
def test_large_sample_size_is_finite(ratio):
    # 171! overflows float64, which previously propagated into the
    # normalization factor and turned every result into NaN.
    s = ratio(200)

    assert np.isfinite(s.factor)
    assert np.isfinite(s.cdf(0.5))
    assert np.isfinite(s.ppf(0.95))


@pytest.mark.parametrize(
    'ratio,expected',
    [
        (dixonstat.r10, 50.0 / 90.0),
        (dixonstat.r11, 50.0 / 80.0),
        (dixonstat.r12, 50.0 / 70.0),
        (dixonstat.r20, 60.0 / 90.0),
        (dixonstat.r21, 60.0 / 80.0),
        (dixonstat.r22, 60.0 / 70.0),
    ],
)
def test_ratiotest_gap(ratio, expected):
    # Hand-computed from the sorted sample [10, 20, 30, 40, 50, 100]:
    # the gap ratio must use the (n - j)-th order statistic, not the
    # n-th (which degenerates to 0 for every j=1 statistic).
    samples = np.array([40.0, 10.0, 100.0, 50.0, 20.0, 30.0])
    gap, _ = dixonstat.ratiotest(ratio, samples, 0.05, 'one-sided')

    np.testing.assert_almost_equal(gap, expected, 6)


@pytest.mark.parametrize(
    'ratio',
    [
        dixonstat.r10,
        dixonstat.r11,
        dixonstat.r12,
        dixonstat.r20,
        dixonstat.r21,
        dixonstat.r22,
    ],
)
@pytest.mark.parametrize('alternative', ['one-sided', 'two-sided'])
def test_normal_ratio(ratio, alternative):
    samples = scipy.stats.norm().rvs(size=30, random_state=42)
    r, pvalue = dixonstat.ratiotest(ratio, samples, 0.05, alternative)

    assert r <= pvalue


@pytest.mark.parametrize(
    'ratio',
    [
        dixonstat.r10,
        dixonstat.r11,
        dixonstat.r12,
        dixonstat.r20,
        dixonstat.r21,
        dixonstat.r22,
    ],
)
def test_wrong_alternative(ratio):
    samples = scipy.stats.norm().rvs(size=30, random_state=42)

    with pytest.raises(ValueError, match='unknown parameter value'):
        dixonstat.ratiotest(ratio, samples, 0.05, 'foo')
