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


def test_accuracy():
    np.testing.assert_almost_equal(dixonstat.r10(5).ppf(0.95), 0.642357, 6)
    np.testing.assert_almost_equal(dixonstat.r10(5).cdf(1.0), 1, 6)

    np.testing.assert_almost_equal(dixonstat.r10(10).ppf(0.95), 0.411860, 5)
    np.testing.assert_almost_equal(dixonstat.r10(10).cdf(1.0), 1, 6)

    np.testing.assert_almost_equal(dixonstat.r10(15).ppf(0.95), 0.338539, 6)
    np.testing.assert_almost_equal(dixonstat.r10(15).cdf(1.0), 1, 6)

    np.testing.assert_almost_equal(dixonstat.r10(20).ppf(0.95), 0.300499, 6)
    np.testing.assert_almost_equal(dixonstat.r10(20).cdf(1.0), 1, 6)

    np.testing.assert_almost_equal(dixonstat.r10(25).ppf(0.95), 0.276421, 6)
    np.testing.assert_almost_equal(dixonstat.r10(25).cdf(1.0), 1, 6)

    np.testing.assert_almost_equal(dixonstat.r10(30).ppf(0.95), 0.259449, 6)
    np.testing.assert_almost_equal(dixonstat.r10(30).cdf(1.0), 1, 6)

    np.testing.assert_almost_equal(dixonstat.r10(40).ppf(0.95), 0.236563, 5)
    np.testing.assert_almost_equal(dixonstat.r10(40).cdf(1.0), 1, 5)

    np.testing.assert_almost_equal(dixonstat.r10(50).ppf(0.95), 0.221436, 5)
    np.testing.assert_almost_equal(dixonstat.r10(50).cdf(1.0), 1, 5)

    np.testing.assert_almost_equal(dixonstat.r10(60).ppf(0.95), 0.210451, 4)
    np.testing.assert_almost_equal(dixonstat.r10(60).cdf(1.0), 1, 4)

    np.testing.assert_almost_equal(dixonstat.r10(70).ppf(0.95), 0.201984, 4)
    np.testing.assert_almost_equal(dixonstat.r10(70).cdf(1.0), 1, 4)

    r11 = dixonstat.RangeRatio(7, 1, 2)
    np.testing.assert_approx_equal(r11.ppf(0.9), 0.532944, 6)


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

    np.testing.assert_array_almost_equal_nulp(r.cdf(R), expected, nulp=64)


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
def test_cdf_is_bounded_by_one(ratio, size):
    try:
        s = ratio(size)
    except ValueError:
        pytest.skip('not enough samples for this combination')

    r = np.linspace(0.0, 1.0, 201)
    assert np.max(s.cdf(r)) <= 1.0


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
