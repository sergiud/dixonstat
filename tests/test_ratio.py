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
