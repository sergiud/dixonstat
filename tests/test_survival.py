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

from dixonstat._survival import _normal_mass
from dixonstat._survival import survival
import mpmath as mp
import numpy as np
import pytest

CLOSED_FORM_DPS = 40

# A small multiple of the deviation from the closed form.
CLOSED_FORM_REL_TOL = 1e-13

# A small multiple of the deviation of the difference of two tail values of the
# normal distribution from its exact value, 7e-15 at eight standard deviations.
NORMAL_MASS_REL_TOL = 64 * np.finfo(float).eps


def three_sample_survival(R):
    # In float64 this cancels against 1 for a ratio close to 1.
    with mp.workdps(CLOSED_FORM_DPS):
        R = mp.mpf(R)
        return float(1 - 3 / mp.pi * (mp.atan((2 * R - 1) / mp.sqrt(3)) + mp.pi / 6))


@pytest.mark.parametrize(
    'R,expected',
    [(-0.5, 1.0), (0.0, 1.0), (1.0, 0.0), (1.5, 0.0)],
)
def test_survival_is_exact_outside_the_open_unit_interval(R, expected):
    assert survival(R, 10, 1, 1) == (expected, 0.0)


def test_survival_propagates_nan():
    value, error = survival(np.nan, 10, 1, 1)

    assert np.isnan(value)
    assert np.isnan(error)


@pytest.mark.parametrize('R', [0.1, 0.5, 0.97, 0.999])
def test_survival_matches_closed_form_for_three_samples(R):
    value, _ = survival(R, 3, 1, 1)

    expected = three_sample_survival(R)

    assert value == pytest.approx(expected, rel=CLOSED_FORM_REL_TOL, abs=0.0)


def test_survival_reports_a_conservative_error_estimate():
    R = 0.97
    expected = three_sample_survival(R)

    value, error = survival(R, 3, 1, 1)

    assert 0.0 < error
    assert abs(value - expected) <= error


def test_survival_honors_the_relative_tolerance():
    _, loose = survival(0.97, 3, 1, 1, rtol=1e-4)
    _, tight = survival(0.97, 3, 1, 1, rtol=1e-11)

    assert tight < loose


def test_survival_warns_at_the_caller_if_the_tolerance_is_not_reached():
    rtol = 1e-11

    with pytest.warns(RuntimeWarning, match='did not reach') as record:
        value, error = survival(0.97, 30, 1, 1, rtol=rtol, max_subdivisions=1)

    assert record[0].filename == __file__
    assert np.isfinite(value)
    assert error > rtol * value


@pytest.mark.parametrize('size,j,i', [(30, 2, 3), (300, 2, 3)])
def test_survival_does_not_warn_for_the_default_tolerance(size, j, i, recwarn):
    survival(0.5, size, j, i)

    assert not recwarn.list


def test_survival_sums_every_binomial_term():
    # With j one above the number of points between the two order statistics the
    # sum covers every outcome. The numerator is then x_n - x_1, so the ratio is 1
    # and exceeds any smaller value with certainty.
    value, _ = survival(0.3, 6, 5, 1)

    assert value == pytest.approx(1.0, rel=CLOSED_FORM_REL_TOL, abs=0.0)


@pytest.mark.parametrize('lower,upper', [(8.0, 8.5), (-8.5, -8.0), (-0.5, 0.5)])
def test_normal_mass_keeps_relative_accuracy(lower, upper):
    with mp.workdps(CLOSED_FORM_DPS):
        expected = float(mp.ncdf(upper) - mp.ncdf(lower))

    actual = _normal_mass(lower, upper)

    assert actual == pytest.approx(expected, rel=NORMAL_MASS_REL_TOL, abs=0.0)
