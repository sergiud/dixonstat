# Identification and rejection of outliers using Dixon's r statistics.
#
# Copyright 2024 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

from dixonstat._quadrature import half_hermgauss
import numpy as np
import pytest


@pytest.mark.parametrize('n_iter', [100, 12])
def test_exp_minus_10x(n_iter):
    '''Evaluate the integral

        ∞
        ⌠
        ⎮           2
        ⎮  -10⋅x  -x
        ⎮ ℯ     ⋅ℯ    dx
        ⌡
        0

    numerically using half-range Gauss-Hermite quadrature and compare against
    the symbolic evaluation.
    '''

    x, w = half_hermgauss(50, gamma=0, n_iter=n_iter)
    inter = np.inner(w, np.exp(-10 * x))
    np.testing.assert_almost_equal(inter, 0.0981094307315388, 7)
