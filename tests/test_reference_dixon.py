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

from scripts import reference_dixon
from scripts.reference_dixon import default_cases
import json
import mpmath as mp
import numpy as np
import pytest

# Digits of the stored precision that the checks of the table leave as a margin
# for rounding in the last places.
RESIDUAL_GUARD_DIGITS = 4

MONTE_CARLO_SAMPLES = 40000
MONTE_CARLO_SEED = 20261103

# The empirical probability is binomial. More than 5 standard deviations do not
# occur by chance across the size of the table.
MONTE_CARLO_SIGMAS = 5

# The float64 seed only has to reach the region of quadratic convergence.
SEED_TOLERANCE = 1e-8


@pytest.fixture
def precision(golden_table):
    dps = golden_table['dps']
    with mp.workdps(dps + RESIDUAL_GUARD_DIGITS):
        yield mp.mpf(10) ** (-dps + RESIDUAL_GUARD_DIGITS)


@pytest.mark.parametrize('q', ['0.5', '0.95', '0.995'])
def test_golden_table_matches_closed_form_for_three_samples(golden_value, precision, q):
    R = golden_value(3, 1, 1, q)
    expected = 1 - 3 / mp.pi * (mp.atan((2 * R - 1) / mp.sqrt(3)) + mp.pi / 6)

    assert abs(expected - (1 - mp.mpf(q))) < precision


def test_golden_table_reflects_r20_to_r10_for_four_samples(golden_value, precision):
    # Negating a sample of size 4 maps r20 to 1 - r10.
    r20 = golden_value(4, 2, 1, '0.95')
    r10 = golden_value(4, 1, 1, '0.05')

    assert abs(r20 + r10 - 1) < precision


def test_golden_table_covers_every_case(golden_table):
    stored = [
        (entry['size'], entry['j'], entry['i'], entry['q'])
        for entry in golden_table['entries']
    ]

    assert stored == default_cases()


def test_golden_table_records_converged_values(golden_table, precision):
    assert not [
        entry
        for entry in golden_table['entries']
        if abs(mp.mpf(entry['residual'])) > precision
    ]


def test_golden_table_matches_the_order_statistics(golden_table):
    rng = np.random.default_rng(MONTE_CARLO_SEED)

    outliers = []
    for entry in golden_table['entries']:
        size, j, i = entry['size'], entry['j'], entry['i']
        q = float(entry['q'])
        x = np.sort(rng.standard_normal((MONTE_CARLO_SAMPLES, size)), axis=1)
        ratio = (x[:, -1] - x[:, -1 - j]) / (x[:, -1] - x[:, i - 1])
        empirical = np.mean(ratio <= float(entry['critical_value']))
        sigma = np.sqrt(q * (1.0 - q) / MONTE_CARLO_SAMPLES)
        if abs(empirical - q) > MONTE_CARLO_SIGMAS * sigma:
            outliers.append((size, j, i, q, empirical))

    assert not outliers


def _closed_form_survival(R, size, j, i, dps):
    assert (size, j, i) == (3, 1, 1)
    with mp.workdps(dps):
        return 1 - 3 / mp.pi * (mp.atan((2 * R - 1) / mp.sqrt(3)) + mp.pi / 6)


def test_survival_reproduces_stored_critical_value(golden_value):
    # A size of 6 with j=2 and i=3 exercises the binomial sum and the power of
    # the lower order statistic, and is the cheapest case that does.
    dps = 4
    q = mp.mpf('0.95')

    with mp.workdps(dps + RESIDUAL_GUARD_DIGITS):
        value = reference_dixon.survival(
            golden_value(6, 2, 3, '0.95'), 6, 2, 3, dps=dps
        )

        assert abs(value - (1 - q)) <= (1 - q) * mp.mpf(10) ** (1 - dps)


def test_secant_iteration_reaches_the_root_from_a_coarse_start(
    golden_value, monkeypatch
):
    monkeypatch.setattr(reference_dixon, 'survival', _closed_form_survival)
    dps = 20

    with mp.workdps(dps):
        R, residual = reference_dixon.critical_value(
            '0.95', 3, 1, 1, dps=dps, start=0.9
        )

        assert abs(R - golden_value(3, 1, 1, '0.95')) < mp.mpf(10) ** (2 - dps)
        assert abs(residual) < mp.mpf(10) ** (2 - dps)


def test_secant_iteration_reports_missing_convergence(monkeypatch):
    monkeypatch.setattr(reference_dixon, 'survival', _closed_form_survival)
    monkeypatch.setattr(reference_dixon, 'MAX_SECANT_STEPS', 1)

    with pytest.raises(RuntimeError, match='did not converge in 1 steps'):
        reference_dixon.critical_value('0.95', 3, 1, 1, dps=20, start=0.9)


def test_critical_value_rejects_precision_below_the_secant_offset(monkeypatch):
    monkeypatch.setattr(reference_dixon, 'survival', _closed_form_survival)
    dps = reference_dixon.MIN_DPS - 1

    with pytest.raises(ValueError, match=f'{dps} digits.*{reference_dixon.MIN_DPS}'):
        reference_dixon.critical_value('0.95', 3, 1, 1, dps=dps, start=0.9)


def test_float_seed_is_close_to_the_stored_critical_value(golden_value):
    seed = reference_dixon._seed(0.95, 6, 2, 3)

    assert abs(seed - float(golden_value(6, 2, 3, '0.95'))) < SEED_TOLERANCE


class _InlinePool:
    """Records the pending work and answers it from a table of finished entries."""

    def __init__(self, entries):
        self.entries = entries
        self.pending = None

    def __call__(self, jobs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap_unordered(self, function, pending):
        self.pending = [case for case, _ in pending]
        return [self.entries[case] for case in self.pending]


def test_generator_resumes_from_a_checkpoint(golden_table, monkeypatch, tmp_path):
    entries = golden_table['entries']
    keys = [(e['size'], e['j'], e['i'], e['q']) for e in entries]
    pool = _InlinePool(dict(zip(keys, entries)))
    monkeypatch.setattr(reference_dixon.multiprocessing, 'Pool', pool)
    # The order of a checkpoint follows the completion of the workers.
    checkpoint = tmp_path / 'checkpoint.jsonl'
    checkpoint.write_text(''.join(json.dumps(e) + '\n' for e in entries[-2::-1]))
    output = tmp_path / 'table.json'

    reference_dixon.main([str(output), '--checkpoint', str(checkpoint)])

    assert pool.pending == keys[-1:]
    assert json.loads(output.read_text()) == {**golden_table, 'mpmath': mp.__version__}
    assert len(checkpoint.read_text().splitlines()) == len(entries)
