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

from scripts.compare_benchmarks import main
import json


def write_benchmark(path, means):
    data = {
        'benchmarks': [
            {'fullname': name, 'stats': {'mean': mean}} for name, mean in means.items()
        ],
    }
    path.write_text(json.dumps(data))


def test_missing_baseline_is_not_a_regression(tmp_path):
    current = tmp_path / 'current.json'
    write_benchmark(current, {'test_foo': 1.0})

    assert main([str(tmp_path / 'missing.json'), str(current)]) == 0


def test_no_regression_when_within_threshold(tmp_path):
    baseline = tmp_path / 'baseline.json'
    current = tmp_path / 'current.json'
    write_benchmark(baseline, {'test_foo': 1.0})
    write_benchmark(current, {'test_foo': 1.5})

    assert main([str(baseline), str(current), '--threshold', '2.0']) == 0


def test_regression_when_over_threshold(tmp_path):
    baseline = tmp_path / 'baseline.json'
    current = tmp_path / 'current.json'
    write_benchmark(baseline, {'test_foo': 1.0})
    write_benchmark(current, {'test_foo': 3.0})

    assert main([str(baseline), str(current), '--threshold', '2.0']) == 1


def test_new_benchmark_without_baseline_entry_is_ignored(tmp_path):
    baseline = tmp_path / 'baseline.json'
    current = tmp_path / 'current.json'
    write_benchmark(baseline, {'test_foo': 1.0})
    write_benchmark(current, {'test_foo': 1.0, 'test_bar': 100.0})

    assert main([str(baseline), str(current), '--threshold', '2.0']) == 0
