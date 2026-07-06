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

#!/usr/bin/env python3
"""Fail if any pytest-benchmark result regressed beyond a threshold."""

import argparse
import json
import sys


def load_means(path):
    with open(path) as f:
        data = json.load(f)

    return {b['fullname']: b['stats']['mean'] for b in data['benchmarks']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', help='path to the baseline benchmark JSON')
    parser.add_argument('current', help='path to the current benchmark JSON')
    parser.add_argument(
        '--threshold',
        type=float,
        default=2.0,
        help='maximum allowed ratio of current mean to baseline mean',
    )
    args = parser.parse_args(argv)

    try:
        baseline = load_means(args.baseline)
    except (OSError, json.JSONDecodeError, KeyError):
        print(f'no usable baseline at {args.baseline!r}, skipping comparison')
        return 0

    current = load_means(args.current)

    regressed = False

    for name, current_mean in current.items():
        baseline_mean = baseline.get(name)

        if baseline_mean is None:
            continue

        ratio = current_mean / baseline_mean
        status = 'REGRESSION' if ratio > args.threshold else 'ok'

        print(
            f'{name}: {current_mean:.6f}s vs baseline {baseline_mean:.6f}s '
            f'({ratio:.2f}x, threshold {args.threshold:.2f}x) [{status}]'
        )

        if ratio > args.threshold:
            regressed = True

    return 1 if regressed else 0


if __name__ == '__main__':
    sys.exit(main())
