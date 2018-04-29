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

from ._ratio import Q
from ._ratio import r10
from ._ratio import r11
from ._ratio import r12
from ._ratio import r20
from ._ratio import r21
from ._ratio import r22
from ._ratio import RangeRatio
from ._ratio import ratiotest

__author__ = 'Sergiu Deitsch'

__all__ = (
    'Q',
    'r10',
    'r11',
    'r12',
    'r20',
    'r21',
    'r22',
    'RangeRatio',
    'ratiotest',
)
