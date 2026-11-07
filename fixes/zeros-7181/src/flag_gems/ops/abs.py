# Copyright 2026 FlagOS Contributors
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

import logging

import triton
import triton.language as tl

from flag_gems.ops.absolute import absolute
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)


def abs(A):
    logger.debug("GEMS ABS")
    if A.is_complex():
        # Triton's type_canonicalisation_dict has no complex entry, so the
        # pointwise kernel must not see a complex pointer. |a+bi| is computed
        # by absolute(), which already reduces the real view with sqrt(re^2+im^2).
        return absolute(A)
    return abs_func(A)


def abs_(A):
    logger.debug("GEMS ABS_")
    if A.is_complex():
        # Match native torch, which rejects in-place abs for complex tensors
        # instead of failing inside the Triton launch path.
        raise RuntimeError("In-place abs is not supported for complex tensors.")
    abs_func(A, out0=A)
    return A
