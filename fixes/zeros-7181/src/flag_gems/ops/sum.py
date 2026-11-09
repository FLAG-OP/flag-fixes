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
import math
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Triton's type_canonicalisation_dict has no complex entry, so a complex
# tensor must never reach a kernel above. Complex reductions are therefore
# decomposed into real/imaginary parts (view_as_real layout) and recombined
# with view_as_complex, matching ATen semantics.
_COMPLEX_TO_REAL = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

# ATen spells dtype names in sum.out's error messages differently from
# torch.dtype reprs (e.g. float32 -> "float", int64 -> "long int").
_ATEN_DTYPE_NAMES = {
    torch.float16: "c10::Half",
    torch.bfloat16: "c10::BFloat16",
    torch.float32: "float",
    torch.float64: "double",
    torch.complex64: "c10::complex<float>",
    torch.complex128: "c10::complex<double>",
    torch.int64: "long int",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
}


def _aten_dtype_name(dtype):
    return _ATEN_DTYPE_NAMES.get(dtype, str(dtype).replace("torch.", ""))


def _real_sum(x, dim, keepdim, dtype):
    """Sum a real tensor with ATen dim/keepdim semantics.

    Routes through the public entries so their empty-tensor branches apply
    (sum_dim_comm would divide by a zero-sized dim).
    """
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        res = sum(x, dtype=dtype)
        if keepdim:
            res = res.reshape([1] * x.ndim)
        return res
    return sum_dim(x, dim, keepdim, dtype=dtype)


def _sum_complex(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    """ATen-compatible sum for complex inputs (and complex dtype requests).

    * dtype None / complex -> sum real and imaginary parts separately and
      recombine (cast the input up first when a wider complex dtype is
      requested, mirroring ATen promotion).
    * dtype real -> ATen casts the complex input to the real dtype first
      (dropping the imaginary part), so summing the real parts matches.
      bool is the exception (c10 TypeCast): a complex value converts to
      True when EITHER component is nonzero, so the cast must see the full
      complex tensor, not just the real part.
    """
    # Materialize an unresolved conj bit first: the .real/.imag accessors
    # hand back views whose raw storage is un-negated, which downstream
    # kernels read literally (dispatch works only because ATen materializes
    # conj inputs before entering this path).
    inp = inp.resolve_conj()
    if inp.is_complex():
        if dtype is None or dtype.is_complex:
            target = dtype if dtype is not None else inp.dtype
            if target != inp.dtype:
                inp = inp.to(target)
            part_dtype = _COMPLEX_TO_REAL[target]
            real_part = _real_sum(inp.real, dim, keepdim, part_dtype)
            imag_part = _real_sum(inp.imag, dim, keepdim, part_dtype)
            res = torch.view_as_complex(torch.stack((real_part, imag_part), dim=-1))
            if res.dtype != target:
                res = res.to(target)
        elif dtype is torch.bool:
            # sum([0+1j], dtype=bool) is True: cast the whole complex tensor
            # with the any-component rule, then reduce (cast-first, ATen).
            # Build the mask from real/imag comparisons: float->bool casts
            # truncate in flag_gems while the nonzero rule must apply.
            res = _real_sum(inp.real.ne(0) | inp.imag.ne(0), dim, keepdim, torch.bool)
        else:
            res = _real_sum(inp.real, dim, keepdim, dtype)
    else:
        # real input with a complex dtype requested: sum in the matching real
        # dtype, then reinterpret the result as complex (imag parts are zero).
        part_dtype = _COMPLEX_TO_REAL[dtype]
        res = _real_sum(inp, dim, keepdim, part_dtype).to(dtype)
    if out is not None:
        if out.shape != res.shape:
            out.resize_(res.shape)
        if out.dtype != res.dtype:
            # ATen sum.out raises on an out dtype that differs from the
            # computed result; copy_ alone would silently cast (review P2).
            raise RuntimeError(
                f"Expected out tensor to have dtype {_aten_dtype_name(res.dtype)}, "
                f"but got {_aten_dtype_name(out.dtype)} instead"
            )
        out.copy_(res)
        return out
    return res


@libentry()
@triton.jit
def sum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def sum_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


def _cast_before_sum(inp, dtype):
    """ATen casts the input to dtype before reducing (cast-first).

    The bool cast must use the nonzero rule (0.5 -> True): flag_gems'
    float->bool cast kernel truncates instead (0.5 -> False), so build the
    mask with a comparison.
    """
    if dtype == inp.dtype:
        return inp
    if dtype is torch.bool:
        return inp.ne(0)
    return inp.to(dtype)


def sum(inp, *, dtype=None):
    logger.debug("GEMS SUM")
    if inp.is_complex() or (dtype is not None and dtype.is_complex):
        return _sum_complex(inp, None, False, dtype=dtype)
    inp = inp.contiguous()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64
    if dtype != inp.dtype:
        # ATen casts the input to dtype before summing (full sum and dim sum
        # both); accumulate in the cast dtype instead of widening afterwards.
        inp = _cast_before_sum(inp, dtype)
    M = inp.numel()

    if M == 0:
        out = torch.zeros([], dtype=dtype, device=inp.device)
        return out

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        sum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS SUM_OUT")
    if inp.is_complex() or (dtype is not None and dtype.is_complex):
        return _sum_complex(inp, None, False, dtype=dtype, out=out)
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64
    if dtype != inp.dtype:
        inp = _cast_before_sum(inp, dtype)
    if out.dtype != dtype:
        # ATen sum.out (full reduction) does NOT adopt out.dtype and raises
        # when it differs from the computed dtype.
        raise RuntimeError(
            f"Expected out tensor to have dtype {_aten_dtype_name(dtype)}, "
            f"but got {_aten_dtype_name(out.dtype)} instead"
        )
    M = inp.numel()
    if M == 0:
        # Empty full reduction: ATen yields a zero scalar of dtype without
        # launching the kernel (the kernel path divides by a zero mid size).
        res = torch.zeros([], dtype=dtype, device=inp.device)
        if out.shape != res.shape:
            out.resize_(res.shape)
        out.copy_(res)
        return out
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        sum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def sum_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        input_ptrs = input_ptr + inp_offset
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        out = tl.sum(inp, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)
    else:
        sum = tl.zeros([TILE_N, TILE_K], dtype=cdtype)

        # specialization does not improve performance inn this example, as tested
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum += inp
        out = tl.sum(sum, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def sum_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + inp_offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        out = tl.sum(inp, axis=0)
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)
    else:
        sum = tl.zeros(
            [
                TILE_N,
            ],
            dtype=cdtype,
        )
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum += inp
        out = tl.sum(sum, axis=0)
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def sum_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0).to(cdtype)
        _sum += a
    sum = tl.sum(_sum, axis=1)[:, None]
    tl.store(out, sum, row_mask)


def sum_dim_comm(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64
    if dim is not None and not isinstance(dim, (list, tuple)):
        # public entries accept a scalar dim (torch.sum(x, 0)); normalize
        # for the iterable-only reduction body below.
        dim = [dim]
    if dtype != inp.dtype:
        # ATen casts the input to dtype before reducing, so for instance
        # f32->i64 truncates each element before summing (summing first and
        # truncating the total can differ by 1); bool uses the nonzero rule.
        inp = _cast_before_sum(inp, dtype)
    if dim is None:
        result = torch.sum(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    if dim == [] or dim == ():
        if not keepdim:
            return sum(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum(inp, dtype=dtype), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]

    if len(dim) == 1:
        dim = dim[0]
        N = inp.shape[dim]
        M = reduce(lambda x, y: x * y, shape[:dim], 1)
        inp = inp.contiguous()
        K = inp.numel() // M // N
        shape[dim] = 1
        _out_provided = out is not None
        if _out_provided:
            # Resize out to the expected output shape, matching native PyTorch
            # sum.out behavior. The caller (e.g. logsumexp) may pass a
            # zero-size placeholder that needs to be resized before use.
            if keepdim:
                out.resize_(shape)
            else:
                out.resize_(shape[:dim] + shape[dim + 1 :])
        else:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            if K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                sum_dim_kernel_non_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                )
            else:
                grid = (M, 1, 1)
                sum_dim_kernel_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                )
        if not keepdim and not _out_provided:
            out = out.squeeze(dim=dim)
        return out
    else:
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N
        _out_provided = out is not None
        if _out_provided:
            dim_set = set(dim)
            if keepdim:
                out.resize_(shape)
            else:
                out.resize_([s for i, s in enumerate(shape) if i not in dim_set])
        else:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            sum_dim_kernel[grid](inp, out, M, N)
        if not keepdim and not _out_provided:
            for d in sorted(dim, reverse=True):
                out = out.squeeze(dim=d)
        return out


def _empty_dim_result(inp, dim, keepdim, dtype):
    """Zeros of the reduced shape for an empty input.

    ATen never launches the reduction for a zero-element input; sharing this
    between sum_dim's empty branch and sum_dim_out keeps both from dividing
    by a zero-sized dim (review P2: sum_dim_out used to ZeroDivisionError).
    """
    out_shape = list(inp.shape)
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        if keepdim:
            out_shape = [1] * len(out_shape)
        else:
            out_shape = []
    else:
        dims_to_reduce = dim if isinstance(dim, (list, tuple)) else [dim]
        if keepdim:
            for d in dims_to_reduce:
                out_shape[d % inp.ndim] = 1
        else:
            sorted_dims_to_remove = sorted(
                dims_to_reduce, key=lambda x: x % inp.ndim, reverse=True
            )
            for d in sorted_dims_to_remove:
                index_to_remove = d % inp.ndim
                out_shape.pop(index_to_remove)
    return torch.zeros(out_shape, dtype=dtype, device=inp.device)


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS SUM_DIM")
    if inp.is_complex() or (dtype is not None and dtype.is_complex):
        return _sum_complex(inp, dim, keepdim, dtype=dtype)
    # support dim = 0, which are consistent with PyTorch
    if inp.numel() == 0:
        if dtype is None:
            dtype = inp.dtype
            if dtype is torch.bool:
                # ATen's empty bool reduction without an explicit dtype
                # promotes to int64; an explicit dtype=torch.bool is kept.
                dtype = torch.int64
        return _empty_dim_result(inp, dim, keepdim, dtype)
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype)


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS SUM_DIM_OUT")
    if dtype is None:
        # ATen sum.dim_IntList_out computes in out.dtype when dtype is omitted
        # (real, complex and bool inputs alike, empty and non-empty); the
        # full-reduction sum.out variant does not adopt and requires a match.
        dtype = out.dtype
    if inp.is_complex() or dtype.is_complex or out.is_complex():
        return _sum_complex(inp, dim, keepdim, dtype=dtype, out=out)
    if dtype != out.dtype:
        # ATen: an explicit dtype that differs from out.dtype raises.
        raise RuntimeError(
            f"Expected out tensor to have dtype {_aten_dtype_name(dtype)}, "
            f"but got {_aten_dtype_name(out.dtype)} instead"
        )
    if inp.numel() == 0:
        # Empty reductions yield zeros of the reduced shape in dtype (bool
        # out included); sum_dim_comm would divide by a zero-sized dim.
        res = _empty_dim_result(inp, dim, keepdim, dtype)
        if out.shape != res.shape:
            out.resize_(res.shape)
        out.copy_(res)
        return out
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype, out=out)
