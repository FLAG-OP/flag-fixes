"""Minimal opt_einsum stand-in for the m2-bert e2e test only.

The test host may not have opt_einsum installable (offline pip). The m2-bert
remote code only uses ``opt_einsum.contract(equation, *tensors)``, which is
substituted with ``torch.einsum`` (same equation language incl. ellipsis).
The real opt_einsum is used when installed: the test tries it first and only
prepends this directory when that import fails.
"""
import torch

contract = torch.einsum

__all__ = ["contract"]
