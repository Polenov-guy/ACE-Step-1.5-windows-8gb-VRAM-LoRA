"""ACE-Step package."""

import os
import sys
import torch
from loguru import logger
from torch.utils.data import DataLoader

# 1. Enable Tensor Cores optimization (beneficial for all platforms)
torch.set_float32_matmul_precision('medium')

# --- WINDOWS COMPATIBILITY PATCHES ---
# These patches are strictly applied only on Windows to prevent
# the notorious 10-minute hang (DDP timeout) and kernel freezes.
if sys.platform == 'win32':
    logger.info("Windows system detected: Applying stability and optimization patches...")

    # A. Disable Flash Attention 2 (Causes kernel freezes/hangs on Windows consumer hardware)
    # We fallback to Scaled Dot Product Attention (SDPA) which is stable.
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        os.environ["ATTN_IMPLEMENTATION"] = "sdpa"
        os.environ["ACE_ATTN_IMPLEMENTATION"] = "sdpa"
        logger.debug("Windows Patch: Flash Attention disabled (forced SDPA)")
    except Exception as e:
        logger.warning(f"Windows Patch: Could not configure attention backend: {e}")

    # B. Global DataLoader Patch
    # Windows has issues with multi-process DataLoaders spawning in this environment.
    # This monkey-patch forces num_workers=0 to prevent the infinite hang at startup.
    _original_init = DataLoader.__init__

    def _windows_safe_init(self, *args, **kwargs):
        # Force single-process loading on Windows
        if kwargs.get('num_workers', 0) > 0:
            # logger.debug(f"Windows Patch: Overriding num_workers={kwargs['num_workers']} to 0")
            kwargs['num_workers'] = 0
            kwargs['persistent_workers'] = False
        _original_init(self, *args, **kwargs)

    DataLoader.__init__ = _windows_safe_init
    logger.info("Windows Patch: DataLoader forced to single-process mode (num_workers=0)")

# -------------------------------------

from .handler import AceStepHandler
