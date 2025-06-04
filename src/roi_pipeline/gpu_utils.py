# src/roi_pipeline/gpu_utils.py

import importlib
import traceback
import pynvml
import numpy as np
from typing import Dict, Optional, List

_GPU_AVAILABLE = None
_RAPIDS_LIBS = {}
_WARNINGS_PRINTED = {
    'pynvml': False,
    'rapids': False
}

def check_gpu_availability(verbose=True) -> bool:
    """Checks for NVIDIA GPU, CUDA drivers (via pynvml), and RAPIDS libraries.

    Sets global flags _GPU_AVAILABLE and _RAPIDS_LIBS.

    Args:
        verbose (bool): If True, print detection status and warnings.

    Returns:
        bool: True if a GPU is detected and RAPIDS libraries are available, False otherwise.
    """
    global _GPU_AVAILABLE, _RAPIDS_LIBS
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    # Check for pynvml (indicates NVIDIA driver presence)
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            if verbose:
                print(f"INFO: Found {device_count} NVIDIA GPU(s). Checking RAPIDS...")
            gpu_detected_via_nvml = True
        else:
            if verbose and not _WARNINGS_PRINTED['pynvml']:
                print("INFO: NVML initialized, but no NVIDIA GPUs detected.")
                _WARNINGS_PRINTED['pynvml'] = True
            gpu_detected_via_nvml = False
        pynvml.nvmlShutdown()
    except Exception as e:
        if verbose and not _WARNINGS_PRINTED['pynvml']:
            print(f"WARNING: pynvml error: {e}. Assuming no GPU.")
            _WARNINGS_PRINTED['pynvml'] = True
        gpu_detected_via_nvml = False

    if not gpu_detected_via_nvml:
        _GPU_AVAILABLE = False
        return False

    # Check for RAPIDS libraries if GPU likely present
    rapids_present = True
    libs_to_check = ['cudf', 'cugraph', 'cupy']
    missing_libs = []
    for lib_name in libs_to_check:
        try:
            lib = importlib.import_module(lib_name)
            _RAPIDS_LIBS[lib_name] = lib
            if verbose: 
                print(f"INFO: Successfully imported RAPIDS lib '{lib_name}'.") 
        except ImportError:
            missing_libs.append(lib_name)
            rapids_present = False
            if verbose: 
                print(f"WARNING: FAILED to import RAPIDS lib '{lib_name}' (ImportError).") 
        except Exception as e:
            if verbose:
                print(f"WARNING: Error importing RAPIDS lib '{lib_name}': {e}")
            missing_libs.append(f"{lib_name} (import error)")
            rapids_present = False

    if rapids_present:
        if verbose:
            print("INFO: RAPIDS libraries found.")
        _GPU_AVAILABLE = True
    else:
        if verbose and not _WARNINGS_PRINTED['rapids']:
            print(f"WARNING: GPU detected, but required RAPIDS libraries ({', '.join(missing_libs)}) not found. GPU acceleration disabled.")
            _WARNINGS_PRINTED['rapids'] = True
        _GPU_AVAILABLE = False

    return _GPU_AVAILABLE

def get_rapids_lib(lib_name: str) -> Optional[object]:
    """Gets a pre-imported RAPIDS library module.

    Args:
        lib_name (str): Name of the RAPIDS library to get

    Returns:
        Optional[object]: The imported library module or None if not found
    """
    return _RAPIDS_LIBS.get(lib_name)

def is_rapids_available() -> bool:
    """Check if RAPIDS libraries are available and working.

    Returns:
        bool: True if RAPIDS is available and working
    """
    return check_gpu_availability(verbose=False) and bool(_RAPIDS_LIBS)

def get_available_rapids_libs() -> List[str]:
    """Get list of available RAPIDS libraries.

    Returns:
        List[str]: List of available RAPIDS library names
    """
    return list(_RAPIDS_LIBS.keys()) 