"""Shared IMC .txt file loading utilities."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

_NON_CHANNEL_COLS = frozenset({"Start_push", "End_push", "Pushes_duration", "X", "Y", "Z"})


def load_imc_txt(path) -> dict:
    """
    Load an IMC .txt pixel file and return a structured dict.

    Returns:
        {
            'channels': dict mapping channel name -> 2D numpy float32 array (h x w),
            'height': int,
            'width': int,
            'channel_names': list of str,
            'df': the raw DataFrame (for callers that need pixel-level access),
        }
    """
    df = pd.read_csv(path, sep="\t")

    x = df["X"].values.astype(int)
    y = df["Y"].values.astype(int)
    h = int(y.max()) + 1
    w = int(x.max()) + 1

    channel_names = [col for col in df.columns if col not in _NON_CHANNEL_COLS]

    channels: Dict[str, np.ndarray] = {}
    for col in channel_names:
        img = np.zeros((h, w), dtype=np.float32)
        img[y, x] = df[col].values.astype(np.float32)
        channels[col] = img

    return {
        'channels': channels,
        'height': h,
        'width': w,
        'channel_names': channel_names,
        'df': df,
    }


def load_imc_images(txt_path) -> Tuple[Dict[str, np.ndarray], int, int]:
    """
    Compatibility wrapper returning (channels_dict, height, width).

    Matches the signature used by generate_spatial_figures.py.
    """
    result = load_imc_txt(txt_path)
    return result['channels'], result['height'], result['width']
