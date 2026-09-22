from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    from scipy.signal import fftconvolve  # type: ignore
except Exception as e:  # pragma: no cover
    fftconvolve = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR = e
else:  # pragma: no cover
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class MatchResult:
    max_val: float
    max_loc: Tuple[int, int]  # (x, y) in image coordinates


def load_rgb(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
        return np.asarray(img)
    except Exception:
        return None


def load_gray(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("L")
        return np.asarray(img)
    except Exception:
        return None


def rgb_to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB uint8 image to grayscale uint8.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected RGB image HxWx3, got shape={arr.shape}")
    # ITU-R BT.601
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)


def _integral_image(a: np.ndarray) -> np.ndarray:
    """
    Integral image with a zero-padded top/left border.
    Output shape: (H+1, W+1)
    """
    a2 = np.asarray(a, dtype=np.float64)
    s = np.cumsum(np.cumsum(a2, axis=0), axis=1)
    out = np.zeros((s.shape[0] + 1, s.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = s
    return out


def _window_sum(ii: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Sum over all h x w windows using an integral image (ii is H+1 x W+1).
    Returns shape: (H-h+1, W-w+1)
    """
    if h <= 0 or w <= 0:
        raise ValueError("invalid window size")
    H1, W1 = ii.shape
    H = H1 - 1
    W = W1 - 1
    if h > H or w > W:
        return np.zeros((0, 0), dtype=np.float64)
    y2 = h
    x2 = w
    # Vectorized 2D summed area table query.
    return ii[y2:, x2:] - ii[:-y2, x2:] - ii[y2:, :-x2] + ii[:-y2, :-x2]


def match_template_ccoeff_normed_gray(image_gray: np.ndarray, template_gray: np.ndarray) -> Optional[MatchResult]:
    """
    Rough replacement for `cv2.matchTemplate(..., TM_CCOEFF_NORMED)` (grayscale).
    Returns best match (max score, max_loc) or None.
    """
    if fftconvolve is None:
        raise RuntimeError(f"scipy_missing:{_SCIPY_IMPORT_ERROR!r}")

    I = np.asarray(image_gray, dtype=np.float32)
    T = np.asarray(template_gray, dtype=np.float32)
    if I.ndim != 2 or T.ndim != 2:
        raise ValueError("expected 2D grayscale arrays")
    h, w = int(T.shape[0]), int(T.shape[1])
    if I.shape[0] < h or I.shape[1] < w:
        return None

    Tc = T - float(T.mean())
    denT = float(np.sum(Tc * Tc))
    if not (denT > 1e-6):
        return None

    # Convolution with flipped template == correlation.
    C = fftconvolve(I, np.flipud(np.fliplr(Tc)), mode="valid")

    ii = _integral_image(I)
    ii2 = _integral_image(I * I)
    sumP = _window_sum(ii, h, w)
    sumP2 = _window_sum(ii2, h, w)
    n = float(h * w)
    varP = sumP2 - (sumP * sumP) / n
    varP = np.maximum(varP, 0.0)
    den = np.sqrt(varP * denT)

    # Avoid division by zero.
    res = np.zeros_like(C, dtype=np.float32)
    m = den > 1e-6
    res[m] = (C[m] / den[m]).astype(np.float32)

    flat_idx = int(np.argmax(res))
    y = int(flat_idx // res.shape[1])
    x = int(flat_idx % res.shape[1])
    return MatchResult(max_val=float(res[y, x]), max_loc=(x, y))


def match_template_ccoeff_normed_color(image_rgb: np.ndarray, template_rgb: np.ndarray) -> Optional[MatchResult]:
    """
    Rough replacement for `cv2.matchTemplate(..., TM_CCOEFF_NORMED)` on RGB images.
    Treats the patch/template as one long vector across channels.
    """
    if fftconvolve is None:
        raise RuntimeError(f"scipy_missing:{_SCIPY_IMPORT_ERROR!r}")

    I = np.asarray(image_rgb, dtype=np.float32)
    T = np.asarray(template_rgb, dtype=np.float32)
    if I.ndim != 3 or I.shape[2] != 3 or T.ndim != 3 or T.shape[2] != 3:
        raise ValueError("expected HxWx3 RGB arrays")
    h, w = int(T.shape[0]), int(T.shape[1])
    if I.shape[0] < h or I.shape[1] < w:
        return None

    Tc = T - float(T.mean())
    denT = float(np.sum(Tc * Tc))
    if not (denT > 1e-6):
        return None

    # Numerator: sum over channels of correlation.
    C = None
    for c in range(3):
        cc = fftconvolve(I[:, :, c], np.flipud(np.fliplr(Tc[:, :, c])), mode="valid")
        C = cc if C is None else (C + cc)
    assert C is not None

    sum_img = I.sum(axis=2)
    sum_img2 = (I * I).sum(axis=2)
    ii = _integral_image(sum_img)
    ii2 = _integral_image(sum_img2)
    sumP = _window_sum(ii, h, w)
    sumP2 = _window_sum(ii2, h, w)
    n = float(h * w * 3)
    varP = sumP2 - (sumP * sumP) / n
    varP = np.maximum(varP, 0.0)
    den = np.sqrt(varP * denT)

    res = np.zeros_like(C, dtype=np.float32)
    m = den > 1e-6
    res[m] = (C[m] / den[m]).astype(np.float32)

    flat_idx = int(np.argmax(res))
    y = int(flat_idx // res.shape[1])
    x = int(flat_idx % res.shape[1])
    return MatchResult(max_val=float(res[y, x]), max_loc=(x, y))

