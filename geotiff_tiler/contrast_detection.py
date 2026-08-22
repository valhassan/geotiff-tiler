"""Low-contrast (no-DRA) detection for Maxar 8-bit ortho-pansharp."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import rasterio
from scipy.signal import find_peaks

_EDR_THRESHOLD = 0.35
_REVIEW_LO, _REVIEW_HI = 0.30, 0.50
_PCTL_LO, _PCTL_HI = 1.0, 99.0
_CLIP_PCTL = 1.0
_CLIP_FRAC = 0.005
_BIMODAL_PROM = 0.25
_MIN_VALID = 0.05
_MAX_DN = 255
_MAX_SIDE = 2048


@dataclass
class BandDiagnostics:
    band: int
    edr: float
    iqr_ratio: float
    crushed_frac: float
    saturated_frac: float
    bimodal: bool
    valid_frac: float


def valid_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    if nodata is None:
        return np.isfinite(arr)
    return np.isfinite(arr) & (arr != nodata)


def _is_bimodal(vals: np.ndarray, max_dn: int) -> bool:
    """Two-peak check on a 3-tap-smoothed histogram (min 4-bin gap)."""
    if vals.size < 100:
        return False
    hist, _ = np.histogram(vals, bins=64, range=(0, max_dn))
    if hist.sum() == 0:
        return False
    smoothed = np.convolve(hist.astype(np.float64), [1, 2, 1], mode="same") / 4.0
    peak_idx, _ = find_peaks(smoothed, distance=4)
    if peak_idx.size < 2:
        return False
    h0, h1 = sorted(smoothed[peak_idx], reverse=True)[:2]
    return bool(h1 / h0 >= _BIMODAL_PROM)


def _band_diagnostics(
    band: np.ndarray, band_idx: int, max_dn: int, nodata: float | None
) -> BandDiagnostics:
    mask = valid_mask(band, nodata)
    if not mask.any():
        return BandDiagnostics(band_idx, 0.0, 0.0, 1.0, 1.0, False, 0.0)
    vals = band[mask].astype(np.float64)
    p_lo, p25, p75, p_hi = np.percentile(vals, [_PCTL_LO, 25, 75, _PCTL_HI])
    interior = vals[(vals > _CLIP_PCTL) & (vals < max_dn - _CLIP_PCTL)]
    return BandDiagnostics(
        band=band_idx,
        edr=float((p_hi - p_lo) / max_dn),
        iqr_ratio=float((p75 - p25) / max_dn),
        crushed_frac=float(np.mean(vals <= _CLIP_PCTL)),
        saturated_frac=float(np.mean(vals >= max_dn - _CLIP_PCTL)),
        bimodal=_is_bimodal(interior, max_dn),
        valid_frac=float(mask.mean()),
    )


def detect_low_contrast(
    bands: np.ndarray,
    max_dn: int = _MAX_DN,
    nodata: float | None = 0,
) -> dict:
    """Per-band EDR diagnostics and a production low-contrast gate."""
    if bands.ndim == 2:
        bands = bands[np.newaxis, ...]
    diags = [
        _band_diagnostics(bands[i], i, max_dn, nodata) for i in range(bands.shape[0])
    ]
    edr_min = float(min(b.edr for b in diags))
    thin = any(b.valid_frac < _MIN_VALID for b in diags)
    if thin:
        status = "insufficient_data"
    elif edr_min < _REVIEW_LO:
        status = "draoff"
    elif edr_min <= _REVIEW_HI:
        status = "review"
    else:
        status = "draon_like"
    rec = {
        "edr_mean": float(np.mean([b.edr for b in diags])),
        "edr_min": edr_min,
        "low_contrast": (edr_min < _EDR_THRESHOLD) and not thin,
        "contrast_status": status,
        "clipped": any(
            b.crushed_frac >= _CLIP_FRAC or b.saturated_frac >= _CLIP_FRAC
            for b in diags
        ),
        "bimodal_any": any(b.bimodal for b in diags),
        "insufficient_valid_data": thin,
    }
    for b in diags:
        rec.update(
            {f"band{b.band}_{k}": v for k, v in asdict(b).items() if k != "band"}
        )
    return rec


def _read_sampled(src: rasterio.DatasetReader, max_side: int = _MAX_SIDE) -> np.ndarray:
    h, w = src.height, src.width
    if not max_side or max(h, w) <= max_side:
        return src.read()
    scale = max(h, w) / max_side
    shape = (src.count, max(1, int(round(h / scale))), max(1, int(round(w / scale))))
    return src.read(out_shape=shape)
