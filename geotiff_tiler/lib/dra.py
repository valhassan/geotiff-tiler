"""DRA simulation transform and low-contrast detection."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

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


_ACTIONS = {
    "draoff": "corrected",
    "draon_like": "passthrough_ok",
    "review": "passthrough_review",
    "insufficient_data": "passthrough_insufficient_data",
}
_REASONS = {
    "corrected": "applied isotonic DRA simulation",
    "passthrough_ok": "already DRA-like",
    "passthrough_review": "review-band, not auto-corrected",
    "passthrough_insufficient_data": "insufficient valid data",
}
_SENSOR_ALIASES = (
    ("worldview-4", "worldview-4"),
    ("worldview-3", "worldview-3"),
    ("worldview-2", "worldview-2"),
    ("geoeye-1", "geoeye-1"),
    ("quickbird-2", "quickbird-2"),
    ("wv4", "worldview-4"),
    ("wv3", "worldview-3"),
    ("wv2", "worldview-2"),
    ("ge01", "geoeye-1"),
    ("qb02", "quickbird-2"),
)


@dataclass
class TargetAnchors:
    sensor: str
    band: int
    q1_on: float
    q99_on: float
    n_pairs: int


@dataclass
class SensorCalibration:
    sensor: str
    curve: dict[int, dict]
    anchors: dict[int, TargetAnchors]


@dataclass
class DraDecision:
    scene_id: str | None
    sensor: str
    contrast_status: str
    edr_min: float
    action: str
    reason: str


def resolve_sensor(name: str | None) -> str | None:
    """Map collection / alias (WV2, worldview-2) to a calibration key."""
    if not name:
        return None
    key = str(name).strip().lower()
    for alias, sensor in _SENSOR_ALIASES:
        if key == alias or alias in key:
            return sensor
    return key


def load_calibration(path: str | Path) -> dict[str, SensorCalibration]:
    """Load the published artifact. Index by sensor at ingest time."""
    payload = json.loads(Path(path).read_text())
    out: dict[str, SensorCalibration] = {}
    for sensor, bands in payload.items():
        curve: dict[int, dict] = {}
        anchors: dict[int, TargetAnchors] = {}
        for k, rec in bands.items():
            b = int(k)
            curve[b] = rec
            anchors[b] = TargetAnchors(
                sensor,
                b,
                float(rec["q1_on"]),
                float(rec["q99_on"]),
                int(rec.get("n_pairs", 0)),
            )
        out[sensor] = SensorCalibration(sensor, curve, anchors)
    return out


def apply_dra_transform(
    band_data: np.ndarray,
    nodata: float | None,
    curve_params: dict,
    target_anchors: TargetAnchors,
    max_dn: float = 255,
    pctl_lo: float = 1.0,
    pctl_hi: float = 99.0,
    clip_pctl: float = 1.0,
) -> np.ndarray:
    """Normalize by scene q1/q99, apply shared curve, rescale to fixed DRAON anchors."""
    mask = valid_mask(band_data, nodata)
    out_dtype = np.uint8 if max_dn <= 255 else np.uint16
    fill = nodata if nodata is not None else 0
    out = np.full(band_data.shape, fill, dtype=out_dtype)
    if not mask.any():
        logger.warning("apply_dra_transform: no valid pixels — returning as-is")
        return out

    vals = band_data[mask].astype(np.float64)
    interior = vals[(vals > clip_pctl) & (vals < max_dn - clip_pctl)]
    if interior.size == 0:
        logger.warning("apply_dra_transform: empty interior — skipping")
        out[mask] = np.clip(vals, 0, max_dn).astype(out_dtype)
        return out

    q1_off, q99_off = np.percentile(interior, [pctl_lo, pctl_hi])
    span_off = q99_off - q1_off
    if span_off < 1e-6:
        logger.warning(
            "apply_dra_transform: degenerate span (q1=%.4f q99=%.4f) — skipping",
            q1_off,
            q99_off,
        )
        out[mask] = np.clip(vals, 0, max_dn).astype(out_dtype)
        return out

    x_norm = np.clip((vals - q1_off) / span_off, 0.0, 1.0)
    xt = np.asarray(curve_params["x_thresholds"], dtype=np.float64)
    yt = np.asarray(curve_params["y_thresholds"], dtype=np.float64)
    y_norm = np.interp(np.clip(x_norm, xt[0], xt[-1]), xt, yt)

    y_dn = np.clip(
        y_norm * (target_anchors.q99_on - target_anchors.q1_on) + target_anchors.q1_on,
        0,
        max_dn,
    )
    out[mask] = y_dn.astype(out_dtype)
    return out


def _decision(
    scene_id: str | None,
    sensor: str,
    rec: dict,
    action: str | None = None,
    reason: str | None = None,
) -> DraDecision:
    status = rec["contrast_status"]
    action = action or _ACTIONS[status]
    return DraDecision(
        scene_id=scene_id,
        sensor=sensor,
        contrast_status=status,
        edr_min=float(rec["edr_min"]),
        action=action,
        reason=reason or _REASONS[action],
    )


def _apply_bands(
    bands: np.ndarray,
    nodata: float | None,
    calibration: SensorCalibration,
    scene_id: str | None,
    max_dn: float,
    clip_pctl: float,
) -> np.ndarray:
    out_bands = []
    missing = []
    for b in range(bands.shape[0]):
        if b in calibration.curve and b in calibration.anchors:
            out_bands.append(
                apply_dra_transform(
                    bands[b],
                    nodata,
                    calibration.curve[b],
                    calibration.anchors[b],
                    max_dn=max_dn,
                    clip_pctl=clip_pctl,
                )
            )
        else:
            missing.append(b)
            out_bands.append(bands[b])
    if missing:
        logger.warning(
            "%s: no calibration for bands %s — passed through", scene_id, missing
        )
    return np.stack(out_bands)


def simulate_dra(
    scene: np.ndarray,
    nodata: float | None,
    calibration: SensorCalibration,
    scene_id: str | None = None,
    max_dn: float = 255,
    clip_pctl: float = 1.0,
) -> tuple[np.ndarray, DraDecision]:
    """Production entry: gate on contrast_status, apply all bands or pass through."""
    squeeze = scene.ndim == 2
    bands = scene[np.newaxis] if squeeze else scene
    rec = detect_low_contrast(bands, max_dn=int(max_dn), nodata=nodata)
    action = _ACTIONS[rec["contrast_status"]]
    if action == "corrected":
        out = _apply_bands(
            bands, nodata, calibration, scene_id, max_dn, clip_pctl
        )
    else:
        out = np.array(bands, copy=True)
        if action != "passthrough_ok":
            logger.warning(
                "%s: %s (edr_min=%.3f) — pass through",
                scene_id,
                rec["contrast_status"],
                rec["edr_min"],
            )
    return (out[0] if squeeze else out), _decision(scene_id, calibration.sensor, rec)


def apply_dra_to_file(
    path: str | Path,
    calibration: SensorCalibration,
    scene_id: str | None = None,
    tmp_dir: str | Path | None = None,
    max_dn: float = 255,
    clip_pctl: float = 1.0,
) -> tuple[Path, DraDecision]:
    """Gate on a downsampled read; apply and write only when status is draoff."""
    path = Path(path)
    with rasterio.open(path) as src:
        sampled = _read_sampled(src)
        nodata = src.nodata
        n_bands = src.count
    rec = detect_low_contrast(sampled, max_dn=int(max_dn), nodata=nodata)
    action = _ACTIONS[rec["contrast_status"]]
    have = set(calibration.curve) & set(calibration.anchors)
    needed = set(range(n_bands))
    if action == "corrected" and not needed.issubset(have):
        logger.warning(
            "%s: band count %d does not match calibration bands %s — pass through",
            scene_id,
            n_bands,
            sorted(have),
        )
        return path, _decision(
            scene_id,
            calibration.sensor,
            rec,
            action="passthrough_ok",
            reason="band count does not match calibration",
        )
    if action != "corrected":
        if action != "passthrough_ok":
            logger.warning(
                "%s: %s (edr_min=%.3f) — pass through",
                scene_id,
                rec["contrast_status"],
                rec["edr_min"],
            )
        return path, _decision(scene_id, calibration.sensor, rec)

    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile
        nodata = src.nodata
    out = _apply_bands(data, nodata, calibration, scene_id, max_dn, clip_pctl)
    dest_dir = Path(tmp_dir) if tmp_dir is not None else path.parent
    dest = dest_dir / f"{path.stem}_dra.tif"
    profile.update(dtype=out.dtype, count=out.shape[0])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(out)
    logger.info("%s: wrote %s", scene_id, dest)
    return dest, _decision(scene_id, calibration.sensor, rec)
