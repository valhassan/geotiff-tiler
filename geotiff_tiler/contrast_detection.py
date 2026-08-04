"""
Low-contrast (no-DRA) detection for Maxar imagery (WV2/3/4, GeoEye-1, QuickBird-2).

Core detection is pure numpy/scipy on in-memory band arrays. The CLI opens
local GeoTIFFs or FOM paired CSVs (STAC ``image_url`` + ``is_draon`` ground
truth) and writes a diagnostics CSV. No correction logic — detection only.

    # FOM DRAON/DRAOFF paired calibration CSV (preferred)
    python -m geotiff_tiler.contrast_detection \\
        --csv notebooks/.../worldview-4_draon_paired.csv \\
        --sensor WV4 -o wv4_contrast_report.csv

    # local rasters
    python -m geotiff_tiler.contrast_detection path/to/scene.tif -o report.csv

Core idea: measure how much of the sensor's available bit depth a scene's
reflective-surface DNs actually occupy, using percentile-based range instead of
raw min/max (robust to hot/cold outlier pixels), plus clipping and bimodality
checks that a plain "low contrast: yes/no" flag would miss.

Calibrated against 201 DRAOFF / 202 DRAON paired FOM scenes (8-bit ortho-
pansharp, WV2/WV3/WV4/GE01/QB02): worst-band EDR (`edr_min`) separates DRAOFF
from DRAON for ~97-98% of pairs. Production gate is `edr_threshold=0.35`
(97.5% DRAOFF catch / 0.5% DRAON false-flag). Scenes with
`edr_review_lo <= edr_min <= edr_review_hi` (default [0.30, 0.50]) get
`contrast_status="review"` — the hard-DRAOFF / weak-DRAON band where a blind
stretch is wrong or pointless. Prefer stretching only when
`contrast_status == "draoff"` (not merely `low_contrast`). `clipped` is NOT a
useful gate for this product (nearly always True) — diagnostic column only.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import rasterio
from scipy.signal import find_peaks

from geotiff_tiler.utils.io import validate_image

logger = logging.getLogger(__name__)

_RASTER_SUFFIXES = {".tif", ".tiff", ".geotiff"}
_DEFAULT_STAC_BANDS = ("red", "green", "blue", "nir")
_FOM_CARRY_COLS = (
    "image_name",
    "image_url",
    "collection",
    "etendue_name",
    "split",
    "id_production",
)
_COLLECTION_SENSOR = (
    ("worldview-4", "WV4"),
    ("worldview-3", "WV3"),
    ("worldview-2", "WV2"),
    ("geoeye-1", "GE01"),
    ("quickbird-2", "QB02"),
)


@dataclass
class BandDiagnostics:
    band: int
    edr: float                  # effective dynamic range, 0-1 (p_hi - p_lo) / max_dn
    iqr_ratio: float             # IQR / max_dn, secondary confirmation of edr
    crushed_frac: float          # fraction of valid pixels at/near 0
    saturated_frac: float        # fraction of valid pixels at/near max_dn
    bimodal: bool                # scene likely has two distinct brightness modes
    valid_frac: float             # fraction of pixels that were valid (not nodata)


@dataclass
class SceneDiagnostics:
    bands: list[BandDiagnostics]
    edr_mean: float
    edr_min: float                # worst band — drives the low_contrast flag
    low_contrast: bool             # edr_min < edr_threshold (production gate)
    contrast_status: str           # "draoff" | "review" | "draon_like" | "insufficient_data"
    clipped: bool                  # diagnostic only, NOT a gate — see note below
    bimodal_any: bool             # any band flagged bimodal
    insufficient_valid_data: bool  # too few valid pixels for diagnostics to be meaningful
    sensor: str | None = field(default=None)

    # NOTE on `clipped`: calibration against 201 DRAOFF / 202 DRAON paired FOM
    # scenes (8-bit ortho-pansharp) found this is nearly always True regardless
    # of DRAOFF/DRAON status, so it is NOT a useful gate for this product. Kept
    # as a raw diagnostic column only. `edr_min` is the validated trigger.


def _valid_mask(band: np.ndarray, nodata: float | None) -> np.ndarray:
    if nodata is None:
        return np.isfinite(band)
    return np.isfinite(band) & (band != nodata)


def _band_diagnostics(
    band: np.ndarray,
    band_idx: int,
    max_dn: int,
    nodata: float | None,
    pctl_lo: float,
    pctl_hi: float,
    clip_pctl: float,
    bimodal_prominence_ratio: float,
) -> BandDiagnostics:
    mask = _valid_mask(band, nodata)
    valid_frac = float(mask.mean())

    if not mask.any():
        return BandDiagnostics(
            band=band_idx, edr=0.0, iqr_ratio=0.0,
            crushed_frac=1.0, saturated_frac=1.0,
            bimodal=False, valid_frac=0.0,
        )

    vals = band[mask].astype(np.float64)

    p_lo, p25, p75, p_hi = np.percentile(vals, [pctl_lo, 25, 75, pctl_hi])
    edr = float((p_hi - p_lo) / max_dn)
    iqr_ratio = float((p75 - p25) / max_dn)

    crushed_frac = float(np.mean(vals <= clip_pctl))
    saturated_frac = float(np.mean(vals >= max_dn - clip_pctl))

    # exclude clipped pixels before bimodality check — a saturation/crush spike
    # at the histogram edge is not a second mode, and left in, it produces
    # false positives on plain single-mode clipped scenes.
    interior_vals = vals[(vals > clip_pctl) & (vals < max_dn - clip_pctl)]
    bimodal = _is_bimodal(interior_vals, max_dn, bimodal_prominence_ratio)

    return BandDiagnostics(
        band=band_idx,
        edr=edr,
        iqr_ratio=iqr_ratio,
        crushed_frac=crushed_frac,
        saturated_frac=saturated_frac,
        bimodal=bimodal,
        valid_frac=valid_frac,
    )


def _is_bimodal(
    vals: np.ndarray,
    max_dn: int,
    prominence_ratio: float,
    n_bins: int = 64,
    min_separation_bins: int = 4,
) -> bool:
    """
    Two-peak check via scipy.signal.find_peaks on a smoothed histogram.

    A 3-tap moving-average smooth removes sampling-noise bumps that would
    otherwise register as spurious adjacent peaks on the same underlying mode.
    A minimum bin separation between candidate peaks additionally rules out
    two noisy local maxima on the same hill being counted as distinct modes.
    """
    if vals.size < 100:
        return False

    hist, _ = np.histogram(vals, bins=n_bins, range=(0, max_dn))
    if hist.sum() == 0:
        return False

    smoothed = np.convolve(hist.astype(np.float64), [1, 2, 1], mode="same") / 4.0

    peak_idx, _ = find_peaks(smoothed, distance=min_separation_bins)
    if peak_idx.size < 2:
        return False

    peak_heights = sorted(smoothed[peak_idx], reverse=True)
    return bool(peak_heights[1] / peak_heights[0] >= prominence_ratio)


def _contrast_status(
    edr_min: float,
    *,
    edr_threshold: float,
    edr_review_lo: float,
    edr_review_hi: float,
    insufficient_valid_data: bool,
) -> str:
    """
    Map edr_min to an action label.

    - draoff: clear low-contrast stretch candidate (below review band)
    - review: ambiguous band [review_lo, review_hi] — audit before stretch
    - draon_like: above review band; do not stretch
    - insufficient_data: stats not trustworthy

    Stretch pipelines should key off ``contrast_status == "draoff"``, not only
    ``low_contrast`` (review-band scenes can still have low_contrast=True when
    edr_min is between review_lo and edr_threshold).
    """
    if insufficient_valid_data:
        return "insufficient_data"
    if edr_min < edr_review_lo:
        return "draoff"
    if edr_min <= edr_review_hi:
        return "review"
    return "draon_like"


def detect_low_contrast(
    bands: np.ndarray,
    max_dn: int = 255,
    nodata: float | None = 0,
    pctl_lo: float = 1.0,
    pctl_hi: float = 99.0,
    clip_pctl: float = 1.0,
    edr_threshold: float = 0.35,
    edr_review_lo: float = 0.30,
    edr_review_hi: float = 0.50,
    clip_frac_threshold: float = 0.005,
    bimodal_prominence_ratio: float = 0.25,
    min_valid_frac: float = 0.05,
    sensor: str | None = None,
) -> SceneDiagnostics:
    """
    Run detection diagnostics on a multi-band image array.

    Parameters
    ----------
    bands : array, shape (n_bands, H, W)
    max_dn : theoretical max digital number for the product bit depth
        (default 255 for 8-bit FOM ortho-pansharp). Use 2047/4095 if you
        ever point this at native 11/12-bit Maxar products — thresholds
        below are calibrated for 8-bit and would need re-deriving.
    nodata : nodata value to exclude from stats. None to skip masking.
    pctl_lo / pctl_hi : percentiles used for the effective dynamic range (EDR)
        measurement. Defaults (1/99) are robust to outlier hot/cold pixels.
    clip_pctl : DN distance from 0 / max_dn counted as "clipped" (accounts for
        near-saturation, not just exact 0 or exact max_dn).
    edr_threshold : EDR below this on the worst band flags low_contrast
        (production gate). Calibrated empirically against 201 DRAOFF / 202
        DRAON paired FOM scenes across WV2/WV3/WV4/GE01/QB02: 0.35 catches
        97.5% of DRAOFF with a 0.5% false-flag rate on DRAON twins. Use 0.40
        for 98.0% catch / 0.5% false-flag if fewer missed DRAOFF scenes
        matters more than a few extra borderline flags.
    edr_review_lo / edr_review_hi : inclusive review band for
        ``contrast_status="review"`` (default [0.30, 0.50]). Hard-DRAOFF /
        weak-DRAON outliers from calibration live here — audit before stretch.
        Require ``edr_review_lo <= edr_review_hi``.
    clip_frac_threshold : fraction of pixels crushed/saturated above which a
        band is flagged as clipped. NOTE: on 8-bit FOM ortho-pansharp this is
        nearly always True regardless of DRAOFF/DRAON status (calibration
        finding) — it is carried as a diagnostic column only, not a gate.
    bimodal_prominence_ratio : second peak height / first peak height above
        which a band is flagged bimodal. Calibration found DRAON scenes are
        more often bimodal than DRAOFF — useful for choosing *how* to stretch
        later, not *whether* to.
    min_valid_frac : minimum fraction of valid (non-nodata) pixels required,
        per band, for that band's diagnostics to be considered meaningful.
        Below this, the scene is flagged insufficient_valid_data rather than
        trusted for low_contrast/clipped/bimodal conclusions.
    sensor : optional tag carried through to the output, for per-sensor
        threshold lookups downstream. WV2 and WV3 showed wider per-sensor
        outlier spread in calibration than WV4/QB02/GE01 — if per-sensor
        thresholds are later derived, this is the join key.

    Returns
    -------
    SceneDiagnostics
    """
    if edr_review_lo > edr_review_hi:
        raise ValueError(
            f"edr_review_lo ({edr_review_lo}) must be <= "
            f"edr_review_hi ({edr_review_hi})"
        )

    if bands.ndim == 2:
        bands = bands[np.newaxis, ...]

    band_diags = [
        _band_diagnostics(
            bands[i], i, max_dn, nodata, pctl_lo, pctl_hi, clip_pctl,
            bimodal_prominence_ratio,
        )
        for i in range(bands.shape[0])
    ]

    edr_values = [b.edr for b in band_diags]
    edr_mean = float(np.mean(edr_values))
    edr_min = float(np.min(edr_values))

    clipped = any(
        (b.crushed_frac >= clip_frac_threshold) or (b.saturated_frac >= clip_frac_threshold)
        for b in band_diags
    )
    bimodal_any = any(b.bimodal for b in band_diags)
    insufficient_valid_data = any(b.valid_frac < min_valid_frac for b in band_diags)
    low_contrast = (edr_min < edr_threshold) and not insufficient_valid_data
    contrast_status = _contrast_status(
        edr_min,
        edr_threshold=edr_threshold,
        edr_review_lo=edr_review_lo,
        edr_review_hi=edr_review_hi,
        insufficient_valid_data=insufficient_valid_data,
    )

    return SceneDiagnostics(
        bands=band_diags,
        edr_mean=edr_mean,
        edr_min=edr_min,
        low_contrast=low_contrast,
        contrast_status=contrast_status,
        clipped=clipped,
        bimodal_any=bimodal_any,
        insufficient_valid_data=insufficient_valid_data,
        sensor=sensor,
    )


def to_record(diag: SceneDiagnostics, image_id: str) -> dict:
    """Flatten SceneDiagnostics into one dict row, for a verification-report CSV."""
    record = {
        "id": image_id,
        "sensor": diag.sensor,
        "edr_mean": diag.edr_mean,
        "edr_min": diag.edr_min,
        "low_contrast": diag.low_contrast,
        "contrast_status": diag.contrast_status,
        "clipped": diag.clipped,
        "bimodal_any": diag.bimodal_any,
        "insufficient_valid_data": diag.insufficient_valid_data,
    }
    for b in diag.bands:
        record[f"band{b.band}_edr"] = b.edr
        record[f"band{b.band}_iqr_ratio"] = b.iqr_ratio
        record[f"band{b.band}_crushed_frac"] = b.crushed_frac
        record[f"band{b.band}_saturated_frac"] = b.saturated_frac
        record[f"band{b.band}_bimodal"] = b.bimodal
        record[f"band{b.band}_valid_frac"] = b.valid_frac
    return record


def parse_is_draon(value: Any) -> bool | None:
    """Parse FOM ``is_draon`` (Yes/No, true/false, 1/0) → bool. None if unknown."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"yes", "y", "true", "t", "1"}:
        return True
    if s in {"no", "n", "false", "f", "0"}:
        return False
    return None


def pair_base_name(image_name: str | None) -> str | None:
    """Strip trailing ``_DRAON`` so DRAON/DRAOFF twins share one pair id."""
    if not image_name:
        return None
    name = str(image_name)
    return name[:-6] if name.endswith("_DRAON") else name


def infer_sensor(collection: str | None, fallback: str | None = None) -> str | None:
    if not collection:
        return fallback
    key = str(collection).lower()
    for needle, sensor in _COLLECTION_SENSOR:
        if needle in key:
            return sensor
    return fallback


def read_bands_sampled(
    src: rasterio.DatasetReader,
    bands: list[int] | None = None,
    max_side: int = 2048,
) -> np.ndarray:
    """Read (n_bands, H, W), downsampling so max(H, W) <= max_side when needed."""
    h, w = src.height, src.width
    if max_side and max(h, w) > max_side:
        scale = max(h, w) / max_side
        out_h = max(1, int(round(h / scale)))
        out_w = max(1, int(round(w / scale)))
        if bands:
            return src.read(bands, out_shape=(len(bands), out_h, out_w))
        return src.read(out_shape=(src.count, out_h, out_w))
    return src.read(bands) if bands else src.read()


def load_bands(
    source: str | Path,
    bands: list[int] | None = None,
    max_side: int = 2048,
    stac_bands: Sequence[str] = _DEFAULT_STAC_BANDS,
) -> tuple[np.ndarray, float | None]:
    """
    Read bands from a local GeoTIFF, STAC item URL, or VRT string.

    STAC items are resolved via :func:`geotiff_tiler.utils.io.validate_image`.
    Large scenes are downsampled with ``max_side`` (percentile EDR does not need
    native resolution).
    """
    source_str = str(source)
    path = Path(source_str)
    if path.is_file() and path.suffix.lower() in _RASTER_SUFFIXES:
        openable: Any = source_str
    elif path.is_file() or source_str.startswith(("http://", "https://")):
        # local path that may be a STAC JSON, or a remote STAC item URL
        openable = validate_image(source_str, bands_requested=list(stac_bands))
    else:
        # already a VRT XML string / GDAL virtual dataset
        openable = source_str

    with rasterio.open(openable) as src:
        data = read_bands_sampled(src, bands=bands, max_side=max_side)
        nodata = src.nodata
    return data, nodata


def diagnose_source(
    source: str | Path,
    image_id: str | None = None,
    max_dn: int = 255,
    nodata: float | None = None,
    bands: list[int] | None = None,
    max_side: int = 2048,
    stac_bands: Sequence[str] = _DEFAULT_STAC_BANDS,
    sensor: str | None = None,
    **detect_kwargs,
) -> dict:
    """Open ``source``, run detection, return a flat CSV-ready record."""
    arr, file_nodata = load_bands(
        source, bands=bands, max_side=max_side, stac_bands=stac_bands
    )
    use_nodata = file_nodata if nodata is None else nodata
    diag = detect_low_contrast(
        arr,
        max_dn=max_dn,
        nodata=use_nodata,
        sensor=sensor,
        **detect_kwargs,
    )
    if image_id is None:
        image_id = Path(str(source)).stem
    return to_record(diag, image_id=image_id)


def diagnose_path(
    path: str | Path,
    max_dn: int = 255,
    nodata: float | None = None,
    bands: list[int] | None = None,
    sensor: str | None = None,
    max_side: int = 2048,
    **detect_kwargs,
) -> dict:
    """Open a local GeoTIFF path, run detection, return a flat CSV-ready record."""
    path = Path(path)
    return diagnose_source(
        path,
        image_id=path.stem,
        max_dn=max_dn,
        nodata=nodata,
        bands=bands,
        max_side=max_side,
        sensor=sensor,
        **detect_kwargs,
    )


def _fom_meta(row: dict[str, str]) -> dict[str, Any]:
    image_name = row.get("image_name") or ""
    is_draon = parse_is_draon(row.get("is_draon"))
    meta: dict[str, Any] = {
        "pair_id": pair_base_name(image_name),
        "is_draon": is_draon,
    }
    for col in _FOM_CARRY_COLS:
        if col in row and row[col] != "":
            meta[col] = row[col]
    return meta


def diagnose_fom_csv(
    csv_path: str | Path,
    max_dn: int = 255,
    nodata: float | None = None,
    bands: list[int] | None = None,
    max_side: int = 2048,
    stac_bands: Sequence[str] = _DEFAULT_STAC_BANDS,
    sensor: str | None = None,
    **detect_kwargs,
) -> tuple[list[dict], int]:
    """
    Run detection on every row of a FOM ``*_draon_paired.csv``.

    Requires ``image_url`` and ``is_draon``. Carries pair metadata through so
    DRAON vs DRAOFF EDR distributions can be compared for threshold calibration.

    Returns
    -------
    records, n_errors
    """
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"empty CSV: {csv_path}")
        fields = set(reader.fieldnames)
        if "image_url" not in fields:
            raise ValueError(f"{csv_path} missing required column: image_url")
        if "is_draon" not in fields:
            raise ValueError(f"{csv_path} missing required column: is_draon")
        rows = list(reader)

    records: list[dict] = []
    errors = 0
    for i, row in enumerate(rows):
        url = (row.get("image_url") or "").strip()
        image_name = (row.get("image_name") or "").strip() or f"row_{i}"
        if not url:
            logger.warning("row %d (%s): empty image_url, skip", i, image_name)
            continue

        row_sensor = sensor or infer_sensor(row.get("collection"))
        meta = _fom_meta(row)
        try:
            logger.info(
                "diagnosing %s (is_draon=%s, pair_id=%s)",
                image_name,
                meta["is_draon"],
                meta["pair_id"],
            )
            record = diagnose_source(
                url,
                image_id=image_name,
                max_dn=max_dn,
                nodata=nodata,
                bands=bands,
                max_side=max_side,
                stac_bands=stac_bands,
                sensor=row_sensor,
                **detect_kwargs,
            )
            merged = {**meta, **record}
            if "image_name" in merged:
                merged["id"] = merged["image_name"]
            records.append(merged)
        except Exception as exc:
            errors += 1
            logger.exception("failed on %s: %s", image_name, exc)
    return records, errors


def _collect_inputs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            found = sorted(
                q
                for q in p.rglob("*")
                if q.is_file() and q.suffix.lower() in _RASTER_SUFFIXES
            )
            if not found:
                logger.warning("no rasters under %s", p)
            paths.extend(found)
        elif p.is_file():
            paths.append(p)
        else:
            logger.warning("skipping missing path: %s", p)
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def _write_csv(records: list[dict], output: Path) -> None:
    if not records:
        raise SystemExit("no records to write")
    # stable front columns for FOM calibration reports
    preferred = [
        "id",
        "pair_id",
        "is_draon",
        "image_name",
        "collection",
        "etendue_name",
        "sensor",
        "edr_mean",
        "edr_min",
        "low_contrast",
        "contrast_status",
        "clipped",
        "bimodal_any",
        "insufficient_valid_data",
    ]
    fieldnames: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if any(key in row for row in records):
            fieldnames.append(key)
            seen.add(key)
    for row in records:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _log_draon_summary(records: list[dict], edr_threshold: float) -> None:
    """Log EDR separation between is_draon True/False — the calibration signal."""
    if not any("is_draon" in r for r in records):
        return
    groups: dict[str, list[float]] = {"DRAON": [], "DRAOFF": [], "unknown": []}
    for r in records:
        edr = r.get("edr_min")
        if edr is None:
            continue
        flag = r.get("is_draon")
        if flag is True:
            groups["DRAON"].append(float(edr))
        elif flag is False:
            groups["DRAOFF"].append(float(edr))
        else:
            groups["unknown"].append(float(edr))

    for label, vals in groups.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        logger.info(
            "%s n=%d edr_min mean=%.4f median=%.4f min=%.4f max=%.4f",
            label,
            arr.size,
            float(arr.mean()),
            float(np.median(arr)),
            float(arr.min()),
            float(arr.max()),
        )

    draoff = np.asarray(groups["DRAOFF"], dtype=np.float64)
    draon = np.asarray(groups["DRAON"], dtype=np.float64)
    if draoff.size and draon.size:
        catch_rate = float((draoff < edr_threshold).mean())
        false_flag_rate = float((draon < edr_threshold).mean())
        n_review = sum(1 for r in records if r.get("contrast_status") == "review")
        logger.info(
            "gate @ edr_threshold=%.2f: catch DRAOFF=%.1f%% false-flag DRAON=%.1f%% "
            "(review-band scenes this run: %d — audit before trusting labels)",
            edr_threshold,
            catch_rate * 100,
            false_flag_rate * 100,
            n_review,
        )


def _parse_bands(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    bands = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not bands:
        raise argparse.ArgumentTypeError("empty --bands")
    if any(b < 1 for b in bands):
        raise argparse.ArgumentTypeError("band indices are 1-based")
    return bands


def _parse_stac_bands(raw: str | None) -> list[str]:
    if raw is None:
        return list(_DEFAULT_STAC_BANDS)
    bands = [x.strip() for x in raw.split(",") if x.strip()]
    if not bands:
        raise argparse.ArgumentTypeError("empty --stac-bands")
    return bands


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m geotiff_tiler.contrast_detection",
        description=(
            "Detect low-contrast (likely DRAOFF) Maxar scenes and write "
            "per-scene diagnostics CSV for DRAON/DRAOFF calibration. "
            "Prefer --csv with a FOM *_draon_paired.csv (image_url + is_draon)."
        ),
    )
    p.add_argument(
        "inputs",
        nargs="*",
        help="GeoTIFF path(s) and/or directories (recursive *.tif/*.tiff)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "FOM paired CSV with image_url + is_draon (e.g. "
            "worldview-4_draon_paired.csv). Uses STAC URLs."
        ),
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("contrast_report.csv"),
        help="output CSV path (default: contrast_report.csv)",
    )
    p.add_argument(
        "--max-dn",
        type=int,
        default=255,
        help="product max DN (default: 255 for 8-bit FOM ortho-pansharp)",
    )
    p.add_argument(
        "--max-side",
        type=int,
        default=2048,
        help="downsample so longest side <= this before stats (default: 2048)",
    )
    p.add_argument(
        "--nodata",
        type=float,
        default=None,
        help="override file nodata; default uses raster nodata tag",
    )
    p.add_argument(
        "--sensor",
        default=None,
        help="sensor tag (e.g. WV4); else inferred from collection when using --csv",
    )
    p.add_argument(
        "--bands",
        type=_parse_bands,
        default=None,
        help="comma-separated 1-based band indices after stack (default: all)",
    )
    p.add_argument(
        "--stac-bands",
        type=_parse_stac_bands,
        default=None,
        help="STAC common-name bands to stack (default: red,green,blue,nir)",
    )
    p.add_argument(
        "--edr-threshold",
        type=float,
        default=0.35,
        help=(
            "EDR below this on worst band -> low_contrast / production gate "
            "(default: 0.35, catches 97.5%% of DRAOFF at 0.5%% false-flag rate "
            "on paired FOM calibration; use 0.40 for 98.0%%/0.5%%)"
        ),
    )
    p.add_argument(
        "--edr-review-lo",
        type=float,
        default=0.30,
        help="inclusive low end of contrast_status=review band (default: 0.30)",
    )
    p.add_argument(
        "--edr-review-hi",
        type=float,
        default=0.50,
        help="inclusive high end of contrast_status=review band (default: 0.50)",
    )
    p.add_argument(
        "--pctl-lo",
        type=float,
        default=1.0,
        help="low percentile for EDR (default: 1)",
    )
    p.add_argument(
        "--pctl-hi",
        type=float,
        default=99.0,
        help="high percentile for EDR (default: 99)",
    )
    p.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.05,
        help="min valid pixel fraction per band (default: 0.05)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log per-file progress",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.csv is None and not args.inputs:
        build_parser().error("provide --csv and/or raster inputs")

    detect_kwargs = {
        "pctl_lo": args.pctl_lo,
        "pctl_hi": args.pctl_hi,
        "edr_threshold": args.edr_threshold,
        "edr_review_lo": args.edr_review_lo,
        "edr_review_hi": args.edr_review_hi,
        "min_valid_frac": args.min_valid_frac,
    }
    stac_bands = args.stac_bands or list(_DEFAULT_STAC_BANDS)

    records: list[dict] = []
    errors = 0

    if args.csv is not None:
        if not args.csv.is_file():
            logger.error("CSV not found: %s", args.csv)
            return 1
        try:
            csv_records, csv_errors = diagnose_fom_csv(
                args.csv,
                max_dn=args.max_dn,
                nodata=args.nodata,
                bands=args.bands,
                max_side=args.max_side,
                stac_bands=stac_bands,
                sensor=args.sensor,
                **detect_kwargs,
            )
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        records.extend(csv_records)
        errors += csv_errors

    if args.inputs:
        paths = _collect_inputs(args.inputs)
        for path in paths:
            try:
                logger.info("diagnosing %s", path)
                records.append(
                    diagnose_path(
                        path,
                        max_dn=args.max_dn,
                        nodata=args.nodata,
                        bands=args.bands,
                        max_side=args.max_side,
                        sensor=args.sensor,
                        **detect_kwargs,
                    )
                )
            except Exception as exc:
                errors += 1
                logger.exception("failed on %s: %s", path, exc)

    if not records:
        logger.error("all inputs failed")
        return 1

    _write_csv(records, args.output)
    n_low = sum(1 for r in records if r.get("low_contrast"))
    n_review = sum(1 for r in records if r.get("contrast_status") == "review")
    n_draoff = sum(1 for r in records if r.get("contrast_status") == "draoff")
    logger.info(
        "wrote %s (%d scenes, %d low_contrast, %d draoff, %d review, %d errors)",
        args.output,
        len(records),
        n_low,
        n_draoff,
        n_review,
        errors,
    )
    _log_draon_summary(records, edr_threshold=args.edr_threshold)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
