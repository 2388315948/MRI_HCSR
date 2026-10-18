"""Prepare DICOM images, 24-class annotations, reference training ROIs, and corrected YOLO test ROIs."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import pydicom
import yaml

from pydicom.dataset import FileMetaDataset
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from pydicom.uid import ExplicitVRBigEndian, ExplicitVRLittleEndian, ImplicitVRLittleEndian


DEFAULT_CONFIG: dict[str, Any] = {


    "data_root": ".",

    "split_folders": {"train": "data_1st_fold/original", "val": "data_2nd_fold/original", "test": "data_test"},

    "csv_filenames": {"train": "mri_comprehensive_annotations.csv", "val": "mri_comprehensive_annotations.csv", "test": "mri_comprehensive_annotations.csv"},


    "filter_csv_to_present_dicom": True,

    "csv_columns": {
        "filename": 'filename',          
        "l_score": 'L_score',              
        "r_score": 'R_score',              

        "coordinate_pattern": r"^(L|R)\s*(1[0-2]|[1-9])(?:\D|$)",
    },

    "dicom": {
        "allow_force_read": True,

        "repair_missing_transfer_syntax": True,

        "multiframe_policy": "error",


        "frame_index": 0,               


        "intensity_mode": "percentile",
        "window_center": 500.0,         
        "window_width": 1000.0,         
        "percentile_low": 0.1,          
        "percentile_high": 99.9,        
        "percentile_ignore_zero": True, 
        "auto_invert_monochrome1": True,
    },

    "image": {
        "target_width": 1024,
        "target_height": 1024,


        "resize_mode": "letterbox",
        "only_upsample": False,

        "jpeg_quality": 95,            
    },

    "annotations": {


        "coordinate_space": "dicom_pixels",


        "class_mode": "24",
        "expected_boxes_per_image": 24, 
        "strict_box_count": True,       
        "out_of_bounds_policy": "error",
        "minimum_polygon_area_pixels": 16.0, 
    },

    "cache": {
        "folder_name": 'data_processing_cache', 
        "roi_point_order_version": "tagger_p1_p2_p3_p4_v2", 
        "resnet_roi_processing_version": "expand3pct_masked_obb_no_rotation_resize_pad_v4", 
        "yolo_cache_layout_version": "images_labels_en_v1", 
        "resnet_image_size": 224,          
        "roi_padding_ratio": 0.03,        
        "reuse_when_unchanged": True,     
        "save_source_image_png": True,    
        "save_ground_truth_jpg": True,    
        "save_each_roi_jpg": True,        
        "save_roi_grid_jpg": True,        
        "roi_grid_columns": 6,            
        "roi_grid_cell_width": 180,       
        "roi_grid_cell_height": 140,      
    },
}


POINT_RE = re.compile(
    r"P([1-4])\s*:\s*\(\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\)", re.I
)

BOX_RE = re.compile(
    r"BOX\s*:\s*\[\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\]", re.I
)


@dataclass
class DicomResult:
    image_u8: np.ndarray
    source_image_u8: np.ndarray
    transform: np.ndarray
    audit: dict[str, Any]

@dataclass
class Annotation:
    group: str
    level: int
    class_id: int
    class_name: str
    points: np.ndarray
    source_points: np.ndarray

@dataclass
class SplitReport:
    split: str                  
    cache_reused: bool          
    total_rows: int             
    successful_images: int      
    failed_images: int          
    successful_boxes: int       
    images_folder: str          
    labels_folder: str          
    ground_truth_folder: str    
    roi_folder: str             
    roi_grid_folder: str        
    audit_csv: str              
    error_report: str           


def make_config(data_root: str | Path | None = None) -> dict[str, Any]:
    """Copy defaults and resolve the data root relative to this module."""

    config = json.loads(json.dumps(DEFAULT_CONFIG, ensure_ascii=False))
    project_folder = Path(__file__).resolve().parent
    configured_root = Path(
        config["data_root"] if data_root is None else data_root
    ).expanduser()
    if not configured_root.is_absolute():
        configured_root = project_folder / configured_root
    config["data_root"] = str(configured_root.resolve())
    return config


def relative_path_text(path: str | Path, project_root: str | Path | None = None) -> str:
    value = Path(path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parent
    )
    try:
        return value.relative_to(root).as_posix()
    except ValueError:

        return value.as_posix()

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:

        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _write_jpg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f'Failed to encode JPG: {path}')
    data.tofile(str(path))


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f'Failed to encode PNG: {path}')
    data.tofile(str(path))


def read_color_jpg(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(Path(path)), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot read image: {path}')
    return image

def _read_csv(path: Path) -> pd.DataFrame:
    last: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            aliases = {
                '\u6587\u4ef6\u540d\u79f0': 'filename',
                '\u6587\u4ef6\u540d': 'filename',
                'L\u8bc4\u5206': 'L_score',
                'R\u8bc4\u5206': 'R_score',
                'l_score': 'L_score',
                'r_score': 'R_score',
            }
            for old_name, new_name in aliases.items():
                if old_name in frame.columns and new_name not in frame.columns:
                    frame = frame.rename(columns={old_name: new_name})
            return frame
        except UnicodeDecodeError as error:
            last = error
    raise RuntimeError(f'Cannot identify CSV encoding: {path}') from last

def _find_csv(folder: Path, configured: str) -> Path:
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = folder / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = sorted(folder.glob("*.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f'{folder} must contain exactly one CSV; found: {candidates}')
    return candidates[0]

def _is_dicom(path: Path, allow_force: bool) -> bool:
    if not path.is_file() or path.suffix.lower() == ".csv":
        return False
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return True

    try:
        pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        return True
    except Exception:
        if not allow_force:
            return False

        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            return hasattr(ds, "Rows") or hasattr(ds, "SOPClassUID")
        except Exception:
            return False

def _dicom_index(folder: Path, config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Path]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    duplicate_names: set[str] = set()
    duplicate_stems: set[str] = set()
    cache_name = str(config["cache"]["folder_name"])

    for path in folder.rglob("*"):

        if cache_name in path.parts or not _is_dicom(path, bool(config["dicom"]["allow_force_read"])):
            continue
        nk, sk = path.name.casefold(), path.stem.casefold()

        if nk in by_name:
            duplicate_names.add(nk)
        else:
            by_name[nk] = path
        if sk in by_stem:
            duplicate_stems.add(sk)
        else:
            by_stem[sk] = path

    for key in duplicate_names:
        by_name.pop(key, None)
    for key in duplicate_stems:
        by_stem.pop(key, None)
    return by_name, by_stem

def _locate(filename: str, by_name: dict[str, Path], by_stem: dict[str, Path]) -> Path | None:
    name = Path(filename).name
    return by_name.get(name.casefold()) or by_stem.get(Path(name).stem.casefold())

def _first_float(value: Any) -> float | None:
    if value is None:
        return None
    try:

        if not isinstance(value, (str, bytes)) and hasattr(value, "__iter__"):
            value = list(value)[0]
        return float(value)
    except (TypeError, ValueError, IndexError):
        return None

def _select_frame(array: np.ndarray, ds: pydicom.Dataset, config: dict[str, Any]) -> tuple[np.ndarray, int, int]:
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    index = int(config["dicom"]["frame_index"])
    policy = str(config["dicom"]["multiframe_policy"])

    if frames > 1:
        if policy == "error":
            raise ValueError(f'DICOM contains {frames} frames; no frame was selected implicitly.')
        if policy != "index" or not 0 <= index < frames:
            raise ValueError(f'Invalid multiframe configuration: policy={policy}, index={index}, frames={frames}')
        array = array[index]
    else:
        index = 0

    if int(getattr(ds, "SamplesPerPixel", 1) or 1) != 1:
        raise ValueError('Only grayscale MRI images are supported.')

    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f'Pixel array is not two-dimensional: {array.shape}')
    return array, frames, index

def _to_uint8(array: np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError('No valid pixels.')
    low = float(finite.min()) if low is None else low
    high = float(finite.max()) if high is None else high

    if high <= low:
        high = low + 1.0

    return np.round((np.clip(array, low, high) - low) / (high - low) * 255).astype(np.uint8)

def _display_pixels(modality: np.ndarray, ds: pydicom.Dataset, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    dc = config["dicom"]
    mode = str(dc["intensity_mode"])

    has_window = hasattr(ds, "VOILUTSequence") or (
        getattr(ds, "WindowCenter", None) is not None
        and getattr(ds, "WindowWidth", None) is not None
    )
    details: dict[str, Any] = {"intensity_mode_requested": mode}


    if mode == "fixed_window":
        center, width = float(dc["window_center"]), float(dc["window_width"])
        if width <= 0:
            raise ValueError('window_width must be positive.')
        low, high = center - width / 2, center + width / 2
        details.update(intensity_mode_used="fixed_window", used_low_value=low, used_high_value=high)
        return _to_uint8(np.clip(modality, low, high), low, high), details


    if mode in {"dicom_window", "dicom_or_percentile"} and has_window:
        try:

            displayed = np.asarray(apply_voi_lut(modality, ds, index=0, prefer_lut=True), dtype=np.float64)
            details.update(
                intensity_mode_used="dicom_window",
                used_window_center=_first_float(getattr(ds, "WindowCenter", None)),
                used_window_width=_first_float(getattr(ds, "WindowWidth", None)),
            )
            return _to_uint8(displayed), details
        except Exception:

            if mode == "dicom_window":
                raise


    if mode == "dicom_window":
        raise ValueError('No usable DICOM window or VOI LUT is available.')

    if mode not in {"percentile", "dicom_or_percentile"}:
        raise ValueError(f'Unknown intensity_mode: {mode}')


    finite = modality[np.isfinite(modality)]
    if bool(dc["percentile_ignore_zero"]):
        nonzero = finite[finite != 0]
        if nonzero.size >= 100:
            finite = nonzero

    low = float(np.percentile(finite, float(dc["percentile_low"])))
    high = float(np.percentile(finite, float(dc["percentile_high"])))

    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + 1.0
    details.update(
        intensity_mode_used="percentile" if mode == "percentile" else "percentile_fallback",
        used_low_value=low,
        used_high_value=high,
    )
    return _to_uint8(np.clip(modality, low, high), low, high), details

def _resize(image: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_h, source_w = image.shape
    ic = config["image"]
    target_w, target_h = int(ic["target_width"]), int(ic["target_height"])

    if bool(ic["only_upsample"]) and (source_w > target_w or source_h > target_h):
        raise ValueError('only_upsample=True, but the source image exceeds the target size.')
    mode = str(ic["resize_mode"])

    if mode == "letterbox":

        scale = min(target_w / source_w, target_h / source_h)
        new_w, new_h = max(1, round(source_w * scale)), max(1, round(source_h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA)

        px, py = (target_w - new_w) // 2, (target_h - new_h) // 2
        output = np.zeros((target_h, target_w), np.uint8)
        output[py:py + new_h, px:px + new_w] = resized

        matrix = np.array([[scale, 0, px], [0, scale, py], [0, 0, 1]], np.float64)
        return output, matrix, dict(resize_mode=mode, scale_x=scale, scale_y=scale, pad_x=px, pad_y=py, target_width=target_w, target_height=target_h)

    if mode == "stretch":

        sx, sy = target_w / source_w, target_h / source_h
        output = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR if sx >= 1 and sy >= 1 else cv2.INTER_AREA)
        matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], np.float64)
        return output, matrix, dict(resize_mode=mode, scale_x=sx, scale_y=sy, pad_x=0, pad_y=0, target_width=target_w, target_height=target_h)

    raise ValueError('resize_mode must be letterbox or stretch.')

def _pixel_data_expected_bytes(ds: pydicom.Dataset) -> int | None:
    try:
        rows = int(ds.Rows)
        columns = int(ds.Columns)
        samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
        frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        bits_allocated = int(ds.BitsAllocated)
        if rows <= 0 or columns <= 0 or samples <= 0 or frames <= 0 or bits_allocated <= 0:
            return None
        total_bits = rows * columns * samples * frames * bits_allocated

        return (total_bits + 7) // 8
    except (AttributeError, TypeError, ValueError):
        return None

def _repair_missing_transfer_syntax(
    ds: pydicom.Dataset,
    enabled: bool,
) -> tuple[str, bool, str]:
    file_meta = getattr(ds, "file_meta", None)
    if file_meta is None:
        file_meta = FileMetaDataset()
        ds.file_meta = file_meta

    existing = getattr(file_meta, "TransferSyntaxUID", None)
    if existing:
        return str(existing), False, "file_meta"

    if not enabled:
        raise RuntimeError(
            'DICOM lacks TransferSyntaxUID; set dicom.repair_missing_transfer_syntax=True.'
        )


    expected_bytes = _pixel_data_expected_bytes(ds)
    pixel_data = getattr(ds, "PixelData", None)
    actual_bytes = len(pixel_data) if isinstance(pixel_data, (bytes, bytearray, memoryview)) else None
    if expected_bytes is not None and actual_bytes is not None:

        if actual_bytes < expected_bytes or actual_bytes > expected_bytes + 2:
            raise RuntimeError(
                f'DICOM lacks TransferSyntaxUID and PixelData does not match uncompressed pixels: actual {actual_bytes} bytes; expected approximately {expected_bytes} bytes. The file may be compressed; its transfer syntax cannot be inferred safely.'
            )


    implicit_vr = getattr(ds, "is_implicit_VR", None)
    little_endian = getattr(ds, "is_little_endian", None)
    original_encoding = getattr(ds, "original_encoding", None)
    if isinstance(original_encoding, tuple) and len(original_encoding) >= 2:
        if implicit_vr is None:
            implicit_vr = original_encoding[0]
        if little_endian is None:
            little_endian = original_encoding[1]


    if little_endian is False:
        if implicit_vr is True:
            raise RuntimeError('Implicit-VR big endian has no corresponding uncompressed DICOM transfer syntax; automatic repair is unavailable.')
        inferred = ExplicitVRBigEndian
        basis = "explicit_vr_big_endian"
    elif implicit_vr is False:
        inferred = ExplicitVRLittleEndian
        basis = "explicit_vr_little_endian"
    else:

        inferred = ImplicitVRLittleEndian
        basis = "implicit_vr_little_endian"

    file_meta.TransferSyntaxUID = inferred
    return str(inferred), True, basis

def read_dicom(path: str | Path, config: dict[str, Any]) -> DicomResult:
    path = Path(path).expanduser().resolve()
    forced = False

    try:
        ds = pydicom.dcmread(str(path), force=False)
    except Exception as error:
        if not bool(config["dicom"]["allow_force_read"]):
            raise RuntimeError(f'Standard DICOM reading failed: {error}') from error
        ds = pydicom.dcmread(str(path), force=True)
        forced = True


    if not any(hasattr(ds, key) for key in ("PixelData", "FloatPixelData", "DoubleFloatPixelData")):
        raise ValueError('DICOM has no pixel data.')


    try:
        transfer_syntax_uid, transfer_syntax_repaired, transfer_syntax_basis = (
            _repair_missing_transfer_syntax(
                ds,
                enabled=bool(config["dicom"].get("repair_missing_transfer_syntax", True)),
            )
        )
    except Exception as error:
        raise RuntimeError(f'DICOM transfer-syntax inspection or repair failed: {error}') from error


    try:
        decoded = ds.pixel_array
    except Exception as error:
        ts = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))
        raise RuntimeError(
            f'Pixel decoding failed, TransferSyntaxUID={ts}; install the appropriate decoder (such as pylibjpeg and its plugins) for compressed transfer syntaxes.'
        ) from error


    raw, frames, frame_index = _select_frame(decoded, ds, config)
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(f'Unsupported {photometric}。')


    modality = np.asarray(apply_modality_lut(raw, ds), dtype=np.float64)

    image, display_info = _display_pixels(modality, ds, config)
    inverted = False

    if bool(config["dicom"]["auto_invert_monochrome1"]) and photometric == "MONOCHROME1":
        image = 255 - image
        inverted = True

    output, transform, resize_info = _resize(image, config)


    ts_obj = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    try:
        compressed = bool(ts_obj.is_compressed)
    except Exception:
        compressed = False

    spacing = getattr(ds, "PixelSpacing", None)
    try:
        spacing_row, spacing_col = float(spacing[0]), float(spacing[1])
    except Exception:
        spacing_row = spacing_col = None


    audit = {
        "dicom_path": str(path),
        "dicom_sha256": _sha256(path),
        "dicom_read_success": True,
        "forced_read": forced,
        "file_size_bytes": path.stat().st_size,
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "transfer_syntax_uid": str(ts_obj or transfer_syntax_uid),
        "transfer_syntax_repaired": transfer_syntax_repaired,
        "transfer_syntax_inference": transfer_syntax_basis,
        "compressed_transfer_syntax": compressed,
        "modality": str(getattr(ds, "Modality", "")),
        "manufacturer": str(getattr(ds, "Manufacturer", "")),
        "source_rows": int(raw.shape[0]),
        "source_columns": int(raw.shape[1]),
        "number_of_frames": frames,
        "selected_frame_index": frame_index,
        "photometric_interpretation": photometric,
        "monochrome1_inverted": inverted,
        "bits_allocated": int(getattr(ds, "BitsAllocated", 0) or 0),
        "bits_stored": int(getattr(ds, "BitsStored", 0) or 0),
        "raw_dtype": str(raw.dtype),
        "raw_min": float(np.nanmin(raw)),
        "raw_max": float(np.nanmax(raw)),
        "raw_mean": float(np.nanmean(raw)),
        "raw_std": float(np.nanstd(raw)),
        "rescale_slope": _first_float(getattr(ds, "RescaleSlope", None)),
        "rescale_intercept": _first_float(getattr(ds, "RescaleIntercept", None)),
        "dicom_window_center": _first_float(getattr(ds, "WindowCenter", None)),
        "dicom_window_width": _first_float(getattr(ds, "WindowWidth", None)),
        "pixel_spacing_row_mm": spacing_row,
        "pixel_spacing_col_mm": spacing_col,
        **display_info,
        **resize_info,
        "output_min": int(output.min()),
        "output_max": int(output.max()),
        "output_mean": float(output.mean()),
        "output_std": float(output.std()),
    }
    return DicomResult(output, image.copy(), transform, audit)

def _class_names(mode: str) -> list[str]:
    if mode != "24":
        raise ValueError("Only the 24 ROI classes L1-L12 and R1-R12 are supported.")
    return [f"L{i}" for i in range(1, 13)] + [f"R{i}" for i in range(1, 13)]

def _class_id(group: str, level: int, mode: str) -> int:
    if mode != "24":
        raise ValueError("Only the 24 ROI classes L1-L12 and R1-R12 are supported.")
    return level - 1 if group == "L" else 12 + level - 1

def _coordinate_columns(columns: Iterable[str], pattern: str) -> dict[tuple[str, int], str]:
    regex = re.compile(pattern, re.I)
    result: dict[tuple[str, int], str] = {}
    for column in columns:
        match = regex.match(str(column).strip())
        if match:
            key = (match.group(1).upper(), int(match.group(2)))
            if key in result:
                raise RuntimeError(f'Duplicate coordinate column: {key}')
            result[key] = str(column)

    expected = {(g, i) for g in ("L", "R") for i in range(1, 13)}
    missing = sorted(expected - set(result))
    if missing:
        raise RuntimeError('Missing coordinate columns: ' + ", ".join(f"{g}{i}" for g, i in missing))
    return result

def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _validate_points_preserve_order(points: np.ndarray, source: str = "OBB") -> np.ndarray:
    """Validate P1-P4 while preserving the annotated corner identities."""
    ordered = np.asarray(points, dtype=np.float64)
    if ordered.shape != (4, 2) or not np.isfinite(ordered).all():
        raise ValueError(f'{source} has invalid four-point coordinates: {ordered}')


    contour = ordered.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        raise ValueError(f'{source} P1-P4 do not form a convex quadrilateral in the specified order.')

    edge_lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    if np.any(edge_lengths <= 1e-6):
        raise ValueError(f'{source} has duplicate or excessively close vertices: {ordered.tolist()}')
    if abs(_signed_area(ordered)) <= 1e-6:
        raise ValueError(f'{source} has nearly zero quadrilateral area.')
    return ordered.copy()


def parse_obb_cell(value: object) -> np.ndarray:
    """Parse P1-P4 or a rotated rectangle without reordering the corners."""
    if pd.isna(value):
        raise ValueError('Empty coordinates.')
    text = str(value)
    matches = POINT_RE.findall(text)

    if len(matches) == 4:
        mapping = {int(i): (float(x), float(y)) for i, x, y in matches}
        if set(mapping) != {1, 2, 3, 4}:
            raise ValueError('Incomplete P1-P4 coordinates.')
        return _validate_points_preserve_order(
            np.array([mapping[i] for i in range(1, 5)], dtype=np.float64),
            source="CSV P1-P4",
        )

    match = BOX_RE.search(text)
    if not match:
        raise ValueError(f'Cannot parse: {text}')
    cx, cy, w, h, angle = map(float, match.groups())
    if w <= 0 or h <= 0:
        raise ValueError('BOX width and height must be positive.')

    theta = math.radians(angle)
    ux = np.array([math.cos(theta), math.sin(theta)]) * w / 2
    uy = np.array([-math.sin(theta), math.cos(theta)]) * h / 2
    c = np.array([cx, cy])

    return _validate_points_preserve_order(
        np.stack([c - ux - uy, c + ux - uy, c + ux + uy, c - ux + uy]),
        source='P1-P4 recovered from CSV BOX',
    )

def _transform_points(points: np.ndarray, matrix: np.ndarray, coordinate_space: str) -> np.ndarray:
    if coordinate_space == "output_pixels":
        return _validate_points_preserve_order(points.copy(), source='output P1-P4 coordinates')
    if coordinate_space != "dicom_pixels":
        raise ValueError('coordinate_space must be dicom_pixels or output_pixels.')

    homogeneous = np.c_[points, np.ones(4)]

    return _validate_points_preserve_order(
        (matrix @ homogeneous.T).T[:, :2],
        source='transformed P1-P4',
    )


def output_points_to_source(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map output-space points back to the original DICOM coordinates."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f'Invalid YOLO four-point coordinates: {points}')
    homogeneous = np.c_[points, np.ones(4)]
    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    source = (inverse @ homogeneous.T).T
    source = source[:, :2] / source[:, 2:3]
    return _validate_points_preserve_order(source, source='inverse-transformed P1-P4')


def _annotation_source_points(
    parsed_points: np.ndarray,
    matrix: np.ndarray,
    coordinate_space: str,
) -> np.ndarray:
    if coordinate_space == "dicom_pixels":
        return _validate_points_preserve_order(
            parsed_points.copy(), source='original DICOM P1-P4 coordinates'
        )
    if coordinate_space == "output_pixels":
        return output_points_to_source(parsed_points, matrix)
    raise ValueError('coordinate_space must be dicom_pixels or output_pixels.')

def _bounds(points: np.ndarray, width: int, height: int, policy: str) -> tuple[np.ndarray, bool]:
    outside = (points[:, 0] < 0).any() or (points[:, 0] > width - 1).any() or (points[:, 1] < 0).any() or (points[:, 1] > height - 1).any()
    if not outside:
        return points, False
    if policy == "error":
        raise ValueError(f'Coordinates out of bounds: {points.tolist()}')
    if policy == "clip":
        clipped = points.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
        return _validate_points_preserve_order(
            clipped, source='P1-P4 after boundary clipping'
        ), True
    raise ValueError('out_of_bounds_policy must be error or clip.')

def _score(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return score if 0 <= score <= 12 else None

def _write_label(path: Path, annotations: Sequence[Annotation], width: int, height: int) -> None:
    lines = []
    for ann in annotations:
        normalized = ann.points.copy()
        normalized[:, 0] /= width
        normalized[:, 1] /= height

        if not ((normalized >= -1e-8).all() and (normalized <= 1 + 1e-8).all()):
            raise ValueError(f'{ann.class_name} has normalized coordinates outside the valid range.')
        lines.append(f"{ann.class_id} " + " ".join(f"{v:.8f}" for v in normalized.reshape(-1)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _draw_gt(image: np.ndarray, annotations: Sequence[Annotation], audit: dict[str, Any]) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.32
    thickness = 1

    for ann in annotations:
        points = np.round(ann.points).astype(np.int32)
        color = (0, 90, 255) if ann.group == "L" else (0, 210, 60)

        cv2.polylines(canvas, [points], True, color, 1, cv2.LINE_AA)

        cx = int(points[:, 0].mean())
        cy = int(points[:, 1].mean())

        text = ann.class_name
        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)

        x1 = max(0, cx - tw // 2 - 3)
        y1 = max(0, cy - th // 2 - 3)
        x2 = min(canvas.shape[1]-1, cx + tw // 2 + 3)
        y2 = min(canvas.shape[0]-1, cy + th // 2 + base + 2)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

        cv2.putText(
            canvas,
            text,
            (cx - tw // 2, cy + th // 2),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0)

    status = f"DICOM READ OK | {audit['source_columns']}x{audit['source_rows']} -> {audit['target_width']}x{audit['target_height']} | {audit['intensity_mode_used']}"
    cv2.putText(canvas, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)

    return canvas

def _roi(
    image: np.ndarray,
    points: np.ndarray,
    padding: float = 0.03,
) -> np.ndarray:
    ordered = _validate_points_preserve_order(
        points, source='reference ROI P1-P4'
    ).astype(np.float32)
    if float(padding) < 0:
        raise ValueError('ROI padding ratio cannot be negative.')
    center = ordered.mean(axis=0)
    expanded = center + (ordered - center) * (1.0 + 2.0 * float(padding))

    width = max(
        float(np.linalg.norm(expanded[1] - expanded[0])),
        float(np.linalg.norm(expanded[2] - expanded[3])),
    )
    height = max(
        float(np.linalg.norm(expanded[2] - expanded[1])),
        float(np.linalg.norm(expanded[3] - expanded[0])),
    )
    output_width = max(1, int(round(width)))
    output_height = max(1, int(round(height)))
    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(expanded, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

def _roi_grid(items: Sequence[tuple[str, np.ndarray]], columns: int, cell_w: int, cell_h: int) -> np.ndarray:

    rows = math.ceil(len(items) / columns)

    canvas = np.zeros((rows * cell_h, columns * cell_w, 3), np.uint8)

    for index, (label, roi) in enumerate(items):

        row, col = divmod(index, columns)

        x0, y0 = col * cell_w, row * cell_h

        usable_h = cell_h - 24


        scale = min(cell_w / max(roi.shape[1], 1), usable_h / max(roi.shape[0], 1))
        rw, rh = max(1, round(roi.shape[1] * scale)), max(1, round(roi.shape[0] * scale))


        resized = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA)

        bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


        ox, oy = x0 + (cell_w - rw) // 2, y0 + 22 + (usable_h - rh) // 2

        canvas[oy:oy + rh, ox:ox + rw] = bgr


        cv2.putText(canvas, label, (x0 + 6, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (80, 80, 80), 1)
    return canvas


def _signature(folder: Path, csv_path: Path, config: dict[str, Any]) -> str:

    digest = hashlib.sha256(json.dumps(
        {k: config[k] for k in ("csv_columns", "dicom", "image", "annotations", "cache")},
        sort_keys=True,
        ensure_ascii=False
    ).encode())


    digest.update(_sha256(csv_path).encode())

    cache_name = str(config["cache"]["folder_name"])

    for path in sorted(p for p in folder.rglob("*")
                       if p.is_file() and p != csv_path and cache_name not in p.parts):
        stat = path.stat()

        digest.update(str(path.relative_to(folder)).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())

    return digest.hexdigest()


def prepare_split(config: dict[str, Any], split: str, force_rebuild: bool = False) -> SplitReport:

    root = Path(config["data_root"]).expanduser().resolve()

    folder = root / str(config["split_folders"][split])

    csv_path = _find_csv(folder, str(config["csv_filenames"].get(split, "")))

    cache = folder / str(config["cache"]["folder_name"])

    signature = _signature(folder, csv_path, config)

    manifest = cache / 'cache_manifest.json'


    if not force_rebuild and bool(config["cache"]["reuse_when_unchanged"]) and manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("signature") == signature:
            report = data["report"]
            report["cache_reused"] = True
            return SplitReport(**report)


    if cache.exists():
        shutil.rmtree(cache)

    # Ultralytics requires paired images/labels directory names.
    images, labels = cache / "images", cache / "labels"
    gt, rois, grids, source_images, audit_dir = (
        cache / 'reference_visualizations',
        cache / 'individual_rois',
        cache / 'roi_grids',
        cache / 'source_resolution_images',
        cache / 'audit',
    )
    for path in (images, labels, gt, rois, grids, source_images, audit_dir):
        path.mkdir(parents=True, exist_ok=True)


    df = _read_csv(csv_path)
    filename_col = str(config["csv_columns"]["filename"])
    if filename_col not in df.columns:
        raise KeyError(f'CSV column not found: {filename_col}')


    columns = _coordinate_columns(df.columns, str(config["csv_columns"]["coordinate_pattern"]))

    by_name, by_stem = _dicom_index(folder, config)
    if bool(config.get("filter_csv_to_present_dicom", False)):
        present_mask = df[filename_col].map(
            lambda value: _locate(str(value).strip(), by_name, by_stem)
            is not None
        )
        df = df.loc[present_mask].copy()
        if df.empty:
            raise RuntimeError(
                f'No reference CSV records match files in the data directory: {folder}；CSV={csv_path}'
            )

    mode = str(config["annotations"]["class_mode"])
    names = _class_names(mode)


    errors, audit_rows, roi_rows = [], [], []
    success_images = success_boxes = 0

    used_stems: set[str] = set()


    for row_index, row in df.iterrows():
        filename = str(row[filename_col]).strip()

        path = _locate(filename, by_name, by_stem)
        if path is None:
            errors.append(f'Fold {row_index + 2}: file not found: {filename}')
            continue

        try:

            dicom = read_dicom(path, config)
            dicom.audit["dicom_path"] = relative_path_text(path, root)
            output_h, output_w = dicom.image_u8.shape
            annotations: list[Annotation] = []
            clipped = 0  


            for group in ("L", "R"):
                for level in range(1, 13):

                    parsed_points = parse_obb_cell(row[columns[(group, level)]])
                    source_points = _annotation_source_points(
                        parsed_points,
                        dicom.transform,
                        str(config["annotations"]["coordinate_space"]),
                    )

                    points = _transform_points(
                        parsed_points,
                        dicom.transform,
                        str(config["annotations"]["coordinate_space"]),
                    )

                    points, was_clipped = _bounds(points, output_w, output_h, str(config["annotations"]["out_of_bounds_policy"]))
                    clipped += int(was_clipped)
                    source_points, _ = _bounds(
                        source_points,
                        dicom.source_image_u8.shape[1],
                        dicom.source_image_u8.shape[0],
                        str(config["annotations"]["out_of_bounds_policy"]),
                    )


                    if abs(_signed_area(points)) < float(config["annotations"]["minimum_polygon_area_pixels"]):
                        raise ValueError(f'{group}{level} has insufficient area.')


                    cid = _class_id(group, level, mode)
                    annotations.append(
                        Annotation(
                            group,
                            level,
                            cid,
                            names[cid],
                            points,
                            source_points,
                        )
                    )


            expected = int(config["annotations"]["expected_boxes_per_image"])
            if len(annotations) != expected and bool(config["annotations"]["strict_box_count"]):
                raise ValueError(f'Expected {expected} boxes; found {len(annotations)} boxes.')


            stem = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", Path(filename).stem) or "unnamed"

            if stem.casefold() in used_stems:
                stem += "_" + hashlib.sha1(str(path).encode()).hexdigest()[:8]
            used_stems.add(stem.casefold())


            image_path, label_path = images / f"{stem}.jpg", labels / f"{stem}.txt"
            _write_jpg(image_path, dicom.image_u8, int(config["image"]["jpeg_quality"]))


            source_image_path_text = ""
            if bool(config["cache"].get("save_source_image_png", True)):
                source_image_path = source_images / f"{stem}.png"
                _write_png(source_image_path, dicom.source_image_u8)
                source_image_path_text = relative_path_text(source_image_path, root)


            _write_label(label_path, annotations, output_w, output_h)


            if bool(config["cache"]["save_ground_truth_jpg"]):
                _write_jpg(gt / f'{stem}_reference.jpg', _draw_gt(dicom.image_u8, annotations, dicom.audit), int(config["image"]["jpeg_quality"]))


            l_score = _score(row.get(str(config["csv_columns"]["l_score"])))
            r_score = _score(row.get(str(config["csv_columns"]["r_score"])))
            roi_items = []


            # Classification inputs are recropped from the lossless source PNG, not these previews.
            for ann in annotations:
                patch = None
                need_preview_roi = bool(config["cache"]["save_each_roi_jpg"]) or bool(
                    config["cache"]["save_roi_grid_jpg"]
                )
                if need_preview_roi:
                    patch = _roi(
                        dicom.source_image_u8,
                        ann.source_points,
                        float(config["cache"].get("roi_padding_ratio", 0.03)),
                    )
                if bool(config["cache"]["save_roi_grid_jpg"]) and patch is not None:
                    roi_items.append((ann.class_name, patch))

                roi_path = ""

                if bool(config["cache"]["save_each_roi_jpg"]) and patch is not None:
                    target = rois / ann.class_name / f"{stem}__{ann.class_name}.png"
                    _write_png(target, patch)
                    roi_path = relative_path_text(target, root)


                score = l_score if ann.group == "L" else r_score
                flat = ann.points.reshape(-1)
                roi_rows.append({
                    "split": split,
                    "filename": filename,
                    "output_jpg": relative_path_text(image_path, root),
                    "group": ann.group,
                    "level": ann.level,
                    "identity": f"{ann.group}{ann.level}",
                    "class_id": ann.class_id,
                    "class_name": ann.class_name,
                    "group_score": score,
                    "visible": int(ann.level <= score) if score is not None else None,
                    "roi_jpg": roi_path,
                    "source_image_png": source_image_path_text,
                    **{f"p{i//2+1}_{'x' if i%2==0 else 'y'}": float(v) for i, v in enumerate(flat)},
                    **{
                        f"source_p{i//2+1}_{'x' if i%2==0 else 'y'}": float(v)
                        for i, v in enumerate(ann.source_points.reshape(-1))
                    },
                })


            if bool(config["cache"]["save_roi_grid_jpg"]):
                grid = _roi_grid(roi_items, int(config["cache"]["roi_grid_columns"]),
                                 int(config["cache"]["roi_grid_cell_width"]),
                                 int(config["cache"]["roi_grid_cell_height"]))
                _write_jpg(grids / f'{stem}_roi_grid.jpg', grid, int(config["image"]["jpeg_quality"]))


            audit_rows.append({
                "csv_row": row_index + 2,
                "csv_filename": filename,
                "output_jpg": relative_path_text(image_path, root),
                "label_txt": relative_path_text(label_path, root),
                "l_score": l_score,
                "r_score": r_score,
                "box_count": len(annotations),
                "clipped_box_count": clipped,
                **dicom.audit
            })

            success_images += 1
            success_boxes += len(annotations)

        except Exception as error:

            errors.append(f'Fold {row_index + 2} row {filename}：{error}')


    audit_csv, error_path = audit_dir / 'dicom_audit.csv', audit_dir / 'error_log.txt'
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(roi_rows).to_csv(audit_dir / 'roi_audit.csv', index=False, encoding="utf-8-sig")
    error_path.write_text("\n".join(errors), encoding="utf-8")


    report = SplitReport(
        split, False, len(df), success_images, len(df) - success_images, success_boxes,
        relative_path_text(images, root),
        relative_path_text(labels, root),
        relative_path_text(gt, root),
        relative_path_text(rois, root),
        relative_path_text(grids, root),
        relative_path_text(audit_csv, root),
        relative_path_text(error_path, root),
    )


    manifest.write_text(
        json.dumps({"signature": signature, "report": asdict(report)}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return report


def reports_to_text(reports: Sequence[SplitReport]) -> str:
    lines = ['DICOM processing report', "=" * 72]
    for r in reports:
        lines += [
            f"[{r.split}] {('reused cache' if r.cache_reused else 'read DICOM and generated cache')}",
            f'  Total CSV rows={r.total_rows}, successful images={r.successful_images}, failed images={r.failed_images}, valid OBBs={r.successful_boxes}',
            f'  Training image directory: {r.images_folder}',
            f'  Reference visualization directory: {r.ground_truth_folder}',
            f'  Individual ROI directory: {r.roi_folder}',
            f'  ROI grid preview directory: {r.roi_grid_folder}',
            f'  Detailed processing audit: {r.audit_csv}',
            f'  Error log: {r.error_report}',
        ]
    return "\n".join(lines)


def extract_obb_roi_no_rotation_resize_with_geometry(
    image: np.ndarray,
    points: np.ndarray,
    size: int = 224,
    padding_ratio: float = 0.03,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Expand and mask the OBB, preserve its orientation, and resize once."""
    if image is None or image.size == 0:
        raise ValueError('Cannot crop an ROI from an empty source image.')
    if int(size) <= 1:
        raise ValueError('size must exceed 1.')
    if float(padding_ratio) < 0:
        raise ValueError('padding_ratio cannot be negative.')
    if interpolation != cv2.INTER_LINEAR:
        raise ValueError("Only bilinear ROI resizing is supported.")

    ordered = _validate_points_preserve_order(
        np.asarray(points, dtype=np.float64), source="ResNet ROI P1-P4"
    ).astype(np.float32)
    center = ordered.mean(axis=0)
    expanded = center + (ordered - center) * (1.0 + 2.0 * float(padding_ratio))

    image_height, image_width = image.shape[:2]
    x1 = max(0, int(np.floor(float(expanded[:, 0].min()))))
    y1 = max(0, int(np.floor(float(expanded[:, 1].min()))))
    x2 = min(int(image_width), int(np.ceil(float(expanded[:, 0].max()))) + 1)
    y2 = min(int(image_height), int(np.ceil(float(expanded[:, 1].max()))) + 1)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f'Invalid expanded ROI canvas: x=[{x1},{x2}), y=[{y1},{y2}), image={image_width}x{image_height}'
        )

    crop = image[y1:y2, x1:x2].copy()
    local_points = expanded - np.array([float(x1), float(y1)], dtype=np.float32)
    polygon = np.rint(local_points).astype(np.int32)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_8)
    valid_mask = mask > 0
    if not np.any(valid_mask):
        raise ValueError('Expanded OBB contains no valid pixels in the cropped canvas.')


    eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edge_mask = valid_mask & (eroded == 0)
    background_samples = crop[edge_mask] if np.any(edge_mask) else crop[valid_mask]
    background_value = int(np.clip(round(float(np.median(background_samples))), 0, 255))
    crop[~valid_mask] = background_value

    source_height, source_width = crop.shape[:2]
    scale = float(size) / float(max(source_width, source_height))
    output_width = max(1, min(int(size), int(round(source_width * scale))))
    output_height = max(1, min(int(size), int(round(source_height * scale))))

    if output_width == source_width and output_height == source_height:
        roi = crop
    else:
        roi = cv2.resize(
            crop,
            (int(output_width), int(output_height)),
            interpolation=int(interpolation),
        )

    geometry = {
        "source_points": ordered,
        "expanded_points": expanded,
        "bbox_x1": int(x1),
        "bbox_y1": int(y1),
        "bbox_x2": int(x2),
        "bbox_y2": int(y2),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "output_width": int(output_width),
        "output_height": int(output_height),
        "scale": float(scale),
        "target_size": int(size),
        "interpolation": int(interpolation),
        "roi_expand_ratio": float(padding_ratio),
        "mask_outside_obb": True,
        "outside_fill_value": int(background_value),
        "rotation_applied": False,
        "perspective_warp_applied": False,
    }
    return roi, geometry

def pad_roi_to_square(
    image: np.ndarray,
    size: int = 224,
    background_value: int | tuple[int, int, int] | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    """Pad an ROI without additional interpolation."""
    if image is None or image.size == 0:
        raise ValueError('Cannot pad an empty ROI.')
    if int(size) <= 0:
        raise ValueError('size must be positive.')
    height, width = image.shape[:2]
    if height > int(size) or width > int(size):
        raise ValueError(
            f'ROI size {width}x{height} exceeds padding canvas {size}x{size}; inconsistent joint-resampling size calculation.'
        )
    if background_value is None:
        background_value = _estimate_background_value(image)

    if image.ndim == 2:
        canvas = np.full((int(size), int(size)), int(background_value), dtype=image.dtype)
    else:
        canvas = np.empty((int(size), int(size), image.shape[2]), dtype=image.dtype)
        canvas[...] = np.asarray(background_value, dtype=image.dtype)

    pad_x = (int(size) - width) // 2
    pad_y = (int(size) - height) // 2
    canvas[pad_y : pad_y + height, pad_x : pad_x + width] = image
    info = {
        "source_width": int(width),
        "source_height": int(height),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
        "size": int(size),
    }
    return (canvas, info) if return_info else canvas


def _estimate_background_value(image: np.ndarray) -> int | tuple[int, int, int]:
    if image.ndim == 2:
        border = np.concatenate(
            [image[0, :], image[-1, :], image[:, 0], image[:, -1]]
        )
        return int(np.median(border))

    border = np.concatenate(
        [image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]],
        axis=0,
    )
    median = np.median(border, axis=0).astype(np.uint8)
    return tuple(int(value) for value in median)


FIVE_FOLD_FOLDERS = (
    "data_1st_fold",
    "data_2nd_fold",
    "data_3rd_fold",
    "data_4th_fold",
    "data_5th_fold",
)
REFERENCE_STANDARD_CSV = "mri_comprehensive_annotations.csv"


def _fold_source_config(
    base_config: dict[str, Any],
    split_name: str,
    relative_folder: str | Path,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config, ensure_ascii=False))
    source = Path(config["data_root"]) / Path(relative_folder)
    own_csv = source / REFERENCE_STANDARD_CSV
    parent_csv = source.parent / REFERENCE_STANDARD_CSV
    if own_csv.is_file():
        csv_name = REFERENCE_STANDARD_CSV
    elif parent_csv.is_file():
        csv_name = f"../{REFERENCE_STANDARD_CSV}"
    else:
        raise FileNotFoundError(
            f'Reference standard not found: {own_csv}; also not found: {parent_csv}'
        )
    config["split_folders"] = {split_name: Path(relative_folder).as_posix()}
    config["csv_filenames"] = {split_name: csv_name}
    config["filter_csv_to_present_dicom"] = True
    return config


def five_fold_sources(validation_fold: int) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    if int(validation_fold) not in range(1, 6):
        raise ValueError('Validation fold must be between 1 and 5.')
    training: list[tuple[str, Path]] = []
    for fold_number, folder in enumerate(FIVE_FOLD_FOLDERS, start=1):
        if fold_number == int(validation_fold):
            continue
        training.extend(
            [
                (f"train_fold{fold_number}_original", Path(folder) / "original"),
                (f"train_fold{fold_number}_augmented", Path(folder) / "augmented"),
            ]
        )
    validation = [
        (
            f"val_fold{int(validation_fold)}_original",
            Path(FIVE_FOLD_FOLDERS[int(validation_fold) - 1]) / "original",
        )
    ]
    return training, validation


def independent_test_sources() -> list[tuple[str, Path]]:
    return [("test_original", Path("data_test"))]


def prepare_source_group(
    config: dict[str, Any],
    sources: Sequence[tuple[str, Path]],
    force_rebuild: bool = False,
) -> tuple[list[SplitReport], pd.DataFrame]:
    reports: list[SplitReport] = []
    frames: list[pd.DataFrame] = []
    project_root = Path(config["data_root"]).resolve()
    for source_name, relative_folder in sources:
        source_config = _fold_source_config(config, source_name, relative_folder)
        report = prepare_split(source_config, source_name, force_rebuild)
        reports.append(report)
        if report.failed_images:
            raise RuntimeError(
                f'{relative_folder} has {report.failed_images} failed images; see {report.error_report}。'
            )
        audit_path = (
            project_root
            / relative_folder
            / str(source_config["cache"]["folder_name"])
            / 'audit'
            / 'roi_audit.csv'
        )
        frame = pd.read_csv(audit_path, encoding="utf-8-sig")
        frame["source_name"] = source_name
        frame["image_key"] = source_name + "::" + frame["filename"].astype(str)

        def _absolute_cached_path(value: object) -> str:
            if pd.isna(value) or not str(value).strip():
                return ""
            return str((project_root / str(value)).resolve())

        if "roi_jpg" in frame.columns:
            frame["roi_jpg"] = frame["roi_jpg"].map(_absolute_cached_path)
        if "source_image_png" not in frame.columns:
            raise KeyError(
                f'{audit_path} lacks source_image_png; rebuild the cache with this dataset.py.'
            )
        frame["source_image_png"] = frame["source_image_png"].map(
            _absolute_cached_path
        )
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)


    # Deduplicate matching labels and coordinates; reject conflicting records.

    duplicate_mask = merged.duplicated(["image_key", "identity"], keep=False)
    if bool(duplicate_mask.any()):
        compare_columns = [
            column for column in (
                "group", "level", "group_score", "visible",
                "source_p1_x", "source_p1_y", "source_p2_x", "source_p2_y",
                "source_p3_x", "source_p3_y", "source_p4_x", "source_p4_y",
            )
            if column in merged.columns
        ]
        conflicts: list[str] = []
        duplicated = merged.loc[duplicate_mask].copy()
        for (image_key, identity), part in duplicated.groupby(
            ["image_key", "identity"], sort=False
        ):
            if len(part) <= 1:
                continue
            for column in compare_columns:
                values = part[column]
                if pd.api.types.is_numeric_dtype(values):
                    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
                    finite = numeric[np.isfinite(numeric)]
                    if finite.size > 1 and not np.allclose(
                        finite, finite[0], rtol=0.0, atol=1e-6
                    ):
                        conflicts.append(f"{image_key}/{identity}/{column}")
                        break
                else:
                    unique = values.fillna("").astype(str).str.strip().unique()
                    if len(unique) > 1:
                        conflicts.append(f"{image_key}/{identity}/{column}")
                        break
        if conflicts:
            raise ValueError(
                'Conflicting duplicate ROI records for the same image cannot be removed automatically: '
                + ", ".join(conflicts[:20])
            )
        before = len(merged)
        merged = merged.drop_duplicates(["image_key", "identity"], keep="first").copy()
        removed = before - len(merged)
        if removed:
            print(f'Safely removed duplicate reference ROI records: {removed} rows.')

    counts = merged.groupby("image_key").size()
    invalid = counts[counts != 24]
    if not invalid.empty:
        raise ValueError(f'Images without exactly 24 ROIs: {invalid.head(20).to_dict()}')
    return reports, merged


def prepare_five_fold_roi_data(
    config: dict[str, Any],
    validation_fold: int,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[SplitReport]]:
    """Use reference ROIs; only the four training folds include augmented images."""
    config = json.loads(json.dumps(config, ensure_ascii=False))


    config["cache"]["save_ground_truth_jpg"] = False
    config["cache"]["save_roi_grid_jpg"] = False
    config["cache"]["save_each_roi_jpg"] = False
    config["cache"]["save_source_image_png"] = True
    training_sources, validation_sources = five_fold_sources(validation_fold)
    train_reports, train_frame = prepare_source_group(
        config, training_sources, force_rebuild
    )
    val_reports, val_frame = prepare_source_group(
        config, validation_sources, force_rebuild
    )
    overlap = set(train_frame["filename"].astype(str)) & set(
        val_frame["filename"].astype(str)
    )
    if overlap:
        raise RuntimeError(
            f'Training and validation sets contain overlapping filenames: {sorted(overlap)[:20]}'
        )
    return train_frame, val_frame, [*train_reports, *val_reports]


def prepare_five_fold_yolo_dataset(
    config: dict[str, Any],
    validation_fold: int,
    force_rebuild: bool = False,
) -> tuple[Path, list[SplitReport]]:
    """Prepare YOLO data with original images only in the validation fold."""
    training_sources, validation_sources = five_fold_sources(validation_fold)
    reports: list[SplitReport] = []
    train_paths: list[str] = []
    val_paths: list[str] = []
    root = Path(config["data_root"]).resolve()
    for source_name, relative_folder in [*training_sources, *validation_sources]:
        source_config = _fold_source_config(config, source_name, relative_folder)
        report = prepare_split(source_config, source_name, force_rebuild)
        reports.append(report)
        image_path = report.images_folder
        if source_name.startswith("train_"):
            train_paths.append(image_path)
        else:
            val_paths.append(image_path)
    names = _class_names(str(config["annotations"]["class_mode"]))

    validation_folder = validation_sources[0][1]
    validation_source = _fold_source_config(config, "val", validation_folder)
    config["split_folders"]["val"] = validation_source["split_folders"]["val"]
    config["csv_filenames"]["val"] = validation_source["csv_filenames"]["val"]
    yaml_path = root / f'yolo_fold_{int(validation_fold)}_dataset.yaml'
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "train": train_paths,
                "val": val_paths,
                "test": [],
                "names": {index: name for index, name in enumerate(names)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path, reports


def prepare_independent_test_yolo_dataset(
    config: dict[str, Any],
    force_rebuild: bool = False,
) -> tuple[Path, list[SplitReport]]:
    source_name, relative_folder = independent_test_sources()[0]
    source_config = _fold_source_config(config, "test", relative_folder)
    report = prepare_split(source_config, "test", force_rebuild)
    config["split_folders"]["test"] = source_config["split_folders"]["test"]
    config["csv_filenames"]["test"] = source_config["csv_filenames"]["test"]
    names = _class_names(str(config["annotations"]["class_mode"]))
    root = Path(config["data_root"]).resolve()
    yaml_path = root / 'yolo_test_dataset.yaml'
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "train": [],
                "val": [],
                "test": [report.images_folder],
                "names": {index: name for index, name in enumerate(names)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path, [report]


def _file_signature(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _yolo_roi_cache_signature(
    yolo_weights: str | Path,
    template_boxes: str | Path,
    source_folder: str | Path,
    yolo_imgsz: int,
    yolo_conf: float,
    yolo_iou: float,
    max_det: int,
) -> str:
    payload = {
        "weights": _file_signature(yolo_weights),
        "template": _file_signature(template_boxes),
        "source_folder": str(Path(source_folder).resolve()),
        "yolo_imgsz": int(yolo_imgsz),
        "yolo_conf": float(yolo_conf),
        "yolo_iou": float(yolo_iou),
        "max_det": int(max_det),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _resolve_existing_relative_to_dataset(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _resolve_best_yolo_weight(yolo_project: str | Path) -> Path:
    project = Path(yolo_project).expanduser()
    if not project.is_absolute():
        project = Path(__file__).resolve().parent / project
    project = project.resolve()
    record_path = project / 'best_yolo_model.json'
    legacy_record = project / '\u6700\u4f73YOLO\u6a21\u578b.json'
    if not record_path.is_file() and legacy_record.is_file():
        record_path = legacy_record
    if not record_path.is_file():
        raise FileNotFoundError(
            f'Not found: {record_path}. Run YOLO train.py to complete five-fold training and best-fold selection first.'
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    weights = project / str(record["weights"])
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return weights


def _source_rows_with_yolo_template_points(
    config: dict[str, Any],
    source_name: str,
    relative_folder: str | Path,
    yolo_weights: str | Path,
    template_boxes: str | Path = "template_boxes.json",
    device: str = "0",
    yolo_imgsz: int = 1024,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.50,
    max_det: int = 40,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace reference ROI coordinates with YOLO-template output; retain reference labels."""
    from ultralytics import YOLO
    # Import locally to avoid a circular dependency with valid.py.
    from valid import load_template_model, correct_yolo_boxes_with_template

    root = Path(config["data_root"]).resolve()
    relative_folder = Path(relative_folder)
    source_folder = root / relative_folder
    if not source_folder.exists():
        raise FileNotFoundError(source_folder)
    weights = _resolve_existing_relative_to_dataset(yolo_weights)
    template_path = _resolve_existing_relative_to_dataset(template_boxes)

    cache_root = root / 'classification_cache' / 'yolo_template_rois' / source_name
    cache_csv = cache_root / 'roi_index.csv'
    audit_csv = cache_root / 'template_correction_audit.csv'
    signature_path = cache_root / 'cache_signature.json'
    signature = _yolo_roi_cache_signature(
        weights, template_path, source_folder, yolo_imgsz, yolo_conf, yolo_iou, max_det
    )
    if (
        not force_rebuild
        and cache_csv.is_file()
        and audit_csv.is_file()
        and signature_path.is_file()
    ):
        try:
            record = json.loads(signature_path.read_text(encoding="utf-8"))
            if record.get("signature") == signature:
                return (
                    pd.read_csv(cache_csv, encoding="utf-8-sig"),
                    pd.read_csv(audit_csv, encoding="utf-8-sig"),
                )
        except Exception:
            pass

    _, base_frame = prepare_source_group(
        config,
        [(source_name, relative_folder)],
        force_rebuild=force_rebuild,
    )
    by_name, by_stem = _dicom_index(source_folder, config)
    model = YOLO(str(weights))
    template = load_template_model(template_path)

    corrected_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for image_key, image_rows in base_frame.groupby("image_key", sort=False):
        filename = str(image_rows.iloc[0]["filename"])
        dicom_path = _locate(filename, by_name, by_stem)
        if dicom_path is None:
            raise FileNotFoundError(f'{source_folder} contains no DICOM matching: {filename}')
        dicom = read_dicom(dicom_path, config)
        yolo_image = cv2.cvtColor(dicom.image_u8, cv2.COLOR_GRAY2BGR)
        result = model.predict(
            source=yolo_image,
            imgsz=int(yolo_imgsz),
            conf=float(yolo_conf),
            iou=float(yolo_iou),
            max_det=int(max_det),
            device=str(device),
            verbose=False,
        )[0]
        if result.obb is None or len(result.obb) == 0:
            boxes = np.empty((0, 4, 2), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=np.int64)
        else:
            boxes = result.obb.xyxyxyxy.detach().cpu().numpy()
            confidences = result.obb.conf.detach().cpu().numpy()
            class_ids = result.obb.cls.detach().cpu().numpy().astype(np.int64)

        correction = correct_yolo_boxes_with_template(
            template,
            boxes,
            confidences,
            class_ids,
            dicom.image_u8.shape,
        )
        summary = correction.summary()
        part = image_rows.copy()
        identities = [f"L{i}" for i in range(1, 13)] + [f"R{i}" for i in range(1, 13)]
        part = part.set_index("identity", drop=False)
        missing = [identity for identity in identities if identity not in part.index]
        if missing:
            raise ValueError(f'{image_key} lacks ROI label records: {missing}')
        part = part.loc[identities].copy()
        for class_id, identity in enumerate(identities):
            source_points = output_points_to_source(
                correction.points[class_id], dicom.transform
            )
            for point_id in range(4):
                part.loc[identity, f"source_p{point_id + 1}_x"] = float(source_points[point_id, 0])
                part.loc[identity, f"source_p{point_id + 1}_y"] = float(source_points[point_id, 1])
            part.loc[identity, "roi_coordinate_source"] = "YOLO11s-OBB+template"
            part.loc[identity, "yolo_template_source"] = str(correction.sources[class_id])
            part.loc[identity, "yolo_template_confidence"] = float(correction.confidences[class_id])
        part = part.reset_index(drop=True)
        corrected_parts.append(part)
        audit_rows.append(
            {
                'image_key': image_key,
                'filename': filename,
                'yolo_weights': str(weights),
                'fit_reliable': summary["fit_reliable"],
                'fit_status': summary["fit_status"],
                'template_scale': summary["template_scale"],
                'template_rotation_deg': summary["template_rotation_degrees"],
                'matched_yolo_boxes': summary["matched_yolo_box_count"],
                'template_added_boxes': summary["template_filled_box_count"],
                'template_relabelled_boxes': summary["template_relabelled_box_count"],
                'template_corrected_boxes': summary["template_corrected_box_count"],
                'output_boxes': summary["corrected_box_count"],
            }
        )

    corrected = pd.concat(corrected_parts, ignore_index=True)
    counts = corrected.groupby("image_key").size()
    if not bool((counts == 24).all()):
        raise ValueError(f'Unexpected corrected YOLO ROI counts: {counts[counts != 24].to_dict()}')
    audit = pd.DataFrame(audit_rows)
    cache_root.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(cache_csv, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_csv, index=False, encoding="utf-8-sig")
    signature_path.write_text(
        json.dumps({"signature": signature}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return corrected, audit


def prepare_best_yolo_test_roi_data(
    config: dict[str, Any],
    yolo_project: str | Path = "runs/mri_line_pairs_yolo11s_obb",
    template_boxes: str | Path = "template_boxes.json",
    device: str = "0",
    yolo_imgsz: int = 1024,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.50,
    max_det: int = 40,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare independent test ROIs using the selected YOLO model and template."""
    weight = _resolve_best_yolo_weight(yolo_project)
    return _source_rows_with_yolo_template_points(
        config=config,
        source_name="test_best_yolo",
        relative_folder=Path("data_test"),
        yolo_weights=weight,
        template_boxes=template_boxes,
        device=device,
        yolo_imgsz=yolo_imgsz,
        yolo_conf=yolo_conf,
        yolo_iou=yolo_iou,
        max_det=max_det,
        force_rebuild=force_rebuild,
    )


