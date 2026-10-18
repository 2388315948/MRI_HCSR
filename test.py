"""Evaluate YOLO models on the independent test set with template correction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from dataset import make_config, prepare_independent_test_yolo_dataset, reports_to_text
import valid as valid_module
from valid import evaluate_split


TEST_MODEL_SCOPE = "best_fold"  
YOLO_TRAIN_PROJECT = "runs/mri_line_pairs_yolo11s_obb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Independent test of YOLO11-OBB MRI line-pair detection.')
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-project", default=YOLO_TRAIN_PROJECT)
    parser.add_argument(
        "--model-scope",
        choices=["best_fold", "all_folds"],
        default=TEST_MODEL_SCOPE,
        help='best_fold evaluates the best fold; all_folds evaluates all five fold models.',
    )
    parser.add_argument(
        "--weights",
        default=None,
        help='Explicit YOLO checkpoint; overrides model-scope.',
    )
    parser.add_argument("--output-dir", default="runs/obb_test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=40)
    parser.add_argument("--template-boxes", default="template_boxes.json")
    parser.add_argument("--disable-template-correction", action="store_true")
    parser.add_argument("--heatmap-alpha", type=float, default=0.45)
    parser.add_argument("--max-visualizations", type=int, default=-1)
    parser.add_argument("--force-data-rebuild", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def best_yolo_weight(train_project: Path) -> tuple[int, Path]:
    record_path = train_project / 'best_yolo_model.json'
    legacy_record = train_project / '\u6700\u4f73YOLO\u6a21\u578b.json'
    if not record_path.is_file() and legacy_record.is_file():
        record_path = legacy_record
    if not record_path.is_file():
        raise FileNotFoundError(
            f'Not found: {record_path}. Run train.py to complete five-fold training first.'
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    fold = int(record["best_fold"])
    weights = train_project / str(record["weights"])
    if not weights.is_file():
        raise FileNotFoundError(f'Best YOLO weights not found: {weights}')
    return fold, weights


def selected_models(args: argparse.Namespace) -> list[tuple[str, int | None, Path]]:
    if args.weights:
        weights = resolve_path(args.weights)
        if not weights.is_file():
            raise FileNotFoundError(f'Explicit checkpoint not found: {weights}')
        return [("manual", None, weights)]

    project = resolve_path(args.train_project)
    if args.model_scope == "best_fold":
        fold, weights = best_yolo_weight(project)
        return [(f"best_fold_{fold}", fold, weights)]

    models: list[tuple[str, int | None, Path]] = []
    for fold in range(1, 6):
        weights = project / f"fold_{fold}" / "weights" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(f'Fold {fold} weights not found: {weights}')
        models.append((f"fold_{fold}", fold, weights))
    return models


def metric_value(results: dict[str, Any], token: str) -> float | None:
    token = token.lower()
    for key, value in results.items():
        if token in str(key).lower():
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def exact_metric(results: dict[str, Any], key: str) -> float | None:
    value = results.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percent(value: float | None) -> float | None:
    return None if value is None else 100.0 * float(value)


_VISUAL_CONTEXT = ""
_VISUAL_STATE: dict[str, dict[str, np.ndarray | None]] = {}


def _write_jpg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f'Failed to save image: {path}')
    encoded.tofile(str(path))


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _information_bar(image: np.ndarray, title: str, details: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        image, 72, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
    )
    width = int(image.shape[1])
    title_scale = max(0.45, min(0.75, width / 900.0))
    detail_scale = max(0.36, title_scale * 0.72)
    cv2.putText(
        canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
        title_scale, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, details, (12, 56), cv2.FONT_HERSHEY_SIMPLEX,
        detail_scale, (205, 205, 205), 1, cv2.LINE_AA,
    )
    return canvas


def _test_save_visuals(
    folder: Path,
    stem: str,
    original: np.ndarray,
    ground_truth: np.ndarray | None,
    prediction: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
) -> None:
    folder = Path(folder)
    height, width = original.shape[:2]
    original = _fit_panel(original, width, height)
    template = (
        _fit_panel(ground_truth, width, height)
        if ground_truth is not None
        else original.copy()
    )
    prediction = _fit_panel(prediction, width, height)
    heatmap_color = cv2.applyColorMap(
        np.round(np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    heatmap_color = _fit_panel(heatmap_color, width, height)
    heatmap_overlay = cv2.addWeighted(
        original, 1.0 - float(alpha), heatmap_color, float(alpha), 0
    )
    details = f"File={stem} | Model={_VISUAL_CONTEXT}"
    panels = {
        "original": _information_bar(original, "Original image", details),
        "template": _information_bar(
            template,
            "Reference/template OBB" if ground_truth is not None else "Reference/template OBB unavailable",
            details,
        ),
        "prediction": _information_bar(prediction, "Raw YOLO prediction", details),
        "heatmap": _information_bar(
            heatmap_overlay, "YOLO multi-scale Grad-CAM", details
        ),
    }
    _write_jpg(folder / 'original_images' / f'{stem}_original.jpg', panels["original"])
    _write_jpg(folder / 'template_images' / f'{stem}_template.jpg', panels["template"])
    _write_jpg(folder / 'prediction_images' / f'{stem}_prediction.jpg', panels["prediction"])
    _write_jpg(folder / "Grad-CAM" / f"{stem}__Grad-CAM.jpg", panels["heatmap"])
    _VISUAL_STATE[stem] = {
        "original": original,
        "template": template,
        "prediction": prediction,
        "heatmap": heatmap_overlay,
    }


def _test_save_corrected_visuals(
    folder: Path,
    stem: str,
    original: np.ndarray,
    ground_truth: np.ndarray | None,
    corrected_prediction: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
) -> None:
    del ground_truth, heatmap, alpha
    folder = Path(folder)
    height, width = original.shape[:2]
    corrected = _fit_panel(corrected_prediction, width, height)
    details = f"File={stem} | Model={_VISUAL_CONTEXT}"
    corrected_panel = _information_bar(
        corrected, "Template-corrected 24 OBB", details
    )
    _write_jpg(folder / 'corrected_images' / f'{stem}_corrected.jpg', corrected_panel)

    state = _VISUAL_STATE.pop(stem, None)
    if state is None:
        raise RuntimeError(f'Cannot create the five-panel comparison; missing {stem} raw visualization cache.')
    panels = [
        _information_bar(state["original"], "Original image", details),
        _information_bar(state["template"], "Reference/template OBB", details),
        corrected_panel,
        _information_bar(state["prediction"], "Raw YOLO prediction", details),
        _information_bar(
            state["heatmap"], "YOLO multi-scale Grad-CAM", details
        ),
    ]
    _write_jpg(
        folder / 'five_panel_comparisons' / f'{stem}_five_panel_comparison.jpg',
        np.concatenate(panels, axis=1),
    )


def install_test_visualization_layout(model_label: str) -> None:
    """Use the six-folder test visualization layout in valid.evaluate_split."""
    global _VISUAL_CONTEXT
    _VISUAL_CONTEXT = str(model_label)
    _VISUAL_STATE.clear()
    valid_module._save_visuals = _test_save_visuals
    valid_module._save_corrected_visuals = _test_save_corrected_visuals


def main() -> None:
    args = parse_args()
    config = make_config(args.data_root)
    dataset_yaml, reports = prepare_independent_test_yolo_dataset(
        config, force_rebuild=args.force_data_rebuild
    )
    print(reports_to_text(reports))

    from ultralytics import YOLO

    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for label, fold, weights_path in selected_models(args):
        print("\n" + "=" * 88)
        print(f'Testing model: {label}, weights: {weights_path}')
        model = YOLO(str(weights_path))
        install_test_visualization_layout(label)
        results = evaluate_split(
            model,
            dataset_yaml,
            config,
            "test",
            output_root / label,
            args.imgsz,
            args.batch,
            args.device,
            args.workers,
            args.conf,
            args.iou,
            args.max_det,
            args.heatmap_alpha,
            args.max_visualizations,
            args.template_boxes,
            not args.disable_template_correction,
        )
        visual_root = output_root / label / 'visualizations'
        visual_names = ['original_images', 'template_images', 'corrected_images', 'prediction_images', "Grad-CAM", 'five_panel_comparisons']
        for visual_name in visual_names:
            (visual_root / visual_name).mkdir(parents=True, exist_ok=True)
        print('Six visualization output types: ', visual_root)
        print('Subdirectories: ', "、".join(visual_names))
        summary_rows.append(
            {
                'model': label,
                'fold': fold,
                'weights': str(weights_path),
                "Precision": metric_value(results, "precision"),
                "Recall": metric_value(results, "recall"),
                "mAP50": metric_value(results, "map50(b)"),
                "mAP50-95": metric_value(results, "map50-95"),
                'raw_24_class_completeness_percent': percent(
                    exact_metric(results, "custom/raw_24class_complete_rate")
                ),
                'corrected_24_class_completeness_percent': percent(
                    exact_metric(results, "custom/corrected_24class_complete_rate")
                ),
                'raw_center_error_mean_px': exact_metric(
                    results, "custom/raw_center_error_mean_px"
                ),
                'raw_center_error_sd_px': exact_metric(
                    results, "custom/raw_center_error_sd_px"
                ),
                'raw_center_error_p95_px': exact_metric(
                    results, "custom/raw_center_error_p95_px"
                ),
                'corrected_center_error_mean_px': exact_metric(
                    results, "custom/corrected_center_error_mean_px"
                ),
                'corrected_center_error_sd_px': exact_metric(
                    results, "custom/corrected_center_error_sd_px"
                ),
                'corrected_center_error_p95_px': exact_metric(
                    results, "custom/corrected_center_error_p95_px"
                ),
                'raw_angle_error_mean_deg': exact_metric(
                    results, "custom/raw_angle_error_mean_deg"
                ),
                'raw_angle_error_sd_deg': exact_metric(
                    results, "custom/raw_angle_error_sd_deg"
                ),
                'corrected_angle_error_mean_deg': exact_metric(
                    results, "custom/corrected_angle_error_mean_deg"
                ),
                'corrected_angle_error_sd_deg': exact_metric(
                    results, "custom/corrected_angle_error_sd_deg"
                ),
                'raw_long_side_relative_error_mean_percent': exact_metric(
                    results, "custom/raw_long_side_error_mean_pct"
                ),
                'corrected_long_side_relative_error_mean_percent': exact_metric(
                    results, "custom/corrected_long_side_error_mean_pct"
                ),
                'raw_short_side_relative_error_mean_percent': exact_metric(
                    results, "custom/raw_short_side_error_mean_pct"
                ),
                'corrected_short_side_relative_error_mean_percent': exact_metric(
                    results, "custom/corrected_short_side_error_mean_pct"
                ),
                'raw_area_relative_error_mean_percent': exact_metric(
                    results, "custom/raw_area_error_mean_pct"
                ),
                'corrected_area_relative_error_mean_percent': exact_metric(
                    results, "custom/corrected_area_error_mean_pct"
                ),
                'raw_IoU_mean': exact_metric(results, "custom/raw_iou_mean"),
                'corrected_IoU_mean': exact_metric(results, "custom/corrected_iou_mean"),
                'template_position_corrected_boxes': exact_metric(
                    results, "custom/template_corrected_box_count"
                ),
                'template_class_relabelled_boxes': exact_metric(
                    results, "custom/template_relabelled_box_count"
                ),
                'template_added_boxes': exact_metric(
                    results, "custom/template_filled_box_count"
                ),
            }
        )

    summary_path = output_root / (
        'yolo_five_fold_test_summary.xlsx'
        if len(summary_rows) > 1
        else 'yolo_test_summary.xlsx'
    )
    pd.DataFrame(summary_rows).to_excel(summary_path, index=False)
    print('\nYOLO test summary: ', summary_path)


if __name__ == "__main__":
    main()
