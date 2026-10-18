"""Train YOLO11s-OBB in five folds and select the best validation model."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from dataset import make_config, prepare_five_fold_yolo_dataset, reports_to_text


DEFAULT_FOLDS = "1,2,3,4,5"  


def parse_fold_list(value: str) -> list[int]:
    folds = sorted(
        set(
            int(item.strip())
            for item in str(value).replace("，", ",").split(",")
            if item.strip()
        )
    )
    if not folds or any(fold not in range(1, 6) for fold in folds):
        raise ValueError('Fold indices must be between 1 and 5.')
    return folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train YOLO11-OBB for MRI line-pair detection in five folds.')
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--folds",
        default=DEFAULT_FOLDS,
        help='Fold indices to run; default: 1,2,3,4,5.',
    )

    parser.add_argument("--fold", type=int, default=None, choices=range(1, 6))
    parser.add_argument("--model", default="yolo11s-obb.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=float, default=4)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="mri_line_pairs_yolo11s_obb")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--force-data-rebuild", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help='Skip training when fold_x/weights/best.pt exists, but re-evaluate validation metrics.',
    )
    return parser.parse_args()


def resolve_relative_to_script(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def resolve_model_source(model_value: str) -> str:
    """Prefer local weights; otherwise use an Ultralytics model name."""
    candidate = Path(str(model_value)).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return str(candidate.resolve())
    if not candidate.is_absolute():
        local_candidate = Path(__file__).resolve().parent / candidate
        if local_candidate.is_file():
            return str(local_candidate.resolve())
    return str(model_value)


def save_model_structure(model: Any, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            model.info(detailed=True, verbose=True)
        except TypeError:
            model.info(detailed=True)
    explanation = 'YOLO11s-OBB architecture\n=======================\nInput: B x 3 x 1024 x 1024. dataset.py converts DICOM to grayscale JPG.\nUltralytics reads three channels to use the official pretrained weights.\n\nBackbone: Conv, C3k2, SPPF, and C2PSA extract multiscale features.\nNeck: upsampling, concatenation, and C3k2 fuse multiscale features.\nOBB head: predicts oriented boxes, confidence, and classes L1-L12 and R1-R12.\n\nThe detector does not enforce exactly 24 boxes per image. valid.py and test.py\nretain raw predictions and optionally apply the 24-box template correction.\n'.strip()
    content = (
        explanation
        + '\n\nOfficial model information\n==========================\n'
        + buffer.getvalue()
        + '\n\nFull network architecture\n=========================\n'
        + str(model.model)
    )
    (output / 'model_architecture.txt').write_text(content, encoding="utf-8")


def rename_yolo_outputs(folder: Path) -> None:
    mapping = {
        "results.png": 'training_results.png',
        "results.csv": 'training_history.csv',
        "args.yaml": 'training_arguments.yaml',
        "confusion_matrix.png": 'confusion_matrix.png',
        "confusion_matrix_normalized.png": 'confusion_matrix_normalized.png',
        "PR_curve.png": 'precision_recall_curve.png',
        "P_curve.png": 'precision_curve.png',
        "R_curve.png": 'recall_curve.png',
        "F1_curve.png": 'f1_curve.png',
        "labels.jpg": 'label_distribution.jpg',
        "labels_correlogram.jpg": 'label_correlogram.jpg',
    }
    for source_name, target_name in mapping.items():
        source = folder / source_name
        target = folder / target_name
        if source == target:
            continue
        if source.is_file():
            if target.exists():
                target.unlink()
            source.replace(target)
    for source in folder.glob("train_batch*.jpg"):
        number = source.stem.replace("train_batch", "")
        source.replace(folder / f'train_batch{number}.jpg')
    for source in folder.glob("val_batch*_labels.jpg"):
        number = source.stem.replace("val_batch", "").replace("_labels", "")
        source.replace(folder / f'val_batch{number}_reference.jpg')
    for source in folder.glob("val_batch*_pred.jpg"):
        number = source.stem.replace("val_batch", "").replace("_pred", "")
        source.replace(folder / f'val_batch{number}_prediction.jpg')


def _float_metric(results_dict: dict[str, Any], candidates: list[str]) -> float:
    normalized = {str(key).lower(): value for key, value in results_dict.items()}
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for key, value in normalized.items():
            if candidate_lower == key or candidate_lower in key:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    return number
    return float("nan")


def validate_best_weight(
    weights_path: Path,
    dataset_yaml: Path,
    fold_folder: Path,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Evaluate each best.pt on its original held-out validation images."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=int(args.batch) if float(args.batch).is_integer() else args.batch,
        device=args.device,
        workers=args.workers,
        conf=0.25,
        iou=0.50,
        max_det=40,
        plots=False,
        save_json=False,
        project=str(fold_folder),
        name='Best-checkpoint validation',
        exist_ok=True,
        verbose=False,
    )
    results_dict = dict(getattr(metrics, "results_dict", {}) or {})
    values = {
        "Precision": _float_metric(results_dict, ["metrics/precision(b)", "metrics/precision", "precision"]),
        "Recall": _float_metric(results_dict, ["metrics/recall(b)", "metrics/recall", "recall"]),
        "mAP50": _float_metric(results_dict, ["metrics/map50(b)", "metrics/map50", "map50"]),
        "mAP50-95": _float_metric(results_dict, ["metrics/map50-95(b)", "metrics/map50-95", "map50-95", "map50_95"]),
        "Fitness": _float_metric(results_dict, ["fitness"]),
    }
    if math.isnan(values["mAP50-95"]):
        raise RuntimeError(
            f'Cannot read mAP50-95 from Ultralytics validation output. Available results_dict keys: {list(results_dict.keys())}'
        )
    return values


def train_one_fold(
    fold: int,
    dataset_yaml: Path,
    root_folder: Path,
    model_source: str,
    args: argparse.Namespace,
) -> Path:
    from ultralytics import YOLO

    fold_folder = root_folder / f"fold_{fold}"
    weights_path = fold_folder / "weights" / "best.pt"
    if args.skip_completed and weights_path.is_file():
        print(f'Fold {fold} already has best.pt; skipping training: {weights_path}')
        return weights_path

    model = YOLO(model_source)
    if fold == 1 or not (root_folder / "model_structure" / 'model_architecture.txt').is_file():
        save_model_structure(model, root_folder / "model_structure")

    parameters: dict[str, Any] = {
        "data": str(dataset_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
        "patience": 80,
        "seed": 42,
        "deterministic": True,
        "amp": False,
        "cache": False,
        "pretrained": True,
        "project": str(root_folder),
        "name": f"fold_{fold}",
        "exist_ok": True,
        "plots": True,
        "save": True,
        "save_period": 10,

        "degrees": 3.0,
        "translate": 0.03,
        "scale": 0.03,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.02,
    }
    if args.resume:
        parameters["resume"] = True

    result = model.train(**parameters)
    actual_folder = Path(getattr(result, "save_dir", fold_folder)).resolve()
    rename_yolo_outputs(actual_folder)
    weights_path = actual_folder / "weights" / "best.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f'Fold {fold} completed but best.pt was not found: {weights_path}')
    print(f'Fold {fold} training completed: {weights_path}')
    return weights_path


def build_five_fold_report(root_folder: Path, rows: list[dict[str, Any]]) -> Path:
    """Rank folds by mAP50-95, mAP50, recall, and precision."""
    frame = pd.DataFrame(rows).sort_values('fold').reset_index(drop=True)
    ranked = frame.sort_values(
        ["mAP50-95", "mAP50", "Recall", "Precision", 'fold'],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )
    best_row = ranked.iloc[0]
    best_fold = int(best_row['fold'])
    frame['is_best_fold'] = frame['fold'].map(
        lambda value: 'yes' if int(value) == best_fold else 'no'
    )

    numeric_columns = ["Precision", "Recall", "mAP50", "mAP50-95", "Fitness"]
    summary_rows: list[dict[str, Any]] = []
    for column in numeric_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        summary_rows.append(
            {
                'metric': column,
                'five_fold_mean': float(series.mean()),
                'five_fold_sd': float(series.std(ddof=1)) if series.notna().sum() > 1 else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows)

    report_path = root_folder / 'yolo_five_fold_validation.xlsx'
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name='fold_results', index=False)
        summary.to_excel(writer, sheet_name='fold_mean_sd', index=False)
        frame[frame['fold'] == best_fold].to_excel(writer, sheet_name='best_fold', index=False)

    relative_weights = f"fold_{best_fold}/weights/best.pt"
    best_record = {
        "best_fold": best_fold,
        "weights": relative_weights,
        "selection_rule": "mAP50-95 > mAP50 > Recall > Precision",
        "metrics": {
            key: (None if pd.isna(best_row[key]) else float(best_row[key]))
            for key in numeric_columns
        },
    }
    (root_folder / 'best_yolo_model.json').write_text(
        json.dumps(best_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root_folder / 'best_yolo_model.txt').write_text(
        f'Best fold: fold_{best_fold}\nWeights: {relative_weights}\nSelection order: mAP50-95 > mAP50 > Recall > Precision\n',
        encoding="utf-8",
    )
    print(f'\nBest YOLO fold: {best_fold} fold')
    print('Best weights: ', root_folder / relative_weights)
    print('Five-fold validation report: ', report_path)
    return report_path


def main() -> None:
    args = parse_args()
    folds = [args.fold] if args.fold is not None else parse_fold_list(args.folds)
    if args.resume and len(folds) != 1:
        raise ValueError('--resume requires one fold; select it with --fold.')

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            'Missing dependencies. Install ultralytics, pydicom, pandas, opencv-python, and openpyxl.'
        ) from error

    project_path = resolve_relative_to_script(args.project)
    root_folder = project_path / args.name
    root_folder.mkdir(parents=True, exist_ok=True)
    model_source = resolve_model_source(args.model)
    if args.resume:
        model_source = str(resolve_relative_to_script(args.resume))

    config = make_config(args.data_root)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for position, fold in enumerate(folds, start=1):
        print("\n" + "=" * 88)
        print(f'YOLO five-fold progress {position}/{len(folds)}: fold {fold} fold')
        try:
            dataset_yaml, reports = prepare_five_fold_yolo_dataset(
                config,
                fold,
                force_rebuild=bool(args.force_data_rebuild and position == 1),
            )
            print(reports_to_text(reports))
            if any(report.failed_images for report in reports):
                raise RuntimeError(
                    'DICOM reading or annotation parsing failed. Check the data-processing cache error log.'
                )
            weights_path = train_one_fold(
                fold, dataset_yaml, root_folder, model_source, args
            )
            metrics = validate_best_weight(
                weights_path, dataset_yaml, root_folder / f"fold_{fold}", args
            )
            rows.append(
                {
                    'fold': fold,
                    **metrics,
                    'model_path': str(weights_path),
                }
            )
            print(
                f"Fold {fold} validation: mAP50-95={metrics['mAP50-95']:.6f}, mAP50={metrics['mAP50']:.6f}, Recall={metrics['Recall']:.6f}, Precision={metrics['Precision']:.6f}"
            )
        except Exception as error:
            failures.append(f"fold_{fold}: {type(error).__name__}: {error}")
            print('Fold failed: ', failures[-1])

    if rows:
        build_five_fold_report(root_folder, rows)
    if failures:
        raise RuntimeError('The following YOLO folds failed during training or validation:\n' + "\n".join(failures))


if __name__ == "__main__":
    main()
