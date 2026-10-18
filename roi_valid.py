"""Evaluate the three ResNet18 configurations out of fold on original reference ROIs."""
from __future__ import annotations


import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dataset import make_config, prepare_five_fold_roi_data
from roi_train import (
    EXPERIMENTS, RoiDataset, binary_metrics,
    class_loss_weights, evaluate, load_checkpoint, parse_folds, parse_list,
    resolve_device, resolve_path, save_excel, whole_image_metrics,
)
from roi_test import (
    predict_fold_with_cam,
    save_input_visualizations,
    save_single_fold_heatmaps,
)


def evaluate_fold(
    exp_id: str,
    fold: int,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    project: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    weight = project / exp_id / f"fold_{fold}" / "weights" / "best.pt"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    model, ckpt, spec = load_checkpoint(weight, device)
    train_frame = train_frame.reset_index(drop=True)
    val_frame = val_frame.reset_index(drop=True)
    mean = ckpt["normalization_mean"]
    std = ckpt["normalization_std"]
    ds = RoiDataset(val_frame, spec, mean, std)
    from roi_train import create_loader
    loader = create_loader(ds, args.batch, args.workers, False, device)
    weights = class_loss_weights(train_frame, device)
    val_loss, pred, _, _ = evaluate(
        model, loader, device, spec, weights, args.focal_gamma, args.boundary_factor
    )
    roi_m = binary_metrics(pred["visible"], pred['resolvability_probability'])
    image_m, image_details = whole_image_metrics(pred)
    row = {
        'experiment_id': exp_id, 'factor': spec.factor, 'level': spec.level, 'fold': fold,
        'validation_loss': val_loss, **roi_m, **image_m,
    }
    pred.insert(0, 'experiment_id', exp_id)
    pred.insert(1, 'fold', fold)
    image_details.insert(0, 'experiment_id', exp_id)
    image_details.insert(1, 'fold', fold)
    del model, ds, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, pred, image_details


def summarize(fold_frame: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in fold_frame.columns if c not in {'experiment_id', 'factor', 'level', 'fold'}]
    rows = []
    for exp_id, part in fold_frame.groupby('experiment_id', sort=False):
        row: dict[str, Any] = {
            'experiment_id': exp_id,
            'factor': str(part.iloc[0]['factor']),
            'level': str(part.iloc[0]['level']),
            'completed_folds': int(len(part)),
        }
        for col in metric_cols:
            values = pd.to_numeric(part[col], errors="coerce")
            row[f'{col}five_fold_mean'] = float(values.mean())
            row[f'{col}five_fold_sd'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _visualization_frame(fold_data: dict[int, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """Collect held-out ROIs for the shared bilinear grayscale input preview."""
    parts = []
    for fold in sorted(fold_data):
        part = fold_data[fold][1].copy().reset_index(drop=True)
        part['visualization_fold'] = int(fold)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    keys = [c for c in ["image_key", "identity"] if c in frame.columns]
    if keys:
        frame = frame.drop_duplicates(keys, keep="first").reset_index(drop=True)
    return frame


def save_validation_fold_gradcam(
    exp_id: str,
    fold: int,
    val_frame: pd.DataFrame,
    project: Path,
    output_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Generate held-out Grad-CAM using only the corresponding fold model."""
    weight = project / exp_id / f"fold_{fold}" / "weights" / "best.pt"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    pred, cams, _ckpt = predict_fold_with_cam(
        weight,
        val_frame.reset_index(drop=True),
        device,
        args.batch,
        args.workers,
        need_cam=True,
    )
    if cams is None:
        raise RuntimeError(f'{exp_id} Fold {fold} did not produce Grad-CAM.')
    save_single_fold_heatmaps(
        output_root / exp_id / f"fold_{fold}" / 'visualizations',
        val_frame.reset_index(drop=True),
        EXPERIMENTS[exp_id],
        cams,
        pred,
        fold,
        args.heatmap_alpha,
        args.max_heatmap_rois,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Out-of-fold validation of the three ResNet18 configurations")
    p.add_argument("--data-root", default=None)
    p.add_argument("--project", default="runs/roi_ablation")
    p.add_argument("--output-dir", default=None, help='Default: project directory. Set a separate output root if needed.')
    p.add_argument("--experiments", default=None)
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--boundary-factor", type=float, default=2.0)
    p.add_argument("--force-data-rebuild", action="store_true")
    p.add_argument("--disable-input-visualizations", action="store_true")
    p.add_argument("--max-input-rois", type=int, default=-1)
    p.add_argument("--disable-heatmaps", action="store_true")
    p.add_argument("--max-heatmap-rois", type=int, default=-1)
    p.add_argument("--heatmap-alpha", type=float, default=0.45)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config(args.data_root)
    project = resolve_path(args.project)
    output_root = resolve_path(args.output_dir) if args.output_dir else project
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    exp_ids = parse_list(args.experiments, list(EXPERIMENTS.keys()))
    folds = parse_folds(args.folds)


    fold_data: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold in folds:
        train_frame, val_frame, _reports = prepare_five_fold_roi_data(
            config,
            validation_fold=fold,
            force_rebuild=args.force_data_rebuild,
        )
        fold_data[int(fold)] = (
            train_frame.reset_index(drop=True),
            val_frame.reset_index(drop=True),
        )

    if not args.disable_input_visualizations:
        save_input_visualizations(
            _visualization_frame(fold_data), output_root, args.max_input_rois
        )

    rows: list[dict[str, Any]] = []
    roi_preds: list[pd.DataFrame] = []
    image_preds: list[pd.DataFrame] = []
    for exp_id in exp_ids:
        print(f'\n===== Combined validation: {exp_id} =====')
        for fold in folds:
            train_frame, val_frame = fold_data[int(fold)]
            row, pred, images = evaluate_fold(
                exp_id, fold, train_frame, val_frame, project, device, args
            )
            rows.append(row)
            roi_preds.append(pred)
            image_preds.append(images)

            if not args.disable_heatmaps:
                print(f'Generating {exp_id} fold_{fold} single-fold Grad-CAM...')
                save_validation_fold_gradcam(
                    exp_id, fold, val_frame, project, output_root, device, args
                )

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fold_frame = pd.DataFrame(rows)
    summary = summarize(fold_frame)
    roi_frame = pd.concat(roi_preds, ignore_index=True) if roi_preds else pd.DataFrame()
    image_frame = pd.concat(image_preds, ignore_index=True) if image_preds else pd.DataFrame()

    sheets: dict[str, pd.DataFrame] = {
        'fold_results': fold_frame,
        'five_fold_summary': summary,
        'oof_roi_predictions': roi_frame,
        'oof_image_scores': image_frame,
    }

    output = output_root / 'resnet18_three_configurations_validation.xlsx'
    save_excel(output, sheets)
    print(f'Validation report: {output}')
    if not args.disable_heatmaps:
        print('Grad-CAM uses only the corresponding held-out fold model, without averaging folds.')


if __name__ == "__main__":
    main()
