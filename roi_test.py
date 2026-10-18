"""Test three ResNet18 configurations on corrected YOLO ROIs using equal five-fold ensembles."""
from __future__ import annotations


import argparse
import gc
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from dataset import make_config, prepare_best_yolo_test_roi_data
from roi_train import (
    EXPERIMENTS, RoiDataset, VISIBLE_THRESHOLD, binary_metrics, create_loader,
    load_checkpoint, parse_list, prepare_input_from_row, resolve_device,
    resolve_path, save_excel, whole_image_metrics,
)


def safe_name(text: Any) -> str:
    value = str(text)
    for token in '<>:"/\\|?*':
        value = value.replace(token, "_")
    return value.strip().rstrip(". ") or "unnamed"


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f'Failed to save PNG: {path}')
    encoded.tofile(str(path))


def _to_u8(channel: np.ndarray) -> np.ndarray:
    x = np.asarray(channel, dtype=np.float32)
    if x.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if float(x.max()) <= 1.0 and float(x.min()) >= 0.0:
        x = x * 255.0
    return np.uint8(np.clip(x, 0, 255))


def information_bar(image: np.ndarray, title: str, details: str) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    top = 82
    canvas = cv2.copyMakeBorder(
        image, top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
    )
    width = max(int(image.shape[1]), 1)
    title_scale = max(0.42, min(0.72, width / 760.0))
    detail_scale = max(0.34, min(0.52, title_scale * 0.72))
    cv2.putText(canvas, title, (10, 31), cv2.FONT_HERSHEY_SIMPLEX,
                title_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, details[:150], (10, 64), cv2.FONT_HERSHEY_SIMPLEX,
                detail_scale, (205, 205, 205), 1, cv2.LINE_AA)
    return canvas


def heat_color(cam: np.ndarray) -> np.ndarray:
    x = np.asarray(cam, dtype=np.float32)
    x -= float(x.min())
    x /= max(float(x.max()), 1e-8)
    return cv2.applyColorMap(np.uint8(np.clip(x * 255, 0, 255)), cv2.COLORMAP_JET)


def overlay_cam(gray: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay Grad-CAM on a grayscale ROI without changing its orientation."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    base = _to_u8(gray)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    resized_cam = cv2.resize(
        np.asarray(cam, dtype=np.float32),
        (base.shape[1], base.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    color = heat_color(resized_cam)
    return cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0.0)


def save_input_visualizations(frame: pd.DataFrame, output_root: Path, maximum: int) -> None:
    root = output_root / 'input_visualizations' / 'bilinear_grayscale' / 'grayscale'
    limit = len(frame) if maximum < 0 else min(len(frame), int(maximum))
    for index in range(limit):
        row = frame.iloc[index]
        stem = f"{safe_name(row['image_key'])}__{safe_name(row['identity'])}"
        details = (
            f"File={safe_name(row['filename'])} | ROI={row['identity']} | "
            f"Group={row['group']} | Level={int(row['level'])}"
        )
        raw = prepare_input_from_row(row, "bilinear", "gray3")
        write_png(
            root / f"{stem}.png",
            information_bar(_to_u8(raw[0]), "Bilinear grayscale", details),
        )
    print(f'Input visualizations: {root}')


class ResNetGradCAM:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.handle = model.layer4[-1].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        del module, inputs
        self.activations = output
        if isinstance(output, torch.Tensor):
            output.retain_grad()

    def close(self) -> None:
        self.handle.remove()

    def generate(self, images: torch.Tensor, target_class: int = 1) -> tuple[torch.Tensor, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(images)
            score = logits[:, int(target_class)].sum()
            score.backward()
        if self.activations is None or self.activations.grad is None:
            raise RuntimeError('Grad-CAM did not capture features or gradients.')
        act = self.activations.detach().float()
        grad = self.activations.grad.detach().float()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=images.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        flat = cam.flatten(1)
        mins = flat.min(dim=1).values[:, None, None]
        maxs = flat.max(dim=1).values[:, None, None]
        cam = (cam - mins) / torch.clamp(maxs - mins, min=1e-8)
        return logits.detach(), cam.detach().cpu().numpy().astype(np.float32)


def predict_fold_with_cam(
    weight: Path, test_frame: pd.DataFrame, device: torch.device,
    batch: int, workers: int, need_cam: bool,
) -> tuple[pd.DataFrame, np.ndarray | None, dict[str, Any]]:
    model, ckpt, spec = load_checkpoint(weight, device)
    ds = RoiDataset(test_frame, spec, ckpt["normalization_mean"], ckpt["normalization_std"])
    loader = create_loader(ds, batch, workers, False, device)
    probs = np.zeros(len(ds), dtype=np.float64)
    cams = np.zeros((len(ds), ds.image_size, ds.image_size), dtype=np.float32) if need_cam else None
    gradcam = ResNetGradCAM(model) if need_cam else None

    for images, labels, _, indices in loader:
        del labels
        idx = indices.numpy()
        images = images.to(device, non_blocking=True)
        if gradcam is None:
            with torch.no_grad():
                logits = model(images)
        else:
            # Use the visible class as the Grad-CAM target in every fold.
            logits, part_cams = gradcam.generate(images, target_class=1)
            cams[idx] = part_cams
        part_probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs[idx] = part_probs
    if gradcam is not None:
        gradcam.close()

    pred = test_frame[["image_key", "filename", "group", "level", "identity", "group_score", "visible"]].copy()
    pred['resolvability_probability'] = probs
    pred['predicted_class'] = (probs >= VISIBLE_THRESHOLD).astype(int)
    del model, ds, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred, cams, ckpt


def ensemble_predictions(fold_predictions: list[pd.DataFrame]) -> pd.DataFrame:
    base_cols = ["image_key", "filename", "group", "level", "identity", "group_score", "visible"]
    base = fold_predictions[0][base_cols].copy().reset_index(drop=True)
    matrix = np.column_stack([p['resolvability_probability'].to_numpy(dtype=float) for p in fold_predictions])
    base['resolvability_probability'] = matrix.mean(axis=1)
    base['predicted_class'] = (base['resolvability_probability'] >= VISIBLE_THRESHOLD).astype(int)
    for fold, p in enumerate(fold_predictions, start=1):
        base[f'Fold {fold} probability'] = p['resolvability_probability'].to_numpy(dtype=float)
    return base


def add_ci_fields(metrics: dict[str, float], details: pd.DataFrame, pred: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = dict(metrics)

    def wilson(success: int, total: int) -> str:
        if total <= 0:
            return ""
        z = 1.959963984540054
        p = success / total
        d = 1 + z*z/total
        c = (p + z*z/(2*total))/d
        h = z*np.sqrt((p*(1-p)+z*z/(4*total))/total)/d
        return f"{100*max(0,c-h):.1f}–{100*min(1,c+h):.1f}"

    for label, column in [
        ('L_score', 'L_score_correct'), ('R_score', 'R_score_correct'),
        ('L_and_R', 'L_and_R_correct'), ('image_score', 'image_score_correct')
    ]:
        out[f"{label}95%CI/%"] = wilson(int(details[column].sum()), int(len(details)))
    roi_correct = int((pred['predicted_class'].astype(int) == pred["visible"].astype(int)).sum())
    out['ROI_accuracy_95CI_percent'] = wilson(roi_correct, int(len(pred)))
    return out


def save_single_fold_heatmaps(
    output: Path, frame: pd.DataFrame, spec, cams: np.ndarray,
    predictions: pd.DataFrame, fold: int, alpha: float, maximum: int,
) -> None:
    """Save held-out Grad-CAM without averaging across folds."""
    cam_folder = output / "Grad-CAM"
    overlay_folder = output / 'GradCAM_overlays'
    limit = len(frame) if maximum < 0 else min(len(frame), int(maximum))
    manifest = []
    for index in range(limit):
        row = frame.iloc[index]
        raw = prepare_input_from_row(row, spec.interpolation, spec.channel)
        gray = raw[0]
        cam = cv2.resize(cams[index], (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        color = heat_color(cam)
        overlay = overlay_cam(gray, cam, alpha)
        pred = predictions.iloc[index]
        filename = f"{safe_name(row['image_key'])}__{safe_name(row['identity'])}.png"
        details = (
            f"File={safe_name(row['filename'])} | ROI={row['identity']} | Exp={spec.experiment_id} | Fold={int(fold)} | True={int(row['visible'])} Pred={int(pred['predicted_class'])} P={float(pred['resolvability_probability']):.4f}"
        )
        write_png(
            cam_folder / filename,
            information_bar(color, f"Fold {int(fold)} Grad-CAM (target=visible)", details),
        )
        write_png(
            overlay_folder / filename,
            information_bar(overlay, f"Fold {int(fold)} Grad-CAM + original ROI", details),
        )
        manifest.append({
            'index': index + 1, 'fold': int(fold), 'image_key': row["image_key"],
            'filename': row["filename"], "ROI": row["identity"],
            'reference_class': int(row["visible"]), 'predicted_class': int(pred['predicted_class']),
            'resolvability_probability': float(pred['resolvability_probability']),
        })
    pd.DataFrame(manifest).to_csv(output / 'gradcam_details.csv', index=False, encoding="utf-8-sig")


def save_experiment_heatmaps(
    output: Path, test_frame: pd.DataFrame, spec, mean_cam: np.ndarray,
    ensemble: pd.DataFrame, alpha: float, maximum: int,
) -> None:
    cam_folder = output / 'five_fold_mean_GradCAM'
    overlay_folder = output / 'five_fold_mean_GradCAM_overlays'
    limit = len(test_frame) if maximum < 0 else min(len(test_frame), int(maximum))
    manifest = []
    for index in range(limit):
        row = test_frame.iloc[index]
        raw = prepare_input_from_row(row, spec.interpolation, spec.channel)
        gray = raw[0]
        cam = cv2.resize(mean_cam[index], (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        color = heat_color(cam)
        overlay = overlay_cam(gray, cam, alpha)
        pred = ensemble.iloc[index]
        filename = f"{safe_name(row['image_key'])}__{safe_name(row['identity'])}.png"
        details = (
            f"File={safe_name(row['filename'])} | ROI={row['identity']} | Exp={spec.experiment_id} | True={int(row['visible'])} Pred={int(pred['predicted_class'])} P={float(pred['resolvability_probability']):.4f}"
        )
        write_png(cam_folder / filename,
                  information_bar(color, "Five-fold mean Grad-CAM (target=visible)", details))
        write_png(overlay_folder / filename,
                  information_bar(overlay, "Five-fold mean Grad-CAM + original ROI", details))
        manifest.append({
            'index': index + 1, 'image_key': row["image_key"], 'filename': row["filename"],
            "ROI": row["identity"], 'reference_class': int(row["visible"]),
            'predicted_class': int(pred['predicted_class']), 'resolvability_probability': float(pred['resolvability_probability']),
            'five_fold_mean_GradCAM': str((cam_folder / filename).name),
            'five_fold_mean_GradCAM_overlays': str((overlay_folder / filename).name),
        })
    pd.DataFrame(manifest).to_csv(output / 'gradcam_details.csv', index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independent testing of three ResNet18 configurations with five-fold soft ensembles and Grad-CAM")
    p.add_argument("--data-root", default=None)
    p.add_argument("--project", default="runs/roi_ablation")
    p.add_argument("--output-dir", default="runs/roi_ablation_test")
    p.add_argument("--yolo-project", default="runs/mri_line_pairs_yolo11s_obb")
    p.add_argument("--template-boxes", default="template_boxes.json")
    p.add_argument("--experiments", default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--yolo-imgsz", type=int, default=1024)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-iou", type=float, default=0.50)
    p.add_argument("--max-det", type=int, default=40)
    p.add_argument("--force-data-rebuild", action="store_true")
    p.add_argument("--disable-input-visualizations", action="store_true")
    p.add_argument("--max-input-rois", type=int, default=-1)
    p.add_argument("--disable-heatmaps", action="store_true")
    p.add_argument("--max-heatmap-rois", type=int, default=-1)
    p.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.25,
        help='Grad-CAM overlay weight, 0 to 1; a lower value makes the original ROI clearer (default: 0.25).',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.heatmap_alpha) <= 1.0:
        raise ValueError('--heatmap-alpha must be between 0 and 1.')
    config = make_config(args.data_root)
    project = resolve_path(args.project)
    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    exp_ids = parse_list(args.experiments, list(EXPERIMENTS.keys()))

    print('Preparing test ROIs using the best YOLO model and template correction...')
    test_frame, yolo_audit = prepare_best_yolo_test_roi_data(
        config, yolo_project=args.yolo_project, template_boxes=args.template_boxes,
        device=args.device, yolo_imgsz=args.yolo_imgsz, yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou, max_det=args.max_det,
        force_rebuild=args.force_data_rebuild,
    )
    test_frame = test_frame.reset_index(drop=True)
    yolo_audit.to_csv(output_root / 'yolo_template_test_audit.csv', index=False, encoding="utf-8-sig")

    if not args.disable_input_visualizations:
        save_input_visualizations(test_frame, output_root, args.max_input_rois)

    summary_rows: list[dict[str, Any]] = []
    for exp_id in exp_ids:
        print(f'\n===== Independent test: {exp_id} =====')
        exp_output = output_root / exp_id
        fold_predictions: list[pd.DataFrame] = []
        fold_metric_rows: list[dict[str, Any]] = []
        cam_sum: np.ndarray | None = None
        spec = EXPERIMENTS[exp_id]

        for fold in range(1, 6):
            weight = project / exp_id / f"fold_{fold}" / "weights" / "best.pt"
            if not weight.is_file():
                raise FileNotFoundError(weight)
            pred, cams, _ckpt = predict_fold_with_cam(
                weight, test_frame, device, args.batch, args.workers,
                need_cam=not args.disable_heatmaps,
            )
            fold_predictions.append(pred)
            roi_m = binary_metrics(pred["visible"], pred['resolvability_probability'])
            image_m, _details = whole_image_metrics(pred)
            fold_metric_rows.append({'fold': fold, **roi_m, **image_m})
            if cams is not None:
                cam_sum = cams if cam_sum is None else cam_sum + cams
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        ensemble = ensemble_predictions(fold_predictions)
        roi_metrics = binary_metrics(ensemble["visible"], ensemble['resolvability_probability'])
        image_metrics, image_details = whole_image_metrics(ensemble)
        metrics = add_ci_fields({**roi_metrics, **image_metrics}, image_details, ensemble)
        metrics.update({
            'experiment_id': exp_id, 'factor': spec.factor, 'level': spec.level,
            'interpolation': spec.interpolation, 'channel': spec.channel,
            'loss': spec.loss, 'sampling': spec.sampling, 'network': spec.network,
        })
        summary_rows.append(metrics)

        fold_sheet = pd.DataFrame(fold_metric_rows)
        save_excel(exp_output / 'independent_test_report.xlsx', {
            'individual_fold_models': fold_sheet,
            'ensemble_metrics': pd.DataFrame([metrics]),
            'ensemble_roi_predictions': ensemble,
            'ensemble_image_scores': image_details,
        })
        if cam_sum is not None:
            save_experiment_heatmaps(
                exp_output, test_frame, spec, cam_sum / 5.0,
                ensemble, args.heatmap_alpha, args.max_heatmap_rois,
            )

    summary = pd.DataFrame(summary_rows)
    save_excel(output_root / 'three_configurations_test_summary.xlsx', {'test_summary': summary})
    print(f'\nAll tests completed: {output_root}')


if __name__ == "__main__":
    main()
