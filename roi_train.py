"""Train baseline, Focal Loss, and boundary-hard sampling ResNet18 models on reference ROIs."""
from __future__ import annotations


import argparse
import contextlib
import gc
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import resnet18

from dataset import (
    extract_obb_roi_no_rotation_resize_with_geometry,
    make_config,
    pad_roi_to_square,
    prepare_five_fold_roi_data,
)

CLASS_NAMES = ["invisible", "visible"]
VISIBLE_THRESHOLD = 0.50
IMAGE_SIZE = 224
ROI_PADDING_RATIO = 0.03
DEFAULT_SEED = 42

INTERPOLATIONS = {"bilinear": ("Bilinear interpolation", cv2.INTER_LINEAR)}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    factor: str
    level: str
    interpolation: str = "bilinear"
    channel: str = "gray3"
    loss: str = "wce"
    sampling: str = "random"
    network: str = "resnet18"


BASELINE = ExperimentSpec("baseline", 'Baseline', 'Baseline')
EXPERIMENTS: dict[str, ExperimentSpec] = {
    "baseline": BASELINE,
    "loss_focal": ExperimentSpec("loss_focal", 'Loss function', "Focal", loss="focal"),
    "sampling_boundary_hard": ExperimentSpec(
        "sampling_boundary_hard", 'Sampling strategy', 'Boundary and hard samples', sampling="boundary_hard"
    ),
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def parse_list(value: str | None, allowed: Sequence[str], default_all: bool = True) -> list[str]:
    if value is None:
        return list(allowed) if default_all else []
    items = [x.strip() for x in str(value).replace("，", ",").split(",") if x.strip()]
    unknown = [x for x in items if x not in allowed]
    if unknown:
        raise ValueError(f'Unknown options: {unknown}; available: {list(allowed)}')
    return list(dict.fromkeys(items))


def parse_folds(value: str) -> list[int]:
    folds = sorted({int(x.strip()) for x in str(value).replace("，", ",").split(",") if x.strip()})
    if not folds or any(x not in range(1, 6) for x in folds):
        raise ValueError('folds must be between 1 and 5.')
    return folds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def resolve_device(value: str) -> torch.device:
    text = str(value).strip()
    if text.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if text.startswith("cuda:"):
        return torch.device(text)
    return torch.device(f"cuda:{text}")


def interpolation_code(name: str) -> int:
    if name not in INTERPOLATIONS:
        raise KeyError(name)
    return int(INTERPOLATIONS[name][1])


def _border_mean(image: np.ndarray) -> float:
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    return float(np.mean(border))


def source_points_from_row(row: pd.Series) -> np.ndarray:
    values = []
    for i in range(1, 5):
        values.append([float(row[f"source_p{i}_x"]), float(row[f"source_p{i}_y"])])
    return np.asarray(values, dtype=np.float32)


@lru_cache(maxsize=128)
def _read_gray_png(path_text: str) -> np.ndarray:
    data = np.fromfile(path_text, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f'Failed to read image: {path_text}')
    return image


def construct_channels(canvas: np.ndarray, channel: str) -> np.ndarray:
    if channel != "gray3":
        raise ValueError("Only replicated three-channel grayscale input is supported.")
    x = canvas.astype(np.float32) / 255.0
    return np.stack([x, x, x], axis=0)


def prepare_input_from_row(
    row: pd.Series,
    interpolation: str,
    channel: str,
    image_size: int = IMAGE_SIZE,
    padding_ratio: float = ROI_PADDING_RATIO,
) -> np.ndarray:
    source = _read_gray_png(str(row["source_image_png"]))
    roi, _ = extract_obb_roi_no_rotation_resize_with_geometry(
        source,
        source_points_from_row(row),
        size=int(image_size),
        padding_ratio=float(padding_ratio),
        interpolation=interpolation_code(interpolation),
    )
    background = int(round(_border_mean(roi)))
    canvas = pad_roi_to_square(roi, size=int(image_size), background_value=background)
    return construct_channels(canvas, channel)


class RoiDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        spec: ExperimentSpec,
        mean: Sequence[float],
        std: Sequence[float],
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.spec = spec
        self.image_size = int(image_size)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
        self.std = np.maximum(self.std, 1e-6)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[int(index)]
        image = prepare_input_from_row(
            row,
            self.spec.interpolation,
            self.spec.channel,
            self.image_size,
        )
        image = (image - self.mean) / self.std
        return (
            torch.from_numpy(image.astype(np.float32)),
            torch.tensor(int(row["visible"]), dtype=torch.long),
            torch.tensor(int(row["level"]), dtype=torch.long),
            torch.tensor(int(index), dtype=torch.long),
        )


def normalization_cache_path(project: Path, fold: int, spec: ExperimentSpec) -> Path:
    return project / "normalization" / f"fold_{fold}_{spec.interpolation}_{spec.channel}.json"


def compute_normalization(
    frame: pd.DataFrame,
    spec: ExperimentSpec,
    project: Path,
    fold: int,
) -> tuple[list[float], list[float]]:
    cache = normalization_cache_path(project, fold, spec)
    if cache.is_file():
        record = json.loads(cache.read_text(encoding="utf-8"))
        return list(map(float, record["mean"])), list(map(float, record["std"]))
    count = 0
    sum_c: np.ndarray | None = None
    sq_c: np.ndarray | None = None
    for _, row in frame.iterrows():
        arr = prepare_input_from_row(row, spec.interpolation, spec.channel, IMAGE_SIZE)
        c = arr.shape[0]
        pixels = arr.reshape(c, -1).astype(np.float64)
        if sum_c is None:
            sum_c = np.zeros(c, dtype=np.float64)
            sq_c = np.zeros(c, dtype=np.float64)
        sum_c += pixels.sum(axis=1)
        sq_c += np.square(pixels).sum(axis=1)
        count += pixels.shape[1]
    assert sum_c is not None and sq_c is not None and count > 0
    mean = sum_c / count
    var = np.maximum(sq_c / count - mean * mean, 1e-12)
    std = np.sqrt(var)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return mean.tolist(), std.tolist()


def _load_pretrained_state() -> dict[str, torch.Tensor]:
    import urllib.request
    local = Path(__file__).resolve().parent / "resnet18-f37072fd.pth"
    if not local.exists():
        print(f'Downloading ResNet18 ImageNet pretrained weights: {local}')
        urllib.request.urlretrieve(
            "https://download.pytorch.org/models/resnet18-f37072fd.pth", local
        )
    try:
        state = torch.load(local, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(local, map_location="cpu")
    return dict(state)


def build_model(spec: ExperimentSpec, pretrained: bool = True, dropout: float = 0.30) -> nn.Module:
    if spec.experiment_id not in EXPERIMENTS:
        raise ValueError(f"Unsupported experiment: {spec.experiment_id}")
    expected = EXPERIMENTS[spec.experiment_id]
    for field in ("interpolation", "channel", "loss", "sampling", "network"):
        if getattr(spec, field) != getattr(expected, field):
            raise ValueError(f"Unsupported {field} for {spec.experiment_id}")
    model = resnet18(weights=None)
    state = _load_pretrained_state() if pretrained else None
    if state is not None:
        state.pop("fc.weight", None)
        state.pop("fc.bias", None)
        model.load_state_dict(state, strict=False)
    model.fc = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(512, 2))
    model.experiment_id = spec.experiment_id
    return model


def class_loss_weights(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = frame["visible"].astype(int).value_counts().reindex([0, 1], fill_value=0).to_numpy(dtype=np.float64)
    if np.any(counts == 0):
        raise ValueError('Training data must include both resolvable and unresolvable ROIs.')
    return torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)


def boundary_flags(frame: pd.DataFrame) -> np.ndarray:
    level = frame["level"].astype(int).to_numpy()
    score = frame["group_score"].astype(int).to_numpy()
    return ((level == score) | (level == score + 1)).astype(np.float64)


def per_sample_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    class_weights: torch.Tensor,
    focal_gamma: float,
) -> torch.Tensor:
    if loss_name not in {"wce", "focal"}:
        raise ValueError(f"Unsupported loss: {loss_name}")
    losses = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    if loss_name == "focal":
        pt = torch.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        losses = losses * torch.pow(1.0 - pt, float(focal_gamma))
    return losses


def initial_sampling_weights(frame: pd.DataFrame, sampling: str, boundary_factor: float) -> np.ndarray | None:
    if sampling == "random":
        return None
    if sampling != "boundary_hard":
        raise ValueError(f"Unsupported sampling: {sampling}")
    weights = np.ones(len(frame), dtype=np.float64)
    weights *= np.where(boundary_flags(frame) > 0, float(boundary_factor), 1.0)
    return weights


def create_loader(
    dataset: RoiDataset,
    batch: int,
    workers: int,
    training: bool,
    device: torch.device,
    sample_weights: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    shuffle = False
    if training:
        if sample_weights is None:
            shuffle = True
        else:
            sampler = WeightedRandomSampler(
                torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(dataset), replacement=True, generator=generator
            )
    return DataLoader(
        dataset, batch_size=int(batch), shuffle=shuffle, sampler=sampler,
        num_workers=int(workers), pin_memory=device.type == "cuda", generator=generator
    )


def continuous_score(levels: Sequence[int], labels: Sequence[int]) -> int:
    mapping = {int(l): int(y) for l, y in zip(levels, labels)}
    score = 0
    for level in range(1, 13):
        if mapping.get(level, 0) != 1:
            break
        score = level
    return score


def whole_image_metrics(predictions: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    key = "image_key" if "image_key" in predictions.columns else "filename"
    rows: list[dict[str, Any]] = []
    for image_key, frame in predictions.groupby(key, sort=False):
        row: dict[str, Any] = {'image_key': image_key, 'filename': str(frame.iloc[0]["filename"])}
        for group in ("L", "R"):
            part = frame[frame["group"] == group].sort_values("level")
            true_score = int(part["group_score"].iloc[0])
            pred_score = continuous_score(part["level"], part['predicted_class'])
            row[f'{group}_reference_score'] = true_score
            row[f'{group}_predicted_score'] = pred_score
            row[f'{group}_score_correct'] = int(true_score == pred_score)
            row[f'{group}_absolute_error'] = abs(true_score - pred_score)
        row['image_reference_score'] = max(row['L_reference_score'], row['R_reference_score'])
        row['image_predicted_score'] = max(row['L_predicted_score'], row['R_predicted_score'])
        row['image_score_correct'] = int(row['image_reference_score'] == row['image_predicted_score'])
        row['L_and_R_correct'] = int(row['L_score_correct'] and row['R_score_correct'])
        true = frame["visible"].astype(int).to_numpy()
        pred = frame['predicted_class'].astype(int).to_numpy()
        row['all_24_rois_correct'] = int(np.array_equal(true, pred))
        row['ROI_accuracy'] = float(np.mean(true == pred))
        rows.append(row)
    details = pd.DataFrame(rows)
    return {
        'L_score_accuracy': float(details['L_score_correct'].mean()),
        'R_score_accuracy': float(details['R_score_correct'].mean()),
        'L_and_R_accuracy': float(details['L_and_R_correct'].mean()),
        'image_score_accuracy': float(details['image_score_correct'].mean()),
        'all_24_rois_accuracy': float(details['all_24_rois_correct'].mean()),
        'mean_image_roi_accuracy': float(details['ROI_accuracy'].mean()),
        "L_MAE": float(details['L_absolute_error'].mean()),
        "R_MAE": float(details['R_absolute_error'].mean()),
    }, details


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    pred = (p >= VISIBLE_THRESHOLD).astype(np.int64)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    accuracy = (tp + tn) / max(len(y), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    if np.unique(y).size == 2:
        ranks = pd.Series(p).rank(method="average").to_numpy(dtype=float)
        n1 = int(np.sum(y == 1)); n0 = int(np.sum(y == 0))
        auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    else:
        auc = float("nan")
    return {
        'ROI_accuracy': float(accuracy), 'ROI_precision': float(precision),
        'ROI_sensitivity': float(recall), 'ROI_specificity': float(specificity),
        "ROI_F1": float(f1), "ROI_AUC": float(auc),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    spec: ExperimentSpec,
    class_weights: torch.Tensor,
    focal_gamma: float,
    boundary_factor: float,
) -> tuple[float, pd.DataFrame, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    probs_all = np.zeros(len(loader.dataset), dtype=np.float64)
    labels_all = np.zeros(len(loader.dataset), dtype=np.int64)
    for images, labels, _, indices in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        idx = indices.numpy()
        logits = model(images)
        sample_losses = per_sample_loss(
            logits, labels_device, spec.loss, class_weights, focal_gamma
        )
        probs = torch.softmax(logits, dim=1)[:, 1]
        pred = (probs >= VISIBLE_THRESHOLD).long()
        losses.extend(sample_losses.detach().cpu().numpy().tolist())
        probs_all[idx] = probs.detach().cpu().numpy()
        labels_all[idx] = labels.numpy()
        for j, row_idx in enumerate(idx):
            source = loader.dataset.frame.iloc[int(row_idx)]
            rows.append({
                "image_key": source["image_key"], "filename": source["filename"],
                "group": source["group"], "level": int(source["level"]),
                "identity": source["identity"], "group_score": int(source["group_score"]),
                "visible": int(labels[j]), 'predicted_class': int(pred[j].cpu()),
                'resolvability_probability': float(probs[j].cpu()),
            })
    return float(np.mean(losses)), pd.DataFrame(rows), probs_all, labels_all


def _autocast(enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def save_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    from openpyxl.styles import Alignment, Font
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.book[name[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center")
            for col in ws.columns:
                width = min(max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2, 40)
                ws.column_dimensions[col[0].column_letter].width = max(width, 10)


def save_training_plot(history: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['epoch'], history['training_loss'], label="Train loss")
    ax.plot(history['epoch'], history['validation_loss'], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    spec: ExperimentSpec, fold: int, epoch: int, mean: Sequence[float], std: Sequence[float],
    args: argparse.Namespace, metrics: dict[str, float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "experiment": asdict(spec), "fold": int(fold), "epoch": int(epoch),
        "normalization_mean": list(map(float, mean)), "normalization_std": list(map(float, std)),
        "image_size": IMAGE_SIZE, "roi_padding_ratio": ROI_PADDING_RATIO,
        "dropout": float(args.dropout), "pretrained": not args.no_pretrained,
        "metrics": metrics,
    }, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any], ExperimentSpec]:
    path = resolve_path(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    spec = ExperimentSpec(**ckpt["experiment"])
    if spec.experiment_id in EXPERIMENTS:
        current = EXPERIMENTS[spec.experiment_id]
        spec = ExperimentSpec(**{**asdict(spec), 'factor': current.factor, 'level': current.level})
    model = build_model(spec, pretrained=False, dropout=float(ckpt.get("dropout", 0.30)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt, spec


def train_one_fold(
    spec: ExperimentSpec, fold: int, train_frame: pd.DataFrame, val_frame: pd.DataFrame,
    device: torch.device, project: Path, args: argparse.Namespace
) -> dict[str, Any]:
    output = project / spec.experiment_id / f"fold_{fold}"
    best_path = output / "weights" / "best.pt"
    if args.skip_completed and best_path.is_file():
        print(f'Skipping completed model: {spec.experiment_id} fold_{fold}')
        return {'experiment_id': spec.experiment_id, 'fold': fold, 'status': 'already_exists'}

    set_seed(args.seed + fold)
    mean, std = compute_normalization(train_frame, spec, project, fold)
    train_ds = RoiDataset(train_frame, spec, mean, std)
    val_ds = RoiDataset(val_frame, spec, mean, std)
    class_weights = class_loss_weights(train_frame, device)
    model = build_model(spec, pretrained=not args.no_pretrained, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-7)
    amp = bool(args.amp and device.type == "cuda")
    scaler = _grad_scaler(amp)

    base_sampling = initial_sampling_weights(train_frame, spec.sampling, args.boundary_factor)
    current_sampling = None if base_sampling is None else base_sampling.copy()
    error_memory = np.full(len(train_ds), 0.5, dtype=np.float64)
    train_eval_loader = create_loader(train_ds, args.batch, args.workers, False, device)
    val_loader = create_loader(val_ds, args.batch, args.workers, False, device)

    best_rank = (-1.0, -1.0, -1.0, -math.inf)
    best_epoch = 0
    no_improve = 0
    history_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loader = create_loader(
            train_ds, args.batch, args.workers, True, device, current_sampling,
            seed=args.seed + fold * 1000 + epoch,
        )
        model.train()
        train_losses: list[float] = []
        for images, labels, _, indices in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            idx = indices.numpy()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp):
                logits = model(images)
                loss = per_sample_loss(
                    logits, labels, spec.loss, class_weights, args.focal_gamma
                ).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        val_loss, val_pred, _, _ = evaluate(
            model, val_loader, device, spec, class_weights,
            args.focal_gamma, args.boundary_factor
        )
        scheduler.step(val_loss)
        roi_metrics = binary_metrics(val_pred["visible"], val_pred['resolvability_probability'])
        image_metrics, image_details = whole_image_metrics(val_pred)
        row = {
            'epoch': epoch, 'learning_rate': float(optimizer.param_groups[0]["lr"]),
            'training_loss': float(np.mean(train_losses)), 'validation_loss': float(val_loss),
            **roi_metrics, **image_metrics,
        }
        history_rows.append(row)
        rank = (
            image_metrics['L_and_R_accuracy'],
            image_metrics['image_score_accuracy'],
            image_metrics['mean_image_roi_accuracy'],
            -float(val_loss),
        )
        if rank > best_rank:
            best_rank = rank; best_epoch = epoch; no_improve = 0
            save_checkpoint(best_path, model, optimizer, spec, fold, epoch, mean, std, args, row)
            save_excel(output / 'best_epoch_validation.xlsx', {'roi_predictions': val_pred, 'image_scores': image_details})
        else:
            no_improve += 1

        if spec.sampling == "boundary_hard":
            _, _, probs, labels_np = evaluate(
                model, train_eval_loader, device, spec, class_weights,
                args.focal_gamma, args.boundary_factor
            )
            errors = np.abs(labels_np.astype(np.float64) - probs)
            error_memory = 0.7 * error_memory + 0.3 * errors
            boundary_base = np.ones(len(train_ds), dtype=np.float64)
            boundary_base *= np.where(boundary_flags(train_frame) > 0, args.boundary_factor, 1.0)
            current_sampling = boundary_base * (1.0 + args.hard_factor * error_memory)

        print(
            f"{spec.experiment_id} fold_{fold} | Epoch {epoch:03d}/{args.epochs} | train={np.mean(train_losses):.4f} val={val_loss:.4f} ROI={roi_metrics['ROI_accuracy']:.4f} L&R={image_metrics['L_and_R_accuracy']:.4f}"
        )
        if no_improve >= args.patience:
            print(f'Early stopping after {args.patience} epochs without improvement.')
            break

    history = pd.DataFrame(history_rows)
    save_excel(output / 'training_history.xlsx', {'training_history': history})
    save_training_plot(history, output / 'training_curves.png')
    (output / 'experiment_config.json').write_text(
        json.dumps({"experiment": asdict(spec), "fold": fold, "best_epoch": best_epoch}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        'experiment_id': spec.experiment_id, 'factor': spec.factor, 'level': spec.level,
        'fold': int(fold), 'best_epoch': int(best_epoch),
        'training_time_s': float(time.perf_counter() - started), 'weights': str(best_path), 'status': 'completed',
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Five-fold ResNet18 training for the three manuscript configurations")
    p.add_argument("--data-root", default=None)
    p.add_argument("--project", default="runs/roi_ablation")
    p.add_argument("--experiments", default=None, help="Comma-separated IDs; defaults to baseline, loss_focal, sampling_boundary_hard")
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--no-summary", action="store_true")
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help='Prepare or inspect reference ROI caches for selected folds and exit without training ResNet18.',
    )
    p.add_argument("--force-data-rebuild", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--boundary-factor", type=float, default=2.0)
    p.add_argument("--hard-factor", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config(args.data_root)
    project = resolve_path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    experiment_ids = parse_list(args.experiments, list(EXPERIMENTS.keys()))
    folds = parse_folds(args.folds)


    # Train on four folds with augmentation; validate on original images of the held-out fold.
    print('Preparing or loading reference ROI caches...')
    fold_data: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold in folds:
        train_frame, val_frame, _reports = prepare_five_fold_roi_data(
            config,
            validation_fold=fold,
            force_rebuild=args.force_data_rebuild,
        )
        train_frame = train_frame.reset_index(drop=True)
        val_frame = val_frame.reset_index(drop=True)
        fold_data[int(fold)] = (train_frame, val_frame)
        print(
            f"Fold {fold} reference ROIs: training={len(train_frame)}, validation={len(val_frame)}; validation images={val_frame['image_key'].nunique()}."
        )

    if args.prepare_only:
        print('Reference ROI cache preparation or inspection completed.')
        return

    rows: list[dict[str, Any]] = []
    for exp_id in experiment_ids:
        spec = EXPERIMENTS[exp_id]
        print(f'\n===== Experiment {exp_id}: {spec.factor} / {spec.level} =====')
        for fold in folds:
            train_frame, val_frame = fold_data[int(fold)]
            rows.append(train_one_fold(spec, fold, train_frame, val_frame, device, project, args))
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if rows and not args.no_summary:
        save_excel(project / 'training_execution_summary.xlsx', {'execution_summary': pd.DataFrame(rows)})
    print("\nThree-configuration training complete. Training and validation use reference ROIs.")


if __name__ == "__main__":
    main()
