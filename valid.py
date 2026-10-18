"""Validate YOLO OBB detection, template correction, geometry, and multi-scale Grad-CAM."""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


from dataset import make_config, prepare_five_fold_yolo_dataset, read_color_jpg, reports_to_text


@dataclass
class TemplateModel:
    path: Path
    class_names: list[str]
    boxes: np.ndarray
    corners: np.ndarray
    reference_width: int
    reference_height: int
    span: float
    settings: dict[str, Any]


@dataclass
class TemplateCorrectionResult:
    points: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    sources: list[str]
    matched_detection_indices: np.ndarray
    original_class_ids: np.ndarray
    center_distance_ratios: np.ndarray
    fit_reliable: bool
    fit_status: str
    scale: float
    rotation_degrees: float
    translation_x: float
    translation_y: float
    anchor_candidate_count: int
    inlier_anchor_count: int
    fit_rmse_fraction: float
    anchor_coverage_fraction: float

    def summary(self) -> dict[str, Any]:
        return {
            "fit_reliable": int(self.fit_reliable),
            "fit_status": self.fit_status,
            "template_scale": float(self.scale),
            "template_rotation_degrees": float(self.rotation_degrees),
            "template_translation_x": float(self.translation_x),
            "template_translation_y": float(self.translation_y),
            "anchor_candidate_count": int(self.anchor_candidate_count),
            "inlier_anchor_count": int(self.inlier_anchor_count),
            "fit_rmse_fraction": float(self.fit_rmse_fraction),
            "anchor_coverage_fraction": float(self.anchor_coverage_fraction),
            "matched_yolo_box_count": int(
                np.sum(self.matched_detection_indices >= 0)
            ),
            "template_filled_box_count": int(
                sum("filled" in source for source in self.sources)
            ),
            "template_relabelled_box_count": int(
                sum("relabelled" in source for source in self.sources)
            ),
            "template_corrected_box_count": int(
                sum("corrected" in source for source in self.sources)
            ),
            "corrected_box_count": int(len(self.class_ids)),
        }


def resolve_template_path(value: str | Path | None = "template_boxes.json") -> Path:
    path = Path(value).expanduser() if value else Path("template_boxes.json")
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f'24-box template not found: {path}')
    return path


def _normalize_angle_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _template_rectangle_points(box: np.ndarray) -> np.ndarray:
    cx, cy, width, height, angle_degrees = [float(value) for value in box]
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    dx = np.asarray([-width / 2, width / 2, width / 2, -width / 2])
    dy = np.asarray([-height / 2, -height / 2, height / 2, height / 2])
    return np.column_stack(
        (
            cx + dx * cosine - dy * sine,
            cy + dx * sine + dy * cosine,
        )
    ).astype(np.float64)


def _polygon_area(points: np.ndarray) -> float:
    x_values = points[:, 0]
    y_values = points[:, 1]
    return float(
        abs(
            np.dot(x_values, np.roll(y_values, 1))
            - np.dot(y_values, np.roll(x_values, 1))
        )
        * 0.5
    )


def load_template_model(value: str | Path | None = "template_boxes.json") -> TemplateModel:
    path = resolve_template_path(value)
    data = json.loads(path.read_text(encoding="utf-8"))
    expected_names = [f"L{i}" for i in range(1, 13)] + [
        f"R{i}" for i in range(1, 13)
    ]
    class_names = [str(item) for item in data["class_names"]]
    if class_names != expected_names:
        raise ValueError(
            'template_boxes.json classes must be ordered L1-L12, R1-R12.'
        )
    boxes = np.asarray(
        [data["boxes"][name] for name in class_names], dtype=np.float64
    )
    if boxes.shape != (24, 5) or not np.all(np.isfinite(boxes)):
        raise ValueError(
            'template_boxes.json must contain 24 valid [cx,cy,w,h,angle_deg] entries.'
        )
    if np.any(boxes[:, 2:4] <= 0):
        raise ValueError('Template box width and height must be positive.')
    corners = np.stack([_template_rectangle_points(box) for box in boxes])
    span_xy = np.ptp(corners.reshape(-1, 2), axis=0)
    reference_size = data.get("reference_image_size", [1024, 1024])
    if len(reference_size) != 2 or min(reference_size) <= 0:
        raise ValueError('reference_image_size must be [width,height].')
    return TemplateModel(
        path=path,
        class_names=class_names,
        boxes=boxes,
        corners=corners,
        reference_width=int(reference_size[0]),
        reference_height=int(reference_size[1]),
        span=max(float(np.linalg.norm(span_xy)), 1.0),
        settings=dict(data.get("correction", {})),
    )


def _apply_similarity(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return (float(scale) * (rotation @ points.T)).T + translation


def _pair_similarity(
    first_template: np.ndarray,
    second_template: np.ndarray,
    first_observed: np.ndarray,
    second_observed: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    template_vector = second_template - first_template
    observed_vector = second_observed - first_observed
    template_length = float(np.linalg.norm(template_vector))
    observed_length = float(np.linalg.norm(observed_vector))
    if template_length <= 1e-8 or observed_length <= 1e-8:
        return None
    scale = observed_length / template_length
    angle = math.atan2(
        observed_vector[1], observed_vector[0]
    ) - math.atan2(template_vector[1], template_vector[0])
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    translation = first_observed - scale * (rotation @ first_template)
    return scale, rotation, translation


def _weighted_similarity(
    template_points: np.ndarray,
    observed_points: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Fit translation, uniform scale, and rotation with weighted Procrustes."""
    if len(template_points) < 2:
        return None
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-6)
    weights /= float(weights.sum())
    template_mean = np.sum(template_points * weights[:, None], axis=0)
    observed_mean = np.sum(observed_points * weights[:, None], axis=0)
    template_centered = template_points - template_mean
    observed_centered = observed_points - observed_mean
    covariance = (
        observed_centered * weights[:, None]
    ).T @ template_centered
    u_matrix, singular_values, v_transpose = np.linalg.svd(covariance)
    sign = 1.0 if np.linalg.det(u_matrix @ v_transpose) >= 0 else -1.0
    rotation = u_matrix @ np.diag([1.0, sign]) @ v_transpose
    denominator = float(
        np.sum(weights * np.sum(template_centered**2, axis=1))
    )
    if denominator <= 1e-12:
        return None
    scale = float(
        np.sum(singular_values * np.asarray([1.0, sign])) / denominator
    )
    if not math.isfinite(scale) or scale <= 0:
        return None
    translation = observed_mean - scale * (rotation @ template_mean)
    return scale, rotation, translation


def _hungarian_assignment(cost: np.ndarray) -> np.ndarray:
    """Solve rectangular assignment without SciPy."""
    cost = np.asarray(cost, dtype=np.float64)
    row_count, column_count = cost.shape
    if column_count < row_count:
        raise ValueError('Hungarian matching requires at least as many columns as rows.')
    u_values = np.zeros(row_count + 1, dtype=np.float64)
    v_values = np.zeros(column_count + 1, dtype=np.float64)
    matching = np.zeros(column_count + 1, dtype=np.int64)
    path = np.zeros(column_count + 1, dtype=np.int64)

    for row in range(1, row_count + 1):
        matching[0] = row
        minimum = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = int(matching[column0])
            delta = np.inf
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                current = (
                    cost[row0 - 1, column - 1]
                    - u_values[row0]
                    - v_values[column]
                )
                if current < minimum[column]:
                    minimum[column] = current
                    path[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            if not math.isfinite(float(delta)):
                raise RuntimeError('Template matching cost matrix has no valid solution.')
            for column in range(column_count + 1):
                if used[column]:
                    u_values[matching[column]] += delta
                    v_values[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matching[column0] == 0:
                break
        while True:
            column1 = int(path[column0])
            matching[column0] = matching[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = np.full(row_count, -1, dtype=np.int64)
    for column in range(1, column_count + 1):
        if matching[column] != 0:
            assignment[matching[column] - 1] = column - 1
    return assignment


def _template_setting(
    template: TemplateModel, name: str, default: Any
) -> Any:
    return template.settings.get(name, default)


def _fallback_template_transform(
    template: TemplateModel,
    points: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    candidate_indices: list[int],
    image_width: int,
    image_height: int,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    rotation = np.eye(2, dtype=np.float64)
    if candidate_indices:
        best_index = max(
            candidate_indices, key=lambda index: float(confidences[index])
        )
        class_id = int(class_ids[best_index])
        detected_area = max(_polygon_area(points[best_index]), 1.0)
        template_area = max(
            float(template.boxes[class_id, 2] * template.boxes[class_id, 3]),
            1.0,
        )
        scale = math.sqrt(detected_area / template_area)
        scale = float(
            np.clip(
                scale,
                _template_setting(template, "minimum_scale", 0.10),
                _template_setting(template, "maximum_scale", 10.0),
            )
        )
        translation = (
            points[best_index].mean(axis=0)
            - scale * template.boxes[class_id, :2]
        )
        return scale, rotation, translation, "fallback_one_anchor"

    scale = min(
        image_width / template.reference_width,
        image_height / template.reference_height,
    )
    reference_center = np.asarray(
        [template.reference_width / 2.0, template.reference_height / 2.0]
    )
    image_center = np.asarray([image_width / 2.0, image_height / 2.0])
    translation = image_center - scale * reference_center
    return float(scale), rotation, translation, "fallback_reference_template"


def _fit_template_transform(
    template: TemplateModel,
    points: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    centers = (
        points.mean(axis=1)
        if len(points)
        else np.empty((0, 2), dtype=np.float64)
    )
    minimum_confidence = float(
        _template_setting(template, "minimum_anchor_confidence", 0.25)
    )
    maximum_per_class = int(
        _template_setting(template, "maximum_candidates_per_class", 2)
    )
    candidates_by_class: dict[int, list[int]] = {}
    for index, (confidence, class_id) in enumerate(
        zip(confidences, class_ids)
    ):
        class_id = int(class_id)
        if 0 <= class_id < 24 and float(confidence) >= minimum_confidence:
            candidates_by_class.setdefault(class_id, []).append(index)

    candidate_indices: list[int] = []
    for class_id in sorted(candidates_by_class):
        ranked = sorted(
            candidates_by_class[class_id],
            key=lambda index: float(confidences[index]),
            reverse=True,
        )
        candidate_indices.extend(ranked[:maximum_per_class])

    minimum_separation = float(
        _template_setting(template, "minimum_pair_separation_fraction", 0.12)
    ) * template.span
    inlier_fraction = float(
        _template_setting(template, "ransac_inlier_distance_fraction", 0.06)
    )
    minimum_scale = float(
        _template_setting(template, "minimum_scale", 0.10)
    )
    maximum_scale = float(
        _template_setting(template, "maximum_scale", 10.0)
    )
    maximum_rotation = float(
        _template_setting(
            template, "maximum_absolute_rotation_degrees", 180.0
        )
    )
    best_hypothesis: tuple[Any, ...] | None = None

    for first_position, first_index in enumerate(candidate_indices):
        first_class = int(class_ids[first_index])
        for second_index in candidate_indices[first_position + 1 :]:
            second_class = int(class_ids[second_index])
            if first_class == second_class:
                continue
            if (
                np.linalg.norm(
                    template.boxes[second_class, :2]
                    - template.boxes[first_class, :2]
                )
                < minimum_separation
            ):
                continue
            hypothesis = _pair_similarity(
                template.boxes[first_class, :2],
                template.boxes[second_class, :2],
                centers[first_index],
                centers[second_index],
            )
            if hypothesis is None:
                continue
            scale, rotation, translation = hypothesis
            rotation_degrees = _normalize_angle_degrees(
                math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
            )
            if (
                not minimum_scale <= scale <= maximum_scale
                or abs(rotation_degrees) > maximum_rotation
            ):
                continue

            transformed_span = max(scale * template.span, 1.0)
            chosen: list[tuple[int, int, float]] = []
            for class_id, indices in candidates_by_class.items():
                expected = _apply_similarity(
                    template.boxes[class_id : class_id + 1, :2],
                    scale,
                    rotation,
                    translation,
                )[0]
                nearest = min(
                    indices,
                    key=lambda index: float(
                        np.linalg.norm(centers[index] - expected)
                    ),
                )
                residual = float(
                    np.linalg.norm(centers[nearest] - expected)
                    / transformed_span
                )
                if residual <= inlier_fraction:
                    chosen.append((class_id, nearest, residual))
            if not chosen:
                continue

            template_inliers = template.boxes[
                [item[0] for item in chosen], :2
            ]
            coverage = (
                float(
                    np.linalg.norm(np.ptp(template_inliers, axis=0))
                    / template.span
                )
                if len(chosen) >= 2
                else 0.0
            )
            rmse = float(
                np.sqrt(np.mean([item[2] ** 2 for item in chosen]))
            )
            confidence_sum = float(
                sum(confidences[item[1]] for item in chosen)
            )
            score = (len(chosen), coverage, confidence_sum, -rmse)
            if best_hypothesis is None or score > best_hypothesis[0]:
                best_hypothesis = (
                    score,
                    scale,
                    rotation,
                    translation,
                    chosen,
                )

    if best_hypothesis is None:
        scale, rotation, translation, status = _fallback_template_transform(
            template,
            points,
            confidences,
            class_ids,
            candidate_indices,
            image_width,
            image_height,
        )
        return scale, rotation, translation, {
            "fit_reliable": False,
            "fit_status": status,
            "anchor_candidate_count": len(candidate_indices),
            "inlier_anchor_count": min(len(candidate_indices), 1),
            "fit_rmse_fraction": float("nan"),
            "anchor_coverage_fraction": 0.0,
        }

    _, scale, rotation, translation, chosen = best_hypothesis
    # Refine the RANSAC inliers with three weighted least-squares iterations.
    for _ in range(3):
        template_fit = template.boxes[
            [item[0] for item in chosen], :2
        ]
        observed_fit = centers[[item[1] for item in chosen]]
        weights = np.asarray(
            [max(float(confidences[item[1]]), 1e-3) for item in chosen]
        )
        refined = _weighted_similarity(template_fit, observed_fit, weights)
        if refined is None:
            break
        scale, rotation, translation = refined
        transformed_span = max(scale * template.span, 1.0)
        updated: list[tuple[int, int, float]] = []
        for class_id, indices in candidates_by_class.items():
            expected = _apply_similarity(
                template.boxes[class_id : class_id + 1, :2],
                scale,
                rotation,
                translation,
            )[0]
            nearest = min(
                indices,
                key=lambda index: float(
                    np.linalg.norm(centers[index] - expected)
                ),
            )
            residual = float(
                np.linalg.norm(centers[nearest] - expected)
                / transformed_span
            )
            if residual <= inlier_fraction:
                updated.append((class_id, nearest, residual))
        if len(updated) < 2:
            break
        chosen = updated

    residuals = np.asarray([item[2] for item in chosen], dtype=np.float64)
    rmse = (
        float(np.sqrt(np.mean(residuals**2)))
        if len(residuals)
        else float("inf")
    )
    template_inliers = template.boxes[[item[0] for item in chosen], :2]
    coverage = (
        float(
            np.linalg.norm(np.ptp(template_inliers, axis=0)) / template.span
        )
        if len(chosen) >= 2
        else 0.0
    )
    left_count = sum(item[0] < 12 for item in chosen)
    right_count = sum(item[0] >= 12 for item in chosen)
    rotation_degrees = _normalize_angle_degrees(
        math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    )
    minimum_each_side = int(
        _template_setting(template, "minimum_anchor_count_each_side", 2)
    )
    fit_reliable = (
        len(chosen)
        >= int(_template_setting(template, "minimum_anchor_count", 6))
        and left_count >= minimum_each_side
        and right_count >= minimum_each_side
        and coverage
        >= float(
            _template_setting(
                template, "minimum_anchor_coverage_fraction", 0.45
            )
        )
        and rmse
        <= float(
            _template_setting(template, "maximum_fit_rmse_fraction", 0.04)
        )
        and minimum_scale <= scale <= maximum_scale
        and abs(rotation_degrees) <= maximum_rotation
    )
    return scale, rotation, translation, {
        "fit_reliable": bool(fit_reliable),
        "fit_status": (
            "reliable_ransac_fit"
            if fit_reliable
            else "low_reliability_ransac_fit"
        ),
        "anchor_candidate_count": len(candidate_indices),
        "inlier_anchor_count": len(chosen),
        "fit_rmse_fraction": rmse,
        "anchor_coverage_fraction": coverage,
    }


def correct_yolo_boxes_with_template(
    template: TemplateModel,
    points: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    image_shape: tuple[int, ...],
) -> TemplateCorrectionResult:
    """Map detections to one corrected ROI for each of the 24 classes."""
    points = np.asarray(points, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if points.size == 0:
        points = np.empty((0, 4, 2), dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (4, 2):
        raise ValueError(
            f'YOLO OBB points must have shape (N,4,2); got {points.shape}'
        )
    if not (len(points) == len(confidences) == len(class_ids)):
        raise ValueError('YOLO box, confidence, and class-ID counts differ.')

    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    finite = np.all(np.isfinite(points), axis=(1, 2)) & np.isfinite(
        confidences
    )
    points = points[finite]
    confidences = confidences[finite]
    class_ids = class_ids[finite]
    scale, rotation, translation, fit = _fit_template_transform(
        template,
        points,
        confidences,
        class_ids,
        image_width,
        image_height,
    )

    corrected_points = np.stack(
        [
            _apply_similarity(item, scale, rotation, translation)
            for item in template.corners
        ]
    )
    corrected_centers = _apply_similarity(
        template.boxes[:, :2], scale, rotation, translation
    )
    corrected_widths = template.boxes[:, 2] * scale
    corrected_heights = template.boxes[:, 3] * scale

    detection_count = len(points)
    dummy_cost = float(
        _template_setting(template, "dummy_match_cost", 1.40)
    )
    cost = np.full((24, detection_count + 24), dummy_cost, dtype=np.float64)
    center_ratios = np.full(
        (24, detection_count), np.inf, dtype=np.float64
    )
    detection_centers = (
        points.mean(axis=1)
        if detection_count
        else np.empty((0, 2), dtype=np.float64)
    )
    detection_areas = np.asarray(
        [max(_polygon_area(item), 1e-6) for item in points],
        dtype=np.float64,
    )

    for class_id in range(24):
        expected_diagonal = max(
            math.hypot(
                corrected_widths[class_id], corrected_heights[class_id]
            ),
            1.0,
        )
        expected_area = max(
            corrected_widths[class_id] * corrected_heights[class_id], 1e-6
        )
        for detection_index in range(detection_count):
            center_ratio = float(
                np.linalg.norm(
                    detection_centers[detection_index]
                    - corrected_centers[class_id]
                )
                / expected_diagonal
            )
            center_ratios[class_id, detection_index] = center_ratio
            class_penalty = (
                0.0
                if int(class_ids[detection_index]) == class_id
                else float(
                    _template_setting(
                        template, "class_mismatch_penalty", 0.65
                    )
                )
            )
            size_penalty = abs(
                math.log(detection_areas[detection_index] / expected_area)
            ) * float(
                _template_setting(template, "size_penalty_weight", 0.20)
            )
            confidence_penalty = (
                1.0
                - float(np.clip(confidences[detection_index], 0.0, 1.0))
            ) * float(
                _template_setting(
                    template, "confidence_penalty_weight", 0.10
                )
            )
            cost[class_id, detection_index] = (
                center_ratio
                + class_penalty
                + size_penalty
                + confidence_penalty
            )
        # Each template ROI has a dummy assignment for a missing detection.
        cost[class_id, detection_count + class_id] = dummy_cost - 1e-6

    assignment = _hungarian_assignment(cost)
    output_confidences = np.zeros(24, dtype=np.float32)
    matched_indices = np.full(24, -1, dtype=np.int64)
    original_class_ids = np.full(24, -1, dtype=np.int64)
    output_distances = np.full(24, np.nan, dtype=np.float32)
    sources: list[str] = []
    maximum_match_distance = float(
        _template_setting(
            template, "match_max_center_box_diagonal", 1.0
        )
    )
    correction_distance = float(
        _template_setting(
            template, "correction_center_box_diagonal", 0.25
        )
    )

    for class_id, assigned_column in enumerate(assignment):
        matched = (
            0 <= assigned_column < detection_count
            and center_ratios[class_id, assigned_column]
            <= maximum_match_distance
            and cost[class_id, assigned_column] < dummy_cost
        )
        if not matched:
            source = "template_filled"
        else:
            detection_index = int(assigned_column)
            matched_indices[class_id] = detection_index
            original_class_ids[class_id] = int(class_ids[detection_index])
            output_confidences[class_id] = float(
                confidences[detection_index]
            )
            output_distances[class_id] = float(
                center_ratios[class_id, detection_index]
            )
            if int(class_ids[detection_index]) != class_id:
                source = "template_relabelled"
            elif center_ratios[class_id, detection_index] > correction_distance:
                source = "template_corrected"
            else:
                source = "template_aligned"
        if not bool(fit["fit_reliable"]):
            source = f"low_reliability_{source}"
        sources.append(source)

    rotation_degrees = _normalize_angle_degrees(
        math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    )
    return TemplateCorrectionResult(
        points=corrected_points.astype(np.float32),
        confidences=output_confidences,
        class_ids=np.arange(24, dtype=np.int64),
        sources=sources,
        matched_detection_indices=matched_indices,
        original_class_ids=original_class_ids,
        center_distance_ratios=output_distances,
        fit_reliable=bool(fit["fit_reliable"]),
        fit_status=str(fit["fit_status"]),
        scale=float(scale),
        rotation_degrees=float(rotation_degrees),
        translation_x=float(translation[0]),
        translation_y=float(translation[1]),
        anchor_candidate_count=int(fit["anchor_candidate_count"]),
        inlier_anchor_count=int(fit["inlier_anchor_count"]),
        fit_rmse_fraction=float(fit["fit_rmse_fraction"]),
        anchor_coverage_fraction=float(fit["anchor_coverage_fraction"]),
    )


def _write_jpg(path: Path, image: np.ndarray, quality: int = 95) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f'Failed to encode JPG: {path}')
    data.tofile(str(path))


def _json_safe(value: Any) -> Any:

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _collect_4d(value: Any) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []

    if torch.is_tensor(value):
        if value.ndim == 4:
            result.append(value)
        return result

    if isinstance(value, (list, tuple)):
        for item in value:
            result.extend(_collect_4d(item))

    elif isinstance(value, dict):
        for item in value.values():
            result.extend(_collect_4d(item))
    return result


class YoloGradCAM:
    """Compute multi-scale Grad-CAM on an independent model copy."""
    def __init__(self, yolo_model: Any):


        source_network = getattr(yolo_model, "model", yolo_model)
        self.network = copy.deepcopy(source_network)
        layers = getattr(self.network, "model", None)
        if layers is None or len(layers) == 0:
            raise RuntimeError('Cannot find internal Ultralytics model layers; Grad-CAM hook registration failed.')
        self.head = layers[-1]
        self.features: list[torch.Tensor] = []
        self.handle = self.head.register_forward_pre_hook(self._hook)

    def _hook(self, module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        del module
        self.features = []
        for feature in _collect_4d(inputs):
            if feature.requires_grad:
                feature.retain_grad()
            self.features.append(feature)

    def reset(self) -> None:
        self.features = []

    def close(self) -> None:
        self.handle.remove()

    @staticmethod
    def _prediction_tensor(output: Any) -> torch.Tensor:
        candidates: list[torch.Tensor] = []

        def collect(value: Any) -> None:
            if torch.is_tensor(value) and value.requires_grad and value.ndim >= 2:
                candidates.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        collect(output)
        if not candidates:
            raise RuntimeError('YOLO Grad-CAM found no differentiable prediction tensor in the forward output.')
        # Prefer the largest decoded prediction tensor with shape (B, C, N).
        three_d = [x for x in candidates if x.ndim == 3]
        pool = three_d if three_d else candidates
        return max(pool, key=lambda x: x.numel())

    def _score(self, output: Any) -> torch.Tensor:
        prediction = self._prediction_tensor(output)
        nc = int(getattr(self.head, "nc", 0) or 0)
        class_scores: torch.Tensor
        if prediction.ndim == 3 and nc > 0:

            if prediction.shape[1] >= 4 + nc:
                class_scores = prediction[:, 4:4 + nc, :]
            elif prediction.shape[2] >= 4 + nc:
                class_scores = prediction[:, :, 4:4 + nc]
            else:
                class_scores = prediction
        else:
            class_scores = prediction
        values = class_scores.reshape(-1)
        if values.numel() == 0:
            raise RuntimeError('YOLO Grad-CAM prediction tensor is empty.')
        # Use the strongest class responses as the Grad-CAM target.
        k = min(20, int(values.numel()))
        return torch.topk(values, k=k, largest=True).values.sum()

    def generate(self, image_bgr: np.ndarray, imgsz: int) -> np.ndarray:
        self.reset()
        network = self.network
        was_training = bool(network.training)
        network.eval()
        network.zero_grad(set_to_none=True)

        source_h, source_w = image_bgr.shape[:2]
        work = image_bgr
        if source_h != int(imgsz) or source_w != int(imgsz):
            work = cv2.resize(work, (int(imgsz), int(imgsz)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        array = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        tensor = torch.from_numpy(array).unsqueeze(0).float() / 255.0
        try:
            parameter = next(network.parameters())
            tensor = tensor.to(device=parameter.device, dtype=parameter.dtype)
        except StopIteration:
            tensor = tensor.to(dtype=torch.float32)

        try:

            with torch.inference_mode(False):
                tensor = tensor.detach().clone().requires_grad_(True)


                if hasattr(self.head, "shape"):
                    self.head.shape = None
                for cache_name in ("anchors", "strides"):
                    cached = getattr(self.head, cache_name, None)
                    if torch.is_tensor(cached):
                        setattr(
                            self.head,
                            cache_name,
                            torch.empty(0, device=tensor.device, dtype=tensor.dtype),
                        )

                with torch.enable_grad():
                    output = network(tensor)
                    score = self._score(output)
                    score.backward()

            maps: list[np.ndarray] = []
            for feature in self.features:
                if feature.grad is None:
                    continue
                activation = feature.detach().float()
                gradient = feature.grad.detach().float()
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
                cam = torch.nn.functional.interpolate(
                    cam, size=(source_h, source_w), mode="bilinear", align_corners=False
                ).squeeze().cpu().numpy()
                cam = np.asarray(cam, dtype=np.float32)
                cam -= float(cam.min())
                maximum = float(cam.max())
                if maximum > 0:
                    cam /= maximum
                maps.append(cam)
            if not maps:
                raise RuntimeError('YOLO Grad-CAM did not obtain valid feature gradients.')
            heatmap = np.mean(np.stack(maps, axis=0), axis=0)
            heatmap -= float(heatmap.min())
            maximum = float(heatmap.max())
            if maximum > 0:
                heatmap /= maximum
            return heatmap.astype(np.float32)
        finally:
            network.zero_grad(set_to_none=True)
            if was_training:
                network.train()


def _colorize(heatmap: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(
        np.clip(heatmap * 255, 0, 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )


_VISUAL_CONTEXT = ""
_VISUAL_STATE: dict[str, dict[str, np.ndarray]] = {}


def _information_bar(image: np.ndarray, title: str, details: str) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas = cv2.copyMakeBorder(
        image, 78, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
    )
    width = max(int(image.shape[1]), 1)
    title_scale = max(0.45, min(0.78, width / 1150.0))
    detail_scale = max(0.34, min(0.56, title_scale * 0.72))
    cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                title_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, details[:180], (12, 61), cv2.FONT_HERSHEY_SIMPLEX,
                detail_scale, (205, 205, 205), 1, cv2.LINE_AA)
    return canvas


def _fit_visual(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def _save_visuals(
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
    original = _fit_visual(original, width, height)
    template = _fit_visual(ground_truth, width, height) if ground_truth is not None else original.copy()
    prediction = _fit_visual(prediction, width, height)
    color = _fit_visual(_colorize(heatmap), width, height)
    gradcam_overlay = cv2.addWeighted(original, 1.0 - float(alpha), color, float(alpha), 0)
    context = f" | {_VISUAL_CONTEXT}" if _VISUAL_CONTEXT else ""
    details = f"File={stem}{context}"

    _write_jpg(folder / 'original_images' / f'{stem}_original.jpg',
               _information_bar(original, "Original image", details))
    _write_jpg(folder / 'template_images' / f'{stem}_template.jpg',
               _information_bar(template, "Reference/template OBB", details))
    _write_jpg(folder / 'prediction_images' / f'{stem}_prediction.jpg',
               _information_bar(prediction, "Raw YOLO prediction", details))
    _write_jpg(folder / "Grad-CAM" / f"{stem}__Grad-CAM.jpg",
               _information_bar(gradcam_overlay, "YOLO multi-scale Grad-CAM", details))

    _VISUAL_STATE[stem] = {
        "original": original,
        "template": template,
        "prediction": prediction,
        "gradcam": gradcam_overlay,
    }


def _save_corrected_visuals(
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
    corrected = _fit_visual(corrected_prediction, width, height)
    context = f" | {_VISUAL_CONTEXT}" if _VISUAL_CONTEXT else ""
    details = f"File={stem}{context}"
    corrected_panel = _information_bar(corrected, "Template-corrected 24 OBB", details)
    _write_jpg(folder / 'corrected_images' / f'{stem}_corrected.jpg', corrected_panel)

    state = _VISUAL_STATE.pop(stem, None)
    if state is None:
        raise RuntimeError(f'Cannot create the five-panel comparison; missing {stem} visualization cache.')
    panels = [
        _information_bar(state["original"], "Original image", details),
        _information_bar(state["template"], "Reference/template OBB", details),
        corrected_panel,
        _information_bar(state["prediction"], "Raw YOLO prediction", details),
        _information_bar(state["gradcam"], "YOLO multi-scale Grad-CAM", details),
    ]
    _write_jpg(
        folder / 'five_panel_comparisons' / f'{stem}_five_panel_comparison.jpg',
        np.concatenate(panels, axis=1),
    )


def _metrics_jpg(metrics: dict[str, Any], path: Path, title: str) -> None:
    entries = list(metrics.items())

    image = Image.new("RGB", (1200, max(180, 90 + 34 * len(entries))), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 25), title, fill=(0, 0, 0), font=font)
    y = 75
    for key, value in entries:

        try:
            display = f"{float(value):.6f}"
        except (TypeError, ValueError):
            display = str(value)
        draw.text((40, y), f"{key}: {display}", fill=(20, 20, 20), font=font)
        y += 34
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


def _convert_pngs(folder: Path) -> None:
    for path in folder.rglob("*.png"):
        try:
            _write_jpg(path.with_suffix(".jpg"), read_color_jpg(path))
        except Exception as error:
            print(f'Warning: {path}Failed to convert to JPG: {error}')


def _obb_arrays(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if result.obb is None or len(result.obb) == 0:
        return (
            np.empty((0, 4, 2), np.float32),
            np.empty((0, 5), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.int64),
        )
    return (
        result.obb.xyxyxyxy.detach().cpu().numpy(),
        result.obb.xywhr.detach().cpu().numpy(),
        result.obb.conf.detach().cpu().numpy(),
        result.obb.cls.detach().cpu().numpy().astype(np.int64),
    )


def draw_obb_labels_center(
    image: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    classes: np.ndarray,
    names: dict | list | None = None,
    sources: list[str] | None = None,
) -> np.ndarray:
    canvas = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.32
    thickness = 1

    for index, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
        pts = np.round(box).astype(np.int32).reshape(-1, 1, 2)

        source = sources[index] if sources is not None else "raw_yolo"
        if "filled" in source:
            color = (255, 0, 255)
            source_mark = "F"
        elif "relabelled" in source:
            color = (0, 165, 255)
            source_mark = "R"
        elif "corrected" in source:
            color = (0, 215, 255)
            source_mark = "C"
        elif "aligned" in source:
            color = (0, 255, 0)
            source_mark = "A"
        else:
            color = (0, 255, 0)
            source_mark = ""
        cv2.polylines(canvas, [pts], True, color, 1, cv2.LINE_AA)

        cx = int(box[:,0].mean())
        cy = int(box[:,1].mean())

        label = str(int(cls_id))
        if names is not None:
            try:
                label = names[int(cls_id)]
            except Exception:
                pass

        text = f"{label} {source_mark} {conf:.2f}" if source_mark else f"{label} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)

        x1 = cx - tw//2 - 3
        y1 = cy - th//2 - 3
        x2 = cx + tw//2 + 3
        y2 = cy + th//2 + base + 2

        overlay = canvas.copy()
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,0,0),-1)
        canvas = cv2.addWeighted(overlay,0.45,canvas,0.55,0)

        cv2.putText(
            canvas,
            text,
            (cx-tw//2, cy+th//2),
            font,
            font_scale,
            (255,255,255),
            thickness,
            cv2.LINE_AA
        )

    return canvas


def _read_ground_truth_obb_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> dict[int, np.ndarray]:
    if not label_path.is_file():
        raise FileNotFoundError(f'YOLO reference labels not found: {label_path}')
    result: dict[int, np.ndarray] = {}
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 9:
            raise ValueError(
                f'{label_path.name}Fold {line_number} is not a standard OBB row: {line}'
            )
        class_id = int(float(parts[0]))
        values = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
        points = values.reshape(4, 2)
        points[:, 0] *= float(image_width)
        points[:, 1] *= float(image_height)
        if class_id in result:
            raise ValueError(f'{label_path.name}Duplicate reference class: {class_id}')
        result[class_id] = points.astype(np.float32)
    return result


def _polygon_geometry(points: np.ndarray) -> dict[str, float]:
    polygon = np.asarray(points, dtype=np.float64).reshape(4, 2)
    center = polygon.mean(axis=0)
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    long_index = int(np.argmax(lengths))
    long_side = float(np.mean(np.sort(lengths)[-2:]))
    short_side = float(np.mean(np.sort(lengths)[:2]))
    vector = edges[long_index]
    angle = math.degrees(math.atan2(vector[1], vector[0])) % 180.0
    hull = cv2.convexHull(polygon.astype(np.float32))
    area = float(abs(cv2.contourArea(hull)))
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "long_side": long_side,
        "short_side": short_side,
        "angle_deg": angle,
        "area": area,
    }


def _angle_difference_degrees(first: float, second: float) -> float:
    """Return the absolute orientation difference modulo 180 degrees."""
    difference = abs(float(first) - float(second)) % 180.0
    return float(min(difference, 180.0 - difference))


def _polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Compute polygon intersection over union for two convex quadrilaterals."""
    first_hull = cv2.convexHull(
        np.asarray(first, dtype=np.float32).reshape(-1, 1, 2)
    )
    second_hull = cv2.convexHull(
        np.asarray(second, dtype=np.float32).reshape(-1, 1, 2)
    )
    first_area = float(abs(cv2.contourArea(first_hull)))
    second_area = float(abs(cv2.contourArea(second_hull)))
    if first_area <= 0 or second_area <= 0:
        return float("nan")
    intersection_area, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    union = first_area + second_area - float(intersection_area)
    if union <= 0:
        return float("nan")
    return float(intersection_area / union)


def _geometry_error_record(
    filename: str,
    class_id: int,
    class_name: str,
    path_name: str,
    truth_points: np.ndarray,
    predicted_points: np.ndarray | None,
    confidence: float | None = None,
    box_source: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "filename": filename,
        "class_id": int(class_id),
        "class_name": class_name,
        "evaluation_path": path_name,
        "matched": int(predicted_points is not None),
        "confidence": confidence,
        "box_source": box_source,
    }
    if predicted_points is None:
        row.update(
            {
                "center_error_px": np.nan,
                "angle_error_deg": np.nan,
                "long_side_relative_error_pct": np.nan,
                "short_side_relative_error_pct": np.nan,
                "area_relative_error_pct": np.nan,
                "iou": np.nan,
            }
        )
        return row

    truth_geometry = _polygon_geometry(truth_points)
    predicted_geometry = _polygon_geometry(predicted_points)
    center_error = math.hypot(
        predicted_geometry["center_x"] - truth_geometry["center_x"],
        predicted_geometry["center_y"] - truth_geometry["center_y"],
    )
    long_error = (
        abs(predicted_geometry["long_side"] - truth_geometry["long_side"])
        / max(truth_geometry["long_side"], 1e-12)
        * 100.0
    )
    short_error = (
        abs(predicted_geometry["short_side"] - truth_geometry["short_side"])
        / max(truth_geometry["short_side"], 1e-12)
        * 100.0
    )
    area_error = (
        abs(predicted_geometry["area"] - truth_geometry["area"])
        / max(truth_geometry["area"], 1e-12)
        * 100.0
    )
    row.update(
        {
            "center_error_px": float(center_error),
            "angle_error_deg": _angle_difference_degrees(
                predicted_geometry["angle_deg"], truth_geometry["angle_deg"]
            ),
            "long_side_relative_error_pct": float(long_error),
            "short_side_relative_error_pct": float(short_error),
            "area_relative_error_pct": float(area_error),
            "iou": _polygon_iou(predicted_points, truth_points),
        }
    )
    return row


def _geometry_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'evaluation_path',
        'reference_roi_count',
        'matched_roi_count',
        'matching_coverage_percent',
        'center_abs_error_mean_px',
        'center_abs_error_sd_px',
        'center_abs_error_p95_px',
        'angle_abs_error_mean_deg',
        'angle_abs_error_sd_deg',
        'angle_abs_error_p95_deg',
        'long_side_relative_error_mean_percent',
        'short_side_relative_error_mean_percent',
        'area_relative_error_mean_percent',
        'IoU_mean',
        'IoU_sd',
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for path_name, part in frame.groupby("evaluation_path", sort=False):
        valid = part[part["matched"].astype(int) == 1].copy()
        total = int(len(part))
        matched = int(len(valid))

        def mean_of(column: str) -> float:
            return float(valid[column].mean()) if matched else float("nan")

        def std_of(column: str) -> float:
            return (
                float(valid[column].std(ddof=1))
                if matched > 1
                else (0.0 if matched == 1 else float("nan"))
            )

        def p95_of(column: str) -> float:
            values = valid[column].dropna().to_numpy(dtype=float)
            return float(np.percentile(values, 95)) if len(values) else float("nan")

        rows.append(
            {
                'evaluation_path': path_name,
                'reference_roi_count': total,
                'matched_roi_count': matched,
                'matching_coverage_percent': 100.0 * matched / max(total, 1),
                'center_abs_error_mean_px': mean_of("center_error_px"),
                'center_abs_error_sd_px': std_of("center_error_px"),
                'center_abs_error_p95_px': p95_of("center_error_px"),
                'angle_abs_error_mean_deg': mean_of("angle_error_deg"),
                'angle_abs_error_sd_deg': std_of("angle_error_deg"),
                'angle_abs_error_p95_deg': p95_of("angle_error_deg"),
                'long_side_relative_error_mean_percent': mean_of("long_side_relative_error_pct"),
                'short_side_relative_error_mean_percent': mean_of("short_side_relative_error_pct"),
                'area_relative_error_mean_percent': mean_of("area_relative_error_pct"),
                'IoU_mean': mean_of("iou"),
                'IoU_sd': std_of("iou"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def evaluate_split(
    model: Any,
    dataset_yaml: Path,
    config: dict[str, Any],
    split: str,
    output_folder: Path,
    imgsz: int = 1024,
    batch: int = 4,
    device: str = "0",
    workers: int = 4,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 40,
    heatmap_alpha: float = 0.45,
    max_visualizations: int = -1,
    template_boxes: str | Path | None = "template_boxes.json",
    generate_corrected_outputs: bool = True,
) -> dict[str, Any]:
    """Evaluate official detection metrics, template correction, and ROI geometry."""
    output_folder.mkdir(parents=True, exist_ok=True)
    template_model = (
        load_template_model(template_boxes)
        if generate_corrected_outputs
        else None
    )
    if template_model is not None:
        if str(config["annotations"]["class_mode"]) != "24":
            raise RuntimeError('Fixed-template correction supports only the 24-class YOLO task.')
        print('24-box template correction enabled; template: ', template_model.path)
    else:
        print('24-box template correction disabled; raw YOLO predictions will still be saved.')


    heatmap_hook = YoloGradCAM(model)
    print("Grad-CAM engine: isolated-autograd-v4")


    metrics = model.val(
        data=str(dataset_yaml), split=split, imgsz=imgsz, batch=batch,
        device=device, workers=workers, conf=conf, iou=iou, max_det=max_det,
        plots=False, save_json=False, project=str(output_folder),
        name="official_metrics", exist_ok=True,
    )

    results_dict = dict(getattr(metrics, "results_dict", {}) or {})


    root = Path(config["data_root"]).expanduser().resolve()
    cache = root / str(config["split_folders"][split]) / str(config["cache"]["folder_name"])
    image_paths = sorted((cache / "images").glob("*.jpg"))
    gt_folder = cache / 'reference_visualizations'
    visual_folder = output_folder / 'visualizations'


    if max_visualizations < 0:
        stems = {p.stem for p in image_paths}
    else:
        stems = {p.stem for p in image_paths[:max_visualizations]}


    prediction_rows: list[dict[str, Any]] = []   
    summary_rows: list[dict[str, Any]] = []      
    corrected_prediction_rows: list[dict[str, Any]] = []
    correction_summary_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    try:

        for image_path in image_paths:
            original = read_color_jpg(image_path)

            heatmap_hook.reset()

            result = model.predict(
                source=str(image_path), imgsz=imgsz, device=device,
                conf=conf, iou=iou, max_det=max_det, verbose=False,
            )[0]

            corners, xywhr, confidences, class_ids = _obb_arrays(result)
            correction: TemplateCorrectionResult | None = None
            if template_model is not None:
                correction = correct_yolo_boxes_with_template(
                    template_model,
                    corners,
                    confidences,
                    class_ids,
                    original.shape,
                )


            label_path = cache / "labels" / f"{image_path.stem}.txt"
            ground_truth_boxes = _read_ground_truth_obb_labels(
                label_path, original.shape[1], original.shape[0]
            )
            # For geometry statistics, keep the highest-confidence detection of each class.

            best_raw_by_class: dict[int, int] = {}
            for raw_index, (raw_class, raw_confidence) in enumerate(
                zip(class_ids, confidences)
            ):
                raw_class = int(raw_class)
                previous = best_raw_by_class.get(raw_class)
                if (
                    previous is None
                    or float(raw_confidence) > float(confidences[previous])
                ):
                    best_raw_by_class[raw_class] = raw_index

            for gt_class_id, truth_points in sorted(ground_truth_boxes.items()):
                raw_index = best_raw_by_class.get(int(gt_class_id))
                geometry_rows.append(
                    _geometry_error_record(
                        image_path.name,
                        int(gt_class_id),
                        result.names[int(gt_class_id)],
                        'raw_yolo',
                        truth_points,
                        None if raw_index is None else corners[raw_index],
                        None if raw_index is None else float(confidences[raw_index]),
                        "raw_yolo" if raw_index is not None else "missing",
                    )
                )
                if correction is not None:
                    corrected_indices = np.where(
                        correction.class_ids.astype(int) == int(gt_class_id)
                    )[0]
                    if len(corrected_indices) == 1:
                        corrected_index = int(corrected_indices[0])
                        geometry_rows.append(
                            _geometry_error_record(
                                image_path.name,
                                int(gt_class_id),
                                template_model.class_names[int(gt_class_id)],
                                'after_template_correction',
                                truth_points,
                                correction.points[corrected_index],
                                float(correction.confidences[corrected_index]),
                                correction.sources[corrected_index],
                            )
                        )
                    else:
                        geometry_rows.append(
                            _geometry_error_record(
                                image_path.name,
                                int(gt_class_id),
                                template_model.class_names[int(gt_class_id)],
                                'after_template_correction',
                                truth_points,
                                None,
                                None,
                                "missing",
                            )
                        )

            counts = Counter(int(v) for v in class_ids)

            expected_ids = {int(k) for k in result.names.keys()}

            predicted_ids = {int(v) for v in class_ids}

            missing = sorted(expected_ids - predicted_ids)
            duplicate = sorted(k for k, count in counts.items() if count > 1)

            expected_count = int(config["annotations"]["expected_boxes_per_image"])


            summary_rows.append({
                "filename": image_path.name,
                "predicted_box_count": len(class_ids),
                "expected_box_count": expected_count,
                "box_count_correct": len(class_ids) == expected_count,
                "mean_confidence": float(confidences.mean()) if len(confidences) else None,
                "minimum_confidence": float(confidences.min()) if len(confidences) else None,
                "missing_class_ids": ",".join(map(str, missing)),
                "missing_class_names": ",".join(result.names[k] for k in missing),
                "duplicate_class_ids": ",".join(map(str, duplicate)),
                "duplicate_class_names": ",".join(result.names[k] for k in duplicate),
            })


            for polygon, box, confidence, class_id in zip(corners, xywhr, confidences, class_ids):
                flat = polygon.reshape(-1)
                prediction_rows.append({
                    "filename": image_path.name,
                    "class_id": int(class_id),
                    "class_name": result.names[int(class_id)],
                    "confidence": float(confidence),
                    "center_x": float(box[0]), "center_y": float(box[1]),
                    "width": float(box[2]), "height": float(box[3]),
                    "rotation_radians": float(box[4]),
                    "p1_x": float(flat[0]), "p1_y": float(flat[1]),
                    "p2_x": float(flat[2]), "p2_y": float(flat[3]),
                    "p3_x": float(flat[4]), "p3_y": float(flat[5]),
                    "p4_x": float(flat[6]), "p4_y": float(flat[7]),
                })


            if correction is not None:
                correction_summary = correction.summary()
                correction_summary.update(
                    {
                        "filename": image_path.name,
                        "raw_yolo_box_count": int(len(class_ids)),
                        "raw_unique_class_count": int(
                            len(set(map(int, class_ids)))
                        ),
                        "raw_missing_class_count": int(
                            24 - len(set(map(int, class_ids)))
                        ),
                        "raw_duplicate_class_count": int(
                            sum(value > 1 for value in counts.values())
                        ),
                    }
                )
                correction_summary_rows.append(correction_summary)
                for index, (
                    polygon,
                    confidence,
                    class_id,
                    source,
                ) in enumerate(
                    zip(
                        correction.points,
                        correction.confidences,
                        correction.class_ids,
                        correction.sources,
                    )
                ):
                    flat = polygon.reshape(-1)
                    center = polygon.mean(axis=0)
                    width = float(np.linalg.norm(polygon[1] - polygon[0]))
                    height = float(np.linalg.norm(polygon[2] - polygon[1]))
                    angle = float(
                        math.atan2(
                            polygon[1, 1] - polygon[0, 1],
                            polygon[1, 0] - polygon[0, 0],
                        )
                    )
                    matched_index = int(
                        correction.matched_detection_indices[index]
                    )
                    corrected_prediction_rows.append(
                        {
                            "filename": image_path.name,
                            "class_id": int(class_id),
                            "class_name": template_model.class_names[
                                int(class_id)
                            ],
                            "confidence": float(confidence),
                            "box_source": source,
                            "matched_raw_detection_index": matched_index,
                            "original_yolo_class_id": int(
                                correction.original_class_ids[index]
                            ),
                            "center_distance_ratio": float(
                                correction.center_distance_ratios[index]
                            ),
                            "template_fit_reliable": int(
                                correction.fit_reliable
                            ),
                            "template_fit_status": correction.fit_status,
                            "center_x": float(center[0]),
                            "center_y": float(center[1]),
                            "width": width,
                            "height": height,
                            "rotation_radians": angle,
                            "p1_x": float(flat[0]),
                            "p1_y": float(flat[1]),
                            "p2_x": float(flat[2]),
                            "p2_y": float(flat[3]),
                            "p3_x": float(flat[4]),
                            "p3_y": float(flat[5]),
                            "p4_x": float(flat[6]),
                            "p4_y": float(flat[7]),
                        }
                    )
                if not correction.fit_reliable:
                    print(
                        f'Warning: {image_path.name}Unreliable template fit; status={correction.fit_status}, inliers={correction.inlier_anchor_count}。'
                    )


            if image_path.stem in stems:
                prediction_boxes, _, prediction_confs, prediction_classes = _obb_arrays(result)
                prediction = draw_obb_labels_center(
                    original,
                    prediction_boxes,
                    prediction_confs,
                    prediction_classes,
                    getattr(model, "names", None),
                )
                try:
                    heatmap = heatmap_hook.generate(original, imgsz)
                except Exception as error:
                    print(f'Warning: {image_path.name}Grad-CAM generation failed: {error}')
                    heatmap = np.zeros(original.shape[:2], np.float32)

                gt_path = gt_folder / f'{image_path.stem}_reference.jpg'
                ground_truth = read_color_jpg(gt_path) if gt_path.exists() else None
                _save_visuals(visual_folder, image_path.stem, original, ground_truth, prediction, heatmap, heatmap_alpha)
                if correction is not None:
                    corrected_prediction = draw_obb_labels_center(
                        original,
                        correction.points,
                        correction.confidences,
                        correction.class_ids,
                        getattr(model, "names", None),
                        correction.sources,
                    )
                    _save_corrected_visuals(
                        visual_folder,
                        image_path.stem,
                        original,
                        ground_truth,
                        corrected_prediction,
                        heatmap,
                        heatmap_alpha,
                    )
    finally:

        heatmap_hook.close()


    metric_names = {
        "metrics/precision(B)": 'precision',
        "metrics/recall(B)": 'recall',
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95",
        "fitness": 'fitness',
    }
    metric_frame = pd.DataFrame(
        [
            {'metric': metric_names.get(str(key), str(key)), 'value': _json_safe(value)}
            for key, value in results_dict.items()
        ]
    )
    column_names = {
        "filename": 'filename', "class_id": 'class_id', "class_name": 'class_name',
        "confidence": 'confidence', "predicted_box_count": 'predicted_box_count', "expected_box_count": 'expected_box_count',
        "box_count_correct": 'box_count_correct', "mean_confidence": 'mean_confidence',
        "minimum_confidence": 'minimum_confidence', "missing_class_ids": 'missing_class_ids',
        "missing_class_names": 'missing_class_names', "duplicate_class_ids": 'duplicate_class_ids',
        "duplicate_class_names": 'duplicate_class_names', "box_source": 'box_source',
        "template_fit_reliable": 'template_fit_reliable', "template_fit_status": 'template_fit_status',
        "raw_yolo_box_count": 'raw_yolo_box_count', "corrected_box_count": 'corrected_box_count',
        "center_x": 'center_x', "center_y": 'center_y', "width": 'width', "height": 'height',
        "rotation_radians": 'rotation_rad', "rotation_degrees": 'rotation_deg',
        "p1_x": 'vertex1_x', "p1_y": 'vertex1_y', "p2_x": 'vertex2_x', "p2_y": 'vertex2_y',
        "p3_x": 'vertex3_x', "p3_y": 'vertex3_y', "p4_x": 'vertex4_x', "p4_y": 'vertex4_y',
        "template_fit_inlier_count": 'template_fit_inlier_count',
        "template_fit_correspondence_count": 'template_fit_correspondence_count',
        "template_fit_reprojection_error": 'template_fit_reprojection_error',
        "template_fit_scale_x": 'template_fit_scale_x', "template_fit_scale_y": 'template_fit_scale_y',
        "template_fit_rotation_degrees": 'template_fit_rotation_deg',
    }
    geometry_frame = pd.DataFrame(geometry_rows)
    geometry_summary_frame = _geometry_summary(geometry_frame)

    raw_complete_count = sum(
        int(
            int(row["predicted_box_count"]) == int(row["expected_box_count"])
            and not str(row["missing_class_ids"]).strip()
            and not str(row["duplicate_class_ids"]).strip()
        )
        for row in summary_rows
    )
    raw_complete_rate = raw_complete_count / max(len(summary_rows), 1)
    corrected_complete_rate = (
        float(
            np.mean(
                [
                    int(row.get("corrected_box_count", 0)) == int(config["annotations"]["expected_boxes_per_image"])
                    for row in correction_summary_rows
                ]
            )
        )
        if correction_summary_rows
        else float("nan")
    )
    results_dict["custom/raw_24class_complete_rate"] = float(raw_complete_rate)
    results_dict["custom/corrected_24class_complete_rate"] = float(corrected_complete_rate)

    if correction_summary_rows:
        results_dict["custom/template_filled_box_count"] = int(
            sum(int(row.get("template_filled_box_count", 0)) for row in correction_summary_rows)
        )
        results_dict["custom/template_relabelled_box_count"] = int(
            sum(int(row.get("template_relabelled_box_count", 0)) for row in correction_summary_rows)
        )
        results_dict["custom/template_corrected_box_count"] = int(
            sum(int(row.get("template_corrected_box_count", 0)) for row in correction_summary_rows)
        )

    for _, summary_row in geometry_summary_frame.iterrows():
        prefix = "raw" if summary_row['evaluation_path'] == 'raw_yolo' else "corrected"
        mapping = {
            'matching_coverage_percent': "geometry_match_coverage_pct",
            'center_abs_error_mean_px': "center_error_mean_px",
            'center_abs_error_sd_px': "center_error_sd_px",
            'center_abs_error_p95_px': "center_error_p95_px",
            'angle_abs_error_mean_deg': "angle_error_mean_deg",
            'angle_abs_error_sd_deg': "angle_error_sd_deg",
            'angle_abs_error_p95_deg': "angle_error_p95_deg",
            'long_side_relative_error_mean_percent': "long_side_error_mean_pct",
            'short_side_relative_error_mean_percent': "short_side_error_mean_pct",
            'area_relative_error_mean_percent': "area_error_mean_pct",
            'IoU_mean': "iou_mean",
            'IoU_sd': "iou_sd",
        }
        for source_column, key_suffix in mapping.items():
            results_dict[f"custom/{prefix}_{key_suffix}"] = float(summary_row[source_column])

    sheets = {
        'official_metrics': metric_frame,
        'raw_yolo_boxes': pd.DataFrame(prediction_rows).rename(columns=column_names),
        'image_summary': pd.DataFrame(summary_rows).rename(columns=column_names),
        'geometry_details': geometry_frame.rename(
            columns={
                "filename": 'filename',
                "class_id": 'class_id',
                "class_name": 'class_name',
                "evaluation_path": 'evaluation_path',
                "matched": 'matched',
                "confidence": 'confidence',
                "box_source": 'box_source',
                "center_error_px": 'center_abs_error_px',
                "angle_error_deg": 'angle_abs_error_deg',
                "long_side_relative_error_pct": 'long_side_relative_error_percent',
                "short_side_relative_error_pct": 'short_side_relative_error_percent',
                "area_relative_error_pct": 'area_relative_error_percent',
                "iou": "IoU",
            }
        ),
        'geometry_summary': geometry_summary_frame,
    }
    if template_model is not None:
        sheets['template_corrected_boxes'] = pd.DataFrame(corrected_prediction_rows).rename(columns=column_names)
        sheets['template_summary'] = pd.DataFrame(correction_summary_rows).rename(columns=column_names)
    report_path = output_folder / ('validation_results.xlsx' if split == "val" else 'test_results.xlsx')
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f'\n{split}Dataset evaluation completed; output root: {output_folder}')
    print('Evaluation results: ', report_path)
    print('Visualization and Grad-CAM directory: ', visual_folder)
    if template_model is not None:
        print('Template-correction details were included in the same workbook.')
    return results_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate YOLO11-OBB MRI line-pair detection.')
    parser.add_argument("--data-root", default=None, help='Dataset root; default: the path configured in dataset.py.')
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, 6),
        help='Held-out original fold; default: validate all five folds.',
    )
    parser.add_argument(
        "--weights",
        default=None,
        help='Validation checkpoint path; default: fold-specific best.pt.',
    )
    parser.add_argument(
        "--train-project",
        default="runs/mri_line_pairs_yolo11s_obb",
        help='Root directory of five-fold YOLO training outputs.',
    )
    parser.add_argument("--output-dir", default="runs/obb_validation", help='Evaluation output directory.')
    parser.add_argument("--imgsz", type=int, default=1024, help='Inference input image size.')
    parser.add_argument("--batch", type=int, default=4, help='Validation batch size.')
    parser.add_argument("--device", default="0", help='GPU index, or cpu.')
    parser.add_argument("--workers", type=int, default=4, help='Number of data-loader workers.')
    parser.add_argument("--conf", type=float, default=0.25, help='Detection confidence threshold.')
    parser.add_argument("--iou", type=float, default=0.5, help='NMS IoU threshold.')
    parser.add_argument("--max-det", type=int, default=40, help='Maximum detections per image.')
    parser.add_argument(
        "--template-boxes",
        default="template_boxes.json",
        help='24-box template JSON; default: the directory containing valid.py.',
    )
    parser.add_argument(
        "--disable-template-correction",
        action="store_true",
        help='Save only raw YOLO predictions, without template correction.',
    )
    parser.add_argument("--heatmap-alpha", type=float, default=0.45, help='Grad-CAM overlay weight.')
    parser.add_argument("--max-visualizations", type=int, default=-1, help='Maximum images to visualize; -1 saves all images.')
    parser.add_argument("--force-data-rebuild", action="store_true", help='Rebuild the dataset cache instead of reusing it.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    folds = [args.fold] if args.fold is not None else list(range(1, 6))


    config = make_config(args.data_root)


    from ultralytics import YOLO

    for fold in folds:
        print("\n" + "=" * 80)
        print(f'Starting fold {fold} validation')
        print("=" * 80)


        dataset_yaml, reports = prepare_five_fold_yolo_dataset(
            config, fold, force_rebuild=args.force_data_rebuild
        )
        print(reports_to_text(reports))

        if args.weights:
            weights_path = Path(args.weights).expanduser()
            if not weights_path.is_absolute():
                weights_path = Path(__file__).resolve().parent / weights_path
            weights_path = weights_path.resolve()
        else:
            train_project = Path(args.train_project).expanduser()
            if not train_project.is_absolute():
                train_project = Path(__file__).resolve().parent / train_project
            weights_path = (
                train_project.resolve()
                / f"fold_{int(fold)}"
                / "weights"
                / "best.pt"
            )

        if not weights_path.is_file():
            raise FileNotFoundError(f'Cannot find fold {fold} validation weights: {weights_path}')

        output_path = Path(args.output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent / output_path

        fold_output = output_path.resolve() / f"fold_{int(fold)}"

        global _VISUAL_CONTEXT
        _VISUAL_CONTEXT = f"Validation fold={int(fold)}"
        _VISUAL_STATE.clear()
        evaluate_split(
            YOLO(str(weights_path)),
            dataset_yaml,
            config,
            "val",
            fold_output,
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

    print('\nFive-fold validation completed.')


if __name__ == "__main__":
    main()
