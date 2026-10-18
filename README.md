# MRI phantom HCSR evaluation

This program evaluates high-contrast spatial resolution (HCSR) in MRI phantom
images. YOLO11s-OBB detects 24 line-pair regions of interest (ROIs), template
correction restores their spatial arrangement, and ResNet18 classifies ROI
resolvability to determine regional and image-level scores.

## Scripts

| Script | Function |
| --- | --- |
| `dataset.py` | DICOM preprocessing, annotation parsing, ROI preparation, and data caching |
| `train.py` | Five-fold YOLO training and best-fold selection |
| `valid.py` | YOLO validation, template correction, geometry analysis, and Grad-CAM |
| `test.py` | YOLO evaluation on the independent test set |
| `roi_train.py` | Five-fold ResNet18 training |
| `roi_valid.py` | Out-of-fold ResNet18 validation |
| `roi_test.py` | Independent testing with template-corrected ROIs and five-fold probability averaging |

## Dependencies

Python, PyTorch, torchvision, Ultralytics, pydicom, NumPy, pandas, OpenCV,
matplotlib, Pillow, PyYAML, and openpyxl.

## Data

Place `template_boxes.json` beside the scripts. The data root contains
`data_1st_fold`, `data_2nd_fold`, `data_3rd_fold`, `data_4th_fold`,
`data_5th_fold`, and `data_test`. Each fold has `original` and `augmented`
subdirectories. The held-out validation fold uses only original images.

Reference CSV files are named `mri_comprehensive_annotations.csv` and contain
`filename`, `L_score`, `R_score`, and the L1-L12 and R1-R12 annotation columns.
Each annotation describes the four ROI corners or a rotated bounding box.
A fold's reference CSV can be placed inside the image subdirectory or its
parent fold directory.

Use `--data-root` to specify the dataset root.

## Classifier configurations

| ID | Loss | Sampling |
| --- | --- | --- |
| `baseline` | Class-weighted cross-entropy | Random shuffle |
| `loss_focal` | Class-weighted Focal Loss | Random shuffle |
| `sampling_boundary_hard` | Class-weighted cross-entropy | Boundary and dynamic hard-sample weighting |

All configurations use ResNet18, bilinear ROI resizing, replicated grayscale
channels, and per-fold training-set normalization. Test predictions are averaged
equally across the five fold models with a decision threshold of 0.50.

## Weights

Place the initial pretrained files beside the scripts:

- `yolo11s-obb.pt`
- `resnet18-f37072fd.pth`

Training attempts to download these files when needed. Trained checkpoints use
the following paths, relative to the script directory:

| Model or configuration | Checkpoint path |
| --- | --- |
| YOLO11s-OBB | `runs/mri_line_pairs_yolo11s_obb/fold_N/weights/best.pt` |
| Baseline | `runs/roi_ablation/baseline/fold_N/weights/best.pt` |
| Focal Loss | `runs/roi_ablation/loss_focal/fold_N/weights/best.pt` |
| Joint weighted sampling | `runs/roi_ablation/sampling_boundary_hard/fold_N/weights/best.pt` |

Here `N` is 1, 2, 3, 4, or 5. The corresponding empty directories are included.
Classifier checkpoints contain the model state, configuration, and per-fold
normalization parameters. Keep the complete `best.pt` files.

The selected YOLO fold is recorded in
`runs/mri_line_pairs_yolo11s_obb/best_yolo_model.json`, using the `best_fold`
and `weights` fields. For testing, place this record alongside the YOLO fold
directories and provide all five classifier checkpoints for each configuration.

## Run

Train and evaluate the detector:

```bash
python train.py
python valid.py
python test.py
```

Train and evaluate the classifiers:

```bash
python roi_train.py
python roi_valid.py
python roi_test.py
```

With trained checkpoints in place, run the validation or test commands directly.
Select one classifier configuration with:

```bash
python roi_test.py --experiments baseline
```

## Outputs

- `yolo_five_fold_validation.xlsx`: detector validation and best-fold results.
- `yolo_test_summary.xlsx`: independent detector test summary.
- `resnet18_three_configurations_validation.xlsx`: classifier validation results.
- `three_configurations_test_summary.xlsx`: classifier test summary.
- `independent_test_report.xlsx`: per-configuration predictions and scores.
- `data_processing_cache`: preprocessed images, annotations, and audit records.
- `classification_cache/yolo_template_rois`: template-corrected test ROIs.

Visualization outputs include ROI previews, detection comparisons, and Grad-CAM
heatmaps and overlays. Run any entry-point script with `--help` for its options.
