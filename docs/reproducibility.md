# Reproducibility and cleanup notes

## Historical evidence

The uploaded project contained three notebooks, a PDF report, two augmentation result figures and a requirements file. The original report is preserved byte-for-byte as `project-report.pdf`. The test-example image was extracted directly from the saved main-notebook output. The two augmentation figures are unchanged apart from their filenames. `results/reported-metrics.json` transcribes the original evaluation outputs and rounded per-class values.

The primary notebook's saved validation/test mIoU outputs match Table 4.2 of the report. They were not regenerated during repository preparation. No dataset, trained weights, training-run logs or measured webcam frame rate are bundled in this repository.

## Changes in the cleaned notebooks

- Consolidated the main and overlay notebooks into one training/evaluation notebook; retained a separate webcam notebook.
- Removed embedded notebook outputs, progress bars, local paths, Colab-only download commands and duplicate function definitions. Historical figures and metrics remain separately accessible.
- Replaced the hard-coded Kaggle runtime path with the actual `kagglehub` download path or `EASYPORTRAIT_DIR`.
- Paired JPEG images with PNG masks by filename stem and ignored metadata files. The original output counted one extra entry in the training mask directory; blindly pairing directory listings could fail.
- Preserved the original U-Net layer architecture, Adam learning rate, image dimensions, nine-class output, normalization and nearest-neighbor mask resize.
- Replaced repeated interactive training calls with one configurable 18-epoch run; added an explicit seed and full-dataset shuffle buffer. These are new-run defaults and can change results relative to the original session.
- Modernized the custom metric reset method, registered metric serialization, and used safe division for zero-union classes. Zero-union classes still contribute zero to the nine-class mean.
- Enabled saving a `.keras` checkpoint, with a guard against overwriting an existing checkpoint, and export of fresh evaluation metrics.
- Corrected webcam BGR/RGB conversion, added camera-open checks and guaranteed camera/window cleanup. No webcam session was opened during repository preparation.
- Added the missing OpenCV and notebook dependencies; removed the need for automatic `gdown` downloads. The source checkpoint URL is documented for manual use.

## Limits of exact reproduction

The original main notebook schedules training in 13 + 2 + 2 + 3 epoch phases, while the last saved phase is interrupted. The report describes 18 epochs, and saved evaluation cells may reflect a later interactive state. A unique, fully reproducible training history cannot be reconstructed from these materials. The reported per-class evaluation is retained as historical evidence, while new runs must report their own metrics.

Table 4.1 and the saved progress output differ on the training mIoU associated with the final reported epoch. The README therefore emphasizes the separate validation/test evaluation from Table 4.2 rather than selecting a training metric from inconsistent progress logs.

The main notebook uses a 20% horizontal-flip probability; the alternate overlay notebook uses 15%. Both saved dataset-construction cells disable augmentation. The supplied five-epoch augmentation figures and table are historical comparisons, not evidence that running the saved notebook unchanged recreates both conditions. The cleaned notebook keeps the main implementation's flip operation with augmentation disabled by default. Before augmentation experiments, verify how the dataset defines left/right brows and eyes and whether label swapping is required by that convention.

The original PDF contains broad claims about generalization and real-time applicability. This repository does not interpret those as statistical significance, measured deployment throughput or a state-of-the-art comparison.

## Validation scope

Repository preparation checks notebook JSON, Python syntax, image/mask pairing behavior, local documentation links, and webcam RGB conversion/resource cleanup using simulated frames and inference failures. Full TensorFlow training, checkpoint loading and physical webcam inference have not been rerun. Use the documented Python 3.11 environment to execute those workflows.
