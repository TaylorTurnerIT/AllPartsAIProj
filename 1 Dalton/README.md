# SDL – Symbol Detection Lab

This repository houses a lightweight pipeline for training and evaluating a ResNet-18 classifier on a custom symbol dataset.  It includes utilities for organizing raw SVG/PNG assets into train/val/test splits, generating class metadata, training the model, and running evaluations or single-image predictions.

## Features
- Scripts to turn scraped symbol assets into clean ImageFolder splits (`make_splits.py`, `organize_classes.py`, `make_classes_json.py`).
- ResNet-18 training loop (`train_model.py`) that saves weights to `symbol_classifier.pth`.
- Evaluation helper (`eval.py`) with classification report + confusion matrix.
- Command-line prediction script (`predict.py`) for quick sanity checks on individual PNGs/SVG renders.

## Requirements
- Python 3.10+
- PyTorch with torchvision (Metal Performance Shaders are used automatically on macOS if available)
- See `requirements.txt` for the exact pip packages.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparing the dataset
### Scrape & convert raw SVGs
1. **Scrape from Radica/GetVecta**
   - `python getvecta_svgs/scrape_all_stencils.py` is the most reliable full-site crawler; it walks every stencil page under `symbols.radicasoftware.com` and saves SVGs into `getvecta_svgs/`.
   - Older, more targeted options (`python scrape_symbols.py`, `python scrape_getvecta.py`) remain for quick runs but the all-stencils script should be your default.
   All scrapers deposit untouched SVGs into `getvecta_svgs/`.
2. **Convert + augment to PNGs**  
   - Use `bash svg_pipeline.sh` for the full pipeline: converts every SVG via Inkscape into multiple resolutions (`dataset/processed/png`) and runs ImageMagick augmentations into `dataset/processed/aug`. Requires Inkscape CLI + ImageMagick installed locally.
   - For a lightweight CairoSVG conversion, run `python svg_to_png.py` to place PNGs in `dataset/png/`.
3. **Organize into class folders**  
   - `python organize_classes.py` walks the processed PNGs (both base and augmented) and copies them into `dataset/classes/<class_name>/image.png` based on the filename pattern.
   - Run `python classes_cleanup.py` afterward to drop derivative folders (blur/noise/etc.) that shouldn't be treated as real classes.

### Generate metadata & splits
1. Run `python make_classes_json.py` to regenerate `classes.json`, which maps numeric IDs to class names.
2. Create stratified train/val/test splits:
   ```bash
   python make_splits.py
   ```
   This populates `dataset/splits/{train,val,test}/<class>/image.png`, matching the layout expected by TorchVision's `ImageFolder`.

> **Note:** The raw dataset, intermediate SVG dumps, and trained weights are ignored via `.gitignore`.  Each collaborator should regenerate them locally.

### 📦 YOLO SLD Dataset (Download)

Download the full dataset (images + labels):

🔗 https://drive.google.com/uc?id=YOUR_FILE_ID

Place it in your project directory and unzip:

```bash
wget -O yolo_sld_2.zip "https://drive.google.com/uc?id=1mdfe9nqis8i8UCKmTsOzKYR4SbfSLBdP&export=download"
unzip -q yolo_sld_2.zip
```

## Step-by-step workflow
Follow this order if you want a verbose roadmap from a blank checkout to a trained model:

1. **Set up Python + packages**  
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   This ensures `torch`, `torchvision`, `pillow`, `numpy`, `scikit-learn`, and `tqdm` match the versions used during development.

2. **Scrape & prepare raw assets**  
   - Run `python getvecta_svgs/scrape_all_stencils.py` (recommended) to download every SVG stencil into `getvecta_svgs/`. Legacy single-endpoint scrapers (`scrape_symbols.py`, `scrape_getvecta.py`) are available if you only need subsets.
   - Convert to PNG (with augmentations) using `bash svg_pipeline.sh` or the simpler `python svg_to_png.py`.
   - Organize files into `dataset/classes/<class_name>/` via `python organize_classes.py`, then clean unwanted pseudo-classes with `python classes_cleanup.py`.

3. **Verify folder structure**  
   Manually spot-check `dataset/classes` to ensure each symbol class has its own directory of PNGs. This structure feeds every later step, so fix naming issues now.

4. **Generate class metadata**  
   ```
   python make_classes_json.py
   ```
   This produces `classes.json`, a numeric ID → class name map consumed by `predict.py` and any downstream tooling.

5. **Build train/val/test splits**  
   ```
   python make_splits.py
   ```
   The script copies files from `dataset/classes` into `dataset/splits/{train,val,test}` with consistent ratios so that TorchVision's `ImageFolder` loaders work out of the box.

6. **Train the model**  
   ```
   python train_model.py
   ```
   Monitor the epoch logs for convergence. At the end you should see `Saved model → symbol_classifier.pth`.

7. **Evaluate on the held-out test set**  
   ```
   python eval.py
   ```
   Requires the weights from Step 5. Prints a detailed classification report plus confusion matrix to validate performance.

8. **Run ad-hoc predictions (optional sanity check)**  
   ```
   python predict.py path/to/image.png
   ```
   Confirms that the exported weights and `classes.json` load correctly and gives you top‑5 probabilities for any single symbol image.

## Training
`train_model.py` fine-tunes a pretrained ResNet-18 using the splits above.

```bash
python train_model.py
```

- Configurable constants live at the top of the script (`DATASET_DIR`, `BATCH`, `EPOCHS`, `LR`).
- The script automatically chooses `mps` on Apple Silicon, otherwise falls back to CPU.
- Model weights are saved to `symbol_classifier.pth` in the project root.

## Evaluation
Use `eval.py` to compute metrics on the held-out test split.

```bash
python eval.py
```

This will print a classification report and confusion matrix. Ensure `symbol_classifier.pth` exists (from training) before running.

## Single-image prediction
`predict.py` loads the trained weights plus `classes.json` and reports the top‑5 predictions for any PNG.

```bash
python predict.py path/to/image.png
```

## Colab training (2-class, GPU)
Use `colab_yolo_2class.ipynb` to train/evaluate YOLOv8 on the reduced two-class dataset (transformer, breaker) on a GPU (A100 recommended):

1. The repository already contains `yolo_sld_2.zip` (two-class dataset). Open the notebook in Colab, upload that zip, and run the install + upload cells. If you regenerate the dataset, re-zip it:
   ```bash
   zip -r yolo_sld_2.zip yolo_sld_2
   ```
2. Train:
   - Baseline (fast): `yolov8s` at 1024px, batch 16, 50 epochs.
   - High-accuracy (A100): `yolov8m` or `yolov8l` at 1280–1408px, batch 20–32, 120–180 epochs, with strong aug (`mosaic=1.0`, `mixup=0.1`, `copy_paste=0.1`, `cos_lr=True`, `close_mosaic=10`).
3. Validate/test: uses the held-out splits; metrics are printed after training.
4. Inference:
   - Direct: `yolo detect predict model=... data=yolo_sld_2/data.yaml imgsz=1024 conf=0.6 iou=0.45`
   - Tiled (for large SLDs): use the tiling section in the notebook to split the SLD into 1280–1408 tiles, then run predict on tiles to reduce duplicate/overlapping detections and improve small-object recall. Keep `conf` high (e.g., 0.6) to hide low-confidence boxes.

Outputs and weights are saved under `/content/runs/detect/<run_name>/`.

## Repository structure
```
images/
├── dataset/                 # Ignored – expected structure: classes/ and splits/
├── train_model.py           # ResNet-18 fine-tuning script
├── eval.py                  # Test-set evaluation helper
├── predict.py               # CLI inference utility
├── make_splits.py           # Build train/val/test folders from dataset/classes
├── make_classes_json.py     # Export class index ↔ name mapping
├── requirements.txt         # Python dependencies
├── README.md                # Project overview & instructions
└── ...                      # Additional helpers for scraping/cleanup (see file headers)
```

## Next steps
- Double-check `.gitignore` before committing to avoid uploading large datasets or checkpoints.
- Document any additional preprocessing scripts directly in their headers for future contributors.
- Consider exporting example notebooks or unit tests if you expand the pipeline further.
