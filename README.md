# Drywall - CLIPSeg Fine-Tuning Pipeline

This repository contains a modular pipeline for fine-tuning CLIPSeg (`CIDAS/clipseg-rd64-refined`) to detect taping areas and cracks on drywall. 

## Requirements
Ensure you have the necessary libraries installed:
```bash
pip install torch torchvision transformers opencv-python pycocotools matplotlib pillow tqdm
```

*Important First Step*: Open ```config.py``` and ensure the ```DATA_ROOT``` and output directories contain the correct paths to your local data.

## Pipeline
4 steps in order to train and evaluate the model

### 1. Data Preparation
Convert the raw COCO ```.json``` annotations into binary ```.png``` masks required for training.
```bash
python data_prep.py
```
*Note*: This script will fill the ```output/ground_truth_masks``` directory. If it fails to find the data, double-check your ```DATA_ROOT``` path in ```config.py```.

### 2. Training
Fine-tune the CLIPSeg decoder on the generated dataset using a custom Focal-Dice loss.
```bash
python train.py
```
*Note*: This script loads the dataset with joint spatial augmentations and trains for n epochs. The resulting weights and processor are saved to ```output/clipseg-finetuned```.

### 3. Evaluate
Find the optimal sigmoid threshold and run final inference on the validation/test datasets.
```bash
python evaluate.py
```
*Note*: The script calculates the best threshold via mIoU, generates the final prediction masks, and saves them to ```output/final_test_predictions```.

### 4. Visualize
Generate a high-quality visual report comparing the Original Image, Ground Truth, and Predictions.
```bash
python visualize.py
```
*Note*: This generates a balanced grid (stratified by class) and saves it to ```output/visual_reports/final_grid_report.png```.


