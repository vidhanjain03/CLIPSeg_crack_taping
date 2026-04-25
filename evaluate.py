import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

import config
from utils import calculate_metrics

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPSegProcessor.from_pretrained(config.MODEL_OUT)
    model     = CLIPSegForImageSegmentation.from_pretrained(config.MODEL_OUT).to(device)
    model.eval()

    # ── 1. Threshold tuning on validation set ──
    print("Tuning sigmoid threshold on validation set...")
    val_mask_paths = glob(os.path.join(config.WORK_DIR, "valid", "*.png"))
    img_dirs_valid = [os.path.join(config.DRYWALL_DIR, "valid"), os.path.join(config.CRACKS_DIR, "valid")]
    THRESHOLDS     = np.arange(0.3, 0.75, 0.05)
    thresh_scores  = {t: [] for t in THRESHOLDS}

    with torch.no_grad():
        for mask_path in tqdm(val_mask_paths, desc="Threshold sweep"):
            filename = os.path.basename(mask_path)
            base_id  = filename.split("__")[0]
            prompt   = filename.split("__")[1].replace(".png", "").replace("_", " ")

            img_path = next((os.path.join(d, base_id + ".jpg") for d in img_dirs_valid if os.path.exists(os.path.join(d, base_id + ".jpg"))), None)
            if img_path is None:
                continue

            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size

            inputs  = processor(text=[prompt], images=[image], padding="max_length", return_tensors="pt").to(device)
            outputs = model(**inputs)

            logits_resized = F.interpolate(
                outputs.logits.squeeze().unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False
            ).squeeze()

            probs    = torch.sigmoid(logits_resized).cpu().numpy()
            gt_mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_bin   = (gt_mask > 127).astype(np.uint8)

            for t in THRESHOLDS:
                pred_bin = (probs > t).astype(np.uint8)
                iou, _   = calculate_metrics(pred_bin, gt_bin)
                if iou is not None:
                    thresh_scores[t].append(iou)

    best_thresh = max(THRESHOLDS, key=lambda t: np.mean(thresh_scores[t]) if thresh_scores[t] else 0)
    print(f"  Best threshold: {best_thresh:.2f} (val mIoU = {np.mean(thresh_scores[best_thresh]):.4f})\n")

    # ── 2. Final unified test evaluation ──
    drywall_valid_masks = glob(os.path.join(config.WORK_DIR, "valid", "*__segment_taping_area.png"))
    cracks_test_masks   = glob(os.path.join(config.WORK_DIR, "test",  "*__segment_crack.png"))
    combined_test_paths = drywall_valid_masks + cracks_test_masks
    img_dirs_test = [os.path.join(config.DRYWALL_DIR, "valid"), os.path.join(config.CRACKS_DIR, "test")]

    metrics_by_prompt = {}
    visuals_saved     = 0
    MAX_VISUALS       = 4

    print(f"Running test inference on {len(combined_test_paths)} images...\n")
    with torch.no_grad():
        for idx, mask_path in enumerate(tqdm(combined_test_paths)):
            filename = os.path.basename(mask_path)
            base_id  = filename.split("__")[0]
            prompt   = filename.split("__")[1].replace(".png", "").replace("_", " ")

            img_path = next((os.path.join(d, base_id + ".jpg") for d in img_dirs_test if os.path.exists(os.path.join(d, base_id + ".jpg"))), None)
            if img_path is None:
                continue

            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size

            inputs  = processor(text=[prompt], images=[image], padding="max_length", return_tensors="pt").to(device)
            outputs = model(**inputs)

            logits_resized = F.interpolate(
                outputs.logits.squeeze().unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False
            ).squeeze()

            probs         = torch.sigmoid(logits_resized).cpu().numpy()
            pred_bin      = (probs > best_thresh).astype(np.uint8)
            pred_mask_255 = pred_bin * 255

            cv2.imwrite(os.path.join(config.PRED_DIR, f"{base_id}__{prompt.replace(' ', '_')}.png"), pred_mask_255)

            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_bin  = (gt_mask > 127).astype(np.uint8)
            iou, dice = calculate_metrics(pred_bin, gt_bin)

            if iou is not None:
                if prompt not in metrics_by_prompt:
                    metrics_by_prompt[prompt] = {"iou": [], "dice": []}
                metrics_by_prompt[prompt]["iou"].append(iou)
                metrics_by_prompt[prompt]["dice"].append(dice)

            step = max(1, len(combined_test_paths) // MAX_VISUALS)
            if visuals_saved < MAX_VISUALS and idx % step == 0:
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(image); axs[0].set_title(f"Original\nPrompt: '{prompt}'"); axs[0].axis("off")
                axs[1].imshow(gt_bin, cmap="gray"); axs[1].set_title("Ground Truth"); axs[1].axis("off")
                axs[2].imshow(pred_bin, cmap="gray"); axs[2].set_title(f"Prediction (IoU: {iou:.2f})" if iou else "Prediction (empty GT)"); axs[2].axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(config.VISUALS_DIR, f"report_visual_{visuals_saved}.png"), dpi=150)
                plt.close()
                visuals_saved += 1

    # ── 3. Print Results ──
    print("\n" + "=" * 55)
    print("  FINAL UNIFIED TEST METRICS")
    print(f"  Threshold used : {best_thresh:.2f}  (tuned on validation)")
    print("=" * 55)

    all_iou, all_dice = [], []
    for prompt, vals in metrics_by_prompt.items():
        ious, dices = vals["iou"], vals["dice"]
        all_iou.extend(ious)
        all_dice.extend(dices)
        print(f"\n  [{prompt}]\n    mIoU : {np.mean(ious):.4f} ± {np.std(ious):.4f}\n    Dice : {np.mean(dices):.4f} ± {np.std(dices):.4f}\n    N    : {len(ious)} non-empty images")

    print("\n  ── Combined (both prompts) ──────────────────────")
    print(f"    Combined mIoU : {np.mean(all_iou):.4f}\n    Combined Dice : {np.mean(all_dice):.4f}\n    Total samples : {len(all_iou)} non-empty images")
    print("=" * 55)

if __name__ == "__main__":
    evaluate()