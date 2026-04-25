import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

# Import centralized paths from config
import config

def find_orig_image(base_id):
    """Helper to find the original RGB image across validation and test splits."""
    for d in [os.path.join(config.DRYWALL_DIR, "valid"), os.path.join(config.CRACKS_DIR, "test")]:
        p = os.path.join(d, base_id + ".jpg")
        if os.path.exists(p): 
            return p
    return None

def generate_qualitative_grid(num_per_class=4, seed=42):
    print("Assembling balanced qualitative grid...")

    # 1. Gather all Ground Truth paths from the test splits
    drywall_gts = glob(os.path.join(config.WORK_DIR, "valid", "*__segment_taping_area.png"))
    crack_gts   = glob(os.path.join(config.WORK_DIR, "test", "*__segment_crack.png"))

    if not drywall_gts or not crack_gts:
        print("Error: Could not find ground truth masks. Did you run data_prep.py?")
        return

    # 2. Select exactly N of each (Stratified Selection)
    # Using a fixed seed so it doesn't randomly change while evaluating
    random.seed(seed) 
    selected_drywall = random.sample(drywall_gts, min(num_per_class, len(drywall_gts)))
    # For cracks, using random.sample as well to ensure variety
    selected_cracks  = random.sample(crack_gts, min(num_per_class, len(crack_gts))) 

    combined_selection = selected_drywall + selected_cracks
    total_rows = len(combined_selection)

    # 3. Setup the Grid (Widened to 16 inches to prevent title overlap)
    # Dynamically scale the height based on how many rows we actually have
    fig, axes = plt.subplots(nrows=total_rows, ncols=3, figsize=(16, 3 * total_rows + 2))
    column_titles = ["Original Image", "Ground Truth", "Prediction"]

    for i, gt_path in enumerate(combined_selection):
        filename = os.path.basename(gt_path)
        base_id  = filename.split("__")[0]
        prompt   = filename.split("__")[1].replace(".png", "").replace("_", " ")

        # Load original image
        orig_path = find_orig_image(base_id)
        if orig_path is None:
            print(f"  Warning: Skipping {base_id}, original image not found.")
            continue
            
        img = Image.open(orig_path).convert("RGB")
        
        # Load ground truth mask
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Load prediction mask (with a fallback if it doesn't exist yet)
        pred_path = os.path.join(config.PRED_DIR, filename)
        if os.path.exists(pred_path):
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        else:
            print(f"  Warning: Prediction missing for {base_id}. Using blank mask.")
            pred_mask = np.zeros_like(gt_mask)

        # Plot Original
        axes[i][0].imshow(img)
        axes[i][0].axis("off")
        
        # Plot Ground Truth
        axes[i][1].imshow(gt_mask, cmap="gray", vmin=0, vmax=255)
        axes[i][1].axis("off")
        
        # Plot Prediction
        axes[i][2].imshow(pred_mask, cmap="gray", vmin=0, vmax=255)
        axes[i][2].axis("off")
        
        # Add Column Titles to the top row only
        if i == 0:
            for col in range(3):
                axes[i][col].set_title(column_titles[col], fontsize=18, pad=20, fontweight='bold')
                
        # FIX: Truncate the excessively long Roboflow IDs so they fit nicely
        short_id = base_id[:18] + "..." if len(base_id) > 18 else base_id
        
        # Add Prompt label to the far left
        axes[i][0].text(-0.05, 0.5, f"[{prompt}]\nID: {short_id}", transform=axes[i][0].transAxes, 
                        fontsize=14, va='center', ha='right', fontweight='bold')

    # FIX: Force spacing manually instead of relying entirely on tight_layout
    plt.subplots_adjust(wspace=0.1, hspace=0.4)

    # Save the final masterpiece
    save_path = os.path.join(config.VISUALS_DIR, "final_grid_report.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    print(f"\nSuccess! Grid saved to: {save_path}")

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(config.VISUALS_DIR, exist_ok=True)
    generate_qualitative_grid()