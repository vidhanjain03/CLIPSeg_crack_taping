import os

# ── Reproducibility ──
SEED = 42

# ── Paths ──
# Update these to match your local or cloud environment
DATA_ROOT = r"D:\CLIPSeg_crack_taping"
DRYWALL_DIR = os.path.join(DATA_ROOT, r"Drywall-Join-Detect.v1-1.coco")
CRACKS_DIR  = os.path.join(DATA_ROOT, r"cracks.v1-1.coco")

WORK_DIR    = r"D:\CLIPSeg_crack_taping\output\ground_truth_masks"
PRED_DIR    = r"D:\CLIPSeg_crack_taping\output\final_test_predictions"
VISUALS_DIR = r"D:\CLIPSeg_crack_taping\output\visual_reports"
MODEL_OUT   = r"D:\CLIPSeg_crack_taping\output\clipseg-finetuned"

# ── Training Hyperparameters ──
MODEL_NAME = "CIDAS/clipseg-rd64-refined"
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0

# Ensure output directories exist
for d in [WORK_DIR, PRED_DIR, VISUALS_DIR, MODEL_OUT]:
    os.makedirs(d, exist_ok=True)