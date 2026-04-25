import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import config

def convert_coco_to_binary_masks(coco_json_path, image_dir, output_dir, prompt_name):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(coco_json_path):
        print(f"  Skipping (not found): {coco_json_path}")
        return
    
    coco    = COCO(coco_json_path)
    img_ids = coco.getImgIds()
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info["file_name"]
        base_id  = os.path.splitext(img_name)[0]
        mask     = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        ann_ids  = coco.getAnnIds(imgIds=img_id)
        
        for ann in coco.loadAnns(ann_ids):
            # 1. Polygon Segmentation
            if isinstance(ann.get("segmentation"), list) and len(ann["segmentation"]) > 0 and len(ann["segmentation"][0]) > 0:
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 255)
            # 2. Bounding Box Fallback
            elif "bbox" in ann and len(ann["bbox"]) == 4:
                x, y, w, h = [int(v) for v in ann["bbox"]]
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                
        cv2.imwrite(os.path.join(output_dir, f"{base_id}__{prompt_name}.png"), mask)

if __name__ == "__main__":
    print("Generating ground-truth masks...")
    for split in ["train", "valid"]:
        convert_coco_to_binary_masks(
            os.path.join(config.DRYWALL_DIR, split, "_annotations.coco.json"),
            os.path.join(config.DRYWALL_DIR, split),
            os.path.join(config.WORK_DIR, split),
            "segment_taping_area",
        )
        convert_coco_to_binary_masks(
            os.path.join(config.CRACKS_DIR, split, "_annotations.coco.json"),
            os.path.join(config.CRACKS_DIR, split),
            os.path.join(config.WORK_DIR, split),
            "segment_crack",
        )
    convert_coco_to_binary_masks(
        os.path.join(config.CRACKS_DIR, "test", "_annotations.coco.json"),
        os.path.join(config.CRACKS_DIR, "test"),
        os.path.join(config.WORK_DIR, "test"),
        "segment_crack",
    )
    print("Ground-truth generation complete.")