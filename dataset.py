import os
import random
import cv2
import numpy as np
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class PromptedSegDataset(Dataset):
    def __init__(self, mask_dir, img_dirs, processor, augment=False):
        self.mask_paths = glob(os.path.join(mask_dir, "*.png"))
        self.img_dirs   = img_dirs
        self.processor  = processor
        self.augment    = augment
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)

    def __len__(self):
        return len(self.mask_paths)

    def _find_image(self, base_id):
        for d in self.img_dirs:
            p = os.path.join(d, base_id + ".jpg")
            if os.path.exists(p):
                return p
        candidates = glob(os.path.join(self.img_dirs[0], base_id + ".*"))
        return candidates[0] if candidates else None

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        filename  = os.path.basename(mask_path)
        base_id   = filename.split("__")[0]
        prompt    = filename.split("__")[1].replace(".png", "").replace("_", " ")

        img_path = self._find_image(base_id)
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {base_id}")

        image    = Image.open(img_path).convert("RGB")
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_pil = Image.fromarray(mask_arr)

        if self.augment:
            if random.random() > 0.5:
                image    = TF.hflip(image)
                mask_pil = TF.hflip(mask_pil)
            angle    = random.uniform(-15, 15)
            image    = TF.rotate(image, angle)
            mask_pil = TF.rotate(mask_pil, angle)
            image = self.color_jitter(image)

        mask_arr = np.array(mask_pil)
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding="max_length",
            return_tensors="pt",
        )
        mask_resized = cv2.resize(mask_arr, (352, 352), interpolation=cv2.INTER_NEAREST)
        mask_tensor  = torch.tensor(mask_resized, dtype=torch.float32) / 255.0

        return {
            "pixel_values":   inputs.pixel_values.squeeze(0),
            "input_ids":      inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels":         mask_tensor,
        }