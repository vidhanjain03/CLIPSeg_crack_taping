import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm

import config
from utils import set_seed
from dataset import PromptedSegDataset
from loss import FocalDiceLoss

def train():
    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPSegProcessor.from_pretrained(config.MODEL_NAME)
    model     = CLIPSegForImageSegmentation.from_pretrained(config.MODEL_NAME).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    base_model = model.module if hasattr(model, "module") else model

    # Freeze CLIP backbone
    for param in base_model.clip.parameters():
        param.requires_grad = False

    # DataLoaders
    img_dirs_train = [os.path.join(config.DRYWALL_DIR, "train"), os.path.join(config.CRACKS_DIR, "train")]
    train_dataset = PromptedSegDataset(os.path.join(config.WORK_DIR, "train"), img_dirs_train, processor, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    # Optimisation
    optimizer = torch.optim.AdamW(base_model.decoder.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    criterion = FocalDiceLoss().to(device)

    print(f"Starting training for {config.NUM_EPOCHS} epochs...\n")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            pixel_values   = batch["pixel_values"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.decoder.parameters(), max_norm=config.MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}\n")

    base_model.save_pretrained(config.MODEL_OUT)
    processor.save_pretrained(config.MODEL_OUT)
    print(f"Model saved to {config.MODEL_OUT}\n")

if __name__ == "__main__":
    train()