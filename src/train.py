import os
import json
import torch
import mlflow
import time
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from model import UNet
from loader import SyntheticSegDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch + 1}/{epochs} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}')
    return history


def calculate_iou(preds, targets, num_classes):
    iou = []
    preds = preds.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    for cls in range(num_classes):
        intersection = np.logical_and(preds == cls, targets == cls).sum()
        union = np.logical_or(preds == cls, targets == cls).sum()
        if union == 0:
            iou.append(np.nan)  # Ignore classes not present in the batch
        else:
            iou.append(intersection / union)
    return np.nanmean(iou)  # Mean IoU across classes


def calculate_dice(preds, targets, num_classes):
    dice = []
    preds = preds.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    for cls in range(num_classes):
        intersection = np.logical_and(preds == cls, targets == cls).sum()
        total = (preds == cls).sum() + (targets == cls).sum()
        if total == 0:
            dice.append(np.nan)  # Ignore classes not present in the batch
        else:
            dice.append(2 * intersection / total)
    return np.nanmean(dice)  # Mean Dice coefficient across classes


if __name__=='__main__':
    """
    """
    train_ds = SyntheticSegDataset('../data', 'train')
    val_ds = SyntheticSegDataset('../data', 'val')

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    print('Train samples:', len(train_ds), 'Val samples:', len(val_ds))

    mlflow.set_experiment("demo-mlops")

    with mlflow.start_run():
        unet = UNet(n_classes=3, base_c=32).to(device)
        print('U-Net params:', sum(p.numel() for p in unet.parameters() if p.requires_grad))

        start = time.time()
        print("Training U-Net...")
        history_unet = train_model(unet, train_loader, val_loader, epochs=10, lr=1e-3)
        time_unet = time.time() - start

        unet.eval()
        iou_scores, dice_scores = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = unet(imgs)
                iou = calculate_iou(outputs, masks, num_classes=3)
                dice = calculate_dice(outputs, masks, num_classes=3)
                iou_scores.append(iou)
                dice_scores.append(dice)

        mean_iou = np.nanmean(iou_scores)
        mean_dice = np.nanmean(dice_scores)

        mlflow.log_metric("mean_iou", mean_iou)
        mlflow.log_metric("mean_dice", mean_dice)
        mlflow.log_metric("train_loss", history_unet['train_loss'][-1])
        mlflow.log_metric("val_loss", history_unet['val_loss'][-1])

        mlflow.pytorch.log_model(unet, "model")

        metrics = {
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "train_loss": history_unet['train_loss'][-1],
            "val_loss": history_unet['val_loss'][-1]
        }
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Model saved in 'models', mean IoU: {mean_iou:.3f}, mean Dice: {mean_dice:.3f}")
