import os
import cv2
import csv
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import KidneyStoneDataset  # dùng lại dataset.py

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMAGE_DIR = "/content/drive/MyDrive/kidney_dataset/test/image"
TEST_LABEL_DIR = "/content/drive/MyDrive/kidney_dataset/test/label"
MODEL_PATH     = "/content/drive/MyDrive/unetpp_from_scratch.pth"

SAVE_DIR = "/content/drive/MyDrive/test_results"
MASK_DIR = os.path.join(SAVE_DIR, "pred_masks")
os.makedirs(MASK_DIR, exist_ok=True)

# ===============================
# DATA
# ===============================
test_dataset = KidneyStoneDataset(TEST_IMAGE_DIR, TEST_LABEL_DIR)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("DEVICE:", DEVICE)
print("Test samples:", len(test_dataset))

# ===============================
# MODEL
# ===============================
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
model.eval()

# ===============================
# METRICS
# ===============================
def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)

# ===============================
# TEST
# ===============================
dice_list, iou_list, names = [], [], []

with torch.no_grad():
    for img, mask in test_loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        pred = torch.sigmoid(model(img))

        dice_list.append(dice_score(pred, mask).item())
        iou_list.append(iou_score(pred, mask).item())

        name = test_dataset.images[len(names)]
        names.append(name)

        pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0] * 255
        cv2.imwrite(os.path.join(MASK_DIR, name), pred_mask)

# ===============================
# SAVE CSV
# ===============================
csv_path = os.path.join(SAVE_DIR, "metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "dice", "iou"])
    for n, d, i in zip(names, dice_list, iou_list):
        writer.writerow([n, d, i])

# ===============================
# RESULT
# ===============================
print("====================================")
print(f"MEAN DICE (TEST): {np.mean(dice_list):.4f}")
print(f"MEAN IoU  (TEST): {np.mean(iou_list):.4f}")
print("Saved masks to :", MASK_DIR)
print("Saved metrics :", csv_path)
print("====================================")
