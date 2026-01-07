# ===============================
# IMPORT
# ===============================
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import KidneyStoneDataset

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

# ===============================
# PATH
# ===============================
IMAGE_DIR = "/content/drive/MyDrive/kidney_dataset/data/image"
LABEL_DIR = "/content/drive/MyDrive/kidney_dataset/data/label"
SAVE_PATH = "/content/drive/MyDrive/unetpp_from_scratch.pth"

# ===============================
# DATASET & LOADER
# ===============================
dataset = KidneyStoneDataset(IMAGE_DIR, LABEL_DIR)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print("Train samples:", len(dataset))

# ===============================
# MODEL
# ===============================
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
).to(device)

# ===============================
# LOSS & OPTIMIZER
# ===============================
criterion = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ===============================
# TRAIN
# ===============================
EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    print(f"Epoch [{epoch}/{EPOCHS}]  Loss: {epoch_loss:.4f}")

# ===============================
# SAVE MODEL
# ===============================
torch.save(
    {
        "epoch": EPOCHS,
        "model_state": model.state_dict()
    },
    SAVE_PATH
)

print("✅ TRAIN DONE - MODEL SAVED")
