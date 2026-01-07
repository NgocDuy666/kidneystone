import os
import cv2
import torch
from torch.utils.data import Dataset

class KidneyStoneDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img_path  = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.label_dir, name)

        img  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img  = img.astype("float32") / 255.0
        mask = mask.astype("float32") / 255.0

        img  = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
