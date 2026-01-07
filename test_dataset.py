from dataset import KidneyStoneDataset

ds = KidneyStoneDataset(
    image_dir=r"D:\kidney_dataset\data\image",
    label_dir=r"D:\kidney_dataset\data\label"
)

img, mask = ds[0]

print(img.shape)     # [1, 512, 512]
print(mask.shape)    # [1, 512, 512]
print(mask.unique()) # tensor([0., 1.])
