import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np

# defien transform pipline
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # resize image
    transforms.ToTensor(),
])

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.JPG', '_mask.png'))
        mask_path = mask_path.replace('.jpg', '_mask.png')
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # tansfer RGB mask to classes
        mask = self.rgb_to_class(mask)

        if self.transform:
            image = self.transform(image)
            mask = transforms.functional.resize(mask, (512, 512), interpolation=transforms.InterpolationMode.NEAREST)
            mask = torch.tensor(np.array(mask), dtype=torch.long)
        return image, mask

    def rgb_to_class(self, mask):
        mask = np.array(mask)
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        # map RGB matrix to class
        class_mask[(mask == [0, 0, 0]).all(axis=2)] = 0  # 黑色 -> 类别0（背景）
        class_mask[(mask == [255, 0, 0]).all(axis=2)] = 1  # 红色 -> 类别1
        class_mask[(mask == [0, 0, 255]).all(axis=2)] = 2  # 蓝色 -> 类别2

        return Image.fromarray(class_mask)




dataset = SegmentationDataset("image", "mask", transform=transform)
