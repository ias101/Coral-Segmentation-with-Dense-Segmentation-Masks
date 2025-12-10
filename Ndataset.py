import os
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # resize picture
    transforms.ToTensor(),          # Transfer to Tensor
])


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        # Adjust mask path
        mask_path = mask_path.replace('.JPG', '_mask.png')
        mask_path = mask_path.replace('.jpg', '_mask.png')

        # Open image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Preprocess the input image using feature_extractor
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # Convert mask to a NumPy array then to a Tensor
        mask = np.array(mask)
        mask = torch.tensor(mask, dtype=torch.long).permute(2, 0, 1)  # Change dimensions to [channels, height, width]

        return inputs["pixel_values"].squeeze(), mask, self.images[idx]  # Return the image filename for later use


# Initialize the new ImageProcessor
feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Create dataset instance
dataset = SegmentationDataset(
    image_dir="image",
    mask_dir="mask",
    feature_extractor=feature_extractor,
    transform=transform
)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Sample inference
for batch in dataloader:
    images, masks, filenames = batch


