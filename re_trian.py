from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch.utils.data import DataLoader
from re_dataset import *
import torch

# upload base model
model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # put model to GPU

# load data
train_loader = DataLoader(dataset, batch_size=12, shuffle=True)

# define optimizer and loss functions
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# train loop
losses = dict()
model.train()
for epoch in range(10):
    losses[epoch] = []
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images, labels=masks)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses[epoch].append(loss.item())

#save the weights
torch.save(model.state_dict(), "segformer_model_both_0.pth")
