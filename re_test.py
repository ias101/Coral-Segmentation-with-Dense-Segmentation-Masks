from PIL import Image
import torch
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
import numpy as np
import os

# initial model
model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
model.load_state_dict(torch.load("segformer_model_both_0.pth"))
model.eval()  # set to evaluation mode

# preprocess the input image
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # resiz image
    transforms.ToTensor(),          # transfei to Tensor
])

# map colors to different classes
colors = {
    0: [0, 0, 0],       # class0 (background)
    1: [255, 0, 0],     # class1 (red)
    2: [0, 0, 255],     # class2 (blue)
}


# define the input path and output path
image_folder = "image"
mask_output_folder = "improved_result"

for image_filename in os.listdir(image_folder):
    if image_filename.endswith((".jpg", ".JPG", ".jpeg")):
        image_path = os.path.join(image_folder, image_filename)
        image = Image.open(image_path).convert("RGB")
        input_image = transform(image).unsqueeze(0)

        # make predict
        with torch.no_grad():
            outputs = model(pixel_values=input_image)

        # get the result of model
        logits = outputs.logits  # [batch_size, num_labels, height, width]
        predicted_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        # transfer the class mask to color
        mask_rgb = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in colors.items():
            mask_rgb[predicted_mask == class_id] = color

        # save to picture
        mask_image = Image.fromarray(mask_rgb)
        mask_output_path = os.path.join(mask_output_folder, f"predict_{os.path.splitext(image_filename)[0]}.png")
        mask_image.save(mask_output_path)
        print(f"Saved mask for {image_filename} as {mask_output_path}")


