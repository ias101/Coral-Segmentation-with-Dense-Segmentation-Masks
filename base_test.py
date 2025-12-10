from transformers import SegformerForSemanticSegmentation
from Ndataset import *
import torch.nn.functional as F

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # move model to GPU

result_dir = "base_result"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def create_color_map(num_classes):
    # define a color map
    color_map = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return color_map

def apply_color_map(prediction, color_map):
    # map 3 categories to color map
    colored_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for label in range(len(color_map)):
        colored_image[prediction == label] = color_map[label]
    return colored_image

def resize_preds_to_labels(preds, labels):
    preds_resized = F.interpolate(preds.unsqueeze(1).float(), size=labels.shape[-2:], mode="nearest").squeeze(1)
    return preds_resized

num_classes = model.config.num_labels
color_map = create_color_map(num_classes)

# prediction
for idx, batch in enumerate(dataloader):
    images,filenames = batch
    images = images.to(device)  # move image to GPU

    # model predict
    with torch.no_grad():
        outputs = model(pixel_values=images)

    logits = outputs.logits  # [batch_size, num_classes, height, width]
    predictions = torch.argmax(logits, dim=1)

    # save the predictions
    for i in range(predictions.shape[0]):
        pred = predictions[i].cpu().numpy()  # transfer to numpy
        colored_pred = apply_color_map(pred, color_map)  # apply color map
        pred_image = Image.fromarray(colored_pred)

        # Use the original filename for saving
        original_filename = filenames[i]
        base_filename = os.path.splitext(original_filename)[0]
        pred_image.save(os.path.join(result_dir, f"{base_filename}_base_prediction.png"))

