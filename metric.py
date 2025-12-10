import os
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
np.set_printoptions(threshold=np.inf)


def rgb_to_binary(image, threshold=128):
    # grayscale = np.mean(image, axis=-1)
    binary_image = (image >= threshold).astype(np.int32)
    return binary_image

def iou_score(y_true, y_pred):
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    iou = intersection / union if union != 0 else 1
    return iou


def f1_score_binary(y_true, y_pred):
    """calculate F1-Score"""
    return f1_score(y_true.flatten(), y_pred.flatten())

def accuracy_binary(y_true, y_pred):
    """calculate Accuracy"""
    return accuracy_score(y_true.flatten(), y_pred.flatten())

def calculate_loss(y_true, y_pred):
    """calculate loss"""
    return np.mean((y_true - y_pred) ** 2)

# define data structures to store metircs for Fine-tuned model
all_ious = []
all_f1_scores = []
all_accuracies = []
all_losses = []
picture_name=[]

#input predictions and calculate metrics
for filename in os.listdir('improved_result'):
    img1 = Image.open('improved_result/'+filename)
    base_filename = filename[8:]
    base_filename_no_ext = os.path.splitext(base_filename)[0]
    new_filename = base_filename_no_ext + '_mask_resized.png'
    img2 = Image.open('all_result/'+new_filename)
    img2 = img2.convert('RGB')
    img_array1 = np.array(img1)
    img_array2 = np.array(img2)
    a = rgb_to_binary(img_array1)
    b = rgb_to_binary(img_array2)
    ious = iou_score(a,b)
    all_ious.append(ious)
    f1 = f1_score_binary(a,b)
    all_f1_scores.append(f1)
    accuracy = accuracy_binary(a,b)
    all_accuracies.append(accuracy)
    loss = calculate_loss(a,b)
    all_losses.append(loss)
    picture_name.append(filename)

data = {
    'Picture name':picture_name,
    'IOU': all_ious,
    'F1-score': all_f1_scores,
    'Accuracy': all_accuracies,
    'Loss': all_losses

}

df = pd.DataFrame(data)
# save metrics to csv file
df.to_csv('improved.csv', index=False)


# define data structures to store metircs for Baseline model
all_ious1 = []
all_f1_scores1 = []
all_accuracies1 = []
all_losses1 = []
picture_name1=[]

#input predictions and calculate metrics
for filename in os.listdir('base_result'):
    img1 = Image.open('base_result/'+filename)
    new_filename = filename[:-20] + '_mask_resized.png'
    img2 = Image.open('all_result/'+new_filename)
    img2 = img2.convert('RGB')
    img_array1 = np.array(img1)
    img_array2 = np.array(img2)
    a = rgb_to_binary(img_array1)
    b = rgb_to_binary(img_array2)
    ious = iou_score(a,b)
    all_ious1.append(ious)
    f1 = f1_score_binary(a,b)
    all_f1_scores1.append(f1)
    accuracy = accuracy_binary(a,b)
    all_accuracies1.append(accuracy)
    loss = calculate_loss(a,b)
    all_losses1.append(loss)
    picture_name1.append(filename)

data1 = {
    'Picture name':picture_name1,
    'IOU': all_ious1,
    'F1-score': all_f1_scores1,
    'Accuracy': all_accuracies1,
    'Loss': all_losses1

}

df = pd.DataFrame(data1)
# save metrics to csv file
df.to_csv('based.csv', index=False)
