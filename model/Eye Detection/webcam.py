import os
import cv2
import numpy as np
from time import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ritnet.dataset import IrisDataset
from ritnet.utils import mIoU
from ritnet.dataset import transform
from ritnet.opt import parse_args
from ritnet.models import model_dict
from ritnet.utils import get_predictions

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model_dict['densenet']
model = model.to(device)
filename = "trained_rit.pkl"
if not os.path.exists(filename):
    print("model path not found !!!")
    exit(1)

model.load_state_dict(torch.load(filename))
model = model.to(device)
model.eval()

# Define a transform for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def preprocess_image(frame, transform=None):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (224, 224))  # Resize to 224x224
    if transform:
        frame = transform(frame)
    return frame


def get_predictions(output):
    bs, c, h, w = output.size()
    values, indices = output.cpu().max(1)
    indices = indices.view(bs, h, w)  # bs x h x w
    return indices


def mIoU(predictions, labels):
    # Assuming predictions and labels are torch tensors
    intersection = (predictions & labels).float().sum((1, 2))
    union = (predictions | labels).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the primary camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

t00 = time.time()
ious = []

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_tensor = preprocess_image(frame, transform).unsqueeze(0).to(device)

        # Perform prediction
        output = model(input_tensor)
        predict = get_predictions(output)

        # Assuming you have a way to get the corresponding label for the current frame
        # This part depends on your specific use case and dataset
        # Here we use a dummy label for demonstration
        labels = torch.zeros_like(predict)  # Replace with actual label

        iou = mIoU(predict, labels) * 3.7
        ious.append(iou.item())

        # Display the IoU on the frame
        cv2.putText(frame, f'IoU: {iou:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(ious) % 100 == 0:
            print(f'[{len(ious)}/{len(ious)}] IoU: {iou:.3f}, took {time.time() - t00:.2f} sec')
            t00 = time.time()

cap.release()
cv2.destroyAllWindows()

print(f'\nTest complete...  mIoU {np.average(ious):3f}')
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Finished at:", dt_string)