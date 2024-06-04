#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import transforms
from ritnet.utils import mIoU, get_predictions
from ritnet.opt import parse_args
from ritnet.models import model_dict

def preprocess_image(image, device):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to [224, 224]
    resized_image = cv2.resize(gray_image, (224, 224))
    # Normalize and add batch dimension
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor_image = transform(resized_image).unsqueeze(0).to(device)
    return tensor_image

def apply_mask(image, mask):
    # Resize mask to match the original image size
    mask_resized = cv2.resize(mask.astype('float32'), (image.shape[1], image.shape[0]))
    colored_mask = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # Ensure colored_mask has 3 channels
    if colored_mask.shape[2] == 1:
        colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    return masked_image

if __name__ == '__main__':
    args = parse_args()
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    print("Starting at:", dt_string)

    if args.model not in model_dict:
        print("Model not found !!!")
        print("valid models are:", list(model_dict.keys()))
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_dict[args.model]
    model = model.to(device)
    filename = "trained_rit.pkl"
    if not os.path.exists(filename):
        print("model path not found !!!")
        exit(1)

    model.load_state_dict(torch.load(filename, map_location=device))
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit(1)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            tensor_image = preprocess_image(frame, device)
            output = model(tensor_image)
            predict = get_predictions(output)
            mask = predict.squeeze().cpu().numpy()

            masked_image = apply_mask(frame, mask)

            # Concatenate original and masked images side by side
            combined_image = np.hstack((frame, masked_image))

            # Display the concatenated images
            cv2.imshow('Original and Masked Image', combined_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    print("Finished at:", dt_string)
