from ritnet.densenet import DenseNet2D
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import cv2
from PIL import Image
import numpy as np

class get_images(Dataset):
    def __init__(self, dir_path, transform):
        self.dir_path = dir_path
        self.transform = transform
        self.lines = []
        for i in os.listdir(self.dir_path):
            self.lines.append(i)
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir_path, self.lines[idx])).convert("L")
        img = img.resize((400, 640))
        W, H = img.width, img.height
        table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
        img = cv2.LUT(np.array(img), table)
        img = self.clahe.apply(np.array(np.uint8(img)))
        img = Image.fromarray(img)
        img = self.transform(img)

        # print("this item is '", self.lines[idx], "'")
        # print("the shape is '", img.shape, "'")
        return img

def get_predictions(output):
    bs, c, h, w = output.size()
    values, indices = output.cpu().max(1)
    indices = indices.view(bs, h, w) # bs x h x w
    return indices


# setting
device = torch.device("cuda")
batch_size = 1  # 32

# sample data load
image_path = '../sight_pred/sample/data'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
images = get_images(image_path, transform)
img_loader = torch.utils.data.DataLoader(
    dataset=images,
    batch_size = batch_size,
    shuffle=False
    # num_workers=2
)

# model load
model = DenseNet2D(dropout=True,prob=0.2)
model = model.to(device)
saved_state_dict = torch.load('./models/trained_rit.pkl')
model.load_state_dict(saved_state_dict)
model = model.to(device)

# set for model
ious = []

model.eval()
count = 0
with (torch.no_grad()):
    for img in img_loader:
        img = img.to(device)
        output = model(img)
        print(output.size())
        # print(output[0])
        # print(output[1])
        # print(output[2])
        # print(output[3])
        # predict = get_predictions(output)
        # # print(labels.shape)
        # iou = mIoU(predict, labels) * 3.7
        # ious.append(iou)
        count += 1
        if count == 1:
            break