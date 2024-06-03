from l2cs.model import L2CS
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class get_images(Dataset):  # NIA2022
    def __init__(self, dir_path, transform):
        self.dir_path = dir_path
        self.transform = transform
        self.lines = []
        for i in os.listdir(self.dir_path):
            self.lines.append(i)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir_path, self.lines[idx]))
        img = self.transform(img)
        print("this item is '", self.lines[idx], "'")
        print("the shape is '", img.shape, "'")

        return img

# setting
cudnn.enabled = True
gpu = torch.device('cuda:0')
batch_size = 1

# sample data load
images_path = "./sample/data"
transform = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
images = get_images(images_path, transform)
# print(images.shape)
img_loader = torch.utils.data.DataLoader(
    dataset=images,
    batch_size=batch_size,
    shuffle=False,
    # num_workers=4,
    pin_memory=True
)

# model load
model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], 90)
saved_state_dict = torch.load('./models/l2cs_trained.pkl')
model.load_state_dict(saved_state_dict)
model.cuda(gpu)
# print(model)

# set for model
idx_tensor = [idx for idx in range(90)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
softmax = torch.nn.Softmax(dim=1)

# prediction
model.eval()
count = 0
with torch.no_grad():
    for img in img_loader:
        img = Variable(img).cuda(gpu)
        print('shape to model', img.shape)
        # total += cont_labels.size(0)

        # label_pitch = cont_labels[:, 0].float() * np.pi / 180
        # label_yaw = cont_labels[:, 1].float() * np.pi / 180

        gaze_pitch, gaze_yaw = model(img)

        # Binned predictions
        _, pitch_bpred = torch.max(gaze_pitch.data, 1)
        _, yaw_bpred = torch.max(gaze_yaw.data, 1)

        # Continuous predictions
        pitch_predicted = softmax(gaze_pitch)
        yaw_predicted = softmax(gaze_yaw)

        # mapping from binned (0 to 28) to angels (-180 to 180)
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 4 - 180

        pitch_predicted = pitch_predicted * np.pi / 180
        yaw_predicted = yaw_predicted * np.pi / 180

        print("pitch_predicted:",pitch_predicted)
        print("yaw_predicted:",yaw_predicted)

        count += 1
        if count == 10:
            break
