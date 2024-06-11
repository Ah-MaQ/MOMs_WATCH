from models.ST_Former import GenerateModel
import torch
from torch.utils import data
import torchvision
from PIL import Image
import numpy as np
import os
import glob


class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)


class Stack(object):
    def __init__(self):
        pass

    def __call__(self, img_group):
        if img_group.mode == 'RGB':
            img_array = np.array(img_group)
            return np.concatenate([img_array for _ in range(16)], axis=2)  # 16 프레임으로 복제


class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        return img.float().div(255) if self.div else img.float()


class ImageDataset(data.Dataset):
    def __init__(self, image_dir, transform, image_size):
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        self.image_paths = glob.glob(os.path.join(self.image_dir, '*.png'))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = torch.reshape(image, (1, 16, 3, self.image_size, self.image_size))  # 모델 입력 형태에 맞게 조정

        return image

    def __len__(self):
        return len(self.image_paths)


def custom_data_loader(image_dir):
    image_size = 112
    transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                Stack(),
                                                ToTorchFormatTensor()])
    dataset = ImageDataset(image_dir=image_dir,
                           transform=transform,
                           image_size=image_size)
    return dataset


model = GenerateModel()
model = torch.nn.DataParallel(model).cuda()
saved_state_dict = torch.load("./dfer_trained.pth")
model.load_state_dict(saved_state_dict)

images_dir = './images'

img_dataset = custom_data_loader(images_dir)
loader = torch.utils.data.DataLoader(img_dataset,
                                     batch_size=1,  # 이미지 단위로 처리
                                     shuffle=False,
                                     pin_memory=True)

# state_keys 사전
state = ['집중', '졸림', '집중 결핍', '집중 하락', '태만']

model.eval()
count = 0
with torch.no_grad():
    for images in loader:
        images = images.cuda().view(-1, 3, 112, 112)  # 배치 차원 축소 및 모델 입력 형태에 맞게 조정
        output = model(images).to('cpu')
        max_index = torch.argmax(output[0])

        # 결과 출력
        print(f"상태: {state[max_index]}")
        count += 1
        if count == 1:
            break