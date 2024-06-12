import numpy as np
import cv2
from PIL import Image
from collections import deque
import time

import torch
import torchvision

from models.dfer.ST_Former import GenerateModel


class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        out_group = list()
        for img in img_group:
            out_group.append(img.resize((self.size, self.size), self.interpolation))
        return out_group


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L' or img_group[0].mode == 'F':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.to(torch.float32).div(255) if self.div else img.to(torch.float32)


image_size = 112

model = GenerateModel()
model = torch.nn.DataParallel(model).cuda()
saved_state_dict = torch.load("./state_dicts/dfer_trained.pth")
model.load_state_dict(saved_state_dict)
model.eval()    # 모델 평가 모드 설정

# 이미지 변환 설정
transform = torchvision.transforms.Compose([GroupResize(image_size),
                                            Stack(),
                                            ToTorchFormatTensor()])

# state_keys 사전
state = ['Concentration',               # '집중'
         'Drowsiness',                  # '졸림'
         'Lack of Concentration',       # '집중 결핍'
         'Decrease in Concentration',   # '집중 하락'
         'Negligence']                  # '태만'

frames = deque(maxlen=16)


def state_awareness(frame):
    # PIL 객체로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # open cv는 BGR 포맷을 사용하므로 RGB로 변환
    pil_frame = [Image.fromarray(frame_rgb)]  # OpenCV 이미지에서 PIL 이미지로 변환
    # print(pil_image)
    frames.extend(pil_frame)
    # print(images)

    if len(frames) == 16:
        # 이미지 전처리
        images = transform(frames)
        images = torch.reshape(images, (-1, 3, image_size, image_size))
        images = images.cuda()

        # 예측
        output = model(images).cpu().detach()
        # print(output[0])
        max_idx = torch.argmax(output[0])

        return state[max_idx]
    pass


if __name__ == "__main__":
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("웹캠을 사용할 수 없습니다.")

    with torch.no_grad():
        while True:
            # 비디오 읽기
            success, frame = cap.read()

            if not success:
                print("프레임을 읽어올 수 없습니다.")
                time.sleep(0.1)
                continue  # 다음 프레임

            cur_state = state_awareness(frame)
            cv2.putText(frame, f'Your now in {cur_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            # 프레임을 화면에 표시
            cv2.imshow('state_awareness', frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 웹캠과 창 해제
    cap.release()
    cv2.destroyAllWindows()
