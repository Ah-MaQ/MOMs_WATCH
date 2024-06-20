import numpy as np
import cv2
from PIL import Image
from collections import deque
import torch
import torchvision
from models.dfer.ST_Former import GenerateModel


class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        return [img.resize((self.size, self.size), self.interpolation) for img in img_group]


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode in ['L', 'F']:
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2) if self.roll \
                else np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


image_size = 112
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = GenerateModel().to(device)
# model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("./cam/state_dicts/new.pth"))
model.eval()  # 모델 평가 모드 설정

# 이미지 변환 설정
transform = torchvision.transforms.Compose([GroupResize(image_size), Stack(), ToTorchFormatTensor()])

# state_keys 사전
state = ['Concentration', 'Drowsiness', 'Lack of Concentration', 'Decrease in Concentration', 'Negligence']

frames = deque(maxlen=16)


def state_awareness(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)
    frames.append(pil_frame)

    if len(frames) == 16:
        frames_list = list(frames)
        images = transform(frames_list)
        images = images.view(-1, 3, image_size, image_size).to(device)

        # 예측
        with torch.no_grad():
            output = model(images).cpu()
        max_idx = torch.argmax(output[0])

        return state[max_idx]
    return None


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
