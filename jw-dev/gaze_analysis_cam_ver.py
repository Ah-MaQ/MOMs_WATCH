import numpy as np
import cv2
import time

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
from torchvision import transforms

from PIL import Image
import mediapipe as mp

from models.l2cs.model import L2CS


def draw_gaze(a, b, c, d, image_in, pitchyaw):
    # bbox
    cv2.rectangle(frame, (a, b), (a + c, b + d), (0, 255, 0), 1)

    # gaze
    image_out = image_in
    length = c
    pos = (int(a + c / 2.0), int(b + d / 3.0))
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), (255, 255, 0),
                    2, cv2.LINE_AA, tipLength=0.18)
    return image_out


# 설정
cudnn.enabled = True
device = torch.device('cpu')     # or 'cuda:0'

# L2CS model 구성
model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)    # ResNet50
saved_state_dict = torch.load('./state_dicts/l2cs_trained.pkl')
model.load_state_dict(saved_state_dict)
model.to(device)

# 얼굴 인식 모델 구성
mp_face_detection = mp.solutions.face_detection

# To get predictions in degrees
softmax = torch.nn.Softmax(dim=1)
idx_tensor = [idx for idx in range(90)]
idx_tensor = torch.FloatTensor(idx_tensor).to(device)

# 이미지 변환 설정
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 모델 평가 모드 설정
model.eval()

# 웹캠 초기화
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("웹캠을 사용할 수 없습니다.")

with torch.no_grad() and mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        # 비디오 읽기
        success, frame = cap.read()

        if not success:
            print("프레임을 읽어올 수 없습니다.")
            time.sleep(0.1)
            continue    # 다음 프레임

        # 초기값
        pitch = [0]
        yaw = [0]

        # 얼굴 인식
        detector = face_detection.process(frame)
        print(detector.detections)
        if detector.detections is not None:
            face = detector.detections[0]  # 1명만

            # 이미지 자르기
            bbox = face.location_data.relative_bounding_box
            x_min = int(bbox.xmin * frame.shape[1])
            if x_min < 0:
                x_min = 0
            y_min = int(bbox.ymin * frame.shape[0])
            if y_min < 0:
                y_min = 0
            width = int(bbox.width * frame.shape[1])
            height = int(bbox.height * frame.shape[0])

            # if face_img.size > 0:   # (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'
            face_img = frame[y_min:y_min+height, x_min:x_min+width]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)    # empty img error!
            face_img = cv2.resize(face_img, (224, 224))

            # 이미지 전처리
            face_img = transform(face_img)
            face_img = face_img.to(device)
            face_img = face_img.unsqueeze(0)

            # 예측
            pitch_predicted, yaw_predicted = model(face_img)
            pitch_predicted = softmax(pitch_predicted)
            yaw_predicted = softmax(yaw_predicted)

            # Get continuous predictions in degrees.
            pitch_predicted = torch.sum(pitch_predicted.data * idx_tensor, dim=1) * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted.data * idx_tensor, dim=1) * 4 - 180

            pitch = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
            yaw = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0
            print(pitch, yaw)

            # bbox & gaze 렌더링
            frame = draw_gaze(x_min, y_min, width, height, frame, (pitch[0], yaw[0]))

        cv2.putText(frame, f'Pitch: {pitch[0]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.putText(frame, f'Yaw: {yaw[0]:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        if -0.3 <= pitch[0] <= 0.3 and -0.3 <= yaw[0] <= 0.3 and detector.detections is not None:
            cv2.putText(frame, 'Look at me, look at me', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        # 프레임을 화면에 표시
        cv2.imshow('Gaze Estimation', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 웹캠과 창 해제
cap.release()
cv2.destroyAllWindows()
