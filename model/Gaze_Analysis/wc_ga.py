import cv2
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
from torchvision import transforms

import mediapipe as mp
from PIL import Image
from l2cs.model import L2CS


def draw_gaze(a, b, c, d, image_in, pitchyaw, prev_pos=None, alpha=0.2):
    # bbox 그리기
    cv2.rectangle(image_in, (a, b), (a + c, b + d), (0, 255, 0), 1)

    # 시선 위치 계산
    length = image_in.shape[1]
    current_pos = (int(a + c / 2.0), int(b + d / 3.0))
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    gaze_pos = tuple(np.round([current_pos[0] + dx, current_pos[1] + dy]).astype(int))

    # 잔상 효과를 위해 이전 위치와 현재 위치를 혼합
    if prev_pos is None:
        blended_pos = gaze_pos
    else:
        blended_pos = (alpha * np.array(gaze_pos) + (1 - alpha) * np.array(prev_pos)).astype(int)

    # 정수 튜플로 변환
    blended_pos = (int(blended_pos[0]), int(blended_pos[1]))

    # 반투명 원형 표시 그리기
    overlay = image_in.copy()
    cv2.circle(overlay, blended_pos, 10, (20, 20, 250), -1)  # 원형 마크
    cv2.addWeighted(overlay, 0.4, image_in, 0.6, 0, image_in)  # 반투명 효과

    return image_in, blended_pos


# 설정
cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# L2CS model 구성
model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)    # ResNet50
saved_state_dict = torch.load('./l2cs_trained.pkl', map_location=device)
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
    start = time.time()
    pitch_err = [0]
    yaw_err = [0]
    prev_pos = None
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
            pitch -= np.mean(pitch_err)
            yaw -= np.mean(yaw_err)
            print(pitch, yaw)

            # bbox & gaze 렌더링
            frame, prev_pos = draw_gaze(x_min, y_min, width, height, frame, (pitch[0], yaw[0]), prev_pos)

        cv2.putText(frame, f'Pitch: {yaw[0]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.putText(frame, f'Yaw: {pitch[0]:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        if -0.3 <= pitch[0] <= 0.3 and -0.3 <= yaw[0] <= 0.3 and detector.detections is not None:
            cv2.putText(frame, 'Look at me, look at me', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        now = time.time()
        if now - start < 3:
            pitch_err.append(pitch[0])
            yaw_err.append(yaw[0])
        else:
            # 프레임을 화면에 표시
            cv2.imshow('Gaze Estimation', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 웹캠과 창 해제
cap.release()
cv2.destroyAllWindows()