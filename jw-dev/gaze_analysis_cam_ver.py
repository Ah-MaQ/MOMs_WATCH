import numpy as np
import cv2
import mediapipe as mp
import time

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms

from models.l2cs.model import L2CS


# 설정
cudnn.enabled = True
device = torch.device('cuda:0')     # 'cpu' or 'cuda:0'

# L2CS model 구성
model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)    # ResNet50
saved_state_dict = torch.load('./state_dicts/l2cs_trained.pkl')
model.load_state_dict(saved_state_dict)
model.to(device)
model.eval()    # 모델 평가 모드 설정

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


# draw face_bbox & gaze with arrow
def draw_gaze(a, b, c, d, image_in, yaw_pitch):
    image_out = image_in

    # bbox
    cv2.rectangle(image_out, (a, b), (a + c, b + d), (0, 255, 0), 1)

    # gaze
    length = c
    pos = (int(a + c / 2.0), int(b + d / 3.0))
    dx = -length * np.sin(yaw_pitch[0]) * np.cos(yaw_pitch[1])
    dy = -length * np.sin(yaw_pitch[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), (255, 255, 0),
                    2, cv2.LINE_AA, tipLength=0.18)
    return image_out


# return 1.face detected?(T,F) 2.face_bbox(x_min, y_min, width, height) 3.yaw 4.pitch
def gaze_analysis(frame):
    with torch.no_grad() and mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # 초기값
        detected = False
        x_min, y_min, width, height = 0, 0, 0, 0
        pitch = [0]
        yaw = [0]

        # 얼굴 인식
        detector = face_detection.process(frame)
        # print(detector.detections)
        if detector.detections:  # is not None
            detected = True
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

            face_img = frame[y_min:y_min + height, x_min:x_min + width]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (224, 224))

            # 이미지 전처리
            face_img = transform(face_img)
            face_img = face_img.to(device)
            face_img = face_img.unsqueeze(0)

            # 예측
            yaw_predicted, pitch_predicted = model(face_img)
            yaw_predicted = softmax(yaw_predicted)
            pitch_predicted = softmax(pitch_predicted)

            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data * idx_tensor, dim=1) * 4 - 180
            pitch_predicted = torch.sum(pitch_predicted.data * idx_tensor, dim=1) * 4 - 180

            yaw = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0
            pitch = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
            # print(yaw, pitch)

        return detected, (x_min, y_min, width, height), yaw[0], pitch[0]


if __name__ == "__main__":
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("웹캠을 사용할 수 없습니다.")

    while True:
        # 비디오 읽기
        success, frame = cap.read()

        if not success:
            print("프레임을 읽어올 수 없습니다.")
            time.sleep(0.1)
            continue    # 다음 프레임

        detected, face_bbox, yaw, pitch = gaze_analysis(frame)

        if detected:
            # bbox & gaze 렌더링
            frame = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch))

        cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        if detected and -0.3 <= yaw <= 0.3 and -0.3 <= pitch <= 0.3:
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
