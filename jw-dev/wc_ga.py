import numpy as np
import cv2

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
from torchvision import transforms

from PIL import Image
import mediapipe as mp

from models.l2cs.model import L2CS

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def draw_gaze(a, b, c, d, image_in, pitchyaw, thickness=2, color=(255, 255, 0), scale=2.0):
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / 2
    pos = (int(a + c / 2.0), int(b + d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out


# 설정
cudnn.enabled = True
gpu = torch.device('cuda:0')

# 모델 로드
model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
saved_state_dict = torch.load('./state_dicts/l2cs_trained.pkl')
model.load_state_dict(saved_state_dict)
model.cuda(gpu)

# 모델 설정
idx_tensor = [idx for idx in range(90)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
softmax = torch.nn.Softmax(dim=1)

# 이미지 변환 설정
transform = transforms.Compose([
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

with torch.no_grad() and mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽어올 수 없습니다.")
            break

        # OpenCV의 BGR 이미지를 PIL 이미지로 변환
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 이미지 변환 적용
        img = transform(pil_img)
        img = img.unsqueeze(0)  # 배치 차원 추가
        img = Variable(img).cuda(gpu)

        # 모델 예측
        gaze_pitch, gaze_yaw = model(img)

        # Binned predictions
        _, pitch_bpred = torch.max(gaze_pitch.data, 1)
        _, yaw_bpred = torch.max(gaze_yaw.data, 1)

        # Continuous predictions
        pitch_predicted = softmax(gaze_pitch)
        yaw_predicted = softmax(gaze_yaw)

        # mapping from binned (0 to 28) to angles (-180 to 180)
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 4 - 180

        pitch_predicted = pitch_predicted * np.pi / 180
        yaw_predicted = yaw_predicted * np.pi / 180

        print("pitch_predicted:", pitch_predicted.item())
        print("yaw_predicted:", yaw_predicted.item())

        # 얼굴 감지
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = face_detection.process(frame)
        # print(face.detections)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if face.detections:
            detection = face.detections[0]
            mp_drawing.draw_detection(frame, detection)

            # 얼굴 영역 정보 추출
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * frame.shape[1])
            y_min = int(bbox.ymin * frame.shape[0])
            width = int(bbox.width * frame.shape[1])
            height = int(bbox.height * frame.shape[0])

            # draw_gaze 함수로 시선 표시
            frame = draw_gaze(x_min, y_min, width, height, frame, (pitch_predicted.item(), yaw_predicted.item()))

        # 결과를 프레임에 표시
        cv2.putText(frame, f'Pitch: {pitch_predicted.item():.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Yaw: {yaw_predicted.item():.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 프레임을 화면에 표시
        cv2.imshow('Gaze Estimation', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 웹캠과 창 해제
cap.release()
cv2.destroyAllWindows()