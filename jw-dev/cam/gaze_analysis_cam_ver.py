import numpy as np
import cv2
import mediapipe as mp
import time
from collections import deque
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from models.l2cs.model import L2CS

# 설정
cudnn.enabled = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# L2CS model 구성
model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)    # ResNet50
saved_state_dict = torch.load('./cam/state_dicts/l2cs_trained.pkl')
model.load_state_dict(saved_state_dict)
model.to(device)
model.eval()    # 모델 평가 모드 설정

# 얼굴 인식 모델 구성
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# To get predictions in degrees
softmax = torch.nn.Softmax(dim=1)
idx_tensor = torch.arange(90, dtype=torch.float32).to(device)

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

# calibration
pos_hist = [deque(maxlen=3), deque(maxlen=3)]
# 30 프레임 동안의 데이터를 저장
pitch_err = deque(maxlen=31)
yaw_err = deque(maxlen=31)
dx_correction = 0
dy_correction = 0


def calculate_correction():
    # 초기화 기간 동안의 평균값을 보정값으로 설정
    dx_correction = -np.mean(yaw_err)
    dy_correction = -np.mean(pitch_err)

    return dx_correction, dy_correction


# return 1.face detected?(T,F) 2.face_bbox(x_min, y_min, width, height) 3.yaw 4.pitch
def gaze_analysis(frame):
    with torch.no_grad():
        # 초기값
        detected = False
        x_min, y_min, width, height = 0, 0, 0, 0
        yaw = 0
        pitch = 0

        # 얼굴 인식
        detector = mp_face_detection.process(frame)
        if detector.detections:
            detected = True
            face = detector.detections[0]

            # 이미지 자르기
            bbox = face.location_data.relative_bounding_box
            x_min = max(int(bbox.xmin * frame.shape[1]), 0)
            y_min = max(int(bbox.ymin * frame.shape[0]), 0)
            width = int(bbox.width * frame.shape[1])
            height = int(bbox.height * frame.shape[0])

            face_img = frame[y_min:y_min + height, x_min:x_min + width]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = transform(face_img).unsqueeze(0).to(device)

            # 예측
            yaw_predicted, pitch_predicted = model(face_img)
            yaw_predicted = softmax(yaw_predicted)
            pitch_predicted = softmax(pitch_predicted)

            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * 4 - 180
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * 4 - 180

            yaw = yaw_predicted.cpu().numpy() * np.pi / 180.0
            pitch = pitch_predicted.cpu().numpy() * np.pi / 180.0
            yaw, pitch = yaw[0], pitch[0]

            if len(yaw_err) < 30:
                yaw_err.append(yaw)
                pitch_err.append(pitch)
            else:
                dx_correction, dy_correction = calculate_correction()
                yaw += dx_correction
                pitch += dy_correction

        return detected, (x_min, y_min, width, height), yaw, pitch


def visualize_gaze_clusters(image_in, gaze_pos):
    if not hasattr(visualize_gaze_clusters, 'gaze_positions'):
        visualize_gaze_clusters.gaze_positions = deque(maxlen=30)
    visualize_gaze_clusters.gaze_positions.append(gaze_pos)

    gaze_positions = visualize_gaze_clusters.gaze_positions

    image_out = image_in
    num_points = len(gaze_positions)
    weights = np.linspace(0.1, 0.6, num_points)

    for i, (x, y) in enumerate(gaze_positions):
        overlay = image_out
        
        color = (200, 20, 20)
        cv2.circle(overlay, (int(x), int(y)), 5, color, -1)

        alpha = weights[i]
        image_out = cv2.addWeighted(overlay, alpha, image_out, 1 - alpha, 0)

    return image_out


# draw face_bbox & gaze with arrow
def draw_gaze(a, b, c, d, image_in, yaw_pitch, mw=1600, mh=900):
    yaw, pitch = yaw_pitch
    
    y1 = (mh - image_in.shape[0]) // 2
    y2 = y1 + image_in.shape[0]
    x1 = (mw - image_in.shape[1]) // 2
    x2 = x1 + image_in.shape[1]
    
    image_out = np.ones((mh, mw, 3), dtype=np.uint8) * 200
    image_out[y1:y2, x1:x2] = image_in

    cv2.rectangle(image_out, (x1+a, y1+b), (x1+a+c, y1+b+d), (0, 255, 0), 1)

    pos = (int(x1 + a + c / 2.0), int(y1 + b + d / 4.0))
    dx = -2 * mw * np.sin(yaw) * np.cos(pitch)
    dy = -2 * mw * np.sin(pitch)

    if not np.isnan(dx) and not np.isnan(dy):
        pos_hist[0].append(pos[0] + dx)
        pos_hist[1].append(pos[1] + dy)

    gaze_pos = [np.mean(pos_hist[0]), np.mean(pos_hist[1])]
    gaze_pos[0] = np.clip(gaze_pos[0], 0, mw)
    gaze_pos[1] = np.clip(gaze_pos[1], 0, mh)
    gaze_pos = tuple(np.round(gaze_pos).astype(int))

    image_out = visualize_gaze_clusters(image_out, gaze_pos)

    overlay = image_out.copy()
    cv2.circle(overlay, gaze_pos, 10, (20, 20, 250), -1)
    cv2.addWeighted(overlay, 0.4, image_out, 0.6, 0, image_out)

    return image_out



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
