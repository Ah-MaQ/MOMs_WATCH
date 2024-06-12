import cv2
import time
import numpy as np
from collections import deque
from sklearn.cluster import DBSCAN

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
from torchvision import transforms

import mediapipe as mp
from PIL import Image
from l2cs.model import L2CS
from pykalman import KalmanFilter


def visualize_gaze_clusters(frame, gaze_pos, max_points=100, eps=30, min_samples=5):
    """
    웹캠 이미지 위에 시선을 시각화하고 클러스터를 표시하는 함수.

    Parameters:
    - frame: 현재 웹캠 이미지 (numpy array).
    - gaze_pos: 최신 시선 좌표 (x, y).
    - max_points: 분석할 최대 시선 좌표 수.
    - eps: DBSCAN 알고리즘의 eps 파라미터.
    - min_samples: DBSCAN 알고리즘의 min_samples 파라미터.

    Returns:
    - result_frame: 시선과 클러스터가 시각화된 이미지.
    """

    # 최근 시선 좌표를 저장하기 위한 리스트
    if not hasattr(visualize_gaze_clusters, 'gaze_positions'):
        visualize_gaze_clusters.gaze_positions = []
    visualize_gaze_clusters.gaze_positions.append(gaze_pos)

    # 최근 시선 좌표만 사용하기 위해 잘라내기
    gaze_positions = visualize_gaze_clusters.gaze_positions[-max_points:]

    # 클러스터링을 위한 DBSCAN 모델 설정
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(gaze_positions)

    # 클러스터 라벨과 중심 좌표 계산
    labels = db.labels_
    unique_labels = set(labels)

    # 결과 이미지 복사
    result_frame = frame.copy()

    # 투명도 조절을 위해 시선 좌표에 대한 시간 가중치 계산
    num_points = len(gaze_positions)
    weights = np.linspace(0.1, 0.6, num_points)

    for i, (x, y) in enumerate(gaze_positions):
        # 시선의 색상 및 투명도 설정
        color = (200, 20, 20)
        alpha = weights[i]

        # 시선을 원으로 표시
        overlay = result_frame.copy()
        cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
        result_frame = cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0)

    # 클러스터 중심 표시
    # for label in unique_labels:
    #     if label == -1:
    #         continue  # 노이즈는 무시
    #
    #     class_member_mask = (labels == label)
    #     xy = np.array(gaze_positions)[class_member_mask]
    #
    #     # 클러스터 중심 계산
    #     centroid = np.mean(xy, axis=0).astype(int)
    #
    #     # 클러스터 중심을 큰 원으로 표시
    #     cv2.circle(result_frame, tuple(centroid), 15, (0, 0, 255), 2)
    #     cv2.putText(result_frame, f'Cluster {label}', tuple(centroid),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return result_frame

# 칼만 필터 초기화
kf_pitch = KalmanFilter(initial_state_mean=0, initial_state_covariance=1, transition_matrices=[1], observation_matrices=[1])
kf_yaw = KalmanFilter(initial_state_mean=0, initial_state_covariance=1, transition_matrices=[1], observation_matrices=[1])

# 초기 상태 및 공분산 행렬
pitch_state_mean = 0
pitch_state_covariance = 1
yaw_state_mean = 0
yaw_state_covariance = 1

def draw_gaze(a, b, c, d, image_in, gaze_angles, alpha = 0.2):
    global pitch_state_mean, pitch_state_covariance
    global yaw_state_mean, yaw_state_covariance

    pitch, yaw = gaze_angles

    # 칼만 필터 업데이트
    pitch_state_mean, pitch_state_covariance = kf_pitch.filter_update(
        filtered_state_mean=pitch_state_mean,
        filtered_state_covariance=pitch_state_covariance,
        observation=pitch
    )
    yaw_state_mean, yaw_state_covariance = kf_yaw.filter_update(
        filtered_state_mean=yaw_state_mean,
        filtered_state_covariance=yaw_state_covariance,
        observation=yaw
    )

    smooth_pitch = pitch_state_mean[0]
    smooth_yaw = yaw_state_mean[0]

    # bbox 그리기
    cv2.rectangle(image_in, (a, b), (a + c, b + d), (0, 255, 0), 1)

    # 시선 위치 계산
    length = 2*image_in.shape[1]
    pos = (int(a + c / 2.0), int(b + d / 4.0))
    dx = -length * np.sin(smooth_yaw) * np.cos(smooth_pitch)
    dy = -length * np.sin(smooth_pitch)

    if not np.isnan(dx) and not np.isnan(dy):
        pos_hist[0].append(pos[0] + dx)
        pos_hist[1].append(pos[1] + dy)

    gaze_pos = [np.mean(pos_hist[0]), np.mean(pos_hist[1])]
    gaze_pos[0] = np.clip(gaze_pos[0], 0, image_in.shape[1])
    gaze_pos[1] = np.clip(gaze_pos[1], 0, image_in.shape[0])
    gaze_pos = tuple(np.round(gaze_pos).astype(int))

    image_out = visualize_gaze_clusters(image_in, gaze_pos)

    # 반투명 원형 표시 그리기
    overlay = image_out.copy()
    cv2.circle(overlay, gaze_pos, 10, (20, 20, 250), -1)
    cv2.addWeighted(overlay, 0.4, image_out, 0.6, 0, image_out)


    return image_out


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
    pitch_hist = deque(maxlen=5)
    yaw_hist = deque(maxlen=5)
    pos_hist = [deque(maxlen=3), deque(maxlen=3)]
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
        frame = cv2.flip(frame, 1) # 좌우 반전
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
            yaw_predicted, pitch_predicted = model(face_img)
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