import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
from sklearn.cluster import DBSCAN

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms

from models.l2cs.model import L2CS

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def gaze_initialize():
    global device, model, mp_face_detection, softmax, idx_tensor, transform, \
        start, pitch_err, yaw_err, pitch_ema, yaw_ema, pos_hist

    # 설정
    cudnn.enabled = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # L2CS model 구성
    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)    # ResNet50
    saved_state_dict = torch.load('./state_dicts/l2cs_trained.pkl', map_location=device)
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

    start = time.time()
    pitch_err = [0]
    yaw_err = [0]
    pitch_ema = None
    yaw_ema = None
    pos_hist = [deque(maxlen=3), deque(maxlen=3)]

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

# draw face_bbox & gaze with arrow
def draw_gaze(a, b, c, d, image_in, gaze_angles):
    global pitch_ema, yaw_ema
    pitch, yaw = gaze_angles

    # EMA 계산
    # alpha = 0.2  # EMA 계수
    # if pitch_ema is None:
    #     pitch_ema = pitch
    #     yaw_ema = yaw
    # else:
    #     pitch_ema = alpha * pitch + (1 - alpha) * pitch_ema
    #     yaw_ema = alpha * yaw + (1 - alpha) * yaw_ema
    #
    # smooth_pitch = pitch_ema
    # smooth_yaw = yaw_ema

    # bbox 그리기
    cv2.rectangle(image_in, (a, b), (a + c, b + d), (0, 255, 0), 1)

    # 시선 위치 계산
    length = 2*image_in.shape[1]
    pos = (int(a + c / 2.0), int(b + d / 4.0))
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)

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
            if x_min < 0: x_min = 0
            y_min = int(bbox.ymin * frame.shape[0])
            if y_min < 0: y_min = 0

            width = int(bbox.width * frame.shape[1])
            height = int(bbox.height * frame.shape[0])

            face_img = frame[y_min:y_min + height, x_min:x_min + width]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

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

            # yaw -= np.mean(yaw_err)
            # pitch -= np.mean(pitch_err)
            # print(yaw, pitch)

            now = time.time()
            if now - start < 3:
                pitch_err.append(pitch[0])
                yaw_err.append(yaw[0])

        return detected, (x_min, y_min, width, height), yaw[0], pitch[0]


if __name__ == "__main__":
    gaze_initialize()

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

        frame = cv2.flip(frame, 1)
        detected, face_bbox, yaw, pitch = gaze_analysis(frame)

        if detected:
            # bbox & gaze 렌더링
            frame = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch))

        cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        if detected and -0.3 <= yaw <= 0.3 and -0.3 <= pitch <= 0.3:
            cv2.putText(frame, 'Look at me, look at me', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # 프레임을 화면에 표시
        cv2.imshow('Gaze Estimation', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠과 창 해제
    cap.release()
    cv2.destroyAllWindows()
