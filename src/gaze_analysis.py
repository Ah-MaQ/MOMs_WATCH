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
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def gaze_initialize():
    global device, model, face_detector, softmax, idx_tensor, transform, \
        err_cnt, pitch_err, yaw_err, pos_hist, dx_correction, dy_correction

    cudnn.enabled = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # L2CS model initialization
    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
    saved_state_dict = torch.load('./state_dicts/l2cs_trained.pkl', map_location=device)
    model.load_state_dict(saved_state_dict)
    model.to(device)
    model.eval()

    # MediaPipe face detection initialization
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Other initializations
    softmax = torch.nn.Softmax(dim=1)
    idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(device)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    err_cnt = time.time()
    pitch_err = deque(maxlen=30)  # 30 프레임 동안의 데이터를 저장
    yaw_err = deque(maxlen=30)
    pos_hist = [deque(maxlen=3), deque(maxlen=3)]

    dx_correction = 0
    dy_correction = 0

def calculate_correction():
    global dx_correction, dy_correction

    # 초기화 기간 동안의 평균값을 보정값으로 설정
    if len(yaw_err) == 30 and len(pitch_err) == 30:
        dx_correction = -np.mean(yaw_err)
        dy_correction = -np.mean(pitch_err)

def visualize_gaze_clusters(frame, gaze_pos, max_points=100, eps=30, min_samples=5):
    if not hasattr(visualize_gaze_clusters, 'gaze_positions'):
        visualize_gaze_clusters.gaze_positions = []
    visualize_gaze_clusters.gaze_positions.append(gaze_pos)

    gaze_positions = visualize_gaze_clusters.gaze_positions[-max_points:]

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(gaze_positions)
    labels = db.labels_
    unique_labels = set(labels)

    result_frame = frame.copy()
    num_points = len(gaze_positions)
    weights = np.linspace(0.1, 0.6, num_points)

    for i, (x, y) in enumerate(gaze_positions):
        color = (200, 20, 20)
        alpha = weights[i]
        overlay = result_frame.copy()
        cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
        result_frame = cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0)

    # # 클러스터 중심 표시
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

def draw_gaze(a, b, c, d, image_in, gaze_angles, mh=900, mw=1600):
    if len(yaw_err) == 30 and len(pitch_err) == 30:
        calculate_correction()  # 보정값 계산

    yaw, pitch = gaze_angles

    y1 = (mh - image_in.shape[0]) // 2
    y2 = y1 + image_in.shape[0]
    x1 = (mw - image_in.shape[1]) // 2
    x2 = x1 + image_in.shape[1]

    output_img = np.ones((mh, mw, 3), dtype=np.uint8) * 200
    output_img[y1:y2, x1:x2] = image_in

    cv2.rectangle(output_img, (x1+a, y1+b), (x1+a+c, y1+b+d), (0, 255, 0), 1)

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

    output_img = visualize_gaze_clusters(output_img, gaze_pos)

    overlay = output_img.copy()
    cv2.circle(overlay, gaze_pos, 10, (20, 20, 250), -1)
    cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0, output_img)

    return output_img, gaze_pos

def gaze_analysis(frame):
    detected = False
    x_min, y_min, width, height = 0, 0, 0, 0
    pitch = [0]
    yaw = [0]

    with torch.no_grad():
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(frame)
        if results.detections:
            detected = True
            face = results.detections[0]
            bbox = face.location_data.relative_bounding_box

            x_min = int(bbox.xmin * frame.shape[1])
            if x_min < 0: x_min = 0
            y_min = int(bbox.ymin * frame.shape[0])
            if y_min < 0: y_min = 0

            width = int(bbox.width * frame.shape[1])
            height = int(bbox.height * frame.shape[0])

            face_img = frame[y_min:y_min + height, x_min:x_min + width]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            face_img = transform(face_img)
            face_img = face_img.to(device)
            face_img = face_img.unsqueeze(0)

            yaw_predicted, pitch_predicted = model(face_img)
            yaw_predicted = softmax(yaw_predicted)
            pitch_predicted = softmax(pitch_predicted)

            yaw_predicted = torch.sum(yaw_predicted.data * idx_tensor, dim=1) * 4 - 180
            pitch_predicted = torch.sum(pitch_predicted.data * idx_tensor, dim=1) * 4 - 180

            yaw = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0
            pitch = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
            yaw, pitch = yaw[0], pitch[0]

            if len(yaw_err) < 30:  # 초기화 기간 동안
                yaw_err.append(yaw)
                pitch_err.append(pitch)
            else:
                # 보정값 적용
                yaw += dx_correction
                pitch += dy_correction

    return detected, (x_min, y_min, width, height), yaw, pitch

def concern(pos):
    return True

if __name__ == "__main__":
    gaze_initialize()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("웹캠을 사용할 수 없습니다.")

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        detected, face_bbox, yaw, pitch = gaze_analysis(frame)

        if detected:
            frame, gaze_pos = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch))
            concentration = concern(gaze_pos)
            # print(concentration)

            cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), 2)
            cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), 2)
            if detected and -0.3 <= yaw <= 0.3 and -0.3 <= pitch <= 0.3:
                cv2.putText(frame, 'Look at me, look at me', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), 2)
            else:
                cv2.putText(frame, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), 2)

            cv2.imshow('Gaze Estimation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
