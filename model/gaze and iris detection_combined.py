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
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.")
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

def gaze_initialize():
    global device, model_gaze, mp_face_detection, softmax, idx_tensor, transform, \
        start, pitch_err, yaw_err, pitch_ema, yaw_ema, pos_hist

    # 설정
    cudnn.enabled = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # L2CS model 구성
    model_gaze = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)  # ResNet50
    saved_state_dict = torch.load('./state_dicts/l2cs_trained.pkl', map_location=device)
    model_gaze.load_state_dict(saved_state_dict)
    model_gaze.to(device)
    model_gaze.eval()  # 모델 평가 모드 설정

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

def visualize_gaze_clusters(frame, gaze_pos, max_points=15, eps=30, min_samples=5):
    # 최근 시선 좌표를 저장하기 위한 리스트
    if not hasattr(visualize_gaze_clusters, 'gaze_positions'):
        visualize_gaze_clusters.gaze_positions = deque(maxlen=max_points)
    visualize_gaze_clusters.gaze_positions.append(gaze_pos)

    # 클러스터링을 위한 DBSCAN 모델 설정
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(visualize_gaze_clusters.gaze_positions)

    # 클러스터 라벨과 중심 좌표 계산
    labels = db.labels_
    unique_labels = set(labels)

    # 결과 이미지 복사
    result_frame = frame.copy()

    # 투명도 조절을 위해 시선 좌표에 대한 시간 가중치 계산
    num_points = len(visualize_gaze_clusters.gaze_positions)
    weights = np.linspace(0.1, 0.6, num_points)

    for i, (x, y) in enumerate(visualize_gaze_clusters.gaze_positions):
        # 시선의 색상 및 투명도 설정
        color = (200, 20, 20)
        alpha = weights[i]

        # 시선을 원으로 표시
        overlay = result_frame.copy()
        cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
        result_frame = cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0)

    return result_frame

def draw_gaze(a, b, c, d, image_in, gaze_angles, mh, mw):
    global pitch_ema, yaw_ema
    yaw, pitch = gaze_angles

    # EMA 계산
    alpha = 0.2  # EMA 계수
    if pitch_ema is None:
        pitch_ema = pitch
        yaw_ema = yaw
    else:
        pitch_ema = alpha * pitch + (1 - alpha) * pitch_ema
        yaw_ema = alpha * yaw + (1 - alpha) * yaw_ema

    smooth_pitch = pitch_ema
    smooth_yaw = yaw_ema

    y1 = (mh - image_in.shape[0]) // 2
    y2 = y1 + image_in.shape[0]
    x1 = (mw - image_in.shape[1]) // 2
    x2 = x1 + image_in.shape[1]

    output_img = np.ones((mh, mw, 3), dtype=np.uint8) * 200
    output_img[y1:y2, x1:x2] = image_in

    # bbox 그리기
    cv2.rectangle(output_img, (x1 + a, y1 + b), (x1 + a + c, y1 + b + d), (0, 255, 0), 1)

    # 시선 위치 계산
    length = mw
    pos = (int(x1 + a + c / 2.0), int(y1 + b + d / 4.0))
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)

    if not np.isnan(dx) and not np.isnan(dy):
        pos_hist[0].append(pos[0] + dx)
        pos_hist[1].append(pos[1] + dy)

    gaze_pos = [np.mean(pos_hist[0]), np.mean(pos_hist[1])]
    gaze_pos[0] = np.clip(gaze_pos[0], 0, mw)
    gaze_pos[1] = np.clip(gaze_pos[1], 0, mh)
    gaze_pos = tuple(np.round(gaze_pos).astype(int))

    output_img = visualize_gaze_clusters(output_img, gaze_pos)

    # 반투명 원형 표시 그리기
    overlay = output_img.copy()
    cv2.circle(overlay, gaze_pos, 10, (20, 20, 250), -1)
    cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0, output_img)

    return output_img, gaze_pos

def gaze_analysis(frame):
    with torch.no_grad() and mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # 초기값
        detected = False
        x_min, y_min, width, height = 0, 0, 0, 0
        pitch = [0]
        yaw = [0]

        # 얼굴 인식
        detector = face_detection.process(frame)
        if detector.detections:
            detected = True
            face = detector.detections[0]

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
            yaw_predicted, pitch_predicted = model_gaze(face_img)
            yaw_predicted = softmax(yaw_predicted)
            pitch_predicted = softmax(pitch_predicted)

            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data * idx_tensor, dim=1) * 4 - 180
            pitch_predicted = torch.sum(pitch_predicted.data * idx_tensor, dim=1) * 4 - 180

            yaw = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0
            pitch = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0

            yaw -= np.mean(yaw_err)
            pitch -= np.mean(pitch_err)

            now = time.time()
            if now - start < 3:
                pitch_err.append(pitch[0])
                yaw_err.append(yaw[0])

        return detected, (x_min, y_min, width, height), yaw[0], pitch[0]

def concern(pos, aoi):
    if aoi['x_min'] <= pos[0] <= aoi['x_max'] and pos[1] >= aoi['y_min']:
        return "Doing well! (attentive)"
    else:
        return "Keep your eyes on the screen"

def evaluate_focus(gaze_positions, std_threshold=10):
    if len(gaze_positions) < 2:
        return "Please stay focused"

    positions = np.array(gaze_positions)
    num_positions = len(positions)
    weights = np.linspace(0.1, 1.0, num_positions)
    mean_position = np.average(positions, axis=0, weights=weights)
    variance = np.average((positions - mean_position) ** 2, axis=0, weights=weights)
    std_dev = np.sqrt(variance)

    if np.any(std_dev * 2.54 / 96 > std_threshold):  # cm 단위로 변환
        return "Please stay focused"
    else:
        return "Doing well! (Focused)"

def define_aoi(dpi, margin_cm=1):
    h = int(dpi * 16)  # 캠 화면크기 임의설정[!]
    w = int(dpi * 17)  # 캠 화면크기 임의설정[!]
    margin_pixels = int(dpi * margin_cm / 2.54)  # cm를 픽셀로 변환 (1인치=2.54cm)

    aoi = {
        'x_min': margin_pixels,
        'x_max': w - margin_pixels,
        'y_min': margin_pixels,
        'y_max': float('inf')  # 아래쪽 마진 없음
    }
    return aoi

def yolo_initialize():
    global model_yolo, classNames
    # Load the YOLOv8 model
    model_yolo = YOLO('./state_dicts/yolo_trained.pt')

    classNames = ["r_iris", "l_iris", "r_eyelid", "l_eyelid", "r_center", "l_center"]

def detect(frame):
    results = model_yolo(frame)

    detected_cls = {}

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = classNames[int(box.cls[0])]

            if cls in classNames[:4]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                xc = (x1 + x2) // 2
                yc = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                eye_r = (width + height) // 4

                detected_cls[cls] = [xc, yc, eye_r]
    return detected_cls

if __name__ == "__main__":
    gaze_initialize()
    yolo_initialize()

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("웹캠을 사용할 수 없습니다.")

    # dpi 계산
    horizontal_res = 1920
    vertical_res = 1080
    diagonal_size_inch = 24  # 모니터 크기 (예: 24인치)
    diagonal_res = (horizontal_res ** 2 + vertical_res ** 2) ** 0.5
    dpi = diagonal_res / diagonal_size_inch

    aoi = None

    # 동공이 감지되지 않은 시간을 추적하기 위한 타이머
    no_r_iris_time = 0
    no_l_iris_time = 0

    start_time_r = None
    start_time_l = None

    warning_message = ""

    while True:
        # 비디오 읽기
        success, frame = cap.read()

        if not success:
            print("프레임을 읽어올 수 없습니다.")
            time.sleep(0.1)
            continue  # 다음 프레임

        frame = cv2.flip(frame, 1)
        if aoi is None:
            aoi = define_aoi(dpi)

        detected_cls = detect(frame)

        current_time = time.time()

        if "r_iris" in detected_cls and "r_eyelid" in detected_cls:
            xc, yc, eye_r = detected_cls["r_iris"]
            cv2.circle(frame, (xc, yc), eye_r, (0, 0, 255), thickness=1)
            no_r_iris_time = 0
            start_time_r = None
        else:
            if start_time_r is None:
                start_time_r = current_time
            no_r_iris_time = current_time - start_time_r

        if "l_iris" in detected_cls and "l_eyelid" in detected_cls:
            xc, yc, eye_r = detected_cls["l_iris"]
            cv2.circle(frame, (xc, yc), eye_r, (0, 0, 255), thickness=1)
            no_l_iris_time = 0
            start_time_l = None
        else:
            if start_time_l is None:
                start_time_l = current_time
            no_l_iris_time = current_time - start_time_l

        if no_r_iris_time >= 5 and no_l_iris_time >= 5:
            warning_message = "Eyes open, sweetheart"
            cv2.putText(frame, warning_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            warning_message = ""

        detected, face_bbox, yaw, pitch = gaze_analysis(frame)

        if detected:
            # bbox & gaze 렌더링
            frame, gaze_pos = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch), mh=int(dpi * 16), mw=int(dpi * 17))
            concentration_status = concern(gaze_pos, aoi)
            focus_status = evaluate_focus(visualize_gaze_clusters.gaze_positions)

            cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
            cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
            cv2.putText(frame, concentration_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
            cv2.putText(frame, focus_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)

        # 프레임을 화면에 표시
        cv2.imshow('Gaze Estimation and Eye Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠과 창 해제
    cap.release()
    cv2.destroyAllWindows()
