import numpy as np
import cv2
import mediapipe as mp
import time
from collections import deque
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from ultralytics import YOLO
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

model_yolo = YOLO('./cam/state_dicts/yolo_trained.pt')
classNames = ["r_iris", "l_iris", "r_eyelid", "l_eyelid", "r_center", "l_center"]

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

    return image_out, gaze_pos


def concern(pos, aoi):
    if aoi['x_min'] <= pos[0] <= aoi['x_max'] and aoi['y_min'] <= pos[1] <= aoi['y_max']:
        return "Doing well! (attentive)"
    else:
        return "Keep your eyes on the screen"


def evaluate_focus(gaze_positions, std_threshold=10, dpi=96):
    if len(gaze_positions) < 2:
        return "Please stay focused"

    positions = np.array(gaze_positions)
    num_positions = len(positions)
    weights = np.linspace(0.1, 1.0, num_positions)
    mean_position = np.average(positions, axis=0, weights=weights)
    variance = np.average((positions - mean_position) ** 2, axis=0, weights=weights)
    std_dev = np.sqrt(variance)

    if np.any(std_dev * 2.54 / dpi > std_threshold):  # cm 단위로 변환
        return "Please stay focused"
    else:
        return "Doing well! (Focused)"


# 관심 영역 설정 (화면의 가장자리 1cm 마진 제외, 아래쪽 제외) 강의실 23인치 모니터
def define_aoi(monitor_height=31.0, monitor_width=52.0, monitor_horizontal_res=1920, monitor_vertical_res=1080, margin_cm=1):
    diagonal_res = (monitor_horizontal_res ** 2 + monitor_vertical_res ** 2) ** 0.5
    diagonal_size_inch = (monitor_height ** 2 + monitor_width ** 2) ** 0.5
    dpi = diagonal_res / diagonal_size_inch

    popup_height = 9  # 팝업창 디멘션. 눈대중으로 지정
    popup_width = 16  # 팝업창 디멘션. 눈대중으로 지정

    h = int(dpi * popup_height)  # 팝업창의 세로 픽셀수
    w = int(dpi * popup_width)  # 팝업창의 가로 픽셀수
    margin_pixels = int(dpi * margin_cm / 2.54)  # cm를 픽셀로 변환 (1인치=2.54cm)

    aoi = {
        'x_min': margin_pixels,
        'x_max': w - margin_pixels,
        'y_min': 6 * margin_pixels,
        'y_max': float('inf') #아래쪽 마진 없음
    }
    return aoi, dpi


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
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("웹캠을 사용할 수 없습니다.")

    aoi = None
    dpi = None

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
            aoi, dpi = define_aoi()

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

        detected, face_bbox, yaw, pitch = gaze_analysis(frame)

        if detected:
            # bbox & gaze 렌더링
            frame, gaze_pos = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch))
            concentration_status = concern(gaze_pos, aoi)
            focus_status = evaluate_focus(visualize_gaze_clusters.gaze_positions, dpi=dpi)

            cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
            cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
            cv2.putText(frame, concentration_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
            cv2.putText(frame, focus_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
            cv2.putText(frame, warning_message, (10, 190), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        # 프레임을 화면에 표시
        cv2.imshow('Gaze Estimation', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠과 창 해제
    cap.release()
    cv2.destroyAllWindows()
    # # 웹캠 초기화
    # cap = cv2.VideoCapture(0)
    #
    # if not cap.isOpened():
    #     raise IOError("웹캠을 사용할 수 없습니다.")
    #
    # while True:
    #     # 비디오 읽기
    #     success, frame = cap.read()
    #
    #     if not success:
    #         print("프레임을 읽어올 수 없습니다.")
    #         time.sleep(0.1)
    #         continue    # 다음 프레임
    #
    #     detected, face_bbox, yaw, pitch = gaze_analysis(frame)
    #
    #     if detected:
    #         # bbox & gaze 렌더링
    #         frame = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch))
    #
    #     cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1, (0, 255, 0), 2)
    #     cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1, (0, 255, 0), 2)
    #     if detected and -0.3 <= yaw <= 0.3 and -0.3 <= pitch <= 0.3:
    #         cv2.putText(frame, 'Look at me, look at me', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
    #                     1, (0, 255, 0), 2)
    #     else:
    #         cv2.putText(frame, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
    #                     1, (0, 255, 0), 2)
    #
    #     # 프레임을 화면에 표시
    #     cv2.imshow('Gaze Estimation', frame)
    #
    #     # 'q' 키를 누르면 종료
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # # 웹캠과 창 해제
    # cap.release()
    # cv2.destroyAllWindows()
