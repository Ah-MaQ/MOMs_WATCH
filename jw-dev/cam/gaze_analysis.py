import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms

from ultralytics import YOLO
from models.l2cs.model import L2CS

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# 설정
cudnn.enabled = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# L2CS model 구성
model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)    # ResNet50
saved_state_dict = torch.load('./state_dicts/l2cs_trained.pkl')
model.load_state_dict(saved_state_dict)
model.to(device)
model.eval()    # 모델 평가 모드 설정

# YOLOv8 모델 로드
yolo = YOLO('./state_dicts/yolo_trained_2.pt')
classNames = ["r_iris", "l_iris", "r_eyelid", "l_eyelid", "r_center", "l_center"]

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
pitch_err = deque(maxlen=31)
yaw_err = deque(maxlen=31)
dx_correction = 0
dy_correction = 0
# 졸음 감지
frame_rate = 10
decision_time = 10
blinking_cnt = deque(maxlen=frame_rate * decision_time)

prev_frame = None

def eye_detection(frame):
    results = yolo(frame)
    detected_cls = {}
    blinking_L = True
    blinking_R = True

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

                detected_cls[cls] = [xc, yc, eye_r, height]

        if "l_iris" in detected_cls and "l_eyelid" in detected_cls:
            _, _, iris, _ = detected_cls["l_iris"]
            _, _, _, height = detected_cls["l_eyelid"]
            if height > 0.5 * iris:
                blinking_L = False

        if "r_iris" in detected_cls and "r_eyelid" in detected_cls:
            _, _, iris, _ = detected_cls["r_iris"]
            _, _, _, height = detected_cls["r_eyelid"]
            if height > 0.5 * iris:
                blinking_R = False

    return (blinking_L, blinking_R), detected_cls

def calculate_correction():
    global dx_correction, dy_correction

    dx_correction = -np.mean(yaw_err)
    dy_correction = -np.mean(pitch_err)

    return dx_correction, dy_correction

def gaze_analysis(frame):
    global prev_frame
    with torch.no_grad():
        blinking, detected_cls = eye_detection(frame)

        if all(blinking) and prev_frame is not None:
            frame = prev_frame

        detector = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if detector.detections:
            detected = True
            face = detector.detections[0]

            bbox = face.location_data.relative_bounding_box
            x_min = max(int(bbox.xmin * frame.shape[1]), 0)
            y_min = max(int(bbox.ymin * frame.shape[0]), 0)
            width = int(bbox.width * frame.shape[1])
            height = int(bbox.height * frame.shape[0])

            face_img = frame[y_min:y_min + height, x_min:x_min + width]
            face_img = transform(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

            prev_frame = frame.copy()

            yaw_predicted, pitch_predicted = model(face_img)
            yaw_predicted = torch.sum(softmax(yaw_predicted) * idx_tensor, dim=1) * 4 - 180
            pitch_predicted = torch.sum(softmax(pitch_predicted) * idx_tensor, dim=1) * 4 - 180

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

            return blinking, detected_cls, detected, (x_min, y_min, width, height), yaw, pitch
        return blinking, detected_cls, False, (0, 0, 0, 0), 0, 0

def weighted_average(data, weights):
    return np.dot(data, weights) / np.sum(weights)

def draw_eyes(blink, eye, frame, x1, y1):
    if "r_iris" in eye and "r_eyelid" in eye:
        xc, yc, eye_r, _ = eye["r_iris"]
        cv2.circle(frame, (x1 + xc, y1 + yc), eye_r, (0, 0, 255), thickness=1)

    if "l_iris" in eye and "l_eyelid" in eye:
        xc, yc, eye_r, _ = eye["l_iris"]
        cv2.circle(frame, (x1 + xc, y1 + yc), eye_r, (0, 0, 255), thickness=1)

    blinking_cnt.append(blink)
    state = "Awake"

    if len(blinking_cnt) < blinking_cnt.maxlen:
        return frame, state

    blink_count = 0
    consecutive_closed = 0
    eye_closed_duration = 0
    blink_threshold = 5
    close_threshold = 2

    weights = np.arange(1, len(blinking_cnt) + 1)

    for (blinking_L, blinking_R) in blinking_cnt:
        if blinking_L and blinking_R:
            consecutive_closed += 1
        else:
            if consecutive_closed > 0:
                blink_count += 1
                eye_closed_duration += consecutive_closed * (1 / frame_rate)
                consecutive_closed = 0

    if consecutive_closed > 0:
        blink_count += 1
        eye_closed_duration += consecutive_closed * (1 / frame_rate)

    weighted_blink_count = weighted_average([blink_count] * len(weights), weights)
    weighted_eye_closed_duration = weighted_average([eye_closed_duration] * len(weights), weights)

    if weighted_blink_count < 1:
        state = "Awake"
    else:
        if weighted_eye_closed_duration >= close_threshold:
            if eye_closed_duration >= close_threshold * 2:
                state = "Sleeping"
            else:
                state = "Drowsy"
        elif weighted_blink_count <= (decision_time * blink_threshold) / 60:
            state = "Drowsy"

    if state != "Awake":
        cv2.putText(frame, f"({state}) Don't Sleep!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2)

    return frame, state

def visualize_gaze(frame, gaze_pos, max_points=30):
    if not hasattr(visualize_gaze, 'gaze_positions'):
        visualize_gaze.gaze_positions = []
    visualize_gaze.gaze_positions.append(gaze_pos)

    gaze_positions = visualize_gaze.gaze_positions[-max_points:]

    result_frame = frame.copy()
    weights = np.linspace(0.1, 0.6, len(gaze_positions))

    for (x, y), alpha in zip(gaze_positions, weights):
        overlay = result_frame.copy()
        cv2.circle(overlay, (int(x), int(y)), 5, (200, 20, 20), -1)
        cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)

    overlay = result_frame.copy()
    cv2.circle(overlay, gaze_pos, 10, (20, 20, 250), -1)
    cv2.addWeighted(overlay, 0.4, result_frame, 0.6, 0, result_frame)

    return result_frame

def concern(pos, aoi):
    if aoi['x_min'] <= pos[0] <= aoi['x_max'] and aoi['y_min'] <= pos[1] <= aoi['y_max']:
        return "Doing well! (attentive)"
    else:
        return "Keep your eyes on the screen"

def draw_gaze(flag, blink, eye, a, b, c, d, image_in, gaze_angles, mh=720, mw=1280):
    y1 = (mh - image_in.shape[0]) // 2
    y2 = y1 + image_in.shape[0]
    x1 = (mw - image_in.shape[1]) // 2
    x2 = x1 + image_in.shape[1]

    image_out = np.ones((mh, mw, 3), dtype=np.uint8) * 200
    image_out[y1:y2, x1:x2] = image_in
    state = "Awake"

    if not flag:
        return image_out, state

    yaw, pitch = gaze_angles

    cv2.rectangle(image_out, (x1 + a, y1 + b), (x1 + a + c, y1 + b + d), (0, 255, 0), 1)

    image_out, state = draw_eyes(blink, eye, image_out, x1, y1)

    pos = (int(x1 + a + c / 2.0), int(y1 + b + d / 4.0))
    dx = -mw * np.sin(yaw) * np.cos(pitch)
    dy = -mw * np.sin(pitch)

    if not np.isnan(dx) and not np.isnan(dy):
        pos_hist[0].append(pos[0] + dx)
        pos_hist[1].append(pos[1] + dy)

    gaze_pos = [np.mean(pos_hist[0]), np.mean(pos_hist[1])]
    gaze_pos[0] = np.clip(gaze_pos[0], 0, mw)
    gaze_pos[1] = np.clip(gaze_pos[1], 0, mh)
    gaze_pos = tuple(np.round(gaze_pos).astype(int))

    image_out = visualize_gaze(image_out, gaze_pos)

    cv2.putText(image_out, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (20, 200, 20), 2)
    cv2.putText(image_out, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (20, 200, 20), 2)

    return image_out, state

if __name__ == "__main__":
    prevTime = time.time()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("웹캠을 사용할 수 없습니다.")

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        blink, detected_cls, detected, face_bbox, yaw, pitch = gaze_analysis(frame)

        frame, state = draw_gaze(detected, blink, detected_cls, face_bbox[0], face_bbox[1],
                                 face_bbox[2], face_bbox[3], frame, (yaw, pitch))

        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime
        fps_str = f"FPS : {fps:.1f}"

        visual_h = 720
        visual_w = 1280
        cv2.putText(frame, fps_str, (visual_w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow('Gaze Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
