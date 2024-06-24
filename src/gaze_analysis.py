import cv2
import time
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
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
saved_state_dict = torch.load('./state_dicts/l2cs_trained.pkl', map_location=device)
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

prev_frame = None
buffer_size = 100  # 10초 동안의 데이터 (0.1초 간격으로 100개)
blinking_buffer = deque(maxlen=buffer_size)
eyes_open_buffer = deque(maxlen=buffer_size)

def eye_detection(frame, fx, fw):
    results = yolo(frame)
    detected_cls = {}
    blinking = True
    # Start with a very large value for comparison
    left_iris_pos = 2e3
    right_iris_pos = 2e3
    left_eye_pos = 2e3
    right_eye_pos = 2e3

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

                if (xc > fx) and (xc < fx + fw):
                    # Check if the detected eye is left or right based on x coordinate
                    if cls in ["l_iris", "r_iris"]:
                        if xc < left_iris_pos:
                            left_iris_pos = xc
                            detected_cls[cls] = [xc, yc, eye_r, height]
                        if xc > right_iris_pos:
                            right_iris_pos = xc
                            detected_cls[cls] = [xc, yc, eye_r, height]

                    elif cls in ["l_eyelid", "r_eyelid"]:
                        if xc < left_eye_pos:
                            left_eye_pos = xc
                            detected_cls[cls] = [xc, yc, eye_r, height]
                        if xc > right_eye_pos:
                            right_eye_pos = xc
                            detected_cls[cls] = [xc, yc, eye_r, height]

        # 눈 깜박임 감지
        iris = []
        eyelid = []
        for key in detected_cls.keys():
            if key in ["l_iris", "r_iris"]:
                _, _, eye_r, _ = detected_cls[key]
                iris.append(0.6 * eye_r)
            else:
                _, _, _, height = detected_cls[key]
                eyelid.append(height)

        if len(iris) > 0 and len(eyelid):
            if min(eyelid) > min(iris):
                blinking = False

    return blinking, detected_cls

def calculate_correction():
    global dx_correction, dy_correction

    dx_correction = -np.mean(yaw_err)
    dy_correction = -np.mean(pitch_err)

    return dx_correction, dy_correction

def gaze_analysis(frame):
    global prev_frame
    with torch.no_grad():
        blinking = False
        detected_cls = {}

        detector = mp_face_detection.process(frame)
        if detector.detections:
            detected = True
            face = detector.detections[0]

            bbox = face.location_data.relative_bounding_box
            x_min = max(int(bbox.xmin * frame.shape[1]), 0)
            y_min = max(int(bbox.ymin * frame.shape[0]), 0)
            width = int(bbox.width * frame.shape[1])
            height = int(bbox.height * frame.shape[0])

            roi_img = frame[y_min:y_min + height, x_min:x_min + width]

            blinking, detected_cls = eye_detection(frame, x_min, width)

            if blinking and prev_frame is not None:
                frame = prev_frame
                roi_img = frame[y_min:y_min + height, x_min:x_min + width]

            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(roi_img)
            face_img = transform(pil_img).unsqueeze(0).to(device)

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


def user_state(blinking, eyes_open):
    # 현재 상태를 버퍼에 추가
    blinking_buffer.append(blinking)
    eyes_open_buffer.append(eyes_open)

    # 버퍼가 꽉 찼을 때만 상태 판별
    if len(blinking_buffer) == buffer_size:
        # 졸음 및 수면 판단 기준 설정
        eyes_closed_threshold = 0.6 * buffer_size  # 10초 동안 눈을 감고 있는 시간의 60% 이상
        continuous_closed_time = 3  # 연속으로 눈을 감고 있는 시간 기준 (3초)

        # 눈 감고 있는 횟수 및 시간 계산
        eyes_closed_count = sum(blinking_buffer)
        continuous_closed_count = 0

        # 연속 눈 감고 있는 시간 계산
        for i in range(len(blinking_buffer)):
            if blinking_buffer[i]:
                continuous_closed_count += 1
                if continuous_closed_count >= continuous_closed_time * 10:
                    return "Sleep"
            else:
                continuous_closed_count = 0

        # 상태 판별
        if eyes_closed_count >= eyes_closed_threshold:
            return "Drowsy"
        else:
            return "Awake"

    # 버퍼가 꽉 차지 않았을 때는 기본 상태 반환
    return "Awake"

def draw_eyes(blinking, eye, frame, dx, dy):
    eyes_open = False
    if "r_iris" in eye:
        xc, yc, eye_r, _ = eye["r_iris"]
        cv2.circle(frame, (dx + xc, dy + yc), eye_r, (0, 0, 255), thickness=1)
        eyes_open = True

    if "l_iris" in eye:
        xc, yc, eye_r, _ = eye["l_iris"]
        cv2.circle(frame, (dx + xc, dy + yc), eye_r, (0, 0, 255), thickness=1)
        eyes_open = True

    state = user_state(blinking, eyes_open)

    return frame, state

def evaluate_focus(coords, state, width, std_threshold=20, decay_rate=0.9):
    n = len(coords)
    if n < 2:
        return True

    margin = 50
    if state != "Awake":
        concern = False

    else: # sleepy or drawsy
        weights = [decay_rate ** (n - i - 1) for i in range(n)]
        weights = np.array(weights) / sum(weights)  # 가중치 합이 1이 되도록 정규화

        x_coords = np.array([coord[0] for coord in coords])
        y_coords = np.array([coord[1] for coord in coords])

        weighted_mean_x = np.sum(weights * x_coords)
        weighted_mean_y = np.sum(weights * y_coords)

        if (margin < weighted_mean_x < width - margin) and (weighted_mean_y > margin):
            concern = True

        else:
            variance_x = np.sum(weights * (x_coords - weighted_mean_x) ** 2)
            variance_y = np.sum(weights * (y_coords - weighted_mean_y) ** 2)

            if variance_x < std_threshold and variance_y < std_threshold:
                concern = True
            else:
                concern = False

    return concern

def visualize_gaze(frame, gaze_pos, state, max_points=10):
    if not hasattr(visualize_gaze, 'gaze_positions'):
        visualize_gaze.gaze_positions = []
    visualize_gaze.gaze_positions.append(gaze_pos)

    gaze_positions = visualize_gaze.gaze_positions[-max_points:]
    concern = evaluate_focus(gaze_positions, state, frame.shape[1])

    result_frame = frame.copy()
    weights = np.linspace(0.1, 0.6, len(gaze_positions))

    for (x, y), alpha in zip(gaze_positions, weights):
        overlay = result_frame.copy()
        cv2.circle(overlay, (int(x), int(y)), 5, (200, 20, 20), -1)
        cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)

    overlay = result_frame.copy()
    cv2.circle(overlay, gaze_pos, 10, (20, 20, 250), -1)
    cv2.addWeighted(overlay, 0.4, result_frame, 0.6, 0, result_frame)

    return result_frame, concern

def draw_gaze(flag, blink, eye, a, b, c, d, image_in, gaze_angles, mh=720, mw=1280):
    y1 = (mh - image_in.shape[0]) // 2
    y2 = y1 + image_in.shape[0]
    x1 = (mw - image_in.shape[1]) // 2
    x2 = x1 + image_in.shape[1]

    image_out = np.ones((mh, mw, 3), dtype=np.uint8) * 200
    image_out[y1:y2, x1:x2] = image_in
    state = "Awake"

    if not flag:
        return image_out, state, False

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

    image_out, concern = visualize_gaze(image_out, gaze_pos, state)

    return image_out, state, concern

def put_Text(frame, yaw, pitch, state, concern, index, value, fps):
    font1 = ImageFont.truetype('fonts/PretendardVariable.ttf', 20)
    font2 = ImageFont.truetype('fonts/Pretendard-SemiBold.ttf', 25)
    font3 = ImageFont.truetype('fonts/Pretendard-SemiBold.ttf', 40)

    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    draw.text((10, 630), f"Yaw  : {yaw:.2f}",   font=font1, fill=(20, 150, 20, 0))
    draw.text((9, 660), f"Pitch : {pitch:.2f}", font=font1, fill=(20, 150, 20, 0))

    states = {'Awake':'정상', 'Drowsy':'졸음', 'Sleep':'수면'}
    draw.text((10, 690), f"State: {states[state]}", font=font1, fill=(20, 150, 20, 0))

    wise = ["지식에 대한 투자는 \n\n최고의 보상을 \n\n가져다 줄 것이다.\n\n\n- Benjamin Franklin -",
            "많은 실패자들은 \n\n포기하기 때문에 \n\n성공이 얼마나 가까웠는지 \n\n깨닫지 못합니다.\n\n\n- Thomas Edison -",
            "미루는 것은 \n\n쉬운 일을 어렵게 만들고 \n\n어려운 일을 더 어렵게 만든다.\n\n\n– Mason Cooley -",
            "더 이상 상황을 \n\n바꿀 수 없을 때 \n\n우리는 스스로를 \n\n변화시켜야 합니다.\n\n\n– Viktor Frankl -",
            "성적이나 결과는 \n\n행동이 아니라 습관입니다.\n\n\n- Aristoteles -",
            "시작하기 위해 \n\n위대해질 필요는 없지만 \n\n위대해지기 위해서는 \n\n시작해야 합니다.\n\n\n- Zig Ziglar -",
            "배움의 아름다운 점은 \n\n아무도 당신에게서 \n\n그것을 빼앗을 수 \n\n없다는 것입니다.\n\n\n- B.B. King -",
            "나는 공부를 좋아하지 않는다. \n\n나는 공부를 싫어한다. \n\n나는 배우는 것을 좋아한다. \n\n배움은 아름답다.\n\n\n- Natalie Portman -",
            "우리는 우리가 하기를 \n\n원하는 무엇이든 할 수 있다. \n\n만약 우리가 그것에 \n\n충분히 오랫동안 매달린다면\n\n\n- Helen Keller -",
            "물에 빠져서가 아니라, \n\n물속에 가라앉은 채로 \n\n있기 때문에 익사하는 것이다.\n\n\n- Paulo Coelho -"]

    draw.text((980, 120), wise[index//10], font=font2, fill=(20, 20, 20, 0))


    if not concern or value < 40:
        v_text = "집중력이 떨어지고 있습니다. 학습에 집중하세요."
    else:
        v_text = " 열심히 집중하고 있군요! 언제나 화이팅입니다."
    draw.text((405, 675), v_text, font=font2, fill=(255, 255, 255, 0))


    draw.text((1180, 690), f"FPS: {fps:.1f}", font=font1, fill=(0, 0, 0, 0))

    draw.text((15,10), "Mom's Watch", font=font3, fill=(7, 121, 255, 0))

    frame = np.array(pil_img)

    bar_width = 440
    x = (1280 - bar_width) // 2
    cv2.rectangle(frame, (x - 10, 630), (x + bar_width + 10, 660), (50, 50, 50), -1)
    fill_width = int(bar_width * (value / 100))

    cv2.rectangle(frame, (x, 635), (x + fill_width, 655), (7, 121, 255), -1)

    return frame


if __name__ == "__main__":
    prevTime = time.time()
    value = 0
    lack_focus = 0.0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("웹캠을 사용할 수 없습니다.")

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        ori_height, ori_width = frame.shape[:2]
        tar_height = 480
        tar_width = int(ori_width * tar_height / ori_height)
        frame = cv2.resize(frame, (tar_width, tar_height))
        if tar_width > 640:
            crop_x = (tar_width - 640) // 2
            frame = frame[:,crop_x:crop_x+640]

        blink, detected_cls, detected, face_bbox, yaw, pitch = gaze_analysis(frame)

        frame, state, concern = draw_gaze(detected, blink, detected_cls, face_bbox[0], face_bbox[1],
                                          face_bbox[2], face_bbox[3], frame, (yaw, pitch))

        curTime = time.time()
        runTime = curTime - prevTime
        fps = 1 / runTime
        prevTime = curTime

        visual_h = 720
        visual_w = 1280

        if concern:
            value += 3 * runTime
            if value > 100: value = 100
        else:
            lack_focus += runTime
            value -= 2 * runTime
            if value < 0: value = 0

        frame = put_Text(frame, yaw, pitch, state, concern, int(curTime%100), value, fps)

        cv2.imshow('Gaze Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
