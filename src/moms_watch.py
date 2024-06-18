import io
import cv2
import time
import threading
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from flask import Flask, request, Response
from flask_cors import CORS

from gaze_analysis import gaze_initialize, gaze_analysis, draw_gaze
from eye_detection import yolo_initialize, detect
# from state_awareness import state_initialize, state_aware

app = Flask(__name__)
CORS(app)

# 전역 변수로 처리된 프레임 저장
processed_frame = None
frame_lock = threading.Lock()  # Add a lock for thread safety

@app.route('/upload_frame', methods=['POST'])
def upload_frames():
    global processed_frame
    try:
        file = request.files['frame']
        frame = Image.open(file.stream)
        frame = np.array(frame)

        # Process the frame
        processed = process_frame(frame)

        with frame_lock:
            processed_frame = processed

        # 프레임 수신 확인 로그 메시지
        app.logger.info("Frame received")

        return '', 204

    except Exception as e:
        app.logger.error(f"Error processing frame: {e}")
        return '', 500

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detected, face_bbox, yaw, pitch = gaze_analysis(frame)

    detected_cls = detect(frame)
    # cur_state = state_aware(frame)
    visualed_img = frame.copy()

    if detected:
        visualed_img, _ = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch))

        y1 = (visualed_img.shape[0] - frame.shape[0]) // 2
        x1 = (visualed_img.shape[1] - frame.shape[1]) // 2
        if "r_iris" in detected_cls and "r_eyelid" in detected_cls:
            xc, yc, eye_r = detected_cls["r_iris"]
            cv2.circle(visualed_img, (x1 + xc, y1 + yc), eye_r, (0, 0, 255), thickness=1)

        if "l_iris" in detected_cls and "l_eyelid" in detected_cls:
            xc, yc, eye_r = detected_cls["l_iris"]
            cv2.circle(visualed_img, (x1 + xc, y1 + yc), eye_r, (0, 0, 255), thickness=1)

        cv2.putText(visualed_img, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (50, 200, 50), 2)
        cv2.putText(visualed_img, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (50, 200, 50), 2)
        if detected and -0.3 <= yaw <= 0.3 and -0.3 <= pitch <= 0.3:
            cv2.putText(visualed_img, 'Look at me, look at me', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
            # cv2.putText(visualed_img, f'Your now in {cur_state}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (50, 200, 50), 2)
        else:
            cv2.putText(visualed_img, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)

    return visualed_img

@app.route('/stream')
def stream():
    def generate():
        while True:
            with frame_lock:
                if processed_frame is not None:
                    _, jpeg = cv2.imencode('.jpg', processed_frame)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.1)  # Adjust the sleep time as needed

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    gaze_initialize()
    yolo_initialize()
    # state_initialize()

    app.run(debug=True)