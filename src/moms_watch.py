import cv2
import time
import queue
import threading
import numpy as np
from PIL import Image

from flask import Flask, request, Response
from flask_cors import CORS

from gaze_analysis import gaze_analysis, draw_gaze

app = Flask(__name__)
CORS(app)

# 전역 큐로 처리되지 않은 프레임과 처리된 프레임을 저장
frame_queue = queue.Queue()
processed_frame_queue = queue.Queue()

def process_frames():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        visual_h = 720
        visual_w = 1280

        blink, detected_cls, detected, face_bbox, yaw, pitch = gaze_analysis(frame)

        visualed_img, state = draw_gaze(detected, blink, detected_cls, face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3],
                                        frame, (yaw, pitch), visual_h, visual_w)

        processed_frame_queue.put(visualed_img)

threading.Thread(target=process_frames, daemon=True).start()

@app.route('/upload_frame', methods=['POST'])
def upload_frames():
    try:
        file = request.files['frame']
        frame = Image.open(file.stream)
        frame = np.array(frame)

        # 프레임 수신 확인 로그 메시지
        app.logger.info("Frame received")

        frame_queue.put(frame)

        return '', 204

    except Exception as e:
        app.logger.error(f"Error processing frame: {e}")
        return '', 500

@app.route('/stream')
def stream():
    def generate():
        while True:
            frame = processed_frame_queue.get()
            if frame is not None:
                # JPEG로 인코딩
                _, jpeg = cv2.imencode('.jpg', frame)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)