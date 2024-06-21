import cv2
import time
import queue
import threading
import numpy as np
from PIL import Image
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from gaze_analysis import gaze_analysis, draw_gaze

app = Flask(__name__)
CORS(app)

# 전역 큐로 처리되지 않은 프레임과 처리된 프레임을 저장
frame_queue = queue.Queue(maxsize=5)  # 큐 크기를 줄여서 지연 감소
processed_frame_queue = queue.Queue(maxsize=5)

prevTime = time.time()
is_there = True
make_alarm = False

def process_frames():
    global prevTime, is_there, make_alarm
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

        is_there = detected
        make_alarm = True if state != "Awake" else False

        # 프레임 수 계산
        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime
        # 프레임 수 문자열에 저장
        fps_str = "FPS : %0.1f" % fps
        cv2.putText(visualed_img, fps_str, (visual_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # 처리된 프레임을 큐에 넣기 전에 큐 크기를 확인하고 오래된 프레임을 버림
        if processed_frame_queue.full():
            processed_frame_queue.get()  # 오래된 프레임 제거

        processed_frame_queue.put(visualed_img)

# 스레드를 통해 프레임 처리 시작
threading.Thread(target=process_frames, daemon=True).start()

@app.route('/upload_frame', methods=['POST'])
def upload_frames():
    try:
        file = request.files['frame']
        frame = Image.open(file.stream)
        frame = np.array(frame)

        # 프레임 수신 확인 로그 메시지
        app.logger.info("Frame received")

        if frame_queue.full():
            frame_queue.get()  # 오래된 프레임 제거

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

@app.route('/get_status', methods=['GET'])
def get_status():
    global is_there, make_alarm

    response = {
        'is_there': is_there,
        'make_alarm': make_alarm
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
