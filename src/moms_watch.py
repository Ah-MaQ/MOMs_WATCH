import cv2
import time
import queue
import threading
import numpy as np
from PIL import Image
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from gaze_analysis import gaze_analysis, draw_gaze, put_Text

app = Flask(__name__)
CORS(app)

# 전역 큐로 처리되지 않은 프레임과 처리된 프레임을 저장
frame_queue = queue.Queue(maxsize=5)  # 큐 크기를 줄여서 지연 감소
processed_frame_queue = queue.Queue(maxsize=5)

prevTime = time.time()
value = 0
lack_focus = 0.0
is_there = True
make_alarm = False

def process_frames():
    global prevTime, value, lack_focus, is_there, make_alarm
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ori_height, ori_width = frame.shape[:2]
        tar_height = 480
        tar_width = int(ori_width * tar_height / ori_height)
        frame = cv2.resize(frame, (tar_width, tar_height))
        if tar_width > 640:
            crop_x = (tar_width - 640) // 2
            frame = frame[:,crop_x:crop_x+640]

        visual_h = 720
        visual_w = 1280

        blink, detected_cls, detected, face_bbox, yaw, pitch = gaze_analysis(frame)

        visualed_img, state, concern = draw_gaze(detected, blink, detected_cls, face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3],
                                                 frame, (yaw, pitch), visual_h, visual_w)

        is_there = detected
        make_alarm = True if state != "Awake" else False

        curTime = time.time()
        runTime = curTime - prevTime
        if runTime > 0.15: runTime = 0.15
        fps = 1 / runTime
        prevTime = curTime

        if concern:
            value += 3 * runTime
            if value > 100: value = 100
        else:
            lack_focus += runTime
            value -= 2 * runTime
            if value < 0: value = 0

        visualed_img = put_Text(visualed_img, yaw, pitch, state, concern, int(curTime%100), value, fps)

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
    global is_there, make_alarm, lack_focus

    response = {
        'is_there': is_there,
        'make_alarm': make_alarm,
        'lack_focus': lack_focus
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
