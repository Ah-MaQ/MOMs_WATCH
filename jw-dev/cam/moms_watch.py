from flask import Flask, request, Response
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import threading
import queue

from gaze_analysis_cam_ver import gaze_analysis, draw_gaze
# from state_awareness_cam_ver import state_awareness

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

        detected, face_bbox, yaw, pitch = gaze_analysis(frame)
        # cur_state = state_awareness(frame)

        if detected:
            frame = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch))

        cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        if detected and -0.3 <= yaw <= 0.3 and -0.3 <= pitch <= 0.3:
            cv2.putText(frame, 'Look at me, look at me', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            # cv2.putText(frame, f'Your now in {cur_state}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        processed_frame_queue.put(frame)


threading.Thread(target=process_frames, daemon=True).start()


@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    try:
        file = request.files['frame']
        img = Image.open(file.stream)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 프레임 수신 확인 로그 메시지
        app.logger.info("Frame received")

        frame = cv2.flip(img, 1)
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
