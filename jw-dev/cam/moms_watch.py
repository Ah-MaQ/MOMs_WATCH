from flask import Flask, request, Response
from flask_cors import CORS
import time
import cv2
import numpy as np
from PIL import Image
import threading
import queue

from gaze_analysis_cam_ver import define_aoi, gaze_analysis, draw_gaze, detect, concern  # , evaluate_focus, visualize_gaze_clusters
# from state_awareness_cam_ver import state_awareness

app = Flask(__name__)
CORS(app)

# 전역 큐로 처리되지 않은 프레임과 처리된 프레임을 저장
frame_queue = queue.Queue()
processed_frame_queue = queue.Queue()


def process_frames():
    aoi, dpi = define_aoi()

    # 동공이 감지되지 않은 시간을 추적하기 위한 타이머
    no_r_iris_time = 0
    no_l_iris_time = 0

    start_time_r = None
    start_time_l = None

    warning_message = ""

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

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
        # cur_state = state_awareness(frame)

        if detected:
            frame, gaze_pos = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch))
            concentration_status = concern(gaze_pos, aoi)
            # focus_status = evaluate_focus(visualize_gaze_clusters.gaze_positions, dpi=dpi)

        cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        if detected:    # and -0.3 <= yaw <= 0.3 and -0.3 <= pitch <= 0.3
            # cv2.putText(frame, 'Look at me, look at me', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 255, 0), 2)
            cv2.putText(frame, concentration_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
            # cv2.putText(frame, focus_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (50, 200, 50), 2)
            cv2.putText(frame, warning_message, (10, 190), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        # else:
        #     cv2.putText(frame, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
        #                 1, (0, 255, 0), 2)

        processed_frame_queue.put(frame)
        app.logger.info("Frame processe")


threading.Thread(target=process_frames, daemon=True).start()


@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    try:
        file = request.files['frame']
        img = Image.open(file.stream)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 프레임 수신 확인 로그 메시지
        # app.logger.info("Frame received")

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
