import cv2
import numpy as np
from flask import Flask, Response

from gaze_analysis import gaze_initialize, gaze_analysis, draw_gaze
from eye_detection import yolo_initialize, detect
# from state_awareness import state_awareness
from tkinter import *

root = Tk()

monitor_height = root.winfo_screenheight()
monitor_width = root.winfo_screenwidth()

app = Flask(__name__)

def generate_frames():
    gaze_initialize()
    yolo_initialize()
    cap = cv2.VideoCapture(0)  # 웹캠에서 영상을 캡처
    while True:
        success, frame = cap.read()  # 프레임 읽기
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            detected, face_bbox, yaw, pitch = gaze_analysis(frame)
            detected_cls = detect(frame)
            # cur_state = state_awareness(frame)

            if detected:
                # bbox & gaze 렌더링
                visualed_img = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch), monitor_height, monitor_width)

            y1 = (monitor_height - frame.shape[0]) // 2
            x1 = (monitor_width - frame.shape[1]) // 2
            if "r_iris" in detected_cls and "r_eyelid" in detected_cls:
                xc, yc, eye_r = detected_cls["r_iris"]
                cv2.circle(visualed_img, (x1+xc, y1+yc), eye_r, (0, 0, 255), thickness=1)

            if "l_iris" in detected_cls and "l_eyelid" in detected_cls:
                xc, yc, eye_r = detected_cls["l_iris"]
                cv2.circle(visualed_img, (x1+xc, y1+yc), eye_r, (0, 0, 255), thickness=1)

            cv2.putText(visualed_img, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
            cv2.putText(visualed_img, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
            if detected and -0.3 <= yaw <= 0.3 and -0.3 <= pitch <= 0.3:
                cv2.putText(visualed_img, 'Look at me, look at me', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (50, 200, 50), 2)
            else:
                cv2.putText(visualed_img, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (50, 200, 50), 2)


            ret, buffer = cv2.imencode('.jpg', visualed_img)
            output_img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output_img + b'\r\n')  # 프레임을 JPEG 형식으로 인코딩


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
