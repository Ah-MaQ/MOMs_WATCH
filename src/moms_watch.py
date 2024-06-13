import cv2
from flask import Flask, Response

from gaze_analysis import gaze_initialize, gaze_analysis, draw_gaze
from eye_detection import yolo_initialize, detect
# from state_awareness import state_awareness


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

            if "r_iris" in detected_cls and "r_eyelid" in detected_cls:
                xc, yc, eye_r = detected_cls["r_iris"]
                cv2.circle(frame, (xc, yc), eye_r, (0, 0, 255), thickness=1)

            if "l_iris" in detected_cls and "l_eyelid" in detected_cls:
                xc, yc, eye_r = detected_cls["l_iris"]
                cv2.circle(frame, (xc, yc), eye_r, (0, 0, 255), thickness=1)

            if detected:
                # bbox & gaze 렌더링
                frame = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch))

            cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            if detected and -0.3 <= yaw <= 0.3 and -0.3 <= pitch <= 0.3:
                cv2.putText(frame, 'Look at me, look at me', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 프레임을 JPEG 형식으로 인코딩


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
