import cv2
from flask import Flask, Response
import threading
import time

from gaze_analysis import gaze_initialize, gaze_analysis, draw_gaze
from eye_detection import yolo_initialize, detect
from state_awareness import state_initialize, state_aware
from tkinter import *

# Tkinter to get screen resolution
root = Tk()
monitor_height = root.winfo_screenheight()
monitor_width = root.winfo_screenwidth()

app = Flask(__name__)

cap = cv2.VideoCapture(0)  # Initialize webcam

# Frame resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set frame rate
desired_fps = 10
frame_time = 1 / desired_fps

# Global frame storage
global_frame = None
lock = threading.Lock()

def capture_frames():
    global global_frame
    gaze_initialize()
    yolo_initialize()
    state_initialize()
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)  # Flip the frame
        with lock:
            global_frame = frame
        time.sleep(frame_time)

def process_frame(frame):
    detected, face_bbox, yaw, pitch = gaze_analysis(frame)
    detected_cls = detect(frame)
    cur_state = state_aware(frame)
    visualed_img = frame.copy()

    if detected:
        visualed_img, _ = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch),
                                    monitor_height, monitor_width)

        y1 = (monitor_height - frame.shape[0]) // 2
        x1 = (monitor_width - frame.shape[1]) // 2
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
            cv2.putText(visualed_img, f'Your now in {cur_state}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)
        else:
            cv2.putText(visualed_img, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (50, 200, 50), 2)

    return visualed_img

def generate_frames():
    while True:
        with lock:
            if global_frame is None:
                continue
            frame = global_frame.copy()

        processed_frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        output_img = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + output_img + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start video capture thread
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)
