import io
import cv2
import time
import threading
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from gaze_analysis import gaze_initialize, gaze_analysis, draw_gaze
from eye_detection import yolo_initialize, detect

app = Flask(__name__)
CORS(app)

processed_frame = None
frame_lock = threading.Lock()

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global processed_frame
    try:
        file = request.files['frame']
        frame = Image.open(file.stream)
        frame = np.array(frame)

        processed = process_frame(frame)

        with frame_lock:
            processed_frame = processed['image']

        response = {
            'status': 'success',
            'message': '열공중' if processed['focused'] else '집중부족',
            'data': processed['data']
        }
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Error processing frame: {e}")
        return jsonify({'status': 'error', 'message': 'Error processing frame'}), 500

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detected, face_bbox, yaw, pitch = gaze_analysis(frame)
    detected_cls = detect(frame)
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

        cv2.putText(visualed_img, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), 2)
        cv2.putText(visualed_img, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), 2)

        focused = -0.3 <= yaw <= 0.3 and -0.3 <= pitch <= 0.3
        message = 'Look at me, look at me' if focused else 'Hey, what r u doing?'

        cv2.putText(visualed_img, message, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), 2)

        return {
            # 'image': visualed_img,
            'focused': focused,
            'data': {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'face_bbox': [float(coord) for coord in face_bbox],
                'eye_landmarks': {k: [float(coord) for coord in v] for k, v in detected_cls.items()}
            }
        }

    return {
        # 'image': frame,
        'focused': False,
        'data': {}
    }

if __name__ == '__main__':
    gaze_initialize()
    yolo_initialize()
    app.run(debug=True)
