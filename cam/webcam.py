from flask import Flask, request, Response
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# 전역 변수로 처리된 프레임 저장
processed_frame = None

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global processed_frame
    try:
        file = request.files['frame']
        img = Image.open(file.stream)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 프레임 수신 확인 로그 메시지
        app.logger.info("Frame received")

        # OpenCV를 사용하여 이미지를 처리 (예: 흑백 변환)
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 처리된 프레임을 전역 변수에 저장
        processed_frame = processed_img

        return '', 204

    except Exception as e:
        app.logger.error(f"Error processing frame: {e}")
        return '', 500

@app.route('/stream')
def stream():
    def generate():
        global processed_frame
        while True:
            if processed_frame is not None:
                # JPEG로 인코딩
                _, jpeg = cv2.imencode('.jpg', processed_frame)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
