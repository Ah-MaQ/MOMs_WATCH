from flask import Flask, request
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # CORS 설정 추가

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    try:
        file = request.files['frame']
        img = Image.open(file.stream)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 로그 메시지로 프레임 수신 확인
        app.logger.info("Frame received")

        # OpenCV를 사용하여 이미지를 처리 (예: 흑백 변환)
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 처리된 이미지를 파일로 저장
        cv2.imwrite('received_frame.jpg', processed_img)

        return '', 204
    except Exception as e:
        app.logger.error(f"Error processing frame: {e}")
        return '', 500

if __name__ == '__main__':
    app.run(debug=True)
