from flask import Flask, request, send_file
from PIL import Image
import io

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']
    image = Image.open(image_file)

    # 이미지 좌우 반전 및 리사이즈
    processed_image = image.transpose(Image.FLIP_LEFT_RIGHT).resize((image.width // 2, image.height // 2))

    img_io = io.BytesIO()
    processed_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('selfsigned.crt', 'selfsigned.key'))
