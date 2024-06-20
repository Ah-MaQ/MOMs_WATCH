import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def annotation_show(filename):
  with open(filename, 'r') as f:
      data = json.load(f)

  annotations = data['Annotations']

  # 이미지를 불러옵니다.
  image_path = 'img.png'
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 형식이므로 RGB로 변환

  # 어노테이션을 그립니다.
  for annotation in annotations['annotations']:
      label = annotation['label']
      points = annotation['points']
      shape = annotation['shape']

      if shape == 'Ellipse':
          cx, cy = annotation['cx'], annotation['cy']
          rx, ry = annotation['rx'], annotation['ry']
          rotate = annotation['rotate']
          center = (int(cx), int(cy))
          axes = (int(rx), int(ry))
          cv2.ellipse(image, center, axes, rotate, 0, 360, (255, 0, 0), 2)

      elif shape == 'Polygon':
          points = np.array(points, np.int32)
          points = points.reshape((-1, 1, 2))
          cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

      elif shape == 'Point':
          point = points[0]
          cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)
  return image

def boundingbox_show(filename):
  with open(filename, 'r') as f:
     data = f.read()

  yolo_labels = data.split('\n')[:-1]

  # 이미지를 불러옵니다.
  image_path = 'img.png'
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 형식이므로 RGB로 변환

  colors = [(255,0,0),(230,20,0),(200,50,20),(100,100,100),(50,200,0),(0,0,250)]

  image_height, image_width = image.shape[:2]
  for label in yolo_labels:
    parts = label.split()
    class_id = int(parts[0])
    cx = float(parts[1]) * image_width
    cy = float(parts[2]) * image_height
    width = float(parts[3]) * image_width
    height = float(parts[4]) * image_height
    # if width==0: width = 1
    # if height==0: height = 1

    x_min = int(cx - width / 2)
    y_min = int(cy - height / 2)
    x_max = int(cx + width / 2)
    y_max = int(cy + height / 2)

    color = colors[class_id]
    # 바운딩 박스를 이미지에 그리기
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(image, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

  return image

#image = annotation_show('label.json')
image = boundingbox_show('label.txt')

# 이미지를 출력합니다.
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()
