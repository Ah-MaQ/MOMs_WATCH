import os
import time
import logging
import json
import random
from PIL import Image

os.chdir("/media/jinsu/My Passport/")

# 로그 파일 설정
logging.basicConfig(filename='yolov8/data/processing_log.txt', level=logging.INFO)

image_list = []
annot_list = []
# 클래스 이름과 ID 매핑
class_mapping = {
    'r_iris': 0,
    'l_iris': 1,
    'r_eyelid': 2,
    'l_eyelid': 3,
    'r_center': 4,
    'l_center': 5
}

for (path, dir, files) in os.walk("./"):
  for filename in files:
    ext = os.path.splitext(filename)[-1]
    if ext == '.png':
      image_list.append("%s/%s" % (path, filename))
    if ext == '.json':
      annot_list.append("%s/%s" % (path, filename))

train_num = int(0.8 * len(image_list))
num = 0

random.shuffle(image_list)

start_time = time.time()
for img_name in image_list:
  img = Image.open(img_name)
  image_width, image_height = img.size
  img_resized = img.resize((640,640))

  if num < train_num:
    output_path = 'yolov8/data/train/'
  else:
    output_path = 'yolov8/data/valid/'

  img_file = os.path.basename(img_name)
  basename = img_file.split('.png')[0]

  img_resized.save(output_path + 'images/' + img_file)

  for annot in annot_list:
    if basename in annot:
      annot_list.remove(annot)
      break

  # 어노테이션 파일을 읽어옵니다.
  with open(annot, 'r') as f:
    data = json.load(f)
  annotations = data['Annotations']['annotations']

  # YOLO 형식의 라벨을 저장할 리스트
  yolo_labels = []

  for annotation in annotations:
      label = annotation['label']
      points = annotation['points']

      # 바운딩 박스의 최대 및 최소 좌표를 계산합니다.
      min_x = min(points, key=lambda x: x[0])[0]
      max_x = max(points, key=lambda x: x[0])[0]
      min_y = min(points, key=lambda x: x[1])[1]
      max_y = max(points, key=lambda x: x[1])[1]

      # 바운딩 박스의 중심과 너비, 높이를 계산합니다.
      bbox_width = max_x - min_x
      bbox_height = max_y - min_y
      center_x = min_x + (bbox_width / 2)
      center_y = min_y + (bbox_height / 2)

      # 이미지 크기에 대한 비율로 변환합니다.
      center_x /= image_width
      center_y /= image_height
      bbox_width /= image_width
      bbox_height /= image_height

      if bbox_width==0: bbox_width = 1/image_width
      if bbox_height==0: bbox_height = 1/image_height

      # 클래스 ID를 가져옵니다.
      class_id = class_mapping[label]

      # YOLO 형식의 라벨을 리스트에 추가합니다.
      yolo_labels.append(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}")


  # YOLO 형식의 라벨을 파일로 저장합니다.
  with open(output_path + 'labels/' + basename + '.txt', 'w') as f:
      for label in yolo_labels:
          f.write(f"{label}\n")

  num += 1

  if num % 1000 == 0:
    elapsed_time = time.time() - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    progress_message = f"Work in progress : {num} / {len(image_list)} - Elapsed time: {elapsed_time_str}"
    print(progress_message)
    logging.info(progress_message)

# 전체 작업 완료 후 로그 기록
total_elapsed_time = time.time() - start_time
total_elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed_time))
completion_message = f"Processing complete: {len(image_list)} images processed - Total elapsed time: {total_elapsed_time_str}"
print(completion_message)
logging.info(completion_message)
logging.info(f"Train : {train_num}, Valid : {len(image_list) - train_num}")
