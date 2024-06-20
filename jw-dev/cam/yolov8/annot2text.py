import json

# 클래스 이름과 ID 매핑
class_mapping = {
    'r_iris': 0,
    'l_iris': 1,
    'r_eyelid': 2,
    'l_eyelid': 3,
    'r_center': 4,
    'l_center': 5
}

# 어노테이션 파일을 읽어옵니다.
with open('label.json', 'r') as f:
    data = json.load(f)

annotations = data['Annotations']['annotations']
image_width = 1920  # 이미지 너비
image_height = 1080  # 이미지 높이

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
with open('label.txt', 'w') as f:
    for label in yolo_labels:
        f.write(f"{label}\n")
