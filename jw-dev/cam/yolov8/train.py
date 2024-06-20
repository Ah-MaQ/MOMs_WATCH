from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data='./data.yaml',  # 데이터 설정 파일 경로
    epochs=50,          # 학습 에포크 수
    batch=64,            # 배치 크기
    lr0=0.01,            # 초기 학습률
    momentum=0.937,      # 모멘텀
    weight_decay=0.0005, # 가중치 감소
    imgsz=640,        # 이미지 크기
    augment=True         # 데이터 증강 사용
)
