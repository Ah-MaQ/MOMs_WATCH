import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from models.ST_Former import GenerateModel

# 환경 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 모델 로드 및 설정
fn_model = "./checkpoint/trained.pth"
model = GenerateModel()
model = torch.nn.DataParallel(model).cuda()

# 모델 가중치 로드
checkpoint = torch.load(fn_model)
model.load_state_dict(checkpoint['state_dict'])

# 손실 함수 정의
criterion = nn.CrossEntropyLoss().cuda()

# 모델 평가 모드 설정
model.eval()


# 결과 시각화 함수
def visualize(frame, output):
    _, pred = torch.max(output, 1)
    cv2.putText(frame, f'Prediction: {pred.item()}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


# 웹캠 초기화 및 이미지 캡처 루프
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # 모델 입력 크기에 맞게 조정
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = img / 255.0  # 정규화
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).cuda()  # 배치 차원 추가 및 GPU로 이동

    # 모델 수행
    with torch.no_grad():
        output = model(img)

    # 결과 시각화
    frame = visualize(frame, output)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
