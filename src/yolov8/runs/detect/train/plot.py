import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# results.csv 파일 읽기
df = pd.read_csv('results.csv')

# 컬럼명 앞뒤 공백 제거
df.columns = [col.strip() for col in df.columns]

# 에폭스 수
current_epochs = len(df)
additional_epochs = 50  # 추가하려는 에폭스 수
total_epochs = current_epochs + additional_epochs

# 함수: 선형 회귀를 사용하여 순차적으로 예측 값 생성
def extend_data_sequential(data, additional_epochs, min_value=None, max_value=None):
    extended_data = data.tolist()
    for i in range(additional_epochs):
        x = np.arange(len(extended_data))
        slope, intercept, _, _, _ = linregress(x, extended_data)
        next_value = intercept + slope * (len(extended_data))
        if min_value is not None:
            next_value = max(next_value, min_value)
        if max_value is not None:
            next_value = min(next_value, max_value)
        extended_data.append(next_value)
    return np.array(extended_data)

# 그래프 그리기
plt.figure(figsize=(12, 8))

# Train losses
plt.subplot(2, 2, 1)
plt.plot(extend_data_sequential(df['train/box_loss'], additional_epochs, min_value=0), label='Train Box Loss')
plt.plot(extend_data_sequential(df['train/cls_loss'], additional_epochs, min_value=0), label='Train Class Loss')
plt.plot(extend_data_sequential(df['train/dfl_loss'], additional_epochs, min_value=0), label='Train DFL Loss')
plt.axvline(x=current_epochs, color='r', linestyle='--', label='Current Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()

# Validation losses
plt.subplot(2, 2, 2)
plt.plot(extend_data_sequential(df['val/box_loss'], additional_epochs, min_value=0), label='Validation Box Loss')
plt.plot(extend_data_sequential(df['val/cls_loss'], additional_epochs, min_value=0), label='Validation Class Loss')
plt.plot(extend_data_sequential(df['val/dfl_loss'], additional_epochs, min_value=0), label='Validation DFL Loss')
plt.axvline(x=current_epochs, color='r', linestyle='--', label='Current Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Losses')
plt.legend()

# Precision, Recall, mAP 그래프 그리기
plt.subplot(2, 2, 3)
plt.plot(extend_data_sequential(df['metrics/precision(B)'], additional_epochs, min_value=0, max_value=1), label='Precision')
plt.plot(extend_data_sequential(df['metrics/recall(B)'], additional_epochs, min_value=0, max_value=1), label='Recall')
plt.plot(extend_data_sequential(df['metrics/mAP50(B)'], additional_epochs, min_value=0, max_value=1), label='mAP50')
plt.plot(extend_data_sequential(df['metrics/mAP50-95(B)'], additional_epochs, min_value=0, max_value=1), label='mAP50-95')
plt.axvline(x=current_epochs, color='r', linestyle='--', label='Current Epochs')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.title('Precision, Recall and mAP')
plt.legend()

# Learning rates
plt.subplot(2, 2, 4)
plt.plot(extend_data_sequential(df['lr/pg0'], additional_epochs, min_value=0), label='Learning Rate pg0')
plt.plot(extend_data_sequential(df['lr/pg1'], additional_epochs, min_value=0), label='Learning Rate pg1')
plt.plot(extend_data_sequential(df['lr/pg2'], additional_epochs, min_value=0), label='Learning Rate pg2')
plt.axvline(x=current_epochs, color='r', linestyle='--', label='Current Epochs')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rates')
plt.legend()

plt.tight_layout()
plt.show()
