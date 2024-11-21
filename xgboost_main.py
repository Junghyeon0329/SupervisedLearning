import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb  # SupervisedLearning.xgboost_main 대신 xgboost로 변경

# iris 데이터셋 로드
iris = datasets.load_iris()

# 특징(Feature)와 레이블(Label) 분리
X = iris.data
y = iris.target

# 데이터를 훈련용 데이터와 테스트용 데이터로 나눔 (80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 정의
model = xgb.XGBClassifier(
    n_estimators=100,      # 트리의 개수
    learning_rate=0.1,     # 학습률
    max_depth=3,           # 트리의 최대 깊이
    random_state=42        # 재현성을 위한 랜덤 시드
)

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 예측 결과 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 중요 변수(feature) 확인
importances = model.feature_importances_
for i, importance in enumerate(importances):
    print(f"Feature: {iris.feature_names[i]}, Importance: {importance:.4f}")

# 모델 시각화 (optional)
import matplotlib.pyplot as plt
xgb.plot_importance(model, importance_type="weight", max_num_features=10)
plt.show()
