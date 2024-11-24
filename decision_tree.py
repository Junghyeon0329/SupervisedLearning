import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# iris 데이터셋 로드
# iris = datasets.load_iris()

# 특징(Feature)와 레이블(Label) 분리
# X = iris.data
# y = iris.target

df = pd.read_csv('otto_train.csv')

# 2. 특징(Feature)와 레이블(Label) 분리
# 여기서는 마지막 열이 레이블이라고 가정
X = df.iloc[:, 1:-1].values  # 첫 번째 컬럼은 ID일 가능성이 높으므로 제외, 마지막 컬럼은 레이블
y = df.iloc[:, -1].values  # 레이블은 마지막 컬럼

# 데이터를 훈련용 데이터와 테스트용 데이터로 나눔 (80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=3,  # 트리의 최대 깊이 제한
    min_samples_split=4,  # 분할을 위한 최소 샘플 수
    min_samples_leaf=2,  # 리프 노드의 최소 샘플 수
    max_features="sqrt"  # 분할에 사용할 최대 특징 개수
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after tuning: {accuracy * 100:.2f}%")

# 결정 트리를 시각화
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True, fontsize=12)
plt.show()
