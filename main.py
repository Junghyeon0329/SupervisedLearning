
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score # 정확도 함수

# 데이터 로드 및 전처리
df = pd.read_csv("./otto_train.csv")
df = df.drop(['id'], axis=1)

# 타겟 변수의 형변환
mapping_dict = {
    'Class_1': 1, 'Class_2': 2, 'Class_3': 3, 'Class_4': 4,
    'Class_5': 5, 'Class_6': 6, 'Class_7': 7, 'Class_8': 8, 'Class_9': 9
}
after_mapping_target = df['target'].apply(lambda x: mapping_dict[x])
feature_columns = list(df.columns.difference(['target']))
X = df[feature_columns]
y = after_mapping_target

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
}

clf = RandomForestClassifier(random_state=0)
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(train_x, train_y)

print("최적의 하이퍼파라미터:", grid_search.best_params_)

best_clf = grid_search.best_estimator_
predict = best_clf.predict(test_x)

print("최종 정확도:", accuracy_score(test_y, predict))
