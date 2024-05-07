import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import numpy as np
from eda_utils import makingData

df_2020 = makingData('statuzGame_2020.json')
df_2021 = makingData('statuzGame_2021.json')
df_2022 = makingData('statuzGame_2022.json')
df_2023 = makingData('statuzGame_2023.json')

combined_df = pd.concat([df_2020, df_2021, df_2022, df_2023], ignore_index=True)
# 특성과 타겟 분리
X = combined_df.drop(['away_result', 'home_result', 'game_id'], axis=1)  # 게임 ID 제거 및 결과 분리
y = combined_df['home_result']  # 홈 팀 기준 결과 사용

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM 모델 학습
svm_model = SVC(probability=True)  # 확률 추정이 가능하도록 설정
svm_model.fit(X_train_scaled, y_train)

# dump(svm_model, 'svm_model.joblib')

# 성능 평가
predictions = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
# 예측 확률을 구함
probabilities = svm_model.predict_proba(X_test_scaled)
# 홈 팀 승리에 대한 확률이 60% 이상이거나 40% 이하인 데이터만 필터링
high_threshold = 0.6
low_threshold = 0.4
selected_indices = np.where((probabilities[:, 1] >= high_threshold) | (probabilities[:, 1] <= low_threshold))[0]
print(f"Number of selected predictions: {len(selected_indices)}")

# 선택된 인덱스를 사용하여 필터링된 테스트 데이터의 레이블과 예측값을 추출
filtered_labels = y_test.iloc[selected_indices]
filtered_predictions = predictions[selected_indices]

# 필터링된 데이터에 대해 정확도 계산
if len(filtered_labels) > 0:
    filtered_accuracy = accuracy_score(filtered_labels, filtered_predictions)
    print(f'Filtered Accuracy (predictions with p>=60% or p<=40%): {filtered_accuracy:.2f}')
else:
    print("No predictions with probability >= 60% or <= 40%.")
