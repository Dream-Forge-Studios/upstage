import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler
from eda_utils import makingData_score_sum
from joblib import dump

df_2019 = makingData_score_sum('statuzGame_2019_score_sum.json')
df_2020 = makingData_score_sum('statuzGame_2020_score_sum.json')
df_2021 = makingData_score_sum('statuzGame_2021_score_sum.json')
df_2022 = makingData_score_sum('statuzGame_2022_score_sum.json')
df_2023 = makingData_score_sum('statuzGame_2023_score_sum.json')
combined_df = pd.concat([df_2019, df_2020, df_2021, df_2022, df_2023],
                        ignore_index=True)


# 특성과 타겟 분리
X = combined_df.drop(['away_result', 'home_result', 'game_id'], axis=1)
y = combined_df['score_sum']

feature_names = X.columns
accuracy_results = []
filtered_accuracy_results = []
filtered_labels_results = []
for random_state in [42, 11, 2, 6, 38]:
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 로지스틱 회귀 모델 생성 및 학습
    log_reg = LogisticRegression(random_state=random_state)
    log_reg.fit(X_train_scaled, y_train)

    # 테스트 데이터에 대한 예측
    predictions = log_reg.predict(X_test_scaled)

    # 성능 평가
    accuracy = accuracy_score(y_test, predictions)
    accuracy_results.append(accuracy)
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(report)

    # 예측 확률을 구함
    probabilities = log_reg.predict_proba(X_test_scaled)
    # 홈 팀 승리에 대한 확률이 60% 이상이거나 40% 이하인 데이터만 필터링
    high_threshold = 0.6
    low_threshold = 0.4
    selected_indices = np.where((probabilities[:, 1] >= high_threshold) | (probabilities[:, 1] <= low_threshold))[0]
    print(f"Number of selected predictions: {len(selected_indices)}")
    filtered_labels_results.append(len(selected_indices))
    # 선택된 인덱스를 사용하여 필터링된 테스트 데이터의 레이블과 예측값을 추출
    filtered_labels = y_test.iloc[selected_indices]
    filtered_predictions = predictions[selected_indices]

    # 필터링된 데이터에 대해 정확도 계산
    if len(filtered_labels) > 0:
        filtered_accuracy = accuracy_score(filtered_labels, filtered_predictions)
        filtered_accuracy_results.append(filtered_accuracy)
        print(f'Filtered Accuracy (predictions with p>=60% or p<=40%): {filtered_accuracy:.2f}')
    else:
        print("No predictions with probability >= 60% or <= 40%.")

    dump(log_reg, f'logreg_model_{random_state}.joblib')

accuracy_average = sum(accuracy_results) / len(accuracy_results)
filtered_accuracy_average = sum(filtered_accuracy_results) / len(filtered_accuracy_results)
filtered_labels_average = sum(filtered_labels_results) / len(filtered_labels_results)
print("정확도:", accuracy_average)
print("60% 정확도:", filtered_accuracy_average)
print("개수:", filtered_labels_average)