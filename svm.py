import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import numpy as np
from eda_utils import makingData
from sklearn.inspection import permutation_importance

df_2010 = makingData('statuzGame_2010.json')
df_2011 = makingData('statuzGame_2011.json')
df_2012 = makingData('statuzGame_2012.json')
df_2013 = makingData('statuzGame_2013.json')
df_2014 = makingData('statuzGame_2014.json')
df_2015 = makingData('statuzGame_2015.json')
df_2016 = makingData('statuzGame_2016.json')
df_2017 = makingData('statuzGame_2017.json')
df_2018 = makingData('statuzGame_2018.json')
df_2019 = makingData('statuzGame_2019.json')
df_2020 = makingData('statuzGame_2020.json')
df_2021 = makingData('statuzGame_2021.json')
df_2022 = makingData('statuzGame_2022.json')
df_2023 = makingData('statuzGame_2023.json')
combined_df = pd.concat([df_2010, df_2011, df_2012, df_2013, df_2014, df_2015, df_2016, df_2017, df_2018, df_2019, df_2020, df_2021, df_2022, df_2023],
                        ignore_index=True)
print(len(combined_df))
# combined_df = pd.concat([df_2020, df_2021, df_2022, df_2023],
#                         ignore_index=True)
# 특성과 타겟 분리
X = combined_df.drop(['away_result', 'home_result', 'game_id'], axis=1)
# X = combined_df.drop(['home_result', 'game_id', 'away_result', 'home_score_10', 'home_conceded_10', 'away_score_10', 'away_conceded_10', 'home_win_rate_10', 'away_win_rate_10'], axis=1)
y = combined_df['home_result']  # 홈 팀 기준 결과 사용

accuracy_results = []
filtered_accuracy_results = []
filtered_labels_results = []
# for random_state in [42, 11, 2, 6, 38]:
for random_state in [38]:
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SVM 모델 학습
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train_scaled, y_train)

    # dump(svm_model, 'svm_model_2015~2023.joblib')

    # 성능 평가
    predictions = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_results.append(accuracy)
    print(accuracy)
    # 예측 확률을 구함
    probabilities = svm_model.predict_proba(X_test_scaled)
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

accuracy_average = sum(accuracy_results) / len(accuracy_results)
filtered_accuracy_average = sum(filtered_accuracy_results) / len(filtered_accuracy_results)
filtered_labels_average = sum(filtered_labels_results) / len(filtered_labels_results)
print("정확도:", accuracy_average)
print("60% 이상 정확도:", filtered_accuracy_average)
print("60% 이상 개수:", filtered_labels_average)

# import matplotlib.pyplot as plt
#
# # 모델 성능 평가 및 중요도 계산
# perm_importance = permutation_importance(svm_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
#
# # 특성 중요도 추출
# feature_importances = perm_importance.importances_mean
#
# # 중요도가 높은 순서로 특성 인덱스 정렬
# sorted_idx = feature_importances.argsort()
#
# # 시각화
# plt.figure(figsize=(12, 8))
# plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
# plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
# plt.xlabel("Permutation Feature Importance")
# plt.title("Feature Importance")
# plt.show()