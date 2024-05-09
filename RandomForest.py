from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from joblib import dump
import numpy as np
import matplotlib.pyplot as plt
from eda_utils import makingData

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

# 특성과 타겟 분리
X = combined_df.drop(['away_result', 'home_result', 'game_id'], axis=1)
# X = combined_df.drop(['home_result', 'game_id', 'away_result', 'home_score_10', 'home_conceded_10', 'away_score_10', 'away_conceded_10', 'away_score', 'home_score', 'away_conceded', 'home_conceded', 'away_win_rate_10', 'home_win_rate_10'], axis=1)
y = combined_df['home_result']  # 홈 팀 기준 결과 사용

feature_names = X.columns
accuracy_results = []
filtered_accuracy_results = []
filtered_labels_results = []
for random_state in [42, 11, 2, 6, 38]:
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 랜덤 포레스트 모델 초기화
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_classifier.fit(X_train, y_train)

    # 특성 중요도 추출
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]  # 중요도에 따라 특성 인덱스를 내림차순으로 정렬

    # 특성 중요도 시각화
    # plt.figure(figsize=(12, 6))
    # plt.title('Feature Importances')
    # plt.bar(range(X_train.shape[1]), importances[indices], color='b', align='center')
    # plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    # plt.xlabel('Feature')
    # plt.ylabel('Importance')
    # plt.show()
    # 모델 학습
    rf_classifier.fit(X_train, y_train)

    dump(rf_classifier, 'RandomForest_model.joblib')

    # 테스트 데이터에 대한 예측
    predictions = rf_classifier.predict(X_test)

    # 성능 평가
    accuracy = accuracy_score(y_test, predictions)
    accuracy_results.append(accuracy)
    report = classification_report(y_test, predictions)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")

    # 예측 확률을 구함
    probabilities = rf_classifier.predict_proba(X_test)
    # 홈 팀 승리에 대한 확률이 60% 이상이거나 40% 이하인 데이터만 필터링
    high_threshold = 0.7
    low_threshold = 0.3
    selected_indices = np.where((probabilities[:, 1] >= high_threshold) | (probabilities[:, 1] <= low_threshold))[0]
    print(f"Number of selected predictions: {len(selected_indices)}")
    filtered_labels_results.append(len(selected_indices))
    # 선택된 인덱스를 사용하여 필터링된 테스트 데이터의 레이블과 예측값을 추출
    filtered_labels = y_test.iloc[selected_indices]
    filtered_predictions = predictions[selected_indices]

    # 필터링된 데이터에 대해 정확도 계산
    if len(filtered_labels) > 0:
        filtered_accuracy = accuracy_score(filtered_labels, filtered_predictions)
        print(f'Filtered Accuracy (predictions with p>=60% or p<=40%): {filtered_accuracy:.2f}')
        filtered_accuracy_results.append(filtered_accuracy)
    else:
        print("No predictions with probability >= 60% or <= 40%.")

accuracy_average = sum(accuracy_results) / len(accuracy_results)
filtered_accuracy_average = sum(filtered_accuracy_results) / len(filtered_accuracy_results)
filtered_labels_average = sum(filtered_labels_results) / len(filtered_labels_results)
print("정확도:", accuracy_average)
print("60% 정확도:", filtered_accuracy_average)
print("개수:", filtered_labels_average)
