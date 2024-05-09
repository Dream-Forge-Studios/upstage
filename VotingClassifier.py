from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from eda_utils import makingData
import pandas as pd
import numpy as np

# 데이터 생성
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
X = combined_df.drop(['away_result', 'home_result', 'game_id'], axis=1)  # 게임 ID 제거 및 결과 분리
y = combined_df['home_result']  # 홈 팀 기준 결과 사용
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 개별 모델 설정
rf = RandomForestClassifier(n_estimators=100, random_state=42)
logreg = LogisticRegression(random_state=42)
svm = SVC(probability=True, random_state=42)
nn = MLPClassifier(random_state=42)

# 소프트 보팅을 사용한 앙상블 모델 설정
ensemble = VotingClassifier(estimators=[
    ('random_forest', rf),
    ('logistic_regression', logreg),
    ('svm', svm),
    ('neural_network', nn)
], voting='soft')

# 모델 훈련
ensemble.fit(X_train, y_train)

# 예측 확률 얻기
y_prob = ensemble.predict_proba(X_test)

# 최대 확률을 구하여 예측값과 비교
y_prob_max = np.max(y_prob, axis=1)  # 각 샘플에 대해 가장 높은 확률을 가진 클래스의 확률
y_pred = np.argmax(y_prob, axis=1)  # 확률이 가장 높은 클래스의 인덱스를 예측값으로 사용

# 전체 정확도 계산
overall_accuracy = accuracy_score(y_test, y_pred)
print(f'Overall Ensemble Accuracy: {overall_accuracy:.4f}')

# 확률이 40% 이하 또는 60% 이상인 샘플에 대한 정확도
selected_mask = (y_prob_max <= 0.4) | (y_prob_max >= 0.6)
selected_y_pred = y_pred[selected_mask]
selected_y_test = y_test.values[selected_mask]  # y_test를 numpy array로 변환하고, 마스크 적용

if len(selected_y_test) > 0:
    selected_accuracy = accuracy_score(selected_y_test, selected_y_pred)
    print(f'Selected Accuracy for Probabilities <= 40% or >= 60%: {selected_accuracy:.4f}')
else:
    print("No data points met the probability criteria for selected accuracy.")