import torch
import torch.nn as nn
import numpy as np
import random
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, parse_qs
import re
from eda_utils import makingData, statizCrawling
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

mean = np.array([5.06798191, 5.04829094, 4.48099303, 4.80324666, 5.1756548 ,
       5.12924439, 5.06101564, 5.05881666, 4.45692858, 4.82904654,
       5.14991521, 5.16668551])
scale = np.array([ 0.48947377,  0.48481628,  3.087156  , 14.41574512,  1.71138018,
        1.62494088,  0.48213828,  0.48131068,  3.75538883, 15.59527533,
        1.65953033,  1.6966742 ])

# 모델 아키텍처 정의
class RegressionWithUncertainty(nn.Module):
    def __init__(self, input_features, num_classes):
        super(RegressionWithUncertainty, self).__init__()
        # 공통 레이어
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 128)

        # 회귀 출력
        self.regression_output = nn.Linear(128, 1)

        # 확률 출력 (분류 문제를 예로 들어)
        self.probability_output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 회귀 값을 예측
        regression = self.regression_output(x)

        # 확률 값을 예측 (분류를 위한 softmax 적용)
        probability = F.softmax(self.probability_output(x), dim=1)

        return regression, probability


urls = [
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240200',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240199',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240198',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240197',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240196',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240192',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240191',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240181',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240185',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240184',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240183',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240182',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240180',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240179',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240177',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240176',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240175',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240174',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240173',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240172',
        ]


results = statizCrawling(urls)

df = makingData(results)

X = df.drop(['away_team', 'game_id', 'home_team', 'away_win_rate', 'home_win_rate', 'away_win_rate_10', 'home_win_rate_10'], axis=1)

away_team = df['away_team']
home_team = df['home_team']

scaler = StandardScaler()
scaler.mean_ = mean
scaler.scale_ = scale

X_scaled = scaler.transform(X)
input_tensor = torch.FloatTensor(X_scaled)

file_path = "score_sum_values.txt"

# 파일에서 데이터 읽어오기
with open(file_path, "r") as f:
    score_sum_values = [float(line.strip()) for line in f]

# 모델 초기화
model = RegressionWithUncertainty(12, len(score_sum_values))

# 모델의 state_dict 로드
model.load_state_dict(torch.load('model_state_dict_score.pth'))


# 모델을 평가 모드로 설정
model.eval()

# 그라디언트 계산 비활성화
with torch.no_grad():
    regression_preds, probability_preds = model(input_tensor)
    probabilities = F.softmax(probability_preds, dim=1)
    max_probabilities, predicted_classes = torch.max(probabilities, 1)

for i in range(len(test_predictions)):
    # 결과 출력
    print(f"{away_team[i]} : {home_team[i]} \n{regression_preds[i]} {'{:.2f}'.format(max_probabilities[i])}%\n")







