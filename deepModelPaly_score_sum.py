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
class Regression(nn.Module):
    def __init__(self, input_features):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.regression_output = nn.Linear(128, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        regression = self.regression_output(x)

        return regression


urls = [
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240210',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240209',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240208',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240207',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240206',
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
model_under = Regression(12)
model_sum = Regression(12)
model_over = Regression(12)

# 모델의 state_dict 로드
model_under.load_state_dict(torch.load('model_score_sum_under.pth'))
model_sum.load_state_dict(torch.load('model_score_sum.pth'))
model_over.load_state_dict(torch.load('model_score_sum_over.pth'))


# 모델을 평가 모드로 설정
model_under.eval()
model_sum.eval()
model_over.eval()

# 그라디언트 계산 비활성화
with torch.no_grad():
    regression_preds_under = model_under(input_tensor)
    regression_preds = model_sum(input_tensor)
    regression_preds_over = model_over(input_tensor)

for i in range(len(regression_preds_under)):
    # 결과 출력
    print(f"{away_team[i]} : {home_team[i]} \n"
          f"model_under: {regression_preds_under[i].item()} {regression_preds_under[i].item() - 2.8}\n"
          f"model_sum: {regression_preds[i].item()} {regression_preds[i].item() + 3.6} {regression_preds[i].item() - 3.7} \n"
          f"model_over: {regression_preds_over[i].item()} {regression_preds_over[i].item() + 5.4}\n\n")







