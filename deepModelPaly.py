import torch
import torch.nn as nn
import numpy as np
import random
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, parse_qs
import re
from eda_utils import makingData, statizCrawling
from sklearn.preprocessing import StandardScaler
import pandas as pd

mean = np.array([0.50189582, 4.97385426, 4.94760493, 4.41647973, 4.71866048,
       0.50454666, 5.03745312, 4.98422933, 0.50036409, 4.96868369,
       4.96110734, 4.41112341, 4.68056081, 0.5004222 , 5.02178961,
       5.01684229])
scale = np.array([8.54432644e-03, 4.81719240e-01, 4.88534128e-01, 4.65446332e+00,
       1.45292983e+01, 2.86352044e-02, 1.59211575e+00, 1.57259465e+00,
       8.58506972e-03, 4.78629255e-01, 4.97199060e-01, 4.62764876e+00,
       1.22634460e+01, 2.82856977e-02, 1.61573882e+00, 1.64970562e+00])
# 모델 아키텍처 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(16, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

urls = [
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240201',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240202',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240203',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240204',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240205',

        ]


results = statizCrawling(urls)

df = makingData(results)

X = df.drop(['away_team', 'game_id', 'home_team'], axis=1)

away_team = df['away_team']
home_team = df['home_team']

X_scaled = (X.values - mean) / scale
input_tensor = torch.FloatTensor(X_scaled)


# 모델 초기화
model = NeuralNetwork()

# 모델의 state_dict 로드
model.load_state_dict(torch.load('model_state_dict_1_6_2015~2023.pth'))
# model.load_state_dict(torch.load('model_state_dict_1_2020~2023.pth'))


# 모델을 평가 모드로 설정
model.eval()

# 그라디언트 계산 비활성화
with torch.no_grad():
    prediction = model(input_tensor)
    print(prediction)
    predicted_class = (prediction > 0.5).float()

for i in range(len(prediction)):
    if predicted_class[i] == torch.tensor([1.]):
        decision = f"{home_team[i]} 승"
    else:
        decision = f"{away_team[i]} 승"

    if float(prediction[i]) < 0.5:
        percent = 100 - float(prediction[i] * 100)
    else:
        percent = float(prediction[i] * 100)
    # 결과 출력
    print(f"{away_team[i]} : {home_team[i]} \n{decision} {'{:.2f}'.format(percent)}%\n")







