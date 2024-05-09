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

mean = np.array([4.76799399, 4.74804658, 4.15105184, 4.40785875, 4.77967693,
       4.73557476, 4.76991736, 4.76381292, 4.11673929, 4.33831705,
       4.76540195, 4.76435011, 9.48271976])
scale = np.array([ 0.41122257,  0.38228446,  2.60566313, 12.19590315,  1.36730673,
        1.33580807,  0.41098654,  0.38630744,  3.38815406, 10.94138304,
        1.36323949,  1.43149017, 23.10695158])

# 모델 아키텍처 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(len(X.columns), 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, len(score_sum_values))
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


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
model = NeuralNetwork()

# 모델의 state_dict 로드
model.load_state_dict(torch.load('model_state_dict_score.pth'))


# 모델을 평가 모드로 설정
model.eval()

# 그라디언트 계산 비활성화
with torch.no_grad():
    test_predictions = model(input_tensor)
    probabilities = F.softmax(test_predictions, dim=1)
    max_probabilities, predicted_classes = torch.max(probabilities, 1)

for i in range(len(test_predictions)):
    # 결과 출력
    print(f"{away_team[i]} : {home_team[i]} \n{predicted_classes[i]} {'{:.2f}'.format(max_probabilities[i])}%\n")







