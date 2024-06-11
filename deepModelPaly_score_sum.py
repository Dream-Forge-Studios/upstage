import torch
import torch.nn as nn
import numpy as np
import random
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, parse_qs
import re
from eda_utils import makingData, statizCrawling, randomAgent, mlbData
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
def oneTeam():
    ['away_score', 'away_conceded', 'away_ERA_all', 'away_ERA_30',
     'away_score_10', 'away_conceded_10', 'home_score', 'home_conceded',
     'home_ERA_all', 'home_ERA_30', 'home_score_10', 'home_conceded_10']
    # wins / (wins + losses)
    # 사용자 입력 데이터
    target = {
        'away_score': 3.68,
        'away_conceded': 4.85,
        'away_ERA_all': 2.85,
        'away_ERA_30': 2.40,
        'away_score_10': 4.0,
        'away_conceded_10': 0.48,
        'home_score': 5.42,
        'home_conceded': 5.12,
        'home_ERA_all': 1.57,
        'home_ERA_30': 3.00,
        'home_score_10': 4.0,
        'home_conceded_10': 0.48,
    }
    # 리스트로 변환된 데이터를 배열로 변환
    input_data = np.array([target[key] for key in sorted(target)])

    return input_data


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

def urlsTake():
    user_agents = randomAgent()
    url = 'https://statiz.sporki.com/schedule/'
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    agent = random.choice(user_agents)

    headers = {
        'User-Agent': agent
    }
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    urls = []
    soup = BeautifulSoup(response.text, 'html.parser')
    day = '15'
    tds = soup.find_all('td')
    for td in tds:
        try:
            if td.find('span').text == day:
                for a in td.find_all('a'):
                    urls.append('https://statiz.sporki.com' + a['href'])
        except:
            continue
    return urls

urls = urlsTake()
results = statizCrawling(urls)
df = makingData(results)
X = df.drop(['away_team', 'game_id', 'home_team', 'away_win_rate', 'home_win_rate', 'away_win_rate_10', 'home_win_rate_10'], axis=1)

# results = mlbData('https://www.covers.com/sport/baseball/mlb/matchup/299763')
# df = makingData(results)
# X = df.drop(['away_team', 'game_id', 'home_team'], axis=1)

away_team = df['away_team']
home_team = df['home_team']

scaler = StandardScaler()
scaler.mean_ = mean
scaler.scale_ = scale

X_scaled = scaler.transform(X)
input_tensor = torch.FloatTensor(X_scaled)

# 모델 초기화
model_sum = Regression(12)

# 모델의 state_dict 로드
model_sum.load_state_dict(torch.load('model_score_sum.pth'))


# 모델을 평가 모드로 설정
model_sum.eval()

# 그라디언트 계산 비활성화
with torch.no_grad():
    regression_preds = model_sum(input_tensor)

for i in range(len(regression_preds)):
    # 결과 출력
    print(f"{away_team[i]} : {home_team[i]} \n"
          f"{regression_preds[i].item() + 4} 80% \n"
          f"{regression_preds[i].item() + 3} 75% \n"
          f"{regression_preds[i].item() + 2} 70% \n"
          f"{regression_preds[i].item() + 1} 63% \n"
          f"{regression_preds[i].item() } 57% \n"
          f"{regression_preds[i].item() - 1} 48% \n"
          f"{regression_preds[i].item() - 2} 39% \n"
          f"{regression_preds[i].item() - 3} 32% \n"
          f"{regression_preds[i].item() - 4} 24% \n"
          f"========================================")







