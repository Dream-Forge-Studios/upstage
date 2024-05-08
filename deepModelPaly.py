import torch
import torch.nn as nn
import numpy as np
import random
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, parse_qs
import re
from eda_utils import makingData
from sklearn.preprocessing import StandardScaler


mean = np.array([0.50132368, 4.78121043, 4.76803538, 4.19554004, 4.46337523,
                 0.50452008, 0.50055998, 4.78575419, 4.7730959, 4.13798417,
                 4.33653166, 0.50197323])
scale = np.array([0.09704761, 0.69991801, 0.61069858, 1.70571644, 3.71480677,
                  0.17187251, 0.09803055, 0.69751297, 0.61295986, 1.95585938,
                  3.50623408, 0.17194038])


def input():
    keyList = ['away_win_rate', 'away_score', 'away_conceded', 'away_ERA_all',
               'away_ERA_30', 'away_win_rate_10', 'home_win_rate', 'home_score',
               'home_conceded', 'home_ERA_all', 'home_ERA_30', 'home_win_rate_10']

    # wins / (wins + losses)
    # 사용자 입력 데이터
    target = {
        'away_win_rate': 0.48,
        'away_score': 5.19,
        'away_conceded': 5.08,
        'away_ERA_all': 4.50,
        'away_ERA_30': 3.00,
        'away_win_rate_10': 0.6,
        'home_win_rate': 0.44,
        'home_score': 5.00,
        'home_conceded': 5.62,
        'home_ERA_all': 1.57,
        'home_ERA_30': 3.00,
        'home_win_rate_10': 0.7,
    }
    # 리스트로 변환된 데이터를 배열로 변환
    input_data = np.array([target[key] for key in keyList])

    # 데이터 표준화
    standardized_data = (input_data - mean) / scale

    # 텐서로 변환
    input_tensor = torch.FloatTensor(standardized_data).unsqueeze(0)  # 배치 차원 추가
    return input_tensor

def statizCrawling(urls):

    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (iPad; CPU OS 13_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/70.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/601.7.7 (KHTML, like Gecko) Version/9.1.2 Safari/601.7.7',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0',
        'Mozilla/5.0 (Windows NT 10.0; rv:68.0) Gecko/20100101 Firefox/68.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36',
        'Mozilla/5.0 (Windows NT 5.1; rv:49.0) Gecko/20100101 Firefox/49.0',
        'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
        'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36',
        'Mozilla/5.0 (X11; Linux i686; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0',
        'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:64.0) Gecko/20100101 Firefox/64.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0',
    ]
    results = {}
    for url in urls:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)

        # 's_no' 파라미터 값 추출
        s_no_value = query_params.get('s_no')[0]

        results[s_no_value] = []

        agent = random.choice(user_agents)

        headers = {
            'User-Agent': agent
        }
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        try:
            results[s_no_value].append({
                'venue': 0
            })

            results[s_no_value].append({
                'venue': 1
            })

            type03 = soup.find('div', class_='table_type03')

            # 팀명
            results[s_no_value][0]['team'] = type03.find_all('th')[1].text
            results[s_no_value][1]['team'] = type03.find_all('th')[2].text

            # 승률
            win = type03.find_all('tr')[2]

            record = win.find_all('td')[1].text
            matches = re.findall(r'\d+', record)
            wins = int(matches[1])
            losses = int(matches[3])
            win_rate = wins / (wins + losses)
            results[s_no_value][0]['win_rate'] = win_rate

            record = win.find_all('td')[2].text
            matches = re.findall(r'\d+', record)
            wins = int(matches[1])
            losses = int(matches[3])
            win_rate = wins / (wins + losses)
            results[s_no_value][1]['win_rate'] = win_rate

            # 득점
            score_temp = type03.find_all('tr')[6]
            results[s_no_value][0]['score'] = float(score_temp.find_all('td')[1].text)
            results[s_no_value][1]['score'] = float(score_temp.find_all('td')[2].text)

            # 실점
            conceded_temp = type03.find_all('tr')[7]
            results[s_no_value][0]['conceded'] = float(conceded_temp.find_all('td')[1].text)
            results[s_no_value][1]['conceded'] = float(conceded_temp.find_all('td')[2].text)

            # 선발 투수 평균 자책
            starting = soup.find_all('div', class_='table_type03')[1]

            ERA_all_temp = starting.find_all('tr')[4]
            results[s_no_value][0]['ERA_all'] = float(ERA_all_temp.find_all('td')[1].text.split(',')[-1])
            results[s_no_value][1]['ERA_all'] = float(ERA_all_temp.find_all('td')[2].text.split(',')[-1])

            ERA_30_temp = starting.find_all('tr')[6]
            results[s_no_value][0]['ERA_30'] = float(ERA_30_temp.find_all('td')[1].text.split(',')[-1])
            results[s_no_value][1]['ERA_30'] = float(ERA_30_temp.find_all('td')[2].text.split(',')[-1])

            # 최근 10경기 승률
            win_10_2 = soup.find_all('div', class_='table_type03')[2]
            losses = 0
            wins = 0
            for span in win_10_2.find_all('span'):
                if span.text == 'L':
                    losses += 1
                elif span.text == 'W':
                    wins += 1
            results[s_no_value][0]['win_rate_10'] = wins / (wins + losses)

            win_10_3 = soup.find_all('div', class_='table_type03')[3]
            losses = 0
            wins = 0
            for span in win_10_3.find_all('span'):
                if span.text == 'L':
                    losses += 1
                elif span.text == 'W':
                    wins += 1
            results[s_no_value][1]['win_rate_10'] = wins / (wins + losses)
        except:
            del results[s_no_value]

    return results
# 모델 아키텍처 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(12, 64)
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
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240191',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240199',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240198',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240197',
        # 'https://statiz.sporki.com/schedule/?m=preview&s_no=20240196',
        ]

results = statizCrawling(urls)

df = makingData(results)

X = df.drop(['away_team', 'game_id', 'home_team'], axis=1)

away_team = df['away_team']
home_team = df['home_team']

X_scaled = (X.values - mean) / scale
input_tensor = torch.FloatTensor(X_scaled)


# input_tensor = input()

# 모델 초기화
model = NeuralNetwork()

# 모델의 state_dict 로드
model.load_state_dict(torch.load('model_state_dict_3.pth'))

# 전체 모델 로드 (필요한 경우)
model.eval()  # 추론 모드 설정


# 모델을 평가 모드로 설정
model.eval()

# 그라디언트 계산 비활성화
with torch.no_grad():
    prediction = model(input_tensor)
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







