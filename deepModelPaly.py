import torch
import torch.nn as nn
import numpy as np
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

# 모델 초기화
model = NeuralNetwork()

# 모델의 state_dict 로드
model.load_state_dict(torch.load('model_state_dict.pth'))

# 전체 모델 로드 (필요한 경우)
model.eval()  # 추론 모드 설정

['away_win_rate', 'away_score', 'away_conceded', 'away_ERA_all',
       'away_ERA_30', 'away_win_rate_10', 'home_win_rate', 'home_score',
       'home_conceded', 'home_ERA_all', 'home_ERA_30', 'home_win_rate_10']
# wins / (wins + losses)
# 사용자 입력 데이터
target = {
    'away_win_rate': 0.58,
    'away_score': 5.13,
    'away_conceded': 4.59,
    'away_ERA_all': 5.11,
    'away_ERA_30': 5.11,
    'away_win_rate_10': 4.0,
    'home_win_rate': 0.48,
    'home_score': 5.42,
    'home_conceded': 5.12,
    'home_ERA_all': 1.57,
    'home_ERA_30': 3.00,
    'home_win_rate_10': 0.7,
}

# 표준화 사용하여 입력 데이터 변환
mean = np.array([0.50132368, 4.78121043, 4.76803538, 4.19554004, 4.46337523,
                 0.50452008, 0.50055998, 4.78575419, 4.7730959, 4.13798417,
                 4.33653166, 0.50197323])
scale = np.array([0.09704761, 0.69991801, 0.61069858, 1.70571644, 3.71480677,
                  0.17187251, 0.09803055, 0.69751297, 0.61295986, 1.95585938,
                  3.50623408, 0.17194038])

# 리스트로 변환된 데이터를 배열로 변환
input_data = np.array([target[key] for key in sorted(target)])

# 데이터 표준화
standardized_data = (input_data - mean) / scale

# 텐서로 변환
input_tensor = torch.FloatTensor(standardized_data).unsqueeze(0)  # 배치 차원 추가

# 모델을 평가 모드로 설정
model.eval()

# 그라디언트 계산 비활성화
with torch.no_grad():
    prediction = model(input_tensor)
    print(prediction)
    predicted_class = (prediction > 0.5).float()

# 결과 출력
print("Predicted Class:", predicted_class.item())







