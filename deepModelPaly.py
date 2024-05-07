import torch
import torch.nn as nn

# 모델 아키텍처 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(13, 64)
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

# 예측을 위한 데이터 준비 (예제)
# 여기서는 X_test를 사용한다고 가정
X_test = torch.FloatTensor(X_test)  # 테스트 데이터를 텐서로 변환
model.eval()  # 모델을 평가 모드로 설정
with torch.no_grad():  # 그라디언트 계산 비활성화
    predictions = model(X_test)
    predicted_classes = (predictions > 0.5).float()