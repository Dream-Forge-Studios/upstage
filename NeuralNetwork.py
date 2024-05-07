import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from eda_utils import makingData

# 데이터 로드 및 전처리 함수 정의
def load_and_preprocess_data():
    df_2020 = makingData('statuzGame_2020.json')
    df_2021 = makingData('statuzGame_2021.json')
    df_2022 = makingData('statuzGame_2022.json')
    df_2023 = makingData('statuzGame_2023.json')
    combined_df = pd.concat([df_2020, df_2021, df_2022, df_2023], ignore_index=True)

    # 예제에서는 target이 'home_result'라고 가정
    X = combined_df.drop(['home_result', 'game_id'], axis=1)
    y = combined_df['home_result']

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# 신경망 모델 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(13, 64)  # 입력 특성 수에 따라 조정 필요
        self.layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)  # 이진 분류를 가정
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x


# 데이터 로드 및 전처리
X, y = load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터를 텐서로 변환
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train.values).unsqueeze(1)
y_test = torch.FloatTensor(y_test.values).unsqueeze(1)

# 데이터 로더 생성
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델 생성 및 파라미터 설정
model = NeuralNetwork()
criterion = nn.BCELoss()  # 이진 분류를 위한 Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련
epochs = 5
for epoch in range(epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 모델의 state_dict 저장
torch.save(model.state_dict(), 'model_state_dict.pth')

# 간단한 정확도 평가
with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes.eq(y_test).sum() / len(y_test)).item()
    print(f'Test Accuracy: {accuracy:.4f}')
