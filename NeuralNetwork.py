import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from eda_utils import makingData
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리 함수 정의
def load_and_preprocess_data():
    df_2020 = makingData('statuzGame_2020.json')
    df_2021 = makingData('statuzGame_2021.json')
    df_2022 = makingData('statuzGame_2022.json')
    df_2023 = makingData('statuzGame_2023.json')
    combined_df = pd.concat([df_2020, df_2021, df_2022, df_2023], ignore_index=True)

    # 예제에서는 target이 'home_result'라고 가정
    X = combined_df.drop(['home_result', 'game_id', 'away_result'], axis=1)
    y = combined_df['home_result']

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# 신경망 모델 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(12, 64)  # 입력 특성 수에 따라 조정 필요
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
epochs = 10
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    train_epoch_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()  # 각 배치의 loss를 더함

    train_epoch_loss /= len(train_loader)  # 각 배치의 loss 평균을 구함
    train_losses.append(train_epoch_loss)  # 현재 epoch의 훈련 손실을 리스트에 추가

    # 테스트 데이터에 대한 손실과 정확도 계산
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions, y_test)
        test_losses.append(test_loss.item())  # 테스트 손실을 리스트에 추가

        # test_predictions가 0.4 이하 또는 0.6 이상인 경우에만 정확도 계산에 포함
        relevant_indices = ((test_predictions <= 0.4) | (test_predictions >= 0.6)).float()
        relevant_test_predictions = test_predictions * relevant_indices
        relevant_y_test = y_test * relevant_indices
        relevant_data_count = relevant_indices.sum().item()  # 0.4 이하 또는 0.6 이상인 데이터 개수 계산
        if relevant_data_count > 0:  # 해당 조건에 해당하는 데이터가 있을 때만 정확도 계산
            test_accuracy = (relevant_test_predictions.round() == relevant_y_test).float().mean()
            test_accuracies.append(test_accuracy.item())  # 테스트 정확도를 리스트에 추가
            print(
                f'Epoch {epoch + 1}, Train Loss: {train_epoch_loss}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Relevant Data Count: {relevant_data_count}')
        else:
            print(
                f'Epoch {epoch + 1}, Train Loss: {train_epoch_loss}, Test Loss: {test_loss}, Test Accuracy: No relevant data')

print(
            f'Epoch {epoch + 1}, Train Loss: {train_epoch_loss}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # test_accuracy = (test_predicted_classes.eq(y_test).sum() / len(y_test)).item()
        # test_accuracies.append(test_accuracy)  # 테스트 정확도를 리스트에 추가
        #
        # print(f'Epoch {epoch + 1}, Train Loss: {train_epoch_loss}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Loss 그래프 그리기
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.show()

# Accuracy 그래프 그리기
plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.show()

# 모델의 state_dict 저장
# torch.save(model.state_dict(), 'model_state_dict.pth')
