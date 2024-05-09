import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from eda_utils import makingData_score_sum
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 데이터 로드 및 전처리 함수 정의
def load_and_preprocess_data():
    df_2014 = makingData_score_sum('statuzGame_2014_score_sum.json')
    df_2015 = makingData_score_sum('statuzGame_2015_score_sum.json')
    df_2016 = makingData_score_sum('statuzGame_2016_score_sum.json')
    df_2017 = makingData_score_sum('statuzGame_2017_score_sum.json')
    df_2018 = makingData_score_sum('statuzGame_2018_score_sum.json')
    df_2019 = makingData_score_sum('statuzGame_2019_score_sum.json')
    df_2020 = makingData_score_sum('statuzGame_2020_score_sum.json')
    df_2021 = makingData_score_sum('statuzGame_2021_score_sum.json')
    df_2022 = makingData_score_sum('statuzGame_2022_score_sum.json')
    df_2023 = makingData_score_sum('statuzGame_2023_score_sum.json')
    combined_df = pd.concat([df_2014, df_2015, df_2016, df_2017, df_2018, df_2019, df_2020, df_2021, df_2022, df_2023],
                            ignore_index=True)

    # 예제에서는 target이 'home_result'라고 가정
    X = combined_df.drop(['home_result', 'game_id', 'away_result', 'away_win_rate', 'home_win_rate', 'away_win_rate_10', 'home_win_rate_10', 'score_sum'], axis=1)
    y = combined_df['score_sum']

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, len(X.columns)

# 신경망 모델 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(inputNum, 64)
        self.layer2 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(64, len(y))
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


# 모델 훈련
epochs = 50
train_losses = []
test_losses = []
test_errors = []
# for random_state in [42, 11, 2, 6, 38]:
for random_state in [42]:
    # 데이터 로드 및 전처리
    X, y, inputNum = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    print(len(y_test))
    # 데이터를 텐서로 변환
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train.values)
    y_test = torch.LongTensor(y_test.values)

    # 데이터 로더 생성
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 모델 생성 및 파라미터 설정
    model = NeuralNetwork()
    # criterion = CustomLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        train_epoch_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            probabilities = F.softmax(output, dim=1)
            max_probabilities, predicted_classes = torch.max(probabilities, 1)
            # penalty = (predicted_classes - target).clamp(min=0)
            penalty = (target - predicted_classes).clamp(min=0)
            penalty_weight = 40  # 페널티 가중치
            penalty_loss = penalty_weight * penalty.float().mean()

            # 총 손실
            total_loss = loss + penalty_loss
            total_loss.backward()
            optimizer.step()

            train_epoch_loss += total_loss.item()  # 각 배치의 loss를 더함

        train_epoch_loss /= len(train_loader)  # 각 배치의 loss 평균을 구함
        train_losses.append(train_epoch_loss)  # 현재 epoch의 훈련 손실을 리스트에 추가

        # 테스트 데이터에 대한 손실과 정확도 계산
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test)
            test_losses.append(test_loss.item())  # 테스트 손실을 리스트에 추가

            # Softmax를 적용하여 각 클래스에 대한 확률을 계산
            probabilities = F.softmax(test_predictions, dim=1)
            # 가장 확률이 높은 클래스와 해당 확률을 가져옴
            max_probabilities, predicted_classes = torch.max(probabilities, 1)

            errors = predicted_classes.float() - y_test.float()
            mean_error = torch.mean(torch.abs(errors))

            negative_count = (errors < 0).sum().item()
            positive_count = (errors > 0).sum().item()
            # print(f"Epoch {epoch + 1}, Train Loss: {train_epoch_loss}, Test Loss: {test_loss}, Mean error: {mean_error} 음수의 개수: {negative_count}, 양수의 개수: {positive_count}")

            # 오차의 평균 계산
            # mean_error = torch.mean(errors)
            #
            # test_errors.append(mean_error)
            # # 확률이 0.006 이상인 경우에만 오차를 계산하고 평균을 구함
            filtered_indices = (max_probabilities >= 0.6)
            filtered_errors = torch.abs(predicted_classes.float()[filtered_indices] - y_test.float()[filtered_indices])
            mean_error_7 = torch.mean(filtered_errors)

            print(
                f'Epoch {epoch + 1}, Train Loss: {train_epoch_loss}, Test Loss: {test_loss}, Mean error: {mean_error}, 7 Mean error: {mean_error_7}, Relevant Data Count: {torch.sum(filtered_indices).item() / len(y_test)}, 음수의 개수: {negative_count}, 양수의 개수: {positive_count}')

# # Loss 그래프 그리기
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.show()
#
# # Accuracy 그래프 그리기
# plt.plot(range(1, epochs + 1), test_errors, label='Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('errors')
# plt.title('Test errors over Epochs')
# plt.legend()
# plt.show()

# torch.save(model.state_dict(), f'model_state_dict_score.pth')