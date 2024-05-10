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


class RegressionWithUncertainty(nn.Module):
    def __init__(self, input_features, num_classes):
        super(RegressionWithUncertainty, self).__init__()
        # 공통 레이어
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 128)

        # 회귀 출력
        self.regression_output = nn.Linear(128, 1)

        # 확률 출력 (분류 문제를 예로 들어)
        self.probability_output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 회귀 값을 예측
        regression = self.regression_output(x)

        # 확률 값을 예측 (분류를 위한 softmax 적용)
        probability = F.softmax(self.probability_output(x), dim=1)

        return regression, probability


def combined_loss(regression_output, probability_output, regression_target, probability_target, beta=0.5,
                  penalty_weight=12):
    # 기존 회귀 손실
    regression_loss = nn.MSELoss()(regression_output, regression_target)
    # 분류 손실
    probability_loss = nn.CrossEntropyLoss()(probability_output, probability_target)

    # 페널티 추가: 예측값이 실제값보다 작을 때 페널티 적용
    penalty = F.relu(regression_target - regression_output).mean()

    # 총 손실 계산
    total_loss = beta * regression_loss + (1 - beta) * probability_loss + penalty_weight * penalty

    return total_loss

# 모델 훈련
epochs = 250
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
    y_train_regression = torch.FloatTensor(y_train.values)  # 회귀 타겟
    y_train_classification = torch.LongTensor(y_train.values)  # 분류 타겟
    y_test_regression = torch.FloatTensor(y_test.values)  # 회귀 타겟
    y_test_classification = torch.LongTensor(y_test.values)  # 분류 타겟
    # y_train = torch.LongTensor(y_train.values)
    # y_test = torch.LongTensor(y_test.values)

    # 데이터 로더 생성
    # train_dataset = TensorDataset(X_train, y_train)
    train_dataset = TensorDataset(X_train, y_train_regression, y_train_classification)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = RegressionWithUncertainty(input_features=inputNum, num_classes=len(y_test))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    for epoch in range(epochs):
        train_epoch_loss = 0.0
        for data, regression_targets, probability_targets in train_loader:
            optimizer.zero_grad()
            regression_preds, probability_preds = model(data)
            loss = combined_loss(regression_preds, probability_preds, regression_targets, probability_targets)

            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()  # 각 배치의 loss를 더함

        train_epoch_loss /= len(train_loader)  # 각 배치의 loss 평균을 구함
        train_losses.append(train_epoch_loss)  # 현재 epoch의 훈련 손실을 리스트에 추가

        # 테스트 데이터에 대한 손실과 정확도 계산
        with torch.no_grad():
            # 모델을 통해 회귀값과 분류 확률값을 동시에 받아옴
            regression_predictions, probability_predictions = model(X_test)

            # 회귀 평가: 회귀 예측에 대한 MSE 손실 계산
            regression_loss = torch.nn.MSELoss()(regression_predictions, y_test_regression.float().unsqueeze(1))
            test_losses.append(regression_loss.item())  # 테스트 손실을 리스트에 추가

            # 분류 평가: Softmax를 적용하여 각 클래스에 대한 확률을 계산
            probabilities = F.softmax(probability_predictions, dim=1)
            # 가장 확률이 높은 클래스와 해당 확률을 가져옴
            max_probabilities, predicted_classes = torch.max(probabilities, 1)

            # 회귀 오차 계산
            regression_errors = regression_predictions.squeeze() - y_test_regression.float()
            mean_regression_error = torch.mean(torch.abs(regression_errors))

            # 음수 오차와 양수 오차의 개수 계산 (회귀)
            negative_count = (regression_errors < 0).sum().item()
            positive_count = (regression_errors > 0).sum().item()

            # 확률이 0.6 이상인 경우에만 오차를 계산하고 평균을 구함 (분류)
            filtered_indices = (max_probabilities >= 0.6)
            filtered_classification_errors = torch.abs(
                predicted_classes.float()[filtered_indices] - y_test_classification.float()[filtered_indices])
            mean_filtered_classification_error = torch.mean(filtered_classification_errors)

            print(
                f'Epoch {epoch + 1}, Train Loss: {train_epoch_loss}, Test Regression Loss: {regression_loss}, Mean Regression Error: {mean_regression_error}, 7 Mean Classification Error: {mean_filtered_classification_error}, Relevant Data Count: {torch.sum(filtered_indices).item() / len(y_test)}, Negative Error Count: {negative_count}, Positive Error Count: {positive_count}')

print()
# # Loss 그래프 그리기
# plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
# plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Test Loss over Epochs')
# plt.legend()
# plt.show()
#
# # Accuracy 그래프 그리기
# plt.plot(range(1, epochs + 1), test_errors, label='Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('errors')
# plt.title('Test errors over Epochs')
# plt.legend()
# plt.show()

# torch.save(model.state_dict(), 'model_state_dict_score_sum.pth')