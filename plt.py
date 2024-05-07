import matplotlib.pyplot as plt
import pandas as pd

# 각 특성에 대해 히스토그램을 그립니다.
def plot_histograms(dataframe):
    for column in dataframe.columns:
        plt.figure(figsize=(10, 4))
        plt.hist(dataframe[column], bins=30, alpha=0.7)
        plt.title(f'Histogram of {column}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        # 저장 경로 설정 (파일 이름 포맷: histogram_[column].png)
        output_path = f"plt/histogram_{column}.png"
        plt.savefig(output_path)  # 히스토그램을 파일로 저장
        plt.close()  # 열린 플롯 창을 닫음


def makingData(file_path):
    data = pd.read_json(file_path)
    # 데이터 프레임 생성
    rows = []
    for game_id, teams in data.items():
        for team in teams:
            game_data = {}
            skip = False  # Add a flag to determine whether to skip this entry
            for key, value in team.items():
                if value == 0.0 and isinstance(value, float):
                    skip = True  # Set skip to True if any value is 0.0
                    break  # Exit the current team loop if a 0.0 value is found
                if key != 'venue':  # 'venue' 정보는 prefix가 이미 구분함
                    game_data[key] = value
            if not skip:
                game_data['game_id'] = game_id
                rows.append(game_data)

    return pd.DataFrame(rows)

df_2020 = makingData('statuzGame_2020.json')
df_2021 = makingData('statuzGame_2021.json')
df_2022 = makingData('statuzGame_2022.json')
df_2023 = makingData('statuzGame_2023.json')

combined_df = pd.concat([df_2020, df_2021, df_2022, df_2023], ignore_index=True)
# 특성과 타겟 분리
X = combined_df.drop(['result', 'game_id'], axis=1)  # 게임 ID 제거 및 결과 분리
y = combined_df['result']

plot_histograms(X)  # X는 위에서 정의한 특성 데이터
