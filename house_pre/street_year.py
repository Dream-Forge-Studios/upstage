import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def street_price_std():
    train_df = pd.read_csv('train.csv')
    std_deviations = train_df.groupby('street')['price'].std()

    print(f"표준편차 평균: {std_deviations.mean()}")
    print(f"표준편차 최댓값: {std_deviations.max()}")
    print(f"표준편차 최솟값: {std_deviations.min()}")

    # 표준편차 내림차순 정렬
    std_deviations = std_deviations.sort_values(ascending=False)

    # 그래프 크기 설정
    plt.figure(figsize=(12, 8))

    # Seaborn 막대 그래프 생성
    sns.barplot(x=std_deviations.index, y=std_deviations.values)

    # 그래프 제목 및 축 레이블 설정
    plt.title('Standard Deviation of Price by Street', fontsize=14)
    plt.xlabel('Street', fontsize=12)
    plt.ylabel('Price Standard Deviation', fontsize=12)

    # x축 레이블 회전 (겹침 방지)
    plt.xticks(rotation=45)

    # 그래프 출력
    plt.tight_layout()
    plt.show()

def street_price_mean():
    # 1. 'train.csv' 파일을 읽어와서 train_df 데이터프레임으로 저장합니다.
    train_df = pd.read_csv('train.csv')

    # 2. 'street' 열에서 랜덤으로 9개의 고유한 값을 선택합니다.
    sampled_streets = np.random.choice(train_df['street'].unique(), 9, replace=False)

    # 3. 선택된 거리에 해당하는 데이터만 필터링합니다.
    filtered_df = train_df[train_df['street'].isin(sampled_streets)]

    # 4. 'date' 열에서 연도 정보를 추출하여 새로운 열 'year'를 생성합니다.
    filtered_df['year'] = pd.to_datetime(filtered_df['date']).dt.year

    # 5. seaborn 라이브러리를 사용하여 꺾은선 그래프를 생성합니다. x축에는 'year' 값, y축에는 'price' 값을 설정합니다.
    # 'street' 열의 각 값에 대해 서로 다른 색상의 선으로 표시합니다.
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=filtered_df, x='year', y='price', hue='street', marker='o')

    # 6. x축 레이블을 'Year', y축 레이블을 'Price'로 설정하고, 그래프 제목을 'Price Trend by Year for Selected Streets'로 설정합니다.
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.title('Price Trend by Year for Selected Streets', fontsize=14)

    # 7. 범례를 표시합니다.
    plt.legend(title='Street')

    # 8. 그래프를 출력합니다.
    plt.show()

def s():
    # 'train.csv' 파일을 읽어옴
    train_df = pd.read_csv('train.csv')
    
    # 'flat_model'별 'price' 평균 계산
    mean_prices = train_df.groupby('type')['price'].mean()

    # 'price' 평균 내림차순 정렬
    mean_prices_sorted = mean_prices.sort_values(ascending=False)

    # Scatter plot 그리기
    plt.figure(figsize=(12, 6))
    plt.scatter(mean_prices_sorted.index, mean_prices_sorted.values)

    # x축 레이블 45도 회전
    plt.xticks(rotation=45, ha='right')

    # 제목 및 레이블 설정
    plt.title('Average Price by Flat Model (Sorted)', fontsize=14)
    plt.xlabel('Flat Model', fontsize=12)
    plt.ylabel('Average Price', fontsize=12)

    # 그래프 출력
    plt.show()

    # block 열의 첫 번째 글자 추출
    train_df['block_first_letter'] = train_df['block'].astype(str).str[:3]

    # street와 block_first_letter를 기준으로 그룹화하여 price의 표준편차 계산
    # std_deviations = train_df.groupby(['street', 'block_first_letter'])['price'].std()
    std_deviations = train_df.groupby(['location'])['price'].std()

    # 결과 출력
    print(f"표준편차 평균: {std_deviations.mean()}")
    print(f"표준편차 최댓값: {std_deviations.max()}")
    print(f"표준편차 최솟값: {std_deviations.min()}")

    # group_sizes = train_df.groupby(['street', 'block_first_letter']).size()
    group_sizes = train_df.groupby(['location']).size()

    # 결과 출력
    print(f"그룹 크기 평균: {group_sizes.mean()}")
    print(f"그룹 크기 최댓값: {group_sizes.max()}")
    print(f"그룹 크기 최솟값: {group_sizes.min()}")

    # 표준편차 내림차순 정렬
    # std_deviations = std_deviations.sort_values(ascending=False)
    #
    # # 멀티 인덱스를 ','로 구분된 문자열로 변환
    # std_deviations.index = [', '.join(x) for x in std_deviations.index]
    #
    # # 그래프 그리기
    # plt.figure(figsize=(12, 8))
    # plt.bar(std_deviations.index, std_deviations.values)
    # plt.title('Standard Deviation of Price by Street and Block First Letter', fontsize=14)
    # plt.xlabel('Street, Block First Letter', fontsize=12)
    # plt.ylabel('Price Standard Deviation', fontsize=12)
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()

s()