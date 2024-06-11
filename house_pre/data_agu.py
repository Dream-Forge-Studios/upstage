import pandas as pd
from collections import defaultdict
import numpy as np

train_df = pd.read_csv('train.csv')

# block 값을 기준으로 유사한 block을 찾는 함수 (수정됨)
def find_most_similar_blocks(blocks, prices):
    prefix_counts = defaultdict(list)
    for block, price in zip(blocks, prices):
        for i in range(1, len(block) + 1):
            prefix = block[:i]
            # 현재 블록과 같은 자릿수의 접두사만 저장
            if all(len(b) == len(prefix) for b in blocks):
                prefix_counts[prefix].append(price)

    # 자릿수가 같은 접두사 중에서 가장 긴 일치 접두사 찾기
    max_prefix = max(prefix_counts, key=lambda k: (len(prefix_counts[k]), len(k)), default="")
    if not max_prefix:  # 일치하는 접두사가 없는 경우
        return None, None

    similar_prices = prefix_counts[max_prefix]
    same_num = len(similar_prices)

    # prePrice 계산 (일치하는 블록이 1개 이상인 경우 평균, 아니면 None)
    if same_num >= 1:
        return np.mean(similar_prices), same_num
    else:
        return None, None

# 'prePrice' 및 'sameNum' 계산 및 새로운 열에 추가
train_df[['prePrice', 'sameNum']] = train_df.groupby('street').apply(
    lambda x: x['block'].apply(find_most_similar_blocks, args=(x['price'],))
).reset_index(drop=True).apply(pd.Series)

train_df['year_of_sale'] = train_df['date'].astype(str).str[:4].astype(int)

train_df_filtered = train_df.dropna(subset=['prePrice', 'sameNum'])

# 필터링된 데이터프레임을 'train_agu.csv' 파일로 저장
train_df_filtered.to_csv('train_agu.csv', index=False)

print("Filtered and saved to train_agu.csv")