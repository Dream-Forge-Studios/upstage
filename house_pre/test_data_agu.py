from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytrie import StringTrie

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

trie = StringTrie()
for block, price in zip(train_df['block'], train_df['price']):
    trie[block] = price

def find_most_similar_blocks_trie(block):
    prefix_matches = trie.keys(block)
    if prefix_matches:
        max_prefix = max(prefix_matches, key=len)
        similar_prices = [trie[key] for key in prefix_matches if len(key) == len(max_prefix)]
        same_num = len(similar_prices)
        if same_num >= 1:
            return np.mean(similar_prices), same_num
    return None, None


# test 데이터에 prePrice 및 sameNum 계산
def add_preprice_and_same_num(test_df):
    # tqdm과 pandas 연동
    tqdm.pandas(desc="Calculating prePrice and sameNum")

    # progress_apply 사용
    result = test_df['block'].progress_apply(find_most_similar_blocks_trie)
    test_df[['prePrice', 'sameNum']] = result.apply(pd.Series)

    return test_df

# 함수 실행
test_df = add_preprice_and_same_num(test_df)

test_df['year_of_sale'] = test_df['date'].astype(str).str[:4].astype(int)

test_df_filtered = test_df.dropna(subset=['prePrice', 'sameNum'])

# 결과 출력
test_df_filtered.to_csv('test_agu.csv', index=False)
