import pandas as pd
from pycaret.regression import *

# 데이터 불러오기
train_data = pd.read_csv('train_agu.csv')
test_df = pd.read_csv('test_agu.csv')

# PyCaret 설정
s = setup(
    data=train_data,
    target='price',
    numeric_features=['area_sqm', 'storey_range', 'year_of_sale', 'sameNum', 'prePrice'],
    categorical_features=['street', 'type'],
    ignore_features=['house_id', 'date', 'location', 'block', 'flat_model', 'commence_date', 'price'],
    test_data=test_df,
)

# 여러 모델 비교 및 최적 모델 선택
best_model = compare_models()

# 최적 모델 학습 및 튜닝
tuned_model = tune_model(best_model)

# 최종 모델 성능 평가
evaluate_model(tuned_model)

# 모델 저장
save_model(tuned_model, 'tuned_house_price_model')
