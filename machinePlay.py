import joblib
import numpy as np
from eda_utils import makingData, statizCrawling

mean = np.array([0.50189582, 4.97385426, 4.94760493, 4.41647973, 4.71866048,
       0.50454666, 5.03745312, 4.98422933, 0.50036409, 4.96868369,
       4.96110734, 4.41112341, 4.68056081, 0.5004222 , 5.02178961,
       5.01684229])
scale = np.array([8.54432644e-03, 4.81719240e-01, 4.88534128e-01, 4.65446332e+00,
       1.45292983e+01, 2.86352044e-02, 1.59211575e+00, 1.57259465e+00,
       8.58506972e-03, 4.78629255e-01, 4.97199060e-01, 4.62764876e+00,
       1.22634460e+01, 2.82856977e-02, 1.61573882e+00, 1.64970562e+00])

# 모델 로드
model = joblib.load('logreg_model_42.joblib')
# model = joblib.load('svm_model_2015~2023.joblib')

urls = [
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240200',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240199',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240198',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240197',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240196',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240192',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240191',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240181',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240185',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240184',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240183',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240182',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240180',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240179',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240177',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240176',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240175',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240174',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240173',
        'https://statiz.sporki.com/schedule/?m=preview&s_no=20240172',
        ]

results = statizCrawling(urls)

df = makingData(results)

X = df.drop(['away_team', 'game_id', 'home_team'], axis=1)

away_team = df['away_team']
home_team = df['home_team']

X_scaled = (X - mean) / scale

# 예측 수행
predicted_class = model.predict(X_scaled)
prediction = model.predict_proba(X_scaled)

# 예측 결과 출력
for i in range(len(prediction)):
    if predicted_class[i] == 1:
        decision = f"{home_team[i]} 승"
    else:
        decision = f"{away_team[i]} 승"

    percent_temp = prediction[i][0]
    if float(percent_temp) < 0.5:
        percent = 100 - float(percent_temp * 100)
    else:
        percent = float(percent_temp * 100)
    # 결과 출력
    print(f"{away_team[i]} : {home_team[i]} \n{decision} {'{:.2f}'.format(percent)}%\n")