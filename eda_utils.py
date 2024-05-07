import pandas as pd

def makingData(file_path):
    data = pd.read_json(file_path)
    # 데이터 프레임 생성
    rows = []
    for game_id, teams in data.items():
        game_data = {}
        skip = False
        for team in teams:
            for key, value in team.items():
                if value == 0.0 and isinstance(value, float):
                    skip = True  # Set skip to True if any value is 0.0
                    break  # Exit the current team loop if a 0.0 value is found
                if key != 'venue':   # 'venue' 정보는 prefix가 이미 구분함
                    prefix = 'away_' if team['venue'] == 0 else 'home_'
                    # if key != 'win_rate_10':
                    #     game_data[prefix + key] = value
                    game_data[prefix + key] = value
        if not skip:
            game_data['game_id'] = game_id
            rows.append(game_data)

    return pd.DataFrame(rows)
