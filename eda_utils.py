import pandas as pd
import random
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, parse_qs
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (iPad; CPU OS 13_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/70.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/601.7.7 (KHTML, like Gecko) Version/9.1.2 Safari/601.7.7',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0',
    'Mozilla/5.0 (Windows NT 10.0; rv:68.0) Gecko/20100101 Firefox/68.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; rv:49.0) Gecko/20100101 Firefox/49.0',
    'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
    'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36',
    'Mozilla/5.0 (X11; Linux i686; rv:60.0) Gecko/20100101 Firefox/60.0',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0',
    'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:64.0) Gecko/20100101 Firefox/64.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0',
]
def mlbData(url):
    results = {}
    agent = random.choice(user_agents)

    headers = {
        'User-Agent': agent
    }
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'

    soup = BeautifulSoup(response.text, 'html.parser')

    s_no_value = url.split('/')[-1]
    results[s_no_value] = []
    results[s_no_value].append({
        'venue': 0
    })

    results[s_no_value].append({
        'venue': 1
    })

    # 팀명
    results[s_no_value][0]['team'] = soup.find_all('span', class_='teamLogo-playerName')[0].text.replace('\n', '')
    results[s_no_value][1]['team'] = soup.find_all('span', class_='teamLogo-playerName')[1].text.replace('\n', '')

    # 득점
    results[s_no_value][0]['score'] = float(soup.find_all('td')[4].text)
    results[s_no_value][1]['score'] = float(soup.find_all('td')[12].text)

    # 실점
    results[s_no_value][0]['conceded'] = float(soup.find_all('td')[5].text)
    results[s_no_value][1]['conceded'] = float(soup.find_all('td')[13].text)

    # 최근 10경기 득점
    score_10_temp = soup.find_all('tbody')[2].find_all('a')
    score_10 = 0
    conceded_10 = 0
    for i in range(1, 38, 4):
        temp = score_10_temp[i].text.split('-')
        score_10 += int(temp[0])
        conceded_10 += int(temp[1])
    results[s_no_value][0]['score_10'] = score_10 / 10
    results[s_no_value][0]['conceded_10'] = conceded_10 / 10

    score_10_temp = soup.find_all('tbody')[4].find_all('a')
    score_10 = 0
    conceded_10 = 0
    for i in range(1, 38, 4):
        temp = score_10_temp[i].text.split('-')
        score_10 += int(temp[0])
        conceded_10 += int(temp[1])
    results[s_no_value][1]['score_10'] = score_10 / 10
    results[s_no_value][1]['conceded_10'] = conceded_10 / 10

    results[s_no_value][0]['ERA_30'] = float(soup.find_all('div', class_='record-value')[9].text)
    results[s_no_value][1]['ERA_30'] = float(soup.find_all('div', class_='record-value')[12].text)

    results[s_no_value][0]['ERA_all'] = 2.79
    results[s_no_value][1]['ERA_all'] = 2.30

    return results

def makingData(file_path):
    if type(file_path) == str:
        data = pd.read_json(file_path)
    elif type(file_path) == dict:
        data = file_path
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
                    # if key != 'away_result':
                    #     game_data[prefix + key] = value
                    game_data[prefix + key] = value
        if not skip:
            game_data['game_id'] = game_id
            rows.append(game_data)

    return pd.DataFrame(rows)

def makingData_score_sum(file_path):
    if type(file_path) == str:
        data = pd.read_json(file_path)
    elif type(file_path) == dict:
        data = file_path
    # 데이터 프레임 생성
    rows = []
    for game_id, teams in data.items():
        game_data = {}
        skip = False
        for team in teams[:-1]:
            for key, value in team.items():
                if value == 0.0 and isinstance(value, float):
                    skip = True  # Set skip to True if any value is 0.0
                    break  # Exit the current team loop if a 0.0 value is found
                if key != 'venue':   # 'venue' 정보는 prefix가 이미 구분함
                    prefix = 'away_' if team['venue'] == 0 else 'home_'
                    # if key != 'away_result':
                    #     game_data[prefix + key] = value
                    game_data[prefix + key] = value
        if not skip:
            game_data['game_id'] = game_id
            game_data['score_sum'] = teams[2]['score_sum']
            rows.append(game_data)

    return pd.DataFrame(rows)
def statizCrawling(urls):
    results = {}
    for url in urls:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)

        # 's_no' 파라미터 값 추출
        s_no_value = query_params.get('s_no')[0]

        results[s_no_value] = []

        agent = random.choice(user_agents)

        headers = {
            'User-Agent': agent
        }
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        try:
            results[s_no_value].append({
                'venue': 0
            })

            results[s_no_value].append({
                'venue': 1
            })

            type03 = soup.find('div', class_='table_type03')

            # 팀명
            results[s_no_value][0]['team'] = type03.find_all('th')[1].text
            results[s_no_value][1]['team'] = type03.find_all('th')[2].text

            # 승률
            win = type03.find_all('tr')[2]

            record = win.find_all('td')[1].text
            matches = re.findall(r'\d+', record)
            wins = int(matches[1])
            losses = int(matches[3])
            win_rate = wins / (wins + losses)
            results[s_no_value][0]['win_rate'] = win_rate

            record = win.find_all('td')[2].text
            matches = re.findall(r'\d+', record)
            wins = int(matches[1])
            losses = int(matches[3])
            win_rate = wins / (wins + losses)
            results[s_no_value][1]['win_rate'] = win_rate

            # 득점
            score_temp = type03.find_all('tr')[6]
            results[s_no_value][0]['score'] = float(score_temp.find_all('td')[1].text)
            results[s_no_value][1]['score'] = float(score_temp.find_all('td')[2].text)

            # 실점
            conceded_temp = type03.find_all('tr')[7]
            results[s_no_value][0]['conceded'] = float(conceded_temp.find_all('td')[1].text)
            results[s_no_value][1]['conceded'] = float(conceded_temp.find_all('td')[2].text)

            # 선발 투수 평균 자책
            starting = soup.find_all('div', class_='table_type03')[1]

            ERA_all_temp = starting.find_all('tr')[4]
            results[s_no_value][0]['ERA_all'] = float(ERA_all_temp.find_all('td')[1].text.split(',')[-1])
            results[s_no_value][1]['ERA_all'] = float(ERA_all_temp.find_all('td')[2].text.split(',')[-1])

            ERA_30_temp = starting.find_all('tr')[6]
            results[s_no_value][0]['ERA_30'] = float(ERA_30_temp.find_all('td')[1].text.split(',')[-1])
            results[s_no_value][1]['ERA_30'] = float(ERA_30_temp.find_all('td')[2].text.split(',')[-1])

            # 최근 10경기 승률
            win_10_2 = soup.find_all('div', class_='table_type03')[2]
            losses = 0
            wins = 0
            for span in win_10_2.find_all('span'):
                if span.text == 'L':
                    losses += 1
                elif span.text == 'W':
                    wins += 1
            results[s_no_value][0]['win_rate_10'] = wins / (wins + losses)

            win_10_3 = soup.find_all('div', class_='table_type03')[3]
            losses = 0
            wins = 0
            for span in win_10_3.find_all('span'):
                if span.text == 'L':
                    losses += 1
                elif span.text == 'W':
                    wins += 1
            results[s_no_value][1]['win_rate_10'] = wins / (wins + losses)

            # 최근 10경기 득점
            score_10_temp = win_10_2.find_all('a')
            score_10 = 0
            conceded_10 = 0
            for i in range(1, 56, 6):
                temp = score_10_temp[i].text.split(':')
                score_10 += int(temp[0])
                conceded_10 += int(temp[1])
            results[s_no_value][0]['score_10'] = score_10 / 10
            results[s_no_value][0]['conceded_10'] = conceded_10 / 10

            score_10_temp = win_10_3.find_all('a')
            score_10 = 0
            conceded_10 = 0
            for i in range(1, 56, 6):
                temp = score_10_temp[i].text.split(':')
                score_10 += int(temp[0])
                conceded_10 += int(temp[1])
            results[s_no_value][1]['score_10'] = score_10 / 10
            results[s_no_value][1]['conceded_10'] = conceded_10 / 10
        except:
            del results[s_no_value]

    return results

def randomAgent():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (iPad; CPU OS 13_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/70.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/601.7.7 (KHTML, like Gecko) Version/9.1.2 Safari/601.7.7',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0',
        'Mozilla/5.0 (Windows NT 10.0; rv:68.0) Gecko/20100101 Firefox/68.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36',
        'Mozilla/5.0 (Windows NT 5.1; rv:49.0) Gecko/20100101 Firefox/49.0',
        'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
        'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36',
        'Mozilla/5.0 (X11; Linux i686; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0',
        'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:64.0) Gecko/20100101 Firefox/64.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0',
    ]
    return random.choice(user_agents)
