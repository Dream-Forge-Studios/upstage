from bs4 import BeautifulSoup
import requests
import random
import json
from urllib.parse import urlparse, parse_qs
import re
from tqdm import tqdm

def statizCrawling(data):
    url = f'https://statiz.sporki.com' + data['link'].replace('summary', 'preview')

    parsed_url = urlparse(data['link'])
    query_params = parse_qs(parsed_url.query)

    # 's_no' 파라미터 값 추출
    s_no_value = query_params.get('s_no')[0]

    results = {
        s_no_value: []
    }

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

    agent = random.choice(user_agents)

    headers = {
        'User-Agent': agent
    }
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'

    soup = BeautifulSoup(response.text, 'html.parser')

    try:
        result = soup.find('div', class_='num').text.split(':')
        results[s_no_value].append({
            'venue': 0
        })

        results[s_no_value].append({
            'venue': 1
        })
        if int(result[0]) > int(result[1]):
            results[s_no_value][0]['result'] = 1
            results[s_no_value][1]['result'] = 0
        elif int(result[0]) < int(result[1]):
            results[s_no_value][0]['result'] = 0
            results[s_no_value][1]['result'] = 1
        else:
            return False
        type03 = soup.find('div', class_='table_type03')
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
    except:
        return False

    return results

for year in ['2023', '2022', '2021', '2020']:
    datas = {}
    with open(f'statuzGame_link_{year}.json', 'r', encoding='utf-8') as file:
        links = json.load(file)

    for link in tqdm(links, desc=f"Processing {year}"):
        results = statizCrawling(link)
        if results:
            datas.update(results)

    with open(f'statuzGame_{year}.json', 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)