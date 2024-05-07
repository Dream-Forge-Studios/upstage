from bs4 import BeautifulSoup
import requests
import random
import time
import json

# pip install beautifulsoup4 requests

def saraminCrawling(text, num):
    results = []
    url = f'https://www.saramin.co.kr/zf_user/search/recruit?searchType=search&searchword={text}&recruitPage={num}'
    print(url)
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

    if soup.find('div', class_='info_no_result'):
        return False
    divs = soup.find_all('div', class_='item_recruit')
    if divs == []:
        print(agent)
    # 찾은 'div' 태그들의 텍스트 출력
    for div in divs:
        company = div.find('a', class_='track_event data_layer').text.strip()

        job_tit = div.find('h2', class_='job_tit')
        title = job_tit.text.strip()
        link = 'https://www.saramin.co.kr' + job_tit.find('a')['href']

        job_condition = div.find('div', class_='job_condition').find_all('span')
        region = job_condition[0].text.strip()
        career = job_condition[1].text.strip()

        deadline = div.find('span', class_='date').text.strip()

        result = {
            'company': company,
            'title': title,
            'link': link,
            'region': region,
            'career': career,
            'deadline': deadline,
        }
        results.append(result)
    print(len(results))
    return results

keywords = [
    {'name': '블록체인', 'keyword': 'blockchain'},
    {'name': '클라우드', 'keyword': 'cloud'},
    {'name': '사이버보안', 'keyword': 'cyberSecurity'},
    {'name': '개인정보보호', 'keyword': 'privacy'},
]

for target in keywords:
    num = 1
    datas = []
    while True:
        temp = saraminCrawling(target['name'], num)
        if temp:
            datas.extend(temp)
        else:
            break
        num += 1
        # time.sleep(random.randint(1, 3))
    key = target['keyword']
    # JSON 파일로 저장
    with open(f'{key}_saramin_school.json', 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)