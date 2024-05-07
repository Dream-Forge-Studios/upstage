from bs4 import BeautifulSoup
import requests
import random
import time
import json

# pip install beautifulsoup4 requests

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    except TypeError as e:
        return False
    return True

def jabkoreaCrawling(text, num):
    results = []
    # num = 213123
    url = f'https://www.jobkorea.co.kr/Search/?stext={text}&tabType=recruit&Page_No={num}'
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

    while True:
        agent = random.choice(user_agents)
        headers = {
            'User-Agent': agent
        }
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        listDefault = soup.find('div', class_='list-default')
        try:
            lis = listDefault.find_all('li', class_='list-post')
            if lis:
                for li in lis:
                    data = li.get('data-gainfo')
                    try:
                        if data:
                            if is_json(data):
                                data = json.loads(data)
                            company = data['dimension48']
                            title = data['dimension45']
                            link = 'https://www.jobkorea.co.kr' + li.find('a')['href']

                            option = li.find('p', class_='option').find_all('span')
                            if len(option) == 4:
                                region = option[2].text.strip()
                                career = option[0].text.strip()
                                deadline = option[-1].text.strip()
                            elif len(option) == 5:
                                region = option[3].text.strip()
                                career = option[0].text.strip()
                                deadline = option[-1].text.strip()
                            elif len(option) == 6:
                                region = option[4].text.strip()
                                career = option[0].text.strip()
                                deadline = option[-1].text.strip()
                            else:
                                print()
                            result = {
                                'company': company,
                                'title': title,
                                'link': link,
                                'region': region,
                                'career': career,
                                'deadline': deadline,
                            }
                            results.append(result)
                        else:
                            return False
                    except:
                        print()
                break
            else:
                print(agent)
        except:
            break
    print(len(results))
    return results

keywords = [
    # {'name': '블록체인', 'keyword': 'blockchain'},
    # {'name': '클라우드', 'keyword': 'cloud'},
    {'name': '사이버보안', 'keyword': 'cyberSecurity'},
    {'name': '개인정보보호', 'keyword': 'privacy'},
]

for target in keywords:
    num = 1
    datas = []
    while True:
        temp = jabkoreaCrawling(target['name'], num)
        if temp:
            datas.extend(temp)
        else:
            break
        num += 1
        # time.sleep(random.randint(1, 3))
    key = target['keyword']
    # JSON 파일로 저장
    with open(f'{key}_job_school.json', 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)