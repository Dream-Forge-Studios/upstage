import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

keywords = [
    # {'name': '웹', 'keyword': 'web'},
    {'name': '서버', 'keyword': 'server'},
    {'name': '프론트', 'keyword': 'front'},
    {'name': '머신러닝', 'keyword': 'machine'},
    {'name': '앱개발', 'keyword': 'app'},
    {'name': '하드웨어', 'keyword': 'hard'},
    {'name': 'DBA', 'keyword': 'dba'},
    {'name': '퍼블리셔', 'keyword': 'publisher'},
]

for target in keywords:
    key = target['keyword']
    with open(f'{key}_saramin.json', 'r', encoding='utf-8') as file:
        datas = json.load(file)

    result = []
    for data in datas:
        try:
            driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
            driver.get(data['link'])
            wait = WebDriverWait(driver, 10)  # 10초 동안 대기
            address_element = wait.until(
                EC.visibility_of_element_located(
                    (By.CLASS_NAME, 'txt_adr'))
            )
            address = address_element.text
            data['address'] = re.sub(r"\([^)]*\)", "", address).strip()
            print(address)
            result.append(data)
        except:
            pass
    # JSON 파일로 저장
    with open(f'{key}_saramin_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)