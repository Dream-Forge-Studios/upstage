import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

keywords = [
    {'name': '블록체인', 'keyword': 'blockchain'},
    {'name': '클라우드', 'keyword': 'cloud'},
    {'name': '사이버보안', 'keyword': 'cyberSecurity'},
    {'name': '개인정보보호', 'keyword': 'privacy'},
]

for target in keywords:
    key = target['keyword']
    name = target['name']
    results = []

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get(f'https://www.wanted.co.kr/search?query={name}&tab=position')
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # 스크롤을 페이지의 맨 아래로 내리기
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # 페이지가 새 데이터를 로드할 시간을 기다림
        time.sleep(3)

        # 새로운 스크롤 높이를 계산하고 이전 스크롤 높이와 비교
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    job_cards = driver.find_elements(By.CLASS_NAME, "JobCard_container__FqChn")
    for card in job_cards:
        a_tag = card.find_element(By.TAG_NAME, 'a')
        link = a_tag.get_attribute('href')
        company = a_tag.get_attribute('data-company-name')
        title = a_tag.get_attribute('data-position-name')
        results.append({
            'link': link,
            'company': company,
            'title': title,
        })
    with open(f'{key}_wanted_school.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)