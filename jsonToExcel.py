import pandas as pd

fileList = [
'blockchain_job_school.json',
'blockchain_saramin_school.json',
'blockchain_wanted_school.json',
'cloud_job_school.json',
'cloud_saramin_school.json',
'cloud_wanted_school.json',
'cyberSecurity_job_school.json',
'cyberSecurity_saramin_school.json',
'cyberSecurity_wanted_school.json',
'privacy_job_school.json',
'privacy_saramin_school.json',
'privacy_wanted_school.json'
]

for file in fileList:
    # JSON 파일 읽기
    with open(file, 'r', encoding='utf-8') as f:
        data = pd.read_json(f)

    # 칼럼명 변경
    data = data.rename(columns={'link': '링크', 'company': '회사', 'title': '제목'})

    names = file.split('_')
    if names[0] == 'blockchain':
        names[0] = '블록체인'
    elif names[0] == 'cloud':
        names[0] = '클라우드'
    elif names[0] == 'cyberSecurity':
        names[0] = '사이버보안'
    elif names[0] == 'privacy':
        names[0] = '개인정보보호'

    if names[1] == 'job':
        names[1] = '잡코리아'
    elif names[1] == 'saramin':
        names[1] = '사람인'
    elif names[1] == 'wanted':
        names[1] = '원티드'

    fileName = f'{names[0]}_{names[1]}.xlsx'

    data = data[['링크', '회사', '제목']]
    # 엑셀 파일로 저장
    data.to_excel(fileName, index=False)
