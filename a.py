import json
import re

# JSON 파일 로드
with open('/media/ksj/hd/code/merged_file.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 불용어 제거 함수
def remove_stop_words(text):
    # 불용어 제거
    cleaned_text = re.sub(r'\\', '', text)
    cleaned_text = re.sub(r'\\n', ' ', cleaned_text)
    cleaned_text = re.sub(r'\\"', '"', cleaned_text)
    cleaned_text = re.sub(r"\\'", "'", cleaned_text)
    return cleaned_text

# 데이터 처리
for item in data:
    document = item['document']
    cleaned_document = remove_stop_words(document)
    item['document'] = cleaned_document

# JSON 파일에 수정된 데이터 저장
with open('/media/ksj/hd/code/merged_file.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)