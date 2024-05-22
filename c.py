import json
import os

def split_json_file(file_path, chunk_size):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_chunks = (len(data) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        chunk_file_path = f"{file_path[:-5]}_chunk_{i}.json"
        with open(chunk_file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=4)

# 사용 예시
split_json_file("/media/ksj/hd/merged_file.json", chunk_size=50000)