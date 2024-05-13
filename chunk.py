import json
import heapq

# JSON 파일 읽기
with open('/media/ksj/hd/merged_file.json', 'r') as f:
    data = json.load(f)

# 데이터를 크기에 따라 정렬
sorted_data = sorted(data, key=lambda x: len(json.dumps(x)), reverse=True)

# 파일 개수
num_files = 3

# 최소 힙을 사용하여 파일 크기 추적
file_sizes = [(0, i) for i in range(num_files)]
heapq.heapify(file_sizes)

# 각 파일에 데이터 분배
files = [[] for _ in range(num_files)]
for item in sorted_data:
    size, index = heapq.heappop(file_sizes)
    files[index].append(item)
    heapq.heappush(file_sizes, (size + len(json.dumps(item)), index))

# 각 파일을 JSON으로 저장
for i, file_data in enumerate(files):
    with open(f'output_{i+1}.json', 'w') as f:
        json.dump(file_data, f, indent=4, ensure_ascii=False)