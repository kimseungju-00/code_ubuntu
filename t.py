import json

with open('/media/ksj/hd/merged_file.json', 'r') as f:
    data = json.load(f)
    print(data)