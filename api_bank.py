import json

# JSON 파일 경로
json_path = 'level-1-api.json'

# JSON 데이터 로드
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# User: 가 정확히 1번만 등장하는 데이터 필터링
filtered_data = [item for item in data if item['input'].count('User:') == 1]

# 필터링된 데이터 출력
for item in filtered_data:
    print(item["input"])
    #print(json.dumps(item, ensure_ascii=False, indent=2))

# 총 데이터 개수 출력
print(f"\n총 데이터 개수: {len(filtered_data)}")
