import json
data = []
with open('BFCL_v3_multi_turn_base.json', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
print(len(data))  # 200 entries
print(data[0]['id'], data[0]['question'][0][0]['content']) # 첫 사례 확인
