import json

# 결과를 저장할 딕셔너리
result = {}

# JSONL 파일 경로 (예: "plan_descriptions.jsonl")
input_file = "apis/api_v3.0.1.jsonl"

# JSONL 파일 한 줄씩 읽어서 결과 딕셔너리에 저장
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.strip()
        if not line:
            continue  # 빈 줄은 건너뜁니다.
        try:
            data = json.loads(line)
            plan_key = data.get("plan")
            # "arguments" 필드가 존재하는 경우 그 키들을 리스트로 만듭니다.
            arguments = data.get("arguments", {})
            arg_keys = list(arguments.keys())
            if plan_key:
                result[plan_key] = arg_keys
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 에러 발생: {e}")

# 결과 딕셔너리를 simple_api.json 파일로 저장
output_file = "apis/simple_api.json"
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(result, outfile, ensure_ascii=False, indent=4)

print(f"결과가 '{output_file}' 파일에 저장되었습니다.")
