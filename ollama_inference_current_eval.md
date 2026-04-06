# `ollama_inference.py` 현재 평가 구조 문서화

이 문서는 `/home/hj153lee/RMA/ollama_inference.py`의 현재 동작을 기준으로, 평가 자동화 리팩터링 전에 구조를 고정해두기 위한 목적의 문서다.

## 1. 목적

현재 스크립트는 Ollama 모델을 대상으로 멀티턴 툴 호출 평가를 수행한다.

- 입력: TSV 테스트셋(`base`, `complex`, 일부 수동셋)
- 처리: 프롬프트 생성 -> Ollama 추론 -> JSON 파싱 -> GT와 exact match 비교
- 출력: 개별 샘플 결과 TSV + 파일별/전체 Accuracy 출력

이 스크립트는 현재 `test_key=base`, `test_key=complex`를 각각 따로 실행하는 사용 방식에 맞춰져 있다.

## 2. 엔트리포인트와 인자

인자 파서는 [`utils/frequently_used_tools.py`](/home/hj153lee/RMA/utils/frequently_used_tools.py#L63) 의 `get_arg_parse()`를 재사용한다.

실제로 `ollama_inference.py`에서 사용하는 인자는 아래 3개다.

- `--t`: 평가 타입. 예: `history-qwen3`, `rewrite-qwen3-base`
- `--o`: 결과 TSV 저장 경로
- `--test_key`: 평가할 데이터셋 묶음 키. 예: `base`, `complex`

실행 시작은 [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L392) 이다.

## 3. 상위 실행 흐름

전체 흐름은 [`main()`](/home/hj153lee/RMA/ollama_inference.py#L137) 안에 거의 모두 들어있다.

1. API metadata 로딩
2. `test_type(--t)`로 모델명과 prompt template 선택
3. `test_key(--test_key)`로 평가 대상 TSV 파일 리스트 선택
4. 각 TSV 파일을 순회하며:
   - dataset 로딩
   - prompt 생성
   - Ollama 호출
   - 결과 파싱
   - GT와 비교하여 `plan`, `arguments`, `all` 판정
   - 파일별 성능 출력
5. 모든 파일 결과를 합쳐 전체 성능 출력
6. 전체 결과를 `--o` 경로에 TSV로 저장

현재는 설정, 데이터셋 선택, 추론, 평가, 출력이 모두 `main()` 내부에 결합되어 있다.

## 4. 주요 함수별 역할

### 4.1 API 스펙 로딩

[`read_apis()`](/home/hj153lee/RMA/ollama_inference.py#L19)

- `simple=False`: `apis/api_v3.0.1.jsonl`을 읽어 `plan -> api_data` dict 생성
- `simple=True`: `apis/simple_api.json` 전체를 JSON dict로 로딩
- `examples`, `returns`, `next_turn_plans`는 제거

`main()`에서는 아래 두 개를 모두 읽는다.

- `apis = read_apis("apis/api_v3.0.1.jsonl", simple=False)`
- `sft_apis = read_apis("apis/simple_api.json", simple=True)`

하지만 실제 전처리에서는 현재 `sft_apis`만 사용한다. [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L333)

즉, `apis`는 현재 사실상 미사용 상태다.

### 4.2 응답 파싱

관련 함수:

- [`parse_response_json()`](/home/hj153lee/RMA/ollama_inference.py#L36)
- [`extract_json_from_markdown()`](/home/hj153lee/RMA/ollama_inference.py#L44)

실제 평가 루프에서는 `parse_response_json()`은 사용되지 않는다.

현재 파싱 순서는 아래와 같다.

1. `ast.literal_eval(raw)` 시도
2. 실패 시 `extract_json_from_markdown(raw)` 시도

주의점:

- `extract_json_from_markdown()` 실패 시 `None`을 반환할 수 있다.
- 이후 `result.get(...)` 호출이 이어지므로, 이 경우 예외가 발생할 수 있다.
- 예외 발생 시 해당 샘플은 모두 `fail` 처리된다.

### 4.3 Ollama 호출

[`generate_text()`](/home/hj153lee/RMA/ollama_inference.py#L71)

- 기본 host: `http://localhost:12434`
- `requests.post("{host}/api/generate")`
- 옵션:
  - `temperature=0.0`
  - `format=json`
  - `num_predict=512`
  - `stream=False`

성공 시 `response.json()["response"]` 문자열을 반환한다.

### 4.4 성능 출력

[`print_eval()`](/home/hj153lee/RMA/ollama_inference.py#L92)

평가 지표는 세 가지다.

- `plan`
- `arguments`
- `all`

계산 방식:

- 각 컬럼에서 `"pass"` 비율을 Accuracy로 계산
- `Macro Accuracy`는 위 세 Accuracy의 단순 평균

즉 현재 `Macro`는:

`(plan_acc + arguments_acc + all_acc) / 3`

추가 특징:

- 콘솔 출력
- `logs/ollama_inference_log.txt`에 append 저장
- `detail=True`일 때 GT plan별 성능 계산 가능

하지만 현재 `main()`에서는 `detail=True`로 호출하지 않으므로 plan별 상세 성능은 기본적으로 출력되지 않는다.

## 5. 모델/프롬프트 선택 구조

`--t` 값은 긴 `if-elif` 체인으로 해석된다. [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L178)

예시:

- `history-qwen3` -> `model_name='qwen3-history:latest'`, history prompt
- `rewrite-qwen3` -> `model_name='qwen3-rewrite:latest'`, rewrite prompt
- `history-qwen3-base` -> `model_name='qwen3-base:latest'`, zero-shot history prompt
- `rewrite-qwen3-base` -> `model_name='qwen3-base:latest'`, zero-shot rewrite prompt

특징:

- `model_name` 목록과 `model_prompts` dict가 따로 관리된다.
- base 계열은 실제 모델명 하나를 공유하면서 prompt만 바꾸는 경우가 있다.
- 문자열 규칙이 코드에 암묵적으로 들어있고, 선언적 설정 형태가 아니다.

자동화 관점에서 이 부분은 확장성과 유지보수성이 낮다.

## 6. 데이터셋 선택 구조

`--test_key`는 `data_files` dict에서 파일 리스트를 고르는 용도다. [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L232)

현재 활성화된 주요 키:

- `base`
- `complex`
- `manual_rewrited`

### 6.1 `base`

현재 `base`는 아래 4개 파일만 포함한다.

- `datasets/tc/it2_nonNR_tc.tsv`
- `datasets/tc/it3_nonNR_tc.tsv`
- `datasets/tc/it4_nonNR_tc.tsv`
- `datasets/tc/it5_nonNR_tc.tsv`

주의:

- `it2_NR_tc.tsv`는 주석 처리되어 빠져 있다.

### 6.2 `complex`

현재 `complex`는 아래 6개 파일을 포함한다.

- `datasets/tc/it3_complex_1_tc.tsv`
- `datasets/tc/it4_complex_1_tc.tsv`
- `datasets/tc/it4_complex_2_tc.tsv`
- `datasets/tc/it5_complex_1_tc.tsv`
- `datasets/tc/it5_complex_2_tc.tsv`
- `datasets/tc/it5_complex_3_tc.tsv`

### 6.3 구조적 특징

현재 구조에서는 한 번 실행 시 `data_files[test_key]` 하나만 순회한다. [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L328)

즉,

- `base` 평가 1회
- `complex` 평가 1회

를 별도로 실행해야 전체 성능을 수동으로 합쳐볼 수 있다.

이것이 현재 자동화를 방해하는 가장 직접적인 구조다.

## 7. 샘플 전처리 구조

전처리는 `main()` 내부의 내부 함수 [`preprocess_example_it()`](/home/hj153lee/RMA/ollama_inference.py#L297) 가 담당한다.

입력 row에서 사용하는 필드는 아래와 같다.

- `candidates`
- `conversation_history`
- `query`
- `rewrited_query`
- `answer`

처리 방식:

1. `candidates` 문자열을 `ast.literal_eval()`로 리스트 복원
2. 각 후보 plan에 대해 API 설명을 합쳐 `tools` 문자열 생성
3. `history` 타입이면:
   - `conversation_history`
   - `query`
   를 prompt에 삽입
4. `rewrite` 타입이면:
   - `rewrited_query`
   를 prompt에 삽입

반환 필드:

- `strprompt`
- `stranswer`
- `candidates`
- `rewrited_query`
- `query`
- `conversation_history`

## 8. 실제 평가 루프

평가 루프는 [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L327) 이후다.

파일 단위 흐름:

1. `load_dataset('csv', delimiter='\t')`로 TSV 로딩
2. `ds.map(...)`으로 prompt 생성
3. 첫 번째 prompt를 콘솔에 출력
4. 각 샘플에 대해 Ollama 호출
5. 결과 파싱
6. GT와 exact match 비교
7. 파일별 DataFrame 생성 및 `print_eval()`
8. 파일 결과를 `all_results`에 누적

### 8.1 정답 판정 방식

GT는 `ex["stranswer"]`를 다시 `ast.literal_eval()`해서 복원한다. [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L356)

판정은 strict exact match다.

- `plan_res = "pass"` if `result["plan"] == gt["plan"]`
- `arg_res = "pass"` if `result["arguments"] == gt["arguments"]`
- `all_res = "pass"` if 둘 다 pass

즉 부분점수, normalization, key order 보정, 값 타입 보정 등은 없다.

## 9. 결과 TSV 구조

최종 저장은 [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L389) 에서 수행한다.

샘플별 저장 컬럼은 아래와 같다.

- `conversation_history`
- `query`
- `rewrited_query`
- `candidates`
- `generation`
- `gt`
- `plan`
- `arguments`
- `all`
- `file`

여기서 `file`은 원본 TSV의 basename이다.

이 컬럼은 이후 파일별 성능 분석과 base/complex 통합 분석의 연결 키로 사용할 수 있다.

## 10. 현재 평균 성능 계산 방식

현재 평균은 두 단계로 존재한다.

### 10.1 파일 내부 평균

각 파일에 대해 `df_file` 기준으로 `plan`, `arguments`, `all` Accuracy를 계산한다. [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L383)

### 10.2 전체 평균

모든 파일 결과를 `all_results`로 합친 뒤, 다시 `print_eval(result)`를 호출한다. [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L387)

즉 현재 "전체 평균"은:

- 현재 실행한 `test_key` 범위 안에서만 계산된다.
- `base`와 `complex`를 동시에 포함하지 않는다.

따라서 현재 구조에서 사용자가 원하는 "전체 평균 성능"을 얻으려면, 실행을 두 번 하고 결과를 따로 합쳐야 한다.

## 11. 현재 코드의 자동화 관점 문제점

### 11.1 `base`와 `complex`를 한 번에 평가할 수 없음

원인:

- `test_key`가 단일 키만 받음
- `data_files[test_key]` 한 묶음만 순회함

영향:

- 실행 2회 필요
- 결과 파일도 2개로 분리됨
- 전체 평균을 사람이 따로 계산해야 함

### 11.2 설정과 실행 로직이 강결합

`main()` 안에 아래 책임이 모두 섞여 있다.

- 모델 선택
- prompt 선택
- 데이터셋 묶음 선택
- prompt 생성
- API 호출
- JSON 파싱
- 채점
- 출력 저장

영향:

- 자동화 옵션 추가가 어려움
- 재사용 가능한 함수 단위가 부족함

### 11.3 모델 선택 로직이 선언적이지 않음

긴 `if-elif` 체인 때문에:

- 새 모델 추가 시 수정 포인트가 많음
- 오타와 매핑 불일치 가능성이 큼

### 11.4 평가 집계가 단일 레벨만 지원

현재 기본 출력은:

- 파일별 집계
- 현재 실행 범위 전체 집계

부족한 것:

- `base`/`complex` split별 집계
- 전체 통합 집계
- turn 수 기준 집계
- plan별 집계의 자동 저장

### 11.5 결과 메타데이터가 부족함

최종 TSV에 아래 정보가 없다.

- `test_key`
- `split` (`base`/`complex`)
- `test_type`
- `model_name`

현재는 출력 파일명과 콘솔 로그에 의존해서 맥락을 복원해야 한다.

통합 자동화 시 이 정보가 컬럼으로 저장되어야 후처리가 쉬워진다.

### 11.6 디버그/에러 처리가 약함

- `raw`가 할당되기 전 예외가 나면 에러 로그에서 혼선 가능
- 파싱 실패 시 구조화된 failure reason이 남지 않음
- 재시도 로직 없음

## 12. `result_anal.py`와의 관계

[`datasets/result/result_anal.py`](/home/hj153lee/RMA/datasets/result/result_anal.py#L4) 는 저장된 결과 TSV를 다시 읽어 plan별/파일별 정확도를 pivot 형태로 보는 후처리 스크립트다.

현재 특징:

- `gt`에서 `plan`을 다시 꺼냄
- `all == pass`를 기준으로 정확도 계산
- 여러 TSV를 옆으로 붙여 비교

하지만 이 스크립트는 현재 주 평가 파이프라인에 통합되어 있지 않다.

즉 "평가 실행"과 "결과 분석"이 분리되어 있고, 사용자가 수동으로 파일 목록을 넣어야 한다.

## 13. 리팩터링 전 기준 결론

현재 `ollama_inference.py`는 "단일 `test_key`에 대해 한번 실행하고, 그 범위의 결과를 저장하는 평가 스크립트"다.

당장 자동화하려면 최소한 아래 2개가 바뀌어야 한다.

1. `base`와 `complex`를 한 번에 돌릴 수 있는 split orchestration
2. split별 성능 + 전체 통합 성능을 동시에 계산하는 집계 계층

그 다음 단계로는 아래가 필요하다.

- 모델/프롬프트 설정 테이블화
- 결과 메타데이터 컬럼 확장
- 분석 스크립트(`result_anal.py`) 일부를 평가 파이프라인 내부로 흡수

## 14. 현재 코드 기준 핵심 참조 위치

- API 로딩: [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L19)
- Ollama 호출: [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L71)
- 성능 출력: [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L92)
- 모델/프롬프트 선택: [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L178)
- 데이터셋 묶음 선택: [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L232)
- 샘플 전처리: [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L297)
- 평가 루프: [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L327)
- 결과 저장: [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py#L389)
- CLI 인자 정의: [`utils/frequently_used_tools.py`](/home/hj153lee/RMA/utils/frequently_used_tools.py#L63)
