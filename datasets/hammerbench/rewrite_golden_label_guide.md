# HammerBench Rewrite Golden Label Guide

목적:
HammerBench `External` 930개 데이터에 대해 SFT용 `rewrite` golden label을 작성한다.

이 작업에서는 `plan`은 만들지 않는다.  
라벨은 `rewrited_query`와 `arguments`만 포함한다.

## Source Data

라벨링 기준 파일은 아래 JSONL을 사용한다.

```text
/home/hj153lee/new-rma/datasets/hammerbench/en/multi-turn.phase1_external_mQsA.turn_gt_1.external.eval.jsonl
```

이 파일은 930개 row로 구성되어 있고, 각 row가 라벨링 단위다.

사용할 필드:

- `id`
- `prior_history`
- `current_user_utterance`
- `gold_call.arguments`

참고:

```text
/home/hj153lee/new-rma/datasets/hammerbench/en/multi-turn.phase1_external_mQsA.turn_gt_1.external.json
```

이 파일도 930개 데이터가 맞지만, 원본 대화 형식에 가까워서 `prior_history`, `current_user_utterance`, `gold_call`이 명시적으로 분리되어 있지 않다. 따라서 라벨 생성에는 `external.eval.jsonl`을 기준으로 사용한다.

## Output Files

원본 파일에는 직접 `rewrited_query` key를 추가하지 않는다.  
새 라벨 파일을 별도로 만든다.

권장 경로:

```text
/home/hj153lee/new-rma/datasets/hammerbench/labels/rewrite_golden.external.jsonl
```

Phase 1 batch 파일은 30개씩 나눠 저장한다.

```text
/home/hj153lee/new-rma/datasets/hammerbench/labels/phase1_batches/batch_001.jsonl
/home/hj153lee/new-rma/datasets/hammerbench/labels/phase1_batches/batch_002.jsonl
...
```

Phase 2 검증 결과도 별도 파일로 둔다.

```text
/home/hj153lee/new-rma/datasets/hammerbench/labels/rewrite_golden.external.verified.jsonl
```

## File Format

최종 라벨은 JSONL로 유지한다.

이유:

- `arguments`가 nested JSON object이므로 TSV보다 안전하다.
- 긴 `rewrited_query`와 특수문자/따옴표가 TSV에서 깨지기 쉽다.
- 기존 HammerBench eval/runs 파일들과 같은 형식이라 후속 스크립트 연결이 쉽다.

TSV는 필요하면 review용 view로만 생성한다.

예:

```text
id<TAB>rewrited_query<TAB>arguments_json
```

## Label Schema

프로젝트 호환성을 위해 `rewrited_query` 필드명을 그대로 사용한다.

```json
{
  "id": "External_10_3",
  "rewrited_query": "Plan a navigation route with departure \"Shanghai Hongqiao Railway Station\" and destination \"Beijing South Railway Station\".",
  "arguments": {
    "departure": "Shanghai Hongqiao Railway Station",
    "destination": "Beijing South Railway Station"
  }
}
```

## Core Rules

1. `arguments`는 planner gold arguments와 exact match여야 한다.
2. `rewrited_query`는 self-contained 문장이어야 한다.
3. `arguments`의 모든 value는 `rewrited_query` 안에 exact substring으로 들어가야 한다.
4. `arguments` value는 paraphrase하면 안 된다.
5. `rewrited_query`에 gold에 없는 extra slot value를 불필요하게 넣지 않는다.
6. 이전 function call에서 이미 확정된 값은 그대로 carry-forward 한다.
7. current turn 기준으로 다음 함수 호출에 필요한 정보만 담는다.

## Exact-Match Principle

`rewrited_query` 전체가 `arguments`와 exact match일 필요는 없다.  
대신 각 argument value는 `rewrited_query` 안에 정확히 포함되어야 한다.

좋은 예:

```text
Search a flight with date "01/May/2023" and departure "Beijing".
```

나쁜 예:

```text
Search a flight leaving from the capital on May 1, 2023.
```

이유:
- `01/May/2023`가 exact match가 아님
- `Beijing`이 exact match로 들어가지 않음

## What To Preserve Exactly

다음 값은 반드시 원문/정답 형태를 유지한다.

- 날짜
- 시간
- 이름
- 주소
- 역/공항/장소 이름
- enum-like canonical values
- 대소문자 차이가 의미 있는 값

예:

```text
8:00 AM, 10th March  -> 유지
01/May/2023          -> 유지
DRIVING              -> 유지
Email                -> 유지
```

## What Not To Do

금지:

- paraphrase
- normalization
- 번역
- surface form 변경
- coarse entity를 더 specific entity로 바꾸기
- 현재 turn에서 아직 확정되지 않은 slot 추가

예:

```text
Beijing -> Capital Airport Terminal T3        (금지)
Today at 3pm -> 3pm today                     (금지)
DRIVING -> drive                              (금지)
Email -> email                                (금지)
```

## Phase 1: Initial Generation

목표:
930개 전체에 대해 1차 golden label 후보를 생성한다.

배치:

- 30개씩 진행
- 총 31 batches

입력:

- conversation history
- current user turn
- planner gold arguments

출력:

- 각 example당 1개 JSON object
- `id`, `rewrited_query`, `arguments`

방법:

1. `arguments`는 planner gold arguments를 그대로 넣는다.
2. `rewrited_query`는 self-contained 문장으로 작성한다.
3. `arguments`의 모든 value가 `rewrited_query`에 exact match로 포함되게 만든다.
4. gold에 없는 extra value는 넣지 않는다.

Phase 1 요청 단위:

- 30개씩 assistant에게 요청
- output은 JSONL only

## Phase 2: Verification And Correction

목표:
1차 생성 라벨을 검증하고 수정한다.

배치:

- 50개씩 진행
- 총 19 batches
- 마지막 batch는 30개

검증 항목:

1. `arguments == planner gold.arguments`
2. `rewrited_query`가 self-contained인지
3. 모든 argument value가 `rewrited_query` 안에 exact substring으로 존재하는지
4. paraphrase가 없는지
5. extra slot value가 없는지
6. carry-forward가 필요한 값이 빠지지 않았는지

Phase 2 요청 단위:

- 50개씩 assistant에게 검증 요청
- 틀린 예시는 수정본까지 함께 출력

## Automatic Checks

스크립트로 최소한 아래를 자동 검사한다.

1. key exact match

```text
label.arguments.keys() == gold.arguments.keys()
```

2. value exact match

```text
label.arguments == gold.arguments
```

3. substring check

```text
for each argument value:
    exact value must appear in rewrited_query
```

4. parse check

```text
JSONL 형식이 깨지지 않았는지
```

## Recommended Workflow

1. Phase 1에서 30개씩 생성
2. 자동 검증 실행
3. 실패 샘플 표시
4. Phase 2에서 50개씩 검증 및 수정
5. 다시 자동 검증
6. 최종 합본 JSONL 생성

## Recommended Output Style

문장은 짧고 직접적으로 쓴다.

좋은 패턴:

```text
Search a flight with date "01/May/2023" and departure "Beijing".
View taxi order detail with time "8:00 AM, 10th March".
Cancel the flight booking with departure "Nanjing", destination "Chengdu", name "Li Si", and time "The day after tomorrow at 4pm".
```

## Summary

이 작업의 핵심은 다음 두 가지다.

1. `arguments`는 planner gold와 exact match
2. `rewrited_query`는 그 exact values를 그대로 포함하는 self-contained sentence

이 원칙만 지키면 Phase 1 자동 생성 + Phase 2 수동 검증 흐름으로 안정적으로 930개 라벨을 만들 수 있다.
