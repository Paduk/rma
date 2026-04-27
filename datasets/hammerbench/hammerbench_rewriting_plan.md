# HammerBench Rewriting Evaluation Plan

## Goal

HammerBench `multi-turn.json`에서 `turn-id > 1`인 follow-up snapshot만 사용해,

- full conversation history -> planner
- rewritten self-contained query -> planner

두 설정의 API planning / argument prediction 성능 차이를 비교한다.

## Base Data

- Source: `data/en/multi-turn.json`
- Unit of evaluation: function-calling snapshot
- Filtering rule: `id = <data-type>_<conversation-id>_<turn-id>`에서 `turn-id > 1`만 사용

## Phase 1

1차 실험 타겟:

- `External`
- `mQsA`

선정 이유:

- `External`: 이전 대화 및 외부 정보(`<EK>`)를 현재 query에 self-contained하게 복원해야 한다.
- `mQsA`: assistant가 여러 slot을 물었지만 user는 일부만 답하는 구조라서, rewrite가 문맥 해소는 하되 없는 slot을 hallucinate하지 않아야 한다.

핵심 검증 포인트:

- external information resolution
- partial answer handling
- omitted slot restoration without fabrication
- downstream argument prediction improvement by rewriting

## Phase 2

2차 실험 타겟:

- `Based`
- `mQmA`
- `mQsA`
- `sQmA`
- `External`

즉, `Diverse Q&A + External` 전체를 사용한다.

선정 이유:

- rewriting의 핵심 failure mode인 follow-up context dependence, incomplete instruction, partial answer, multi-slot answer, external reference를 함께 평가할 수 있다.
- 1차 실험에서 확인한 rewrite 효과가 broader multi-turn setting에서도 유지되는지 검증할 수 있다.

## Labeling Principle For Oracle Rewrite

- 현재 snapshot의 user utterance를 self-contained query로 재작성한다.
- 이전 대화와 `<EK>`에 명시된 정보만 사용한다.
- 대화에 없는 slot 값은 새로 만들지 않는다.
- user가 일부 slot만 답한 경우, rewrite도 그 일부만 명시한다.
- 이전 slot이 correction 된 경우 가장 최신 값을 반영한다.

## Suggested Evaluation Comparison

- original current query + full conversation history -> planner
- oracle rewritten query -> planner
- model rewritten query -> planner

주요 지표:

- API accuracy
- argument / slot accuracy
- exact-match on final function call
