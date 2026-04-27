# HammerBench Eval Implementation Plan

## Goal

Phase 1 (`External`, `mQsA`, `turn-id > 1`) 기준으로

- baseline: `full history -> planner`
- ours: `full history -> rewrite -> rewritten_query -> planner`

평가 파이프라인을 순차적으로 구현한다.

## Step 1

Canonical eval JSONL 생성기 구현

- input: phase1 subset JSON
- output fields:
  - `id`
  - `data_type`
  - `prior_history`
  - `current_user_utterance`
  - `planner_history`
  - `tool_schema` (`required` 제거)
  - `gold_call`
  - `oracle_rewritten_query`

## Step 2

프롬프트 생성기 구현

- baseline planner prompt
- rewrite prompt
- planner-from-rewrite prompt
- optional ablation prompt (`full_history + rewritten_query + tool`)

## Step 3

예측 결과 포맷 및 실행 흐름 구현

- baseline prediction JSONL
- rewrite prediction JSONL
- ours prediction JSONL
- optional oracle prediction JSONL

## Step 4

평가 스크립트 구현

- API accuracy
- argument accuracy
- exact match on final function call

## Note

- `gold_call`은 평가용 정답으로만 사용하고 프롬프트에는 넣지 않는다.
- `oracle_rewritten_query`는 우선 `null` placeholder로 두고, 이후 별도 라벨링 파일로 채운다.
