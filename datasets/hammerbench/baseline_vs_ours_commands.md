# HammerBench Baseline vs Ours

기본 평가 대상은 `External` subset입니다.

빠른 테스트만 하고 싶으면 각 명령어 끝에 `--limit 100`을 추가하면 됩니다.

## Baseline

설명:
전체 `conversation history + tools`를 planner에 그대로 넣고, 모델이 바로 다음 function call의 `name`과 `arguments`를 예측합니다.  
Rewrite 단계는 없습니다.

명령어:

```bash
python3 /home/hj153lee/new-rma/datasets/hammerbench/run_hammerbench_pipeline.py \
  --mode baseline \
  --planner-backend cloud \
  --planner-model gpt-5.4-mini \
  --planner-reasoning-effort medium
```

## Ours

설명:
먼저 5-shot structured rewrite를 수행해서 `rewritten_query`와 `confirmed_arguments`를 만들고,  
그 다음 planner가 이 구조화된 정보를 바탕으로 최종 function call을 예측합니다.

명령어:

```bash
python3 /home/hj153lee/new-rma/datasets/hammerbench/run_hammerbench_pipeline.py \
  --mode ours \
  --planner-backend cloud \
  --planner-model gpt-5.4-mini \
  --planner-reasoning-effort medium \
  --rewrite-backend cloud \
  --rewrite-model gpt-5.4-mini \
  --rewrite-reasoning-effort medium \
  --rewrite-prompt-key structured_fewshot_full_5
```
