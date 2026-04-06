# Failure Taxonomy Guide

## Purpose

This guide defines the label system for first-pass failure analysis on exact-match evaluation data.

The goal is to make failure review consistent across:

- automated pre-labeling
- human review
- later error analysis and aggregation

This taxonomy is designed for tasks where correctness depends on exact match of:

- `plan`
- `arguments`

## Labeling Levels

Each reviewed example should be described using:

1. `primary_label`
2. `error_field`
3. `error_subtype`
4. optional reviewer notes

These levels should be interpreted in order.

## 1. Primary Label

### `plan_wrong`

Meaning:

- The predicted `plan` does not exactly match the ground-truth `plan`.

Use when:

- the model chose the wrong action entirely
- the prediction cannot be meaningfully analyzed only as an argument mismatch

### `argument_wrong`

Meaning:

- The predicted `plan` is correct, but one or more `arguments` do not exactly match ground truth.

Use when:

- the plan is right
- failure comes from missing, wrong, vague, hallucinated, or mis-grounded argument values

### `generation_failure`

Meaning:

- The model output could not be parsed or was malformed in a way that prevents normal comparison.

Use when:

- parsing failed
- runtime or formatting error occurred
- output is structurally unusable

### `gt_suspect`

Meaning:

- The ground-truth annotation itself may be wrong, inconsistent, or not grounded in the conversation.

Use when:

- the expected value is not supported by conversation history
- the expected value appears to reference the wrong turn
- the model output appears grounded, but the ground truth does not

Important:

- `gt_suspect` is a review flag, not an automatic proof that the annotation is wrong.

## 2. Error Field

This identifies which argument field is problematic.

Common values:

- `to`
- `message`
- `attachments`
- `uri`
- `title`
- `name`
- `content`
- `datetime`
- `location`
- `contact`
- `multiple`
- `none`

Guidance:

- Use the exact argument key when possible.
- Use `multiple` if more than one field is wrong.
- Use `none` for `plan_wrong`, `generation_failure`, or when no meaningful field can be isolated.

## 3. Error Subtype

This identifies how the failure happened.

### `arg_missing`

Meaning:

- A required argument key is absent.

Example:

- attachment expected, but no attachment field predicted

### `arg_extra`

Meaning:

- An unnecessary argument key was added.

### `arg_value_mismatch`

Meaning:

- The key is correct, but the value is wrong.

Use as the default subtype when a more specific subtype below does not clearly apply.

### `arg_wrong_turn_reference`

Meaning:

- The model copied a value from the wrong history turn.

Use when:

- the correct value exists in history
- the predicted value also exists in history
- but it came from a different turn than the one intended by the bucket/task setup

### `arg_vague_or_placeholder`

Meaning:

- The predicted value is underspecified or templatic.

Examples:

- `that info`
- `text here`
- `the file`

### `arg_hallucinated`

Meaning:

- The predicted value is not supported by the query or conversation history.

### `arg_format_mismatch`

Meaning:

- The semantic value is close, but exact match fails because of formatting differences.

Examples:

- different punctuation
- spacing differences
- URI formatting mismatch

Use carefully:

- only use this when the value is substantively the same and the failure is mostly surface form

### `arg_partial_copy`

Meaning:

- The model copied only part of the required value.

Example:

- copied a filename but omitted the full URI
- copied only part of a message string

### `arg_incomplete`

Meaning:

- The value is partially correct but missing one or more required components.

Example:

- copied an address but omitted city/postcode
- copied one attachment when multiple are needed

### `arg_over_specific`

Meaning:

- The prediction adds unsupported detail beyond what should have been copied.

### `parse_or_runtime_failure`

Meaning:

- Use under `generation_failure` when the actual cause is parser/runtime failure.

### `gt_not_grounded_in_history`

Meaning:

- Use under `gt_suspect` when the ground-truth value cannot be found in the conversation history.

### `gt_turn_mismatch`

Meaning:

- Use under `gt_suspect` when the ground truth appears to refer to the wrong turn.

### `gt_value_inconsistent`

Meaning:

- Use under `gt_suspect` when the annotation is inconsistent with query/history/task semantics.

## Recommended Label Combinations

### Wrong plan

- `primary_label = plan_wrong`
- `error_field = none`
- `error_subtype = none`

### Wrong message value

- `primary_label = argument_wrong`
- `error_field = message`
- `error_subtype = arg_value_mismatch`

### Missing attachment

- `primary_label = argument_wrong`
- `error_field = attachments`
- `error_subtype = arg_missing`

### Picked attachment from wrong turn

- `primary_label = argument_wrong`
- `error_field = attachments`
- `error_subtype = arg_wrong_turn_reference`

### Placeholder message

- `primary_label = argument_wrong`
- `error_field = message`
- `error_subtype = arg_vague_or_placeholder`

### Parser failure

- `primary_label = generation_failure`
- `error_field = none`
- `error_subtype = parse_or_runtime_failure`

### Suspicious annotation

- `primary_label = gt_suspect`
- `error_field = message` or other relevant field
- `error_subtype = gt_not_grounded_in_history` or related subtype

## Suggested Review Columns

Recommended columns for manual or automatic labeling:

- `primary_label`
- `error_field`
- `error_subtype`
- `expected_turn`
- `predicted_source_turn`
- `gt_supported_in_history`
- `pred_supported_in_history`
- `review_notes`

## Review Efficiency Columns

The labeled review TSVs may also include helper columns intended for faster sorting and filtering during manual review.

### `error_signature`

Meaning:

- A compact combined key in the form:
  - `primary_label|error_field|error_subtype`

Use when:

- grouping near-identical failure patterns
- sorting spreadsheet rows by failure type
- pivoting counts by exact review bucket

### `review_priority`

Meaning:

- A coarse review order bucket for triage.

Current values:

- `p0_generation_failure`
- `p1_plan_wrong`
- `p2_gt_suspect`
- `p3_multi_field_argument`
- `p4_structural_argument`
- `p5_attachment_argument`
- `p6_simple_argument`

Use when:

- reviewing the hardest or most important cases first
- separating structural failures from high-volume simple mismatches

### `is_plan_case`

Meaning:

- `true` if the row is mainly a wrong-plan case.

### `is_attachment_case`

Meaning:

- `true` if the isolated failure field is `attachments`.

Use when:

- quickly batching attachment-copy failures

### `is_message_case`

Meaning:

- `true` if the isolated failure field is `message`.

Use when:

- reviewing paraphrase, hallucination, or under-copy message failures together

### `is_to_case`

Meaning:

- `true` if the isolated failure field is `to`.

Use when:

- reviewing recipient resolution failures together

### `is_multi_field_case`

Meaning:

- `true` if the row is labeled as `multiple`.

Use when:

- separating compound failures from single-field failures

### `is_gt_suspect_candidate`

Meaning:

- `true` if the row is labeled `gt_suspect` or looks like a candidate for GT review based on support signals.

Use when:

- creating a shortlist for annotation-audit review
- checking whether a failure may reflect label quality rather than only model quality

## Korean One-Line Summary

### `primary_label`

- `plan_wrong`: 아예 다른 액션을 예측한 경우
- `argument_wrong`: 액션은 맞지만 인자값이 틀린 경우
- `generation_failure`: 출력 파싱 자체가 안 되거나 구조가 깨진 경우
- `gt_suspect`: 정답 라벨 자체가 대화 근거와 안 맞을 가능성이 있는 경우

### `error_subtype`

- `arg_missing`: 필요한 인자가 빠진 경우
- `arg_extra`: 불필요한 인자가 추가된 경우
- `arg_value_mismatch`: 같은 필드지만 값이 다른 경우
- `arg_wrong_turn_reference`: 다른 턴의 값을 잘못 참조한 경우
- `arg_vague_or_placeholder`: `that info`, `text here` 같은 모호한 값인 경우
- `arg_hallucinated`: 대화나 쿼리에 근거 없는 값을 만든 경우
- `arg_format_mismatch`: 의미는 비슷하지만 표면 형식 차이로 exact match가 깨진 경우
- `arg_partial_copy`: 필요한 값의 일부만 복사한 경우
- `arg_incomplete`: 값은 맞는 방향이지만 필수 구성요소가 빠진 경우
- `arg_over_specific`: 근거 없이 필요 이상으로 구체화한 경우
- `parse_or_runtime_failure`: 파서/런타임 오류로 정상 비교가 불가능한 경우
- `gt_not_grounded_in_history`: GT 값이 대화 이력에서 확인되지 않는 경우
- `gt_turn_mismatch`: GT가 다른 턴을 기준으로 잡힌 것 같은 경우
- `gt_value_inconsistent`: GT가 쿼리/이력/태스크 의미와 불일치하는 경우

## Notes for Human Labelers

- Prefer the most specific subtype that is clearly justified.
- If uncertain, use:
  - `argument_wrong`
  - the best available `error_field`
  - `arg_value_mismatch`
- Use `gt_suspect` conservatively.
- If both annotation suspicion and model error are possible, note both in `review_notes`, but keep `primary_label` focused on the strongest evidence.
