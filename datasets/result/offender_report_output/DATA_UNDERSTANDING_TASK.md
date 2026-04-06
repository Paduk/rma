# Data Understanding Task

## Goal

Analyze failure cases where success is determined by exact match against the ground-truth `plan` and `arguments`.

The main question is not only whether a prediction failed, but:

- which field failed (`plan` vs `arguments`)
- which argument key/value was wrong
- whether the model copied the value from the wrong turn
- whether the model failed to retrieve an available value from history
- whether the model hallucinated a value not supported by the conversation

## Input Structure

Each example contains:

- `conversation_history`
- `query`
- model output fields such as `generation`
- ground-truth `gt`

`conversation_history` is a list of prior user/assistant turns.
Each item looks like:

- `turn N: user utterance -> assistant response`

The answer needed for `arguments` is expected to be recoverable from `conversation_history`.

## Difficulty / Reference Buckets

The bucket name indicates which turn in `conversation_history` should be referenced to recover the relevant value.

- `it_complex_1`: reference turn 1
- `it_complex_2`: reference turn 2
- `it_complex_3`: reference turn 3
- `nonNR`: reference the last turn in the history

Examples:

- In `Turn 4 / it4_complex_1`, the correct attachment may come from history turn 1, while later turns contain distractors.
- In `Turn 5 / nonNR`, the correct value should come from the latest turn.

## Evaluation Semantics

- `plan = pass/fail` indicates whether the predicted plan exactly matched the ground truth.
- `arguments = pass/fail` indicates whether the predicted arguments exactly matched the ground truth.
- `all = pass/fail` is the final exact-match result over both.

This means many important failures are `plan=pass, arguments=fail`.

## Practical Review Focus

When reviewing failure cases, prioritize:

1. Wrong value copied from the wrong history turn
2. Placeholder or underspecified value instead of grounded value
3. Missing attachment / wrong attachment URI
4. Partial paraphrase that fails exact match
5. Parsing/runtime failure in generation

## Current Review Workflow

Priority offender review TSVs mix:

- all failure rows
- a small number of pass comparison rows

and use:

- `review_label = fail | pass_sample`

This is intended for side-by-side human inspection of exact-match failure patterns.
