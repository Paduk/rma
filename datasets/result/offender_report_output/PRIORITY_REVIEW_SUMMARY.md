# Priority Review Summary

## Purpose

This note summarizes what to review first from:

- `common_offenders.csv`
- `offender_details.csv`

The goal is not to rank the lowest-accuracy plans only, but to identify plans that repeatedly break the expected turn/bucket difficulty structure across models and therefore deserve manual failure taxonomy review first.

## Key Takeaways

- `send_message` is the clearest cross-model turn-trend anomaly.
- `ACTION_OPEN_CONTENT` is the clearest turn-5 bucket-order anomaly.
- `ACTION_VIEW_CONTACT` is the most reusable review target for wrong-turn retrieval patterns.
- `ACTION_NAVIGATE_TO_LOCATION` is also useful, but is slightly less clean as a first taxonomy-building target than `ACTION_VIEW_CONTACT`.

## Review Priority

### 1. `send_message` (`turn_trend`)

Why first:

- It is the strongest shared `turn_offenders` case across all 5 models.
- Accuracy tends to increase from later turns instead of degrading with longer context.
- This suggests dataset/task-structure effects in addition to ordinary model failure.

What to inspect:

- whether later turns are easier because of surface-form repetition
- whether `turn_5` examples are more directly copyable than `turn_2`
- whether failures are mostly `argument_wrong` rather than `plan_wrong`
- whether message content is paraphrased, partially copied, or copied from the wrong turn

Likely failure subtypes to check first:

- `arg_wrong_turn_reference`
- `arg_partial_copy`
- `arg_format_mismatch`
- `arg_vague_or_placeholder`
- `arg_value_mismatch`

Why it matters:

- This is the best candidate for checking whether the expected "longer history = harder task" assumption actually holds for this dataset slice.

### 2. `ACTION_OPEN_CONTENT` (`turn5_bucket_offenders`)

Why second:

- It is the strongest shared turn-5 bucket offender across all 5 models.
- The expected order is that later buckets, especially `nonNR`, should become easier.
- Instead, `complex_3` and/or `nonNR` often collapse, which strongly suggests structural retrieval/copy problems.

What to inspect:

- whether the target `uri` or content reference is copied incompletely
- whether the model selects the wrong attachment/content source turn
- whether the model outputs a shortened filename instead of the full required URI
- whether `nonNR` examples contain annotation or formatting traps

Likely failure subtypes to check first:

- `arg_wrong_turn_reference`
- `arg_partial_copy`
- `arg_missing`
- `arg_format_mismatch`
- `arg_hallucinated`

Likely error fields:

- `uri`
- `attachments`
- `content`

Why it matters:

- This is the cleanest target for building a high-signal taxonomy around structural argument failures.

### 3. `ACTION_VIEW_CONTACT` (`turn3_bucket_offenders`, then `turn5_bucket_offenders`)

Why third:

- It appears as a shared offender in both turn-3 and turn-5 bucket analyses.
- Overall accuracy is often high, but specific buckets, especially `nonNR`, still break the expected ordering.
- That makes it a good "controlled" review case: the task is not globally bad, but the retrieval rule fails under certain reference conditions.

What to inspect:

- whether the model grabs the wrong contact from earlier history
- whether name/contact fields are semantically right but not exact-match identical
- whether `nonNR` examples trigger confusion between latest mention and earlier distractors

Likely failure subtypes to check first:

- `arg_wrong_turn_reference`
- `arg_value_mismatch`
- `arg_format_mismatch`
- `arg_incomplete`

Likely error fields:

- `contact`
- `name`
- `multiple`

Why it matters:

- This is the best candidate for validating a reusable wrong-turn retrieval pattern beyond message/content tasks.

## Secondary Target

### `ACTION_NAVIGATE_TO_LOCATION`

Use this after the top 3 if more review bandwidth is available.

Why:

- It is shared across models in turn-trend and bucket-order analyses.
- The pattern looks real, but as a taxonomy-seeding target it is slightly less clean than `ACTION_VIEW_CONTACT`.

What to inspect:

- location string mismatch
- partial address copy
- wrong-turn location retrieval
- overspecified or underspecified place names

## Practical Review Order

1. Review `send_message` rows first to validate whether the turn-trend assumption itself is unstable.
2. Review `ACTION_OPEN_CONTENT` rows next to build structural argument-failure labels.
3. Review `ACTION_VIEW_CONTACT` rows third to confirm whether wrong-turn retrieval generalizes across entity-copy tasks.
4. Use `ACTION_NAVIGATE_TO_LOCATION` as a follow-up validation set.

## Reading Caution

- These offender reports are best used for triage, not as final causal proof.
- Some `nonNR` buckets have relatively small support, so bucket-level swings can be noisy.
- A shared offender means "repeated anomaly across models", not necessarily "lowest absolute accuracy".

## Suggested Next Labeling Focus

If manual review starts now, the highest-value subtype order is:

1. `arg_wrong_turn_reference`
2. `arg_partial_copy` / `arg_incomplete`
3. `arg_format_mismatch`
4. `arg_value_mismatch`
5. `gt_suspect` checks for suspicious `nonNR` rows
