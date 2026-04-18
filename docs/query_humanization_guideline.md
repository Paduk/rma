# Query Humanization Guideline

## Column Roles

- `ori_query`: original follow-up user query.
- `ori_rewrited_query`: original explicit rewrite with context resolved.
- `new_query`: clean, natural, human-like follow-up query.
- `rewrited_query`: clean explicit rewrite that resolves context and preserves all slots.
- `query`: ASR/noisy version of `new_query`.

## Core Rules

- Preserve the target intent and `plan`.
- Do not change required arguments from `answer`.
- Keep `new_query` and `rewrited_query` clean and grammatical.
- Put typo/ASR noise only in `query`.
- `new_query` may keep pronouns such as `it`, `that`, `them`, `there` when natural.
- `rewrited_query` should resolve those references using conversation history.
- Do not invent new files, contacts, places, dates, times, apps, or message contents.

## Protected Values

Do not corrupt or paraphrase these in `rewrited_query`:

- URI values: `content://...`, `file://...`, `https://...`
- filenames: `Report.pdf`, `style.css`, `MeetingNotes.docx`
- emails and phone numbers
- dates, times, durations, alarm IDs
- event titles, contact names, app names, playlist names
- addresses and coordinates

## ASR Noise Rules for `query`

Allowed light noise:

- typo: `please -> plase`, `beginning -> beggining`
- homophone: `to -> two`, `for -> four`, `there -> their`
- minor missing punctuation
- light word confusion: `track -> trek`, `video -> vdeo`

Avoid:

- changing intent words too much
- corrupting filenames, URIs, emails, phone numbers
- changing dates, times, IDs, durations
- making the sentence impossible to understand
