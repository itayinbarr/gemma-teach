---
name: edit-file
target_tool: Edit
intent_tags: [update, change, fix, modify, rewrite, edit, adjust, revise]
token_cost: 130
---
### Edit
Replaces an EXACT block of text in an existing file. The `old_text` must match the file's current content character-for-character, including whitespace. Only the first match is replaced; include surrounding context to make the match unique.

REQUIRED: `path` (string), `old_text` (string), `new_text` (string)

EXAMPLE — update a single bullet:
{"name":"Edit","args":{"path":"student.md","old_text":"- Likes painting","new_text":"- Likes painting and digital art"}}

To make several changes, emit several Edit calls — one per location. Do not retry the same Edit twice if it errors; read the file first to confirm the current content.
