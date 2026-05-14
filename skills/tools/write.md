---
name: write-file
target_tool: Write
intent_tags: [create, write, new file, save, produce, generate, draft]
token_cost: 110
---
### Write
Creates a NEW file inside the working directory. Refuses if the file already exists — use Edit for changes.

REQUIRED: `path` (string), `content` (string)

EXAMPLE — create a student profile:
{"name":"Write","args":{"path":"student.md","content":"# Maya\n\n## Snapshot\n..."}}

If you see an "already exists" error, switch to Edit (see Edit skill).
