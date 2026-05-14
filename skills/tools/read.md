---
name: read-file
target_tool: Read
intent_tags: [read, open, view, show, load, contents]
token_cost: 90
---
### Read
Reads a UTF-8 text file inside the session's working directory. Paths are relative to the working directory; absolute paths or paths that escape the working directory are refused.

REQUIRED: `path` (string)
OPTIONAL: `max_bytes` (integer, default 200000)

EXAMPLE — read a markdown file:
{"name":"Read","args":{"path":"student.md"}}
