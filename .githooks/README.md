# Git hooks

Versioned hooks for this repository. Enable them once per clone:

```bash
git config core.hooksPath .githooks
```

## `commit-msg`

Rejects any `Co-authored-by:` trailer. Commits in this repository have exactly one
author — the human making the commit. AI assistants (Copilot, Codex, etc.) are never
credited as co-authors.
