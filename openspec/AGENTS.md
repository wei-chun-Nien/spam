<!--
OpenSpec agent workflow and guidance for contributors and automated agents.
-->
# OpenSpec agents & workflow

This repository uses a lightweight OpenSpec process under `openspec/`.
The main artifacts are:

- `openspec/project.md` — project context and conventions.
- `openspec/AGENTS.md` — this file: workflow and how to work with the assistant.
- `openspec/proposals/*.md` — change proposals (drafts, accepted specs).

Roles
- Author — writes proposals
- Reviewer — reviews proposals and PRs
- Implementer — implements code and tests
- Agent — the assistant: drafts proposals, scaffolds code, runs local checks

Proposal lifecycle (recommended)
1. Draft: create `openspec/proposals/<NNN>-short-name.md` with summary, motivation, spec, tests, and plan.
2. Discuss: iterate via PR or issue comments.
3. Accept: mark `Status: Accepted` and attach an owner and implementation plan.
4. Implement: open PR(s) referencing the proposal and include tests & docs.
5. Release: merge, tag, and update changelog.

How to work with the assistant
- Ask clearly what step you want: `draft proposal`, `scaffold implementation`, `implement and run tests`, or `create PR`.
- If unspecified, the assistant will make reasonable stack assumptions (see `project.md`) and note them in generated files.
- Use the todo list the assistant creates to track work and progress.

Proposal template (recommended)
- Title
- Status (Draft / Accepted / Rejected)
- Summary
- Motivation
- Specification (API, data shapes)
- Backwards compatibility / migration notes
- Tests / acceptance criteria
- Implementation plan & timeline
- Alternatives and open questions

