<!--
Project context for the repository. This file captures the project's
purpose, assumed tech stack, conventions, and contact points for contributors.
If any assumption is incorrect, update this file or tell the assistant to
regenerate it with your preferred stack.
-->
# Project: mail

Short description
- Lightweight mail processing and experimentation workspace. Contains tools
  and prototypes for message processing, classification, and admin UIs.

Assumed tech stack (changeable)
- Language: Python 3.11
- Web / UI: FastAPI (API) and Streamlit (demos)
- ML: scikit-learn, pandas
- Storage: PostgreSQL (for production), local files for demos
- Packaging: Poetry or pip + requirements.txt

Repository layout (convention)
- `src/` - application source code
- `apps/` - demo apps (e.g., Streamlit)
- `openspec/` - proposals and process documentation
- `tests/` - unit and integration tests
- `docker/` - optional container/dev files

Git & PR conventions
- Default branch: `main`
- Feature branches: `feat/<short-desc>`
- Commit messages: short prefix (e.g., `feat:`, `fix:`) and concise body
- PRs should reference OpenSpec proposals when implementing spec changes

CI & quality
- CI should run lint (ruff/flake8), typecheck (mypy), and tests (pytest) on PRs

Contract (quick)
- Inputs: incoming messages (API, SMTP, or dataset files)
- Outputs: processed messages, logs, metrics, model artifacts
- Error modes: transient errors retried, malformed inputs rejected

Next steps
- Confirm or edit the stack choices above.
- If confirmed, the assistant can scaffold a starter project (pyproject/requirements, basic app, CI) and implement selected proposals.

