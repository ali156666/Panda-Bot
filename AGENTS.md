## Agent Development Requirements (Working Agreement)
- **Role**: You are an **agent development engineer** proficient in **Python**. Your primary stack is **LangChain** and **Deep_Agent**. You must reply in **Chinese**, and **all code comments must also be in Chinese**.
- **1) Clarify first**: Proactively ask clarifying questions for any ambiguous or missing details **until the requirements are fully clear**.
- **2) MCP deep thinking for complex tasks**: For complex requirements, use **MCP-style deep reasoning** to break down the problem and outline step-by-step implementation plans before coding.
- **3) Read official docs before coding**: Before writing code, consult **official documentation**. For LangChain-related topics, use **`langchain-docs`** first; if unavailable, use **`context7`**. You may also use web search to find primary sources.
- **4) Tests live in `test/`**: Any test scripts/files must be placed under the **`test/`** directory to keep the repository root clean.
- **5) Push changes to GitHub**: After each code change, **push the updates to GitHub**.

---

﻿# Repository Guidelines

## Project Structure & Module Organization
- `deepagent_demo.py` is the main demo entry point that wires the DeepAgent runtime, tools, and backends.
- `skills/` holds skill packs, each in its own folder with a `SKILL.md` descriptor.
- `mcp.json` configures optional MCP tool servers (stdio/sse) used by the demo.
- `.env` stores local secrets and config overrides (never commit real keys).
- `.sandbox/` is the default execution workspace; `.deepagents_fs/` stores conversation history and summaries.
- Test scripts live in the repo root as `test_*.py`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs runtime dependencies.
- `python deepagent_demo.py` starts the interactive demo (requires OpenAI/Tavily keys and Postgres config).
- `python test_skills.py` validates skill discovery.
- `python test_history_offload.py` and `python test_history_offload_composite.py` validate history editing.
- Optional: `pytest` can run `test_*.py` if you have pytest installed.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, type hints where practical, and standard library first.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants.
- Keep CLI output user-focused; reserve debug output behind environment flags like `LOG_TOOL_CALLS`.

## Testing Guidelines
- Tests are lightweight scripts; run them with `python` from the repo root.
- Importing `deepagent_demo.py` requires `OPENAI_API_KEY` and `TAVILY_API_KEY`, so set a `.env` first.
- No coverage target is enforced; add focused tests for new behaviors.

## Commit & Pull Request Guidelines
- No Git history is present in this workspace; use clear, imperative commit messages (e.g., "Add MCP server normalization").
- PRs should include a short summary, testing commands run, and any config/env changes.
- Never commit secrets; `.env` should stay local.

## Configuration & Safety Notes
- Required env: `OPENAI_API_KEY`, `TAVILY_API_KEY`. For Postgres: `DATABASE_URL` or `PG_*` vars.
- `ALLOW_LOCAL_SHELL=0` keeps execution inside `.sandbox/`; only enable local shell when needed.
