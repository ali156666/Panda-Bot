# Repository Guidelines

## Project Structure & Module Organization
- Core entry scripts are `deepagent_demo.py` (interactive runtime) and `feishu_deepagent_bot.py` (Feishu bot + scheduler loop).
- Main packages:
  - `deepagent/session/`: session lifecycle and persistence helpers.
  - `browser/`: browser tool integration used by agent workflows.
  - `scheduler/`: cron, heartbeat, run logs, and SQLite-backed scheduling state.
  - `prompts/`: prompt fragments loaded into runtime system prompts.
  - `skills/`: local skill packs (`SKILL.md`, optional `scripts/`, optional `data/`).
- Runtime/state artifacts include `memory/`, `scheduler.sqlite`, `.deepagents_fs/`, and `feishu_bot.log`; treat these as generated data, not source.

## Build, Test, and Development Commands
- `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`: create and activate local env (PowerShell).
- `pip install -r requirements.txt`: install runtime dependencies.
- `python deepagent_demo.py`: run the local interactive agent.
  - Requires `OPENAI_API_KEY` and `TAVILY_API_KEY`.
- `python feishu_deepagent_bot.py`: run Feishu long-connection bot.
  - Also requires `APP_ID` and `APP_SECRET`.
- `python -m unittest skills/pdf/scripts/check_bounding_boxes_test.py`: run the currently committed unit test.

## Coding Style & Naming Conventions
- Target Python 3.12+ with 4-space indentation and UTF-8 files.
- Use type hints for new or changed functions.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants/env keys `UPPER_SNAKE_CASE`.
- Keep startup side effects in script guards (`if __name__ == "__main__":`).

## Testing Guidelines
- No centralized test suite is configured yet; add focused tests with each behavioral change.
- Prefer `test_*.py` names; existing script tests may use `*_test.py` (for example in `skills/pdf/scripts/`).
- For scheduler/session changes, include deterministic persistence checks (write/read/update paths).

## Commit & Pull Request Guidelines
- Current history uses short imperative subjects (for example, `Add ...`).
- Recommended format: `<Verb> <scope>: <summary>` (example: `Fix scheduler: handle empty cron spec`).
- PRs should include purpose, env/config changes, verification steps, and logs/screenshots when bot behavior changes.
- Link related issues/tasks and call out any migration impact on persisted files (`scheduler.sqlite`, `memory/`).

## Security & Configuration Tips
- Keep secrets in `.env`; do not commit API keys, tokens, or chat identifiers.
- Redact sensitive values before sharing `feishu_bot.log` or copied runtime output.
