[简体中文](README.md) | [English](README_EN.md)

<div align="center">
  <img src="assets/images/panda-mascot.jpg" alt="Panda Bot Mascot" width="320">
  <h1>🐼 Panda Bot: Lightweight Personal AI Assistant 🤖</h1>
  <p>
    <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python">
    <img src="https://img.shields.io/badge/runtime-DeepAgents%20%2B%20LangGraph-00A36C" alt="Runtime">
    <img src="https://img.shields.io/badge/channel-Feishu-1E80FF" alt="Feishu">
    <img src="https://img.shields.io/badge/tools-MCP%20%2B%20Browser-7A52F4" alt="Tools">
  </p>
</div>

`Panda Bot` is a local-first Python agent project focused on practical personal assistant workflows with minimal complexity.

- `deepagent_demo.py`: interactive CLI agent
- `feishu_deepagent_bot.py`: Feishu long-connection bot (with Cron + Heartbeat)

It integrates browser automation, MCP tools, extendable skills (`skills/`), and persistent session/memory management.

## 📰 News

- **2026-02-11**: `README_EN.md` was synchronized with the latest Chinese README layout and image showcase.

## ✨ Key Features

- Dual runtime entrypoints: CLI + Feishu bot.
- Built-in scheduling: add/list/remove/run Cron jobs.
- Heartbeat wake-up with interval and active-hour windows.
- Browser automation based on `DrissionPage`.
- MCP tool integration via `mcp.json`.
- Auto-discovered local skills from `skills/**/SKILL.md`.
- Long-term memory with SQLite + embeddings + summaries.

## 🎬 Showcase

<table align="center">
  <tr align="center">
    <th>Generate and send weather image</th>
    <th>AI weather voice broadcast</th>
  </tr>
  <tr>
    <td align="center"><img src="assets/images/showcase-weather-image.jpg" alt="Weather image generation and sending" width="280"></td>
    <td align="center"><img src="assets/images/showcase-weather-voice.jpg" alt="AI weather voice report" width="280"></td>
  </tr>
  <tr align="center">
    <th>Export daily report to Word</th>
    <th>Retrieve and deliver local file</th>
  </tr>
  <tr>
    <td align="center"><img src="assets/images/showcase-report-export.jpg" alt="Daily report export to Word" width="280"></td>
    <td align="center"><img src="assets/images/showcase-file-delivery.jpg" alt="Local file lookup and delivery" width="280"></td>
  </tr>
</table>

## 🏗️ Architecture

```text
User / Feishu
   |
   +--> feishu_deepagent_bot.py --------+
   |                                     |
   +--> deepagent_demo.py                |
                                         v
                                deepagent runtime
                           (model + tools + memory)
                             |        |        |
                             |        |        +--> memory/ (SQLite)
                             |        +----------> browser/ + tool.py
                             +-------------------> mcp.json (MCP servers)
```

## 📁 Project Structure

```text
.
├── deepagent_demo.py            # interactive entrypoint
├── feishu_deepagent_bot.py      # Feishu bot entrypoint
├── deepagent/
│   └── session/                 # session lifecycle and persistence
├── browser/                     # browser tool integration
├── scheduler/                   # Cron / Heartbeat / Store
├── prompts/                     # prompt fragments
├── skills/                      # local skill packs (SKILL.md)
├── tool.py                      # image and TTS tools
├── mcp.json                     # MCP server config
├── .env.example                 # environment template
└── requirements.txt
```

## 📦 Install

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1) Configure environment variables

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

Minimum for `deepagent_demo.py`:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

Additional for `feishu_deepagent_bot.py`:

- `APP_ID`
- `APP_SECRET`

### 2) Run

Interactive mode:

```bash
python deepagent_demo.py
```

Feishu bot mode:

```bash
python feishu_deepagent_bot.py
```

## 🔧 Key Environment Variables

Use `.env.example` as the source of truth.

- Runtime and model
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (default: `gpt-5.3-codex`)
  - `OPENAI_BASE_URL`
- Search and media
  - `TAVILY_API_KEY`
  - `GEMINI_API_KEY`
  - `TTS_API_KEY` / `TTS_BASE_URL`
- Memory
  - `MEMORY_ROOT`
  - `MEMORY_DB_PATH`
  - `MEMORY_SUMMARY_MODEL`
  - `MEMORY_EMBEDDING_MODEL`
- Feishu
  - `APP_ID`
  - `APP_SECRET`
- Misc
  - `LOCAL_SHELL_ROOT`
  - `LOG_TAIL_LINES`

## 💬 Feishu Bot Setup

1. Create a Feishu app and enable bot capability.
2. Subscribe to event: `im.message.receive_v1`.
3. Add required message permissions (send/receive).
4. Put `APP_ID` and `APP_SECRET` in `.env`, then run `python feishu_deepagent_bot.py`.

## 🧩 MCP & Skills

- MCP servers are loaded from `mcp.json` by default.
- Local skills are discovered from `skills/<skill-name>/SKILL.md`.
- Inject third-party keys by environment variables instead of hard-coding.

## ✅ Test

Current committed example test:

```bash
python -m unittest skills/pdf/scripts/check_bounding_boxes_test.py
```

## 🔐 Security Notes

- Do not commit `.env`, logs, SQLite files, or runtime caches.
- Redact secrets before sharing logs.
- Rotate credentials immediately if exposed.

## 🛠️ Troubleshooting

- `Missing required environment variables`: check `.env` keys and spelling.
- MCP tools not loaded: verify commands in `mcp.json` are executable.
- Feishu bot not replying: verify app permissions, subscriptions, and `APP_ID`/`APP_SECRET`.
