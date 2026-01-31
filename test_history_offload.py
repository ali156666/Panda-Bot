import os
from pathlib import Path

from deepagent_demo import HistoryBackend, history_root

backend = HistoryBackend(root_dir=history_root, virtual_mode=True)
path = "/conversation_history/test-offload.md"
real_path = Path(history_root) / "conversation_history" / "test-offload.md"
if real_path.exists():
    real_path.unlink()

# 先写入一个带 \n 的内容（在 Windows 文本写入会落地为 \r\n）
write_res = backend.write(path, "Line1\nLine2\n")
print("write_error:", getattr(write_res, "error", None))

# 模拟 SummarizationMiddleware: download -> decode -> edit
resp = backend.download_files([path])[0]
existing = resp.content.decode("utf-8") if resp.content else ""
new_section = "## Summarized at 2026-01-30T00:00:00+00:00\n\nHuman: 你好\n\n"
combined = existing + new_section
edit_res = backend.edit(path, existing, combined)
print("edit_error:", getattr(edit_res, "error", None))
print("occurrences:", getattr(edit_res, "occurrences", None))

# 读取最终内容长度，确认写入成功
final_text = real_path.read_text(encoding="utf-8") if real_path.exists() else ""
print("final_len:", len(final_text))
