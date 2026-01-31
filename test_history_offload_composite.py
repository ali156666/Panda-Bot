from pathlib import Path
import uuid

from deepagent_demo import HistoryBackend, history_root
from deepagents.backends import CompositeBackend

history_backend = HistoryBackend(root_dir=history_root, virtual_mode=True)
backend = CompositeBackend(default=history_backend, routes={"/conversation_history/": history_backend})

# 用唯一文件名避免占用/权限问题
file_id = uuid.uuid4().hex[:8]
path = f"/conversation_history/test-composite-{file_id}.md"
real_path = Path(history_root) / f"test-composite-{file_id}.md"

write_res = backend.write(path, "Line1\nLine2\n")
print("write_error:", getattr(write_res, "error", None))

resp = backend.download_files([path])[0]
existing = resp.content.decode("utf-8") if resp.content else ""
new_section = "## Summarized at 2026-01-30T00:00:00+00:00\n\nHuman: 你好\n\n"
combined = existing + new_section
edit_res = backend.edit(path, existing, combined)
print("edit_error:", getattr(edit_res, "error", None))
print("occurrences:", getattr(edit_res, "occurrences", None))
print("final_len:", len(real_path.read_text(encoding="utf-8")) if real_path.exists() else 0)
