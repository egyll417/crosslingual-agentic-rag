import hashlib
import json
from collections import Counter
from pathlib import Path

from crag.data.schema import Question


def write_snapshot(questions: list[Question], out_dir: str | Path, name: str, meta: dict) -> Path:
    """Write questions as JSONL plus a manifest recording where they came from."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(q.to_dict(), ensure_ascii=False) + "\n" for q in questions)
    path = out_dir / f"{name}.jsonl"
    path.write_text(payload, encoding="utf-8")
    manifest = {
        **meta,
        "file": path.name,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "records": len(questions),
        "records_per_lang": dict(sorted(Counter(q.lang for q in questions).items())),
    }
    manifest_path = out_dir / f"{name}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def read_snapshot(path: str | Path) -> list[Question]:
    with Path(path).open(encoding="utf-8") as f:
        return [Question.from_dict(json.loads(line)) for line in f if line.strip()]
