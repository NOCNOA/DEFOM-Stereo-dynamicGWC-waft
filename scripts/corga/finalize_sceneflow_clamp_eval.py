"""Wait for the three fast3 baseline shards, then validate and summarize."""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "diagnostics/large_delta_eval"
STATUS = OUT / "sceneflow_clamp_eval_status.json"
EXPECTED = {0: 1457, 2: 1457, 3: 1456}


def write_status(state, **extra):
    data = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(STATUS)


def finished(gpu, expected):
    stem = f"sceneflow_full_fast3_70k_cap8_gpu{gpu}"
    path = OUT / f"{stem}.jsonl"
    log = OUT / f"{stem}.log"
    if not path.exists() or not log.exists():
        return False
    count = path.read_bytes().count(b"\n")
    if count > expected:
        raise ValueError(f"Too many result rows in {path}: {count} > {expected}")
    return count == expected and "Mixed pixel weighted:" in log.read_text(errors="replace")


def main():
    write_status("running", expected_images=sum(EXPECTED.values()))
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        if all(finished(gpu, count) for gpu, count in EXPECTED.items()):
            command = [sys.executable, str(ROOT / "scripts/corga/summarize_sceneflow_unclamped_models.py")]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            print(result.stdout, flush=True)
            summary = OUT / "sceneflow_unclamped_models_summary.json"
            write_status("complete", summary=str(summary), expected_images=sum(EXPECTED.values()))
            return
        time.sleep(30)
    raise TimeoutError("SceneFlow fast3 baseline shards did not finish within two hours")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status("error", error=str(exc))
        raise
