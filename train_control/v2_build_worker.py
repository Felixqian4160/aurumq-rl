"""Persistent worker for a WaveHunter v2 panel build."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write_runtime(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def main() -> int:
    runtime = Path(sys.argv[1])
    log_path = Path(sys.argv[2])
    root = Path(sys.argv[3])
    pool, start_date, end_date, include_fundamental, task_id = sys.argv[4:9]
    runtime_data = json.loads(runtime.read_text(encoding='utf-8'))
    cmd = ['/usr/bin/python3', str(root / 'scripts' / 'build_wavehunter_panel.py'),
           pool, start_date, end_date, 'v2', include_fundamental]
    runtime_data['status'] = 'running'
    runtime_data['pid'] = os.getpid()
    write_runtime(runtime, runtime_data)
    rc = -1
    try:
        with log_path.open('a', encoding='utf-8', buffering=1) as stream:
            stream.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] worker started pid={os.getpid()}\n')
            stream.flush()
            proc = subprocess.Popen(cmd, cwd=str(root), stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=False, text=True)
            rc = proc.wait()
    except Exception as exc:
        with log_path.open('a', encoding='utf-8', buffering=1) as stream:
            stream.write(f'worker error: {exc}\n')
    started_at = runtime_data.get('started_at', time.time())
    candidates = sorted((root / 'data').glob(f'wavehunter_v2_{pool}_*.parquet'),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    artifact = next((p for p in candidates if p.stat().st_mtime >= started_at and p.exists()), None)
    done = rc == 0 and artifact is not None and artifact.exists()
    runtime_data.update({
        'status': 'done' if done else 'failed',
        'running': False,
        'returncode': rc,
        'artifact': artifact.name if artifact else None,
        'result': ({'panel': artifact.name, 'size_gb': round(artifact.stat().st_size / 1024**3, 3),
                    'version': 'v2', 'features': ['A1', 'A2', '10d', '20d', '30d']}
                   if done else None),
        'progress': '完成' if done else f'失败 exit={rc}' if rc != 0 else '失败：未找到 wavehunter_v2 面板',
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    write_runtime(runtime, runtime_data)
    return 0 if done else 1


if __name__ == '__main__':
    raise SystemExit(main())
