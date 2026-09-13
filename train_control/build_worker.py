"""Persistent worker for v1 factor and v3 WaveHunter panel builds."""
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


def append(log_path: Path, text: str) -> None:
    with log_path.open('a', encoding='utf-8', buffering=1) as stream:
        stream.write(text.rstrip() + '\n')


def main() -> int:
    runtime = Path(sys.argv[1])
    log_path = Path(sys.argv[2])
    root = Path(sys.argv[3])
    mode = sys.argv[4]
    config = json.loads(sys.argv[5])
    state = json.loads(runtime.read_text(encoding='utf-8'))
    state.update(status='running', running=True, pid=os.getpid())
    write_runtime(runtime, state)
    started_at = state.get('started_at', time.time())
    rc = -1
    artifact = None
    try:
        append(log_path, f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] worker started pid={os.getpid()} mode={mode}')
        if mode == 'v1':
            sys.path.insert(0, str(root / 'scripts'))
            import build_factor_panel as bfp
            def callback(message):
                append(log_path, message)
                state['progress'] = message[:160]
                write_runtime(runtime, state)
            bfp.add_log_callback(callback)
            stats = bfp.build_pool(stop_flag=None, **config)
            rc = 0
            state['result'] = {'status': 'ok', **(stats if isinstance(stats, dict) else {'stats': stats})}
            pattern = f"factor_panel_{config['pool']}_*.parquet"
            candidates = sorted((root / 'data').glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            artifact = next((p for p in candidates if p.stat().st_mtime >= started_at and p.exists()), None)
        elif mode == 'v3':
            if config['pool'] == 'cs800':
                import polars as pl
                stamp = f"{config['start_date']}_{config['end_date']}"
                hs_path = root / 'data' / f'wavehunter_hs300_{stamp}.parquet'
                cs_path = root / 'data' / f'wavehunter_csi500_{stamp}.parquet'
                artifact = root / 'data' / f'wavehunter_cs800_{stamp}.parquet'
                hs_codes = pl.scan_parquet(str(hs_path)).select('ts_code').unique().collect()['ts_code'].to_list()
                pl.concat([pl.scan_parquet(str(hs_path)),
                           pl.scan_parquet(str(cs_path)).filter(~pl.col('ts_code').is_in(hs_codes))],
                          how='vertical_relaxed').sink_parquet(str(artifact), compression='zstd')
                rc = 0
                state['result'] = {'status': 'ok', 'panel': artifact.name}
                return_code = 0
            else:
                return_code = None
            cmd = ['/usr/bin/python3', str(root / 'scripts' / 'build_wavehunter_panel.py'),
                   config['pool'], config['start_date'], config['end_date']]
            if return_code is None:
                with log_path.open('a', encoding='utf-8', buffering=1) as stream:
                    proc = subprocess.Popen(cmd, cwd=str(root), stdout=stream, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, text=True)
                    rc = proc.wait()
                artifact = root / 'data' / f"wavehunter_{config['pool']}_{config['start_date']}_{config['end_date']}.parquet"
                if not artifact.exists():
                    artifact = None
        elif mode == 'v8':
            cmd = ['/usr/bin/python3', str(root / 'scripts' / 'build_wavehunter_panel.py'),
                   config['pool'], config['start_date'], config['end_date'], 'v8',
                   '1' if config.get('include_fundamental', True) else '0']
            with log_path.open('a', encoding='utf-8', buffering=1) as stream:
                proc = subprocess.Popen(cmd, cwd=str(root), stdout=stream, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, text=True, start_new_session=True)
                rc = proc.wait()
            pattern = f"wavehunter_v8_{config['pool']}_*.parquet"
            candidates = sorted((root / 'data').glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            artifact = next((p for p in candidates if p.stat().st_mtime >= started_at and p.exists()), None)
            if artifact:
                state['result'] = {'status': 'ok', 'panel': artifact.name, 'version': 'v8'}

    except Exception as exc:
        append(log_path, f'worker error: {exc}')
    done = rc == 0 and artifact is not None and artifact.exists()
    state.update(status='done' if done else 'failed', running=False, returncode=rc,
                 artifact=artifact.name if artifact else None,
                 progress='完成' if done else f'失败 exit={rc}',
                 finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_runtime(runtime, state)
    return 0 if done else 1


if __name__ == '__main__':
    raise SystemExit(main())
