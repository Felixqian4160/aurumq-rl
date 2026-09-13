"""v10.1 WebUI simulation API."""
from __future__ import annotations
import json, os, subprocess, threading, time, uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from .common import AURUMQ_ROOT, PYTHON_EXE
from core.process import pid_alive

router=APIRouter(tags=['v10.1-simulation'])
_RUNTIME=AURUMQ_ROOT/'data'/'task_runtime';_RUNTIME.mkdir(parents=True,exist_ok=True)
_PROC:Optional[subprocess.Popen]=None

class V101SimulationRequest(BaseModel):
 model_dir:str
 panel:str
 initial_capital:float=100000.
 start_date:str='2025-01-01'
 end_date:str='2025-12-31'
 top_k:int=20
 cost_bps:float=15.
 slippage_bps:float=10.
 rebalance_days:int=20
 max_position_pct:float=.05
 a1_threshold:float=.55
 a2_threshold:float=.55
 peak_threshold:float=.50
 b1_threshold:float=.55

def _write(p,d):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));os.replace(tmp,p)
def _read(t):
 p=_RUNTIME/f'{t}.json'
 if not p.exists():return {'version':'v10.1','task_id':t,'status':'not_found','running':False}
 try:d=json.loads(p.read_text())
 except: return {'version':'v10.1','task_id':t,'status':'failed','running':False}
 if d.get('status')=='running' and not pid_alive(d.get('pid')):d.update(status='failed',running=False,progress='模拟进程已退出');_write(p,d)
 d['running']=d.get('status')=='running';return d

def _latest():
 ps=sorted(_RUNTIME.glob('v10_1_sim_*.json'),key=lambda p:p.stat().st_mtime,reverse=True);return _read(ps[0].stem) if ps else {'version':'v10.1','status':'idle','running':False}
def _watch(t,proc):
 rc=proc.wait();d=_read(t);d.update(status='done' if rc==0 else 'failed',running=False,returncode=rc,finished_at=time.time(),progress='ledger已验证' if rc==0 else f'模拟进程退出 {rc}');_write(_RUNTIME/f'{t}.json',d)

@router.post('/api/v10_1/simulation/start')
def start_v101_sim(req:V101SimulationRequest):
 global _PROC
 if _PROC and _PROC.poll() is None:raise HTTPException(409,'v10.1 模拟已在运行')
 if not req.model_dir.startswith('whv10_1_'):raise HTTPException(400,'模型必须为 whv10_1_')
 model=AURUMQ_ROOT/'models'/req.model_dir;panel=AURUMQ_ROOT/'data'/req.panel
 if not (model/'policy.onnx').is_file() or not (model/'metadata.json').is_file():raise HTTPException(400,'v10.1 模型缺少 ONNX/metadata')
 if not panel.is_file() or not panel.name.startswith('wavehunter_v10_1_'):raise HTTPException(400,'必须选择 v10.1 面板')
 if not 0<=req.cost_bps<=100 or not 0<=req.slippage_bps<=100:raise HTTPException(400,'cost/slippage 必须在 0~100 bps')
 t=f'v10_1_sim_{req.model_dir}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}';cfg=_RUNTIME/f'{t}.config.json';state=_RUNTIME/f'{t}.json';log=_RUNTIME/f'{t}.log'
 payload=req.model_dump();payload.update(version='v10.1',model_dir=str(model),panel_path=str(panel));cfg.write_text(json.dumps(payload,ensure_ascii=False,indent=2));_write(state,{'version':'v10.1','task_id':t,'status':'starting','running':False,'model':req.model_dir,'panel':req.panel,'out_dir':str(model),'log_path':str(log),'started_at':time.time()})
 runner=AURUMQ_ROOT/'scripts/v10_1'/'run_sim_v10_1_v3.py';
 with log.open('w') as out:_PROC=subprocess.Popen([PYTHON_EXE,str(runner),str(cfg),str(state)],cwd=AURUMQ_ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1'})
 d=json.loads(state.read_text());d.update(pid=_PROC.pid,status='running',running=True,command=[PYTHON_EXE,str(runner),str(cfg),str(state)]);_write(state,d);threading.Thread(target=_watch,args=(t,_PROC),daemon=True).start();return {'version':'v10.1','status':'started','task_id':t,'log_path':str(log)}

@router.get('/api/v10_1/simulation/status')
def v101_sim_status(task_id:str=''):return _read(task_id) if task_id else _latest()
@router.get('/api/v10_1/simulation/log')
def v101_sim_log(task_id:str=''):
 d=v101_sim_status(task_id);p=Path(d.get('log_path',''));return {'version':'v10.1','task_id':d.get('task_id'),'status':d,'lines':p.read_text(errors='replace').splitlines()[-500:] if p.is_file() else []}
@router.post('/api/v10_1/simulation/stop')
def v101_sim_stop(task_id:str=''):
 d=v101_sim_status(task_id)
 if not d.get('task_id'):return {'version':'v10.1','status':'idle'}
 p=_RUNTIME/f"{d['task_id']}.json";d.update(status='stopping',stop_requested_at=time.time());_write(p,d);pid=d.get('pid')
 if isinstance(pid,int) and pid_alive(pid):
  try:os.killpg(pid,15)
  except:pass
 return {'version':'v10.1','task_id':d['task_id'],'status':'stopping'}
