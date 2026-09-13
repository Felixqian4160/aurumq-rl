#!/usr/bin/env python3
"""WaveHunter v10.1 isolated training entry.

Only v10.1 panels and labels are accepted. Output namespace is models/whv10_1_*.
"""
from __future__ import annotations
import os, sys, argparse, datetime, json, random
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts/v10_1'))
import numpy as np
import polars as pl
from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter, build_tradeable_mask
from aurumq_rl.lstm_weight_env import LstmWeightConfig
from aurumq_rl.v10_1.env import WaveHunterV10_1Env
from aurumq_rl.v10_1.policy import WaveHunterV10_1Policy
from aurumq_rl.v10_1.contract import build_training_contract
from train_loop_v10_1 import V10_1PPO, V10_1Monitor


def args():
 p=argparse.ArgumentParser(description='Train isolated WaveHunter v10.1')
 p.add_argument('--panel',required=True);p.add_argument('--start-date',default='2023-01-01');p.add_argument('--end-date',default='2023-12-31');p.add_argument('--out-dir',required=True)
 p.add_argument('--window',type=int,default=20);p.add_argument('--top-k',type=int,default=20);p.add_argument('--n-factors',type=int,default=0);p.add_argument('--forward-period',type=int,default=20);p.add_argument('--universe-filter',default='main_board_non_st')
 p.add_argument('--max-position-pct',type=float,default=.02);p.add_argument('--rebalance-days',type=int,default=20);p.add_argument('--cost-bps',type=float,default=15.);p.add_argument('--total-timesteps',type=int,default=10000)
 p.add_argument('--learning-rate',type=float,default=3e-5);p.add_argument('--lstm-hidden',type=int,default=64);p.add_argument('--lstm-layers',type=int,default=1);p.add_argument('--seed',type=int,default=42);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--n-steps',type=int,default=32);p.add_argument('--max-grad-norm',type=float,default=.5);p.add_argument('--target-kl',type=float,default=.02)
 p.add_argument('--aux-lambda',type=float,default=.1);p.add_argument('--a1-lambda',type=float,default=1.);p.add_argument('--a2-lambda',type=float,default=1.);p.add_argument('--peak-lambda',type=float,default=1.5);p.add_argument('--b1-lambda',type=float,default=.3)
 p.add_argument('--a1-pos-weight',type=float,default=20.);p.add_argument('--a2-pos-weight',type=float,default=40.);p.add_argument('--peak-pos-weight',type=float,default=80.);p.add_argument('--b1-pos-weight',type=float,default=10.)
 p.add_argument('--reward-type',default='absolute_return_v3');p.add_argument('--reward-scale',type=float,default=10.);p.add_argument('--reward-w-abs',type=float,default=.6);p.add_argument('--reward-w-dd',type=float,default=.25);p.add_argument('--reward-w-hit',type=float,default=.15);p.add_argument('--reward-norm-vol',type=float,default=.02)
 p.add_argument('--use-shared-sm',action=argparse.BooleanOptionalAction,default=False,help='Use frozen TradingStateMachine for shared execution contract (P0 Phase 3 opt-in)')
 p.add_argument('--sm-cost-bps',type=float,default=None,help='SM cost_bps (None=use --cost-bps)')
 p.add_argument('--sm-slippage-bps',type=float,default=10.,help='SM slippage_bps')
 p.add_argument('--sm-initial-capital',type=float,default=100000.,help='SM initial capital')
 p.add_argument('--sm-replace-reward',action=argparse.BooleanOptionalAction,default=False,help='Replace LstmWeightEnv reward with SM NAV delta (Phase 4a experimental; REQUIRES --use-shared-sm)')
 p.add_argument('--bayes-vol-enabled',action=argparse.BooleanOptionalAction,default=False);p.add_argument('--bayes-vol-alpha0',type=float,default=2.);p.add_argument('--bayes-vol-beta0',type=float,default=1e-4);p.add_argument('--bayes-vol-decay',type=float,default=.98);p.add_argument('--bayes-vol-uncertainty-weight',type=float,default=.5);p.add_argument('--bayes-vol-min',type=float,default=.005)
 p.add_argument('--cvar-enabled',action=argparse.BooleanOptionalAction,default=False);p.add_argument('--hhi-enabled',action=argparse.BooleanOptionalAction,default=False)
 return p.parse_args()


def pivot(df,col,dates,codes):
 sub=df.select(['trade_date','ts_code',col]).group_by(['trade_date','ts_code']).agg(pl.col(col).mean())
 piv=sub.pivot(values=col,index='trade_date',on='ts_code').sort('trade_date');present=[c for c in piv.columns if c!='trade_date'];arr=piv.drop('trade_date').to_numpy()
 out=np.full((len(dates),len(codes)),np.nan,np.float32);di={str(d)[:10]:i for i,d in enumerate(dates)}
 for r in piv.iter_rows(named=True):
  t=di.get(str(r['trade_date'])[:10]);
  if t is None:continue
  for j,c in enumerate(codes):
   if c in present and r.get(c) is not None:out[t,j]=float(r[c])
 return out


def main():
 a=args(); panel_path=Path(a.panel); panel_path=panel_path if panel_path.is_absolute() else ROOT/'data'/panel_path
 if not panel_path.name.startswith('wavehunter_v10_1_'): raise ValueError('v10.1 训练只接受 wavehunter_v10_1_ 面板')
 sd=datetime.date.fromisoformat(a.start_date);ed=datetime.date.fromisoformat(a.end_date)
 if ed>datetime.date.today() or sd>=ed:raise ValueError('非法训练日期')
 if a.lstm_hidden>64 or a.n_steps>32 or a.batch_size>32:raise ValueError('v10.1 smoke安全边界: hidden<=64,n_steps<=32,batch<=32')
 random.seed(a.seed);np.random.seed(a.seed);import torch;torch.manual_seed(a.seed)
 out=Path(a.out_dir);out=out if out.is_absolute() else ROOT/'models'/out.name;out.mkdir(parents=True,exist_ok=True)
 uf={'main_board_non_st':UniverseFilter.MAIN_BOARD_NON_ST,'all_a':UniverseFilter.ALL_A}.get(a.universe_filter,UniverseFilter.MAIN_BOARD_NON_ST)
 loader=FactorPanelLoader(parquet_path=panel_path); panel=loader.load_panel(sd,ed,n_factors=a.n_factors if a.n_factors>0 else None,forward_period=a.forward_period,universe_filter=uf)
 dates=panel.dates;codes=panel.stock_codes
 required=['v10_1_a1_point','v10_1_a2_interval','v10_1_zig_peak','v10_1_peak_zone','v10_1_b1_interval']
 label_columns=['trade_date','ts_code'] + required
 available_columns=pl.scan_parquet(str(panel_path)).collect_schema().names()
 missing=[c for c in label_columns if c not in available_columns]
 if missing:raise ValueError(f'v10.1 标签缺失: {missing}')
 # 只读取训练区间的标签列，避免把 2M 行 × 345 列完整物化进训练进程。
 df=(pl.scan_parquet(str(panel_path))
       .filter((pl.col('trade_date') >= sd) & (pl.col('trade_date') <= ed))
       .select(label_columns)
       .collect())
 labs={c:np.where(np.isfinite(pivot(df,c,dates,codes)),pivot(df,c,dates,codes),-1).astype(np.int32) for c in required}
 factor_bad=[c for c in panel.factor_names if c.startswith(('v10_1_','v10_','v9_'))]
 if factor_bad:raise RuntimeError(f'标签泄漏进入 observation: {factor_bad[:10]}')
 from stable_baselines3.common.vec_env import DummyVecEnv
 mask=build_tradeable_mask(panel)
 # Phase 4a: when both --use-shared-sm and --sm-replace-reward are set, swap
 # the env's ``step`` method for ``step_with_sm_reward`` so SB3 PPO trains on
 # the SM path's NAV delta instead of the legacy cost model.  This must
 # happen *before* the env instance is created so the bound method is on the
 # class; we restore ``step`` after DummyVecEnv construction to keep the rest
 # of the codebase clean.
 _sm_reward_active = bool(a.sm_replace_reward and a.use_shared_sm)
 if _sm_reward_active:
     WaveHunterV10_1Env.step = WaveHunterV10_1Env.step_with_sm_reward  # type: ignore[assignment]
 def make():
  cfg=LstmWeightConfig(start_date=sd,end_date=ed,n_factors=panel.factor_array.shape[2],window=a.window,reward_type=a.reward_type,cost_bps=a.cost_bps,max_position_pct=a.max_position_pct,top_k=a.top_k,rebalance_days=a.rebalance_days,reward_scale=a.reward_scale)
  cfg.reward_w_abs=a.reward_w_abs;cfg.reward_w_dd=a.reward_w_dd;cfg.reward_w_hit=a.reward_w_hit;cfg.reward_norm_vol=a.reward_norm_vol
  cfg.__dict__.update({'bayes_vol_enabled':a.bayes_vol_enabled,'bayes_vol_alpha0':a.bayes_vol_alpha0,'bayes_vol_beta0':a.bayes_vol_beta0,'bayes_vol_decay':a.bayes_vol_decay,'bayes_vol_uncertainty_weight':a.bayes_vol_uncertainty_weight,'bayes_vol_min':a.bayes_vol_min,'cvar_enabled':a.cvar_enabled,'hhi_enabled':a.hhi_enabled})
  return WaveHunterV10_1Env(config=cfg,factor_panel=panel.factor_array,return_panel=panel.return_array,pct_change_panel=panel.pct_change_array,is_st_panel=panel.is_st_array,is_suspended_panel=panel.is_suspended_array,days_since_ipo_panel=panel.days_since_ipo_array,tradeable_mask=mask,v10_1_a1_point=labs['v10_1_a1_point'],v10_1_a2_interval=labs['v10_1_a2_interval'],v10_1_peak=labs['v10_1_peak_zone'],v10_1_b1=labs['v10_1_b1_interval'],use_shared_sm=a.use_shared_sm,sm_cost_bps=a.sm_cost_bps,sm_slippage_bps=a.sm_slippage_bps,sm_initial_capital=a.sm_initial_capital,open_array=getattr(panel,'open_array',None),close_array=getattr(panel,'close_array',None))
 env=DummyVecEnv([make])
 # NOTE: ``step_with_sm_reward`` swap is intentionally not restored; SB3
 # PPO holds a reference to the bound method at construction time, so the
 # patched version sticks for the lifetime of this training run.  Other
 # Python processes start fresh and are unaffected.
 kwargs={'lstm_hidden':a.lstm_hidden,'lstm_layers':a.lstm_layers,'_n_stocks':panel.factor_array.shape[1],'_window':a.window,'_n_factors':panel.factor_array.shape[2],'aux_lambda':a.aux_lambda,'a1_lambda':a.a1_lambda,'a2_lambda':a.a2_lambda,'peak_lambda':a.peak_lambda,'b1_lambda':a.b1_lambda,'a1_pos_weight':a.a1_pos_weight,'a2_pos_weight':a.a2_pos_weight,'peak_pos_weight':a.peak_pos_weight,'b1_pos_weight':a.b1_pos_weight}
 model=V10_1PPO(policy=WaveHunterV10_1Policy,env=env,learning_rate=a.learning_rate,aux_lambda=a.aux_lambda,n_steps=a.n_steps,batch_size=a.batch_size,n_epochs=2,gamma=.99,gae_lambda=.95,clip_range=.2,max_grad_norm=a.max_grad_norm,target_kl=a.target_kl,seed=a.seed,verbose=1,policy_kwargs=kwargs)
 model.learn(total_timesteps=a.total_timesteps,progress_bar=False,callback=V10_1Monitor(model,out/'reward_metrics.jsonl'))
 model.save(str(out/'ppo_final.zip'))
 import torch as t
 model.policy.eval();model.policy.cpu();nf=panel.factor_array.shape[2];dummy=t.zeros(1,panel.factor_array.shape[1]*a.window*nf)
 class Wrap(t.nn.Module):
  def __init__(self,s):super().__init__();self.s=s
  def forward(self,x):return self.s.forward_with_heads(x)
 t.onnx.export(Wrap(model.policy),dummy,str(out/'policy.onnx'),opset_version=14,input_names=['observation'],output_names=['action','a1','a2','peak','b1'],dynamic_axes={'observation':{0:'batch'},'action':{0:'batch'},'a1':{0:'batch'},'a2':{0:'batch'},'peak':{0:'batch'},'b1':{0:'batch'}},dynamo=False)
 meta={'algorithm':'PPO_V10_1','model':'wavehunter_v10_1_4heads','panel':str(panel_path),'obs_shape':[panel.factor_array.shape[1]*a.window*nf],'action_shape':[panel.factor_array.shape[1]],'n_stocks':panel.factor_array.shape[1],'n_factors':nf,'training_timesteps':a.total_timesteps,'label_columns':required,'reward_type':a.reward_type,'version':'v10.1','train_start_date':a.start_date,'train_end_date':a.end_date,'stock_codes':list(panel.stock_codes),'factor_names':list(panel.factor_names),'hyperparams':vars(a),'p0_bridge':{'use_shared_sm':a.use_shared_sm,'sm_cost_bps':a.sm_cost_bps,'sm_slippage_bps':a.sm_slippage_bps,'sm_initial_capital':a.sm_initial_capital,'trading_state_machine_version':'v10.1-tradingstate-bridge-v1'}}
 # ── P0-3 contract (2026-09-13): 写训练合同到 metadata，模拟端强制 diff ──
 meta['training_contract'] = build_training_contract(meta)
 (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));(out/'training_summary.json').write_text(json.dumps({'model':'wavehunter_v10_1_4heads','panel':str(panel_path),'total_timesteps':a.total_timesteps,'n_stocks':panel.factor_array.shape[1],'n_factors':nf,'stock_codes':list(panel.stock_codes),'factor_names':list(panel.factor_names),'aux_stats':model.aux_stats,'training_contract':meta['training_contract']},ensure_ascii=False,indent=2));print(json.dumps(meta,ensure_ascii=False))
if __name__=='__main__':main()
