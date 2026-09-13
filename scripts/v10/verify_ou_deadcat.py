#!/usr/bin/env python3
"""验证 OU 高确认分但未接 A1 的实例是否随后再破低。"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, polars as pl
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from aurumq_rl.v10.label_engine import fit_ou_params,post_ou_confirmation_features,stop_falling_confirmation_score

def main():
 p=argparse.ArgumentParser();p.add_argument('--panel',default='data/wavehunter_v10_hs300_20040102_20260804.parquet');p.add_argument('--stocks',type=int,default=10);p.add_argument('--high-quantile',type=float,default=2/3);p.add_argument('--out',default='data/v10_ou_deadcat_analysis.json');a=p.parse_args();panel=Path(a.panel);panel=panel if panel.is_absolute() else ROOT/panel
 l=pl.scan_parquet(str(panel));codes=l.select('ts_code').unique().sort('ts_code').limit(a.stocks).collect()['ts_code'].to_list();df=l.filter(pl.col('ts_code').is_in(codes)).select(['trade_date','ts_code','adj_close','v10_zig_peak','v10_zig_valley','v10_a1_point']).sort(['ts_code','trade_date']).collect();instances=[]
 for code in codes:
  d=df.filter(pl.col('ts_code')==code);prices=d['adj_close'].to_numpy().astype(float);pk=d['v10_zig_peak'].to_numpy()==1;va=d['v10_zig_valley'].to_numpy()==1;a1=d['v10_a1_point'].to_numpy()==1;ps=np.flatnonzero(pk);vs=np.flatnonzero(va)
  for peak in ps:
   f=vs[vs>peak]
   if not len(f):continue
   valley=int(f[0]);span=valley-int(peak)
   if span<35:continue
   lo=int(peak)+max(31,int(np.ceil(span*.5)));hi=valley;c=[]
   for end in range(lo,hi+1):
    q=prices[end-30:end];par=fit_ou_params(q,min_samples=10)
    if par is None or par['theta']<.2:continue
    if np.sum(np.abs(q[-3:]-par['mu'])<=max(par['sigma'],np.finfo(float).eps))>=2:
     feat=post_ou_confirmation_features(prices,end,5,5);c.append((end,stop_falling_confirmation_score(feat)))
   if c:
    end,score=max(c,key=lambda x:x[1]);a1_after=bool(np.any(a1[end+3:min(len(a1),end+8)]));vlow=float(prices[valley]);future=prices[end+10:min(len(prices),end+21)];rebreak=bool(len(future)>0 and np.nanmin(future)<vlow);instances.append({'code':code,'peak':int(peak),'valley':valley,'ou_end':int(end),'score':score,'a1_after':a1_after,'valley_low':vlow,'future_10_20_min':float(np.nanmin(future)) if len(future) else None,'rebreak_10_20':rebreak})
 scores=np.array([x['score'] for x in instances]);cut=float(np.quantile(scores,a.high_quantile)) if len(scores) else float('nan');high=[x for x in instances if x['score']>=cut];high_no=[x for x in high if not x['a1_after']]
 def summary(xs):return {'count':len(xs),'rebreak_count':sum(x['rebreak_10_20'] for x in xs),'rebreak_pct':100*sum(x['rebreak_10_20'] for x in xs)/max(len(xs),1),'a1_count':sum(x['a1_after'] for x in xs),'a1_pct':100*sum(x['a1_after'] for x in xs)/max(len(xs),1)}
 # 用全部实例的每实例随机位置作为随机基线，固定选择 valley前半段/后半段候选结束逻辑不与高分组混用。
 out={'sample':{'stocks':len(codes),'rows':len(df),'dates':f'{df["trade_date"].min()}~{df["trade_date"].max()}'},'config':vars(a),'instances':len(instances),'high_cut_score':cut,'all':summary(instances),'high':summary(high),'high_without_a1':summary(high_no),'high_without_a1_rebreak_definition':'minimum price from ou_end+10 through ou_end+20 < same instance valley price','instance_rows':instances}
 outp=Path(a.out);outp=outp if outp.is_absolute() else ROOT/outp;outp.write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({'out':str(outp),'instances':len(instances),'high':len(high),'high_without_a1':len(high_no),'rebreak_pct':out['high_without_a1']['rebreak_pct']},ensure_ascii=False))
if __name__=='__main__':main()
