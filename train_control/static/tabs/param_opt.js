/**
 * tabs/param_opt.js — 📊 参数优化统一 Tab
 * 3 个子面板：参数扫描 / Walk-forward / 结果总览
 * 后端 API 不动，前端聚合。
 */
(function(){
  'use strict';

  // ══════════════════════════════════════════════════
  // 子 Tab 切换
  // ══════════════════════════════════════════════════
  function switchOptTab(name){
    document.querySelectorAll('.po-subtab').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.po-panel').forEach(p=>{p.style.display='none';p.classList.remove('active');});
    const btn=document.getElementById('po-tab-'+name);
    const pane=document.getElementById('po-panel-'+name);
    if(btn) btn.classList.add('active');
    if(pane){pane.style.display='';pane.classList.add('active');}
    // 懒加载
    if(name==='scan') poLoadScanStatus();
    if(name==='wf') poLoadWfPanels();
    if(name==='overview') poLoadOverview();
  }
  window.switchOptTab=switchOptTab;

  // ══════════════════════════════════════════════════
  // 子面板 1：参数扫描 (复用 /api/wavehunter/v2/matrix/*)
  // ══════════════════════════════════════════════════
  let _poScanPoll=null;

  function poLoadScanStatus(){
    Promise.all([
      fetch('/api/wavehunter/v2/matrix/status').then(r=>r.ok?r.json():{status:'idle',done:0,total:0,running:false}),
      fetch('/api/wavehunter/v2/matrix/runs').then(r=>r.ok?r.json():{runs:[]}),
    ]).then(([s,r])=>{
      const badge=document.getElementById('poScanBadge');
      const prog=document.getElementById('poScanProgress');
      const tbl=document.getElementById('poScanTable');
      if(badge){
        if(s.running){badge.className='status-badge status-running';badge.textContent='运行中';}
        else if(s.done>0){badge.className='status-badge status-done';badge.textContent=`${s.done}/${s.total||'?'} 完成`;}
        else{badge.className='status-badge status-idle';badge.textContent='空闲';}
      }
      if(prog) prog.textContent=s.running?`进度: ${s.done||0}/${s.total||'?'} | 当前: ${s.current_run||''}`:(s.done?`已完成 ${s.done}/${s.total||'?'}`:'');
      poRenderScanRuns(r);
      if(s.running){clearTimeout(_poScanPoll);_poScanPoll=setTimeout(poLoadScanStatus,5000);}
    }).catch(e=>{if(typeof addLog==='function')addLog('❌ 扫描状态: '+(e.message||e));});
  }

  function poRenderScanRuns(d){
    const runs=(d&&d.runs)||[];
    if(!runs.length){document.getElementById('poScanTable').innerHTML='<div style="color:#667788;padding:8px">暂无记录</div>';return;}
    let h='<table style="width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed">';
    h+='<thead><tr style="border-bottom:1px solid #2a3a4a">';
    ['#','跨度','步数','超额','Sharpe','WR','PF','状态'].forEach(c=>{h+=`<th style="padding:6px;text-align:left">${c}</th>`;});
    h+='</tr></thead><tbody>';
    runs.forEach((r,i)=>{
      const cls=r.status==='done'?'color:#81c784':r.status==='running'?'color:#4fc3f7':r.status==='failed'?'color:#ef5350':'color:#667788';
      h+=`<tr style="border-bottom:1px solid #1a2a3a">`;
      h+=`<td style="padding:6px">${i+1}</td>`;
      h+=`<td style="padding:6px">${r.span||'?'}</td>`;
      h+=`<td style="padding:6px">${r.steps||'?'}</td>`;
      const m=(r.metrics)||{};
      const excess=m.excess_return;
      h+=`<td style="padding:6px">${excess!=null?(excess>0?'+':'')+excess.toFixed(2)+'%':'—'}</td>`;
      h+=`<td style="padding:6px">${m.sharpe!=null?(m.sharpe).toFixed(3):'—'}</td>`;
      h+=`<td style="padding:6px">${m.win_rate!=null?m.win_rate.toFixed(1)+'%':'—'}</td>`;
      h+=`<td style="padding:6px">${m.profit_factor!=null?m.profit_factor.toFixed(2):'—'}</td>`;
      h+=`<td style="padding:6px;${cls}">${r.status||'pending'}</td>`;
      h+=`</tr>`;
    });
    h+='</tbody></table>';
    document.getElementById('poScanTable').innerHTML=h;
  }

  function poStartScan(){
    const confirmBox=document.getElementById('poScanConfirm');
    if(!confirmBox?.checked){if(typeof addLog==='function')addLog('⚠ 请先确认当前股票池面板和参数');return;}
    const pool=document.querySelector('input[name="poScanPool"]:checked')?.value||'hs300';
    const phase=document.querySelector('input[name="poScanPhase"]:checked')?.value||'smoke';
    fetch('/api/wavehunter/v2/matrix/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm:true,pool,phase,pause_after_run:false})})
    .then(r=>r.ok?r.json():r.json().then(e=>{throw e;})).then(d=>{
      if(d.error){if(typeof addLog==='function')addLog('❌ '+d.error);return;}
      if(typeof addLog==='function')addLog('🚀 参数扫描已启动: '+pool+' '+phase);
      poLoadScanStatus();
    }).catch(e=>{if(typeof addLog==='function')addLog('❌ 启动失败: '+(e.message||e.detail||JSON.stringify(e)));});
  }

  function poStopScan(){
    fetch('/api/wavehunter/v2/matrix/stop',{method:'POST'}).then(r=>r.json()).then(()=>{
      if(typeof addLog==='function')addLog('⏹ 参数扫描已停止');
      clearTimeout(_poScanPoll);
      poLoadScanStatus();
    }).catch(e=>{if(typeof addLog==='function')addLog('❌ 停止失败: '+e.message);});
  }

  function poResetScan(){
    if(!confirm('确认重置矩阵状态？已完成的结果保留。'))return;
    fetch('/api/wavehunter/v2/matrix/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm:true})})
    .then(r=>r.json()).then(()=>{
      if(typeof addLog==='function')addLog('🔄 矩阵已重置');
      poLoadScanStatus();
    }).catch(e=>{if(typeof addLog==='function')addLog('❌ 重置失败: '+e.message);});
  }

  window.poStartScan=poStartScan;
  window.poStopScan=poStopScan;
  window.poResetScan=poResetScan;

  // ══════════════════════════════════════════════════
  // 子面板 2：Walk-forward (复用 /api/walkforward/*)
  // ══════════════════════════════════════════════════
  let _poWfPoll=null;

  function poLoadWfPanels(){
    fetch('/api/train/panels').then(r=>r.ok?r.json():[]).then(d=>{
      const sel=document.getElementById('poWfPanel');
      if(!sel)return;
      const panels=Array.isArray(d)?d:(d.panels||[]);
      sel.innerHTML='<option value="">-- 选择面板 --</option>'+panels.map(p=>{
        const name=typeof p==='string'?p:(p.name||p.path||'');
        return `<option value="${name}">${name.split('/').pop()}</option>`;
      }).join('');
    }).catch(()=>{});
  }

  function poWfStart(){
    const num=v=>{const n=parseFloat(v);return isNaN(n)?-1:n;};
    const params={
      panel:document.getElementById('poWfPanel').value,
      steps:parseInt(document.getElementById('poWfSteps').value)||100000,
      train_years:parseInt(document.getElementById('poWfTrainYears').value)||2,
      val_years:parseInt(document.getElementById('poWfValYears').value)||1,
      top_k:num(document.getElementById('poWfTopK')?.value),
      cost_bps:num(document.getElementById('poWfCost')?.value),
      stop_loss_pct:num(document.getElementById('poWfStopLoss')?.value),
      max_position_pct:num(document.getElementById('poWfMaxPos')?.value),
      rebalance_days:num(document.getElementById('poWfRebalance')?.value),
      max_holding_days:num(document.getElementById('poWfMaxHold')?.value),
    };
    if(!params.panel){if(typeof addLog==='function')addLog('❌ 请选择面板');return;}
    document.getElementById('poWfBadge').className='status-badge status-running';
    document.getElementById('poWfBadge').textContent='运行中';
    document.getElementById('poWfStartBtn').disabled=true;
    document.getElementById('poWfStopBtn').disabled=false;
    document.getElementById('poWfResults').style.display='none';
    fetch('/api/walkforward/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)})
    .then(r=>r.ok?r.json():r.json().then(e=>{throw e;})).then(d=>{
      if(d.error||d.detail){if(typeof addLog==='function')addLog('❌ '+(d.error||d.detail));poWfResetBadge();return;}
      if(typeof addLog==='function')addLog('🚀 Walk-forward 已启动');
      poPollWfStatus();
    }).catch(e=>{if(typeof addLog==='function')addLog('❌ '+e.message);poWfResetBadge();});
  }

  function poWfStop(){
    fetch('/api/walkforward/stop',{method:'POST'}).then(r=>r.json()).then(()=>{
      if(typeof addLog==='function')addLog('⏹ Walk-forward 已停止');
      poWfResetBadge();
    }).catch(e=>{if(typeof addLog==='function')addLog('❌ '+e.message);});
  }

  function poWfResetBadge(){
    const b=document.getElementById('poWfBadge');
    if(b){b.className='status-badge status-idle';b.textContent='空闲';}
    const s=document.getElementById('poWfStartBtn');if(s)s.disabled=false;
    const t=document.getElementById('poWfStopBtn');if(t)t.disabled=true;
  }

  function poPollWfStatus(){
    fetch('/api/walkforward/status').then(r=>r.ok?r.json():{}).then(d=>{
      const prog=document.getElementById('poWfProgress');
      if(d.running){
        if(prog) prog.textContent=d.progress||'';
        clearTimeout(_poWfPoll);_poWfPoll=setTimeout(poPollWfStatus,5000);
      }else{
        poWfResetBadge();
        if(d.completed) poLoadWfResults();
      }
    }).catch(()=>{clearTimeout(_poWfPoll);_poWfPoll=setTimeout(poPollWfStatus,10000);});
  }

  function poLoadWfResults(){
    fetch('/api/walkforward/result').then(r=>r.ok?r.json():{}).then(d=>{
      const win=d.windows||[];
      if(!win.length){if(typeof addLog==='function')addLog('⚠ 无窗口结果');return;}
      const el=document.getElementById('poWfResults');if(el)el.style.display='block';
      // 表格
      let h='<table style="width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed">';
      h+='<thead><tr style="border-bottom:1px solid #2a3a4a">';
      ['#','训练期','验证期','Sharpe','超额','收益','回撤'].forEach(c=>{h+=`<th style="padding:6px;text-align:left">${c}</th>`;});
      h+='</tr></thead><tbody>';
      let sumSharpe=0,sumExcess=0,posCount=0;
      win.forEach((w,i)=>{
        const s=w.sharpe||0,e=w.excess||0,r=w.total_return||0,dd=w.max_dd||0;
        sumSharpe+=s;sumExcess+=e;if(s>0)posCount++;
        h+=`<tr style="border-bottom:1px solid #1a2a3a">`;
        h+=`<td style="padding:6px">${i+1}</td>`;
        h+=`<td style="padding:6px;font-size:11px">${(w.train_start||'').slice(0,7)}~${(w.train_end||'').slice(0,7)}</td>`;
        h+=`<td style="padding:6px;font-size:11px">${(w.val_start||'').slice(0,7)}~${(w.val_end||'').slice(0,7)}</td>`;
        h+=`<td style="padding:6px;color:${s>0?'#81c784':'#ef5350'}">${s.toFixed(3)}</td>`;
        h+=`<td style="padding:6px;color:${e>0?'#81c784':'#ef5350'}">${e>0?'+':''}${e.toFixed(2)}%</td>`;
        h+=`<td style="padding:6px">${r>0?'+':''}${r.toFixed(2)}%</td>`;
        h+=`<td style="padding:6px;color:#ef5350">${dd.toFixed(1)}%</td>`;
        h+=`</tr>`;
      });
      h+='</tbody></table>';
      document.getElementById('poWfTable').innerHTML=h;
      // 统计摘要
      const avgS=(sumSharpe/win.length).toFixed(3);
      const avgE=(sumExcess/win.length).toFixed(2);
      const wr=((posCount/win.length)*100).toFixed(0);
      document.getElementById('poWfSummary').innerHTML=
        `<span style="color:#4fc3f7">平均 Sharpe: ${avgS}</span> | `+
        `<span style="color:#81c784">平均超额: ${avgE}%</span> | `+
        `<span style="color:#ffd54f">正 Sharpe 窗口: ${wr}% (${posCount}/${win.length})</span>`;
    }).catch(e=>{if(typeof addLog==='function')addLog('❌ 加载WF结果: '+e.message);});
  }

  window.poWfStart=poWfStart;
  window.poWfStop=poWfStop;

  // ══════════════════════════════════════════════════
  // 子面板 3：结果总览
  // ══════════════════════════════════════════════════
  function poLoadOverview(){
    // 聚合: sim ledgers + matrix results + WF results
    Promise.all([
      fetch('/api/sim/ledgers').then(r=>r.ok?r.json():{ledgers:[]}).catch(()=>({ledgers:[]})),
      fetch('/api/wavehunter/v2/matrix/results').then(r=>r.ok?r.json():{results:[]}).catch(()=>({results:[]})),
      fetch('/api/wavehunter/v2/models').then(r=>r.ok?r.json():{models:[]}).catch(()=>({models:[]})),
    ]).then(([leds,mtx,models])=>{
      const ledgerList=leds.ledgers||[];
      const mtxList=mtx.results||[];
      const modelList=models.models||[];

      // 合并: 优先用 ledger 数据（有模拟指标）
      const rows=[];
      ledgerList.forEach(l=>{
        rows.push({
          name:l.model||l.id,
          total_return:l.total_return,
          excess:l.excess_return??l.excess,
          sharpe:l.sharpe,
          win_rate:l.win_rate,
          profit_factor:l.profit_factor,
          max_dd:l.max_dd,
          trades:l.trades,
          saved_at:l.saved_at,
        });
      });
      // 补充 matrix 里有但 ledger 里没有的
      mtxList.forEach(m=>{
        if(!rows.find(r=>r.name===m.name)){
          rows.push({name:m.name,excess:m.excess,sharpe:m.sharpe,win_rate:m.win_rate,
            profit_factor:m.profit_factor,max_dd:m.max_dd});
        }
      });

      // 排序 (默认 Sharpe desc)
      const sortKey=document.getElementById('poOverviewSort')?.value||'sharpe';
      rows.sort((a,b)=>(b[sortKey]||0)-(a[sortKey]||0));

      // 渲染排行
      let h='<table style="width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed">';
      h+='<thead><tr style="border-bottom:1px solid #2a3a4a">';
      ['#','模型','总收益','超额','Sharpe','WR','PF','回撤','交易'].forEach(c=>{h+=`<th style="padding:6px;text-align:left;cursor:pointer">${c}</th>`;});
      h+='</tr></thead><tbody>';
      rows.slice(0,30).forEach((r,i)=>{
        const nameShort=(r.name||'').replace('whv2_','').replace('whv4_','').slice(0,30);
        h+=`<tr style="border-bottom:1px solid #1a2a3a;cursor:pointer" onclick="poOverviewSelect('${r.name||''}')">`;
        h+=`<td style="padding:6px">${i+1}</td>`;
        h+=`<td style="padding:6px;font-size:11px" title="${r.name||''}">${nameShort}</td>`;
        h+=`<td style="padding:6px;color:${(r.total_return||0)>0?'#81c784':'#ef5350'}">${r.total_return!=null?(r.total_return>0?'+':'')+r.total_return.toFixed(2)+'%':'—'}</td>`;
        h+=`<td style="padding:6px;color:${(r.excess||0)>0?'#81c784':'#ef5350'}">${r.excess!=null?(r.excess>0?'+':'')+r.excess.toFixed(2)+'%':'—'}</td>`;
        h+=`<td style="padding:6px;color:${(r.sharpe||0)>0?'#81c784':'#ef5350'}">${r.sharpe!=null?r.sharpe.toFixed(3):'—'}</td>`;
        h+=`<td style="padding:6px">${r.win_rate!=null?r.win_rate.toFixed(1)+'%':'—'}</td>`;
        h+=`<td style="padding:6px">${r.profit_factor!=null?r.profit_factor.toFixed(2):'—'}</td>`;
        h+=`<td style="padding:6px;color:#ef5350">${r.max_dd!=null?r.max_dd.toFixed(1)+'%':'—'}</td>`;
        h+=`<td style="padding:6px">${r.trades||'—'}</td>`;
        h+=`</tr>`;
      });
      h+='</tbody></table>';
      document.getElementById('poOverviewTable').innerHTML=h;
      // 统计
      const best=rows[0];
      document.getElementById('poOverviewBest').innerHTML=best?
        `🏆 最佳: <b style="color:#4fc3f7">${best.name||'?'}</b> | Sharpe <b style="color:#ffd54f">${(best.sharpe||0).toFixed(3)}</b> | 超额 <b style="color:#81c784">${best.excess!=null?(best.excess>0?'+':'')+best.excess.toFixed(2)+'%':'—'}</b>`:'';
    }).catch(e=>{if(typeof addLog==='function')addLog('❌ 结果总览: '+e.message);});
  }

  function poOverviewSelect(name){
    if(typeof addLog==='function')addLog('📋 选中: '+name+' (图表Tab联动开发中)');
  }

  window.poLoadOverview=poLoadOverview;

  // ══════════════════════════════════════════════════
  // 初始化
  // ══════════════════════════════════════════════════
  window.initParamOpt=function(){
    poLoadScanStatus();
    poLoadWfPanels();
    // 不自动加载 overview (懒加载)
  };

})();
