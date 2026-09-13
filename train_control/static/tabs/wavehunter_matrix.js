// tabs/wavehunter_matrix.js — 🌐 全局规律矩阵 Tab
// 从 index.html inline JS 抽离；依赖全局 fetch/addLog
// 🌐 WaveHunter 全局规律矩阵 Tab
// ══════════════════════════════════════════════════
let whmTimer=null;
function whmEsc(x){const d=document.createElement('div');d.textContent=x==null?'':String(x);return d.innerHTML;}
function loadWhMatrix(){Promise.all(['/api/wavehunter/matrix/config','/api/wavehunter/matrix/runs','/api/wavehunter/matrix/status','/api/wavehunter/matrix/events'].map(u=>fetch(u).then(r=>r.json()))).then(([c,r,s,e])=>{const ps=Object.entries(c.pools||{}).map(([k,v])=>k+':' +(v.exists?'✅':'❌')).join(' | ');document.getElementById('whmConfigSummary').textContent='72 Run | '+ps+' | 跨度 '+(c.spans||[]).join('/')+' | 步数 '+(c.steps||[]).map(x=>x/1000+'k').join('/');document.getElementById('whmProgress').textContent=(s.status||'idle')+' '+(s.progress||'')+' '+(s.run_id||'');const b=document.getElementById('whmBadge');b.textContent=s.running?'运行中':(s.status==='done'?'完成':(s.status==='stopped'?'已停止':'未开始'));b.className='status-badge '+(s.running?'status-running':(s.status==='done'?'status-done':'status-idle'));document.getElementById('whmStartBtn').disabled=!!s.running;document.getElementById('whmStopBtn').disabled=!s.running;let h='<table style="width:100%;border-collapse:collapse;font-size:11px;table-layout:fixed"><colgroup><col style="width:22%"><col style="width:7%"><col style="width:7%"><col style="width:8%"><col style="width:8%"><col style="width:10%"><col style="width:10%"><col style="width:10%"><col style="width:10%"><col style="width:8%"></colgroup><tr style="color:#4fc3f7"><th style="padding:4px 6px;text-align:center">Run</th><th>池</th><th>跨度</th><th>步数</th><th>状态</th><th>收益</th><th>超额</th><th>Sharpe</th><th>胜率</th><th>Ledger</th></tr>';(r.runs||[]).forEach(x=>{const m=x.metrics||{};const td='padding:4px 6px;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';h+='<tr style="border-top:1px solid #1f3a52"><td style="'+td+'">'+whmEsc(x.run_id)+'</td><td style="'+td+'">'+x.pool+'</td><td style="'+td+'">'+x.span+'</td><td style="'+td+'">'+(x.steps/1000)+'k</td><td style="'+td+'">'+x.status+'</td><td style="'+td+'">'+(m.total_return??'')+'</td><td style="'+td+'">'+(m.excess_return??'')+'</td><td style="'+td+'">'+(m.sharpe_ratio??m.sharpe??'')+'</td><td style="'+td+'">'+(m.win_rate??'')+'</td><td style="'+td+'">'+((x.artifacts||{}).ledger?'✅':'-')+'</td></tr>';});document.getElementById('whmRunTable').innerHTML=h+'</table>';document.getElementById('whmEvents').innerHTML=(e.events||[]).slice(-100).map(x=>'<div>'+whmEsc(x.time)+' ['+whmEsc(x.kind)+'] '+whmEsc(x.run_id)+' '+whmEsc(JSON.stringify(x))+'</div>').join('')||'暂无事件';if(s.running){clearTimeout(whmTimer);whmTimer=setTimeout(loadWhMatrix,5000);}}).catch(e=>addLog('❌ 矩阵刷新失败: '+e.message));}
function startWhMatrix(){if(!document.getElementById('whmConfirm').checked){addLog('⚠ 请先确认三个 v3 面板已完成审计');return;}fetch('/api/wavehunter/matrix/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm:true,resume:true})}).then(r=>r.json()).then(d=>{if(d.error){addLog('❌ '+d.error);return;}addLog('🚀 全局规律矩阵已启动: '+d.total+' Run，串行复用现有训练/模拟 Tab');loadWhMatrix();}).catch(e=>addLog('❌ '+e.message));}
function stopWhMatrix(){fetch('/api/wavehunter/matrix/stop',{method:'POST'}).then(()=>{addLog('⏹ 矩阵停止请求已发送');loadWhMatrix();});}
function buildWhMatrixPanel(){const pool=document.getElementById('whmPanelPool').value;const badge=document.getElementById('whmPanelBadge');const btn=document.getElementById('whmPanelBuildBtn');badge.textContent='构建中';badge.className='status-badge status-running';btn.disabled=true;fetch('/api/wavehunter/panel/build/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pool,start_date:document.getElementById('whmPanelStart').value,end_date:document.getElementById('whmPanelEnd').value})}).then(r=>r.ok?r.json():r.json().then(e=>{throw new Error(e.detail||'HTTP '+r.status);})).then(d=>{if(d.detail){throw new Error(d.detail);}addLog('🌊 v3面板构建已启动: '+pool,'panel_build');pollWhMatrixPanel();}).catch(e=>{addLog('❌ v3面板构建失败: '+e.message,'panel_build');badge.textContent='失败';badge.className='status-badge status-error';btn.disabled=false;});}
function pollWhMatrixPanel(){fetch('/api/wavehunter/panel/build/status').then(r=>r.ok?r.json():r.json().then(e=>{throw new Error(e.detail||'HTTP '+r.status);})).then(d=>{appendBuildRuntimeLogs(d,'v3面板构建');document.getElementById('whmPanelProgress').textContent=d.progress||d.error||'';if(d.running){setTimeout(pollWhMatrixPanel,3000);}else{const b=document.getElementById('whmPanelBadge');const btn=document.getElementById('whmPanelBuildBtn');b.textContent=d.result&&d.result.status==='ok'?'完成':(d.status==='done'?'完成':'失败');b.className='status-badge '+(d.result&&d.result.status==='ok'||d.status==='done'?'status-done':'status-error');btn.disabled=false;loadWhMatrix();}}).catch(e=>{addLog('❌ v3面板状态读取失败: '+e.message,'panel_build');setTimeout(pollWhMatrixPanel,5000);});}

// 🌊 WaveHunter 主升浪猎手 Tab
// ══════════════════════════════════════════════════
const WH_DEFAULTS = {
    whPanel: 'wavehunter_hs300_20040102_20260804.parquet',
    whSteps: '200000', whStartDate: '2023-01-01', whEndDate: '2023-12-31',
    whWindow: '20', whTopK: '20', whMaxPos: '0.02', whRebal: '20',
    whLr: '0.0001', whHidden: '128',
    // 公共参数 (对齐训练 Tab)
    whUniverseFilter: 'main_board_non_st', whNFactors: '0', whFwd: '20',
    whCostBps: '15', whNEnvs: '1', whTargetKl: '0.02', whSeed: '42',
    whBatchSize: '64', whNSteps: '64', whMaxGradNorm: '1.0',
    // WaveHunter 专属
    whAuxLambda: '0.1', whA2PosWeight: '25', whLblWindow: '20', whA2Thr: '0.10'
};
let whBadgeTimer = null;

function whDefaultParams() { return { ...WH_DEFAULTS }; }

function whReadParams() {
    const p = {};
    for (const k of Object.keys(WH_DEFAULTS)) {
        const el = document.getElementById(k);
        p[k.replace('wh', '').toLowerCase()] = el ? el.value : WH_DEFAULTS[k];
    }
    return p;
}
function whWriteParams(p) {
    for (const k of Object.keys(WH_DEFAULTS)) {
        const el = document.getElementById(k);
        if (el) el.value = p[k] ?? WH_DEFAULTS[k];
    }
}
// 读取面板列表 (懒加载)
function loadWhPanels() {
    fetch('/api/wavehunter/panels').then(r => r.json()).then(d => {
        const sel = document.getElementById('whPanel');
        if (!sel) return;
        const panels = d.panels || [];
        sel.innerHTML = panels.map(p => `<option value="${p.name}">${p.name} (${p.size_mb} MB)</option>`).join('') || '<option value="">无面板</option>';
        // 默认选中 hs300 v3 面板
        const preferred = panels.find(p => p.name.includes('wavehunter_hs300'));
        if (preferred) sel.value = preferred.name;
    }).catch(() => {});
}
// 模型列表
function loadWhModels() {
    fetch('/api/wavehunter/models').then(r => r.json()).then(d => {
        const box = document.getElementById('whModelList');
        if (!box) return;
        const models = d.models || [];
        if (!models.length) { box.innerHTML = '<div style="color:#667788;padding:8px">暂无训练模型</div>'; return; }
        box.innerHTML = models.map(m => `
            <div style="border:1px solid #2a2f3a;border-radius:6px;padding:8px 10px;margin-bottom:8px;background:#1a1f2a">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-family:monospace;font-size:12px;color:#eee">${m.name}</span>
                    <span style="display:flex;gap:4px">
                        ${m.has_ppo ? '<span class="status-badge status-ok" style="font-size:10px">✅ 已训练</span>' : '<span class="status-badge status-idle" style="font-size:10px">无权重</span>'}
                        <button class="btn btn-sm btn-danger" onclick="deleteWhModel('${m.name}')" style="padding:2px 6px;font-size:10px">🗑</button>
                    </span>
                </div>
                <div style="font-size:11px;color:#8899aa;margin-top:4px">
                    ${m.panel ? '📦 ' + m.panel : ''}
                    ${m.total_timesteps ? ' | 🏃 ' + m.total_timesteps.toLocaleString() + ' 步' : ''}
                    ${m.ppo_size_mb ? ' | 💾 ' + m.ppo_size_mb + ' MB' : ''}
                </div>
                ${m.aux_stats ? `
                <div style="font-size:10px;color:#667788;margin-top:4px">
                    A1_loss=${(m.aux_stats.a1_loss||0).toFixed(3)} A1_acc=${((m.aux_stats.a1_acc||0)*100).toFixed(1)}%
                    | A2_loss=${(m.aux_stats.a2_loss||0).toFixed(3)} A2_acc=${((m.aux_stats.a2_acc||0)*100).toFixed(1)}%
                </div>` : ''}
            </div>`).join('');
    }).catch(() => {});
}
function deleteWhModel(name) {
    if (!confirm(`删除模型 ${name}？`)) return;
    fetch('/api/wavehunter/models/' + name, { method: 'DELETE' }).then(r => r.json()).then(() => {
        loadWhModels(); addLog(`🗑 已删除模型 ${name}`);
    }).catch(e => addLog('❌ 删除失败: ' + e.message));
}
function whSetBadge(text, cls) {
    const b = document.getElementById('whTrainBadge');
    if (!b) return;
    b.textContent = text;
    b.className = 'status-badge ' + cls;
}
// 参数预设
function loadWhConfigs() {
    fetch('/api/wavehunter/configs').then(r => r.json()).then(d => {
        const sel = document.getElementById('whSavedConfigs');
        if (!sel) return;
        const cfgs = d.configs || [];
        sel.innerHTML = '<option value="">-- 已保存参数 --</option>' +
            cfgs.map(c => `<option value="${c.name}">${c.name}</option>`).join('');
    }).catch(() => {});
}
function loadWhConfig(name) {
    if (!name) return;
    fetch('/api/wavehunter/configs').then(r => r.json()).then(d => {
        const c = (d.configs || []).find(x => x.name === name);
        if (c) whWriteParams(c.params);
        addLog(`📥 已载入参数预设: ${name}`);
    }).catch(() => {});
}

// ⚖️ 权重矩阵子账本（内嵌矩阵 Tab，不新建 Tab，不新增训练入口）
// 复用 /api/wavehunter/v2/train/* 与 /api/sim/ledgers，单变量扫 a1/a2 pos_weight
let _weightTimer = null;
let _weightSweepQueue = [];
let _weightRunning = false;
function weightEsc(x){const d=document.createElement('div');d.textContent=x==null?'':String(x);return d.innerHTML;}
function loadWeightMatrix(){
  const summary=document.getElementById('weightMatrixSummary');
  const table=document.getElementById('weightMatrixTable');
  const badge=document.getElementById('weightBadge');
  const prog=document.getElementById('weightProgress');
  const eventsEl=document.getElementById('weightMatrixEvents');
  Promise.all([fetch('/api/wavehunter/v2/models').then(r=>r.json()), fetch('/api/wavehunter/v2/train/status').then(r=>r.json()), fetch('/api/sim/ledgers').then(r=>r.json())]).then(([m,st,led])=>{
    const models=(m.models||[]).slice();
    // Filter weight-related models: name contains whv2 or hyperparams pos_weight deviates
    // For display: group by a1w/a2w from training_summary hyperparams
    let rows=[];
    for(const mm of models){
      // fetch hyperparams from summary if available via /api/wavehunter/v2/models detail is not per-model, need local cache: use list already includes hyperparams? fallback to name parse
      const hp = mm.hyperparams || mm.config || {};
      rows.push({name:mm.name, a1w: hp.a1_pos_weight ?? '', a2w: hp.a2_pos_weight ?? '', metrics: mm.metrics || mm.aux_stats || {}, ledger: (led.ledgers||[]).find(l=>l.model===mm.name)});
    }
    // Sort by a1w then a2w then name
    rows.sort((a,b)=> (Number(a.a1w)||0)-(Number(b.a1w)||0) || (Number(a.a2w)||0)-(Number(b.a2w)||0) || a.name.localeCompare(b.name));
    // Summary
    const trainState = st.running ? 'training '+(st.current||0)+'/'+(st.total||'?') : (st.status||'idle');
    if(summary) summary.textContent = 'Models: '+models.length+' | Train: '+trainState+' | Sweeps: '+rows.length+' | Filter: a1/a2 pos_weight in training_summary hyperparams';
    if(badge){ badge.textContent = st.running ? 'training' : (st.status==='finished'?'done':(st.status||'idle')); badge.className='status-badge '+(st.running?'status-running':(st.status==='finished'?'status-done':'status-idle')); }
    if(prog) prog.textContent = st.running ? ('current '+(st.current||0)+'/'+(st.total||'')+' '+(st.percent||0)+'%') : '';
    if(table){
      let h='<table style="width:100%;border-collapse:collapse;font-size:11px"><tr style="color:#4fc3f7"><th style="padding:4px">Model</th><th>a1w</th><th>a2w</th><th>P(A1)</th><th>R(A1)</th><th>P(A2)</th><th>R(A2)</th><th>Ledger</th></tr>';
      for(const r of rows){
        const ax=r.metrics;
        h+='<tr style="border-top:1px solid #1f3a52"><td style="padding:4px">'+weightEsc(r.name)+'</td><td style="text-align:center">'+weightEsc(r.a1w)+'</td><td style="text-align:center">'+weightEsc(r.a2w)+'</td><td style="text-align:center">'+weightEsc(ax.a1_precision!=null?Number(ax.a1_precision).toFixed(3):'')+'</td><td style="text-align:center">'+weightEsc(ax.a1_recall!=null?Number(ax.a1_recall).toFixed(3):'')+'</td><td style="text-align:center">'+weightEsc(ax.a2_precision!=null?Number(ax.a2_precision).toFixed(3):'')+'</td><td style="text-align:center">'+weightEsc(ax.a2_recall!=null?Number(ax.a2_recall).toFixed(3):'')+'</td><td style="text-align:center">'+(r.ledger?'ok: '+weightEsc(r.ledger.id):'-')+'</td></tr>';
      }
      table.innerHTML = h+'</table>';
    }
    if(eventsEl) eventsEl.textContent = 'train cmd: '+(st.cmd||'').slice(0,300);
    if(st.running){ clearTimeout(_weightTimer); _weightTimer=setTimeout(loadWeightMatrix,5000); }
  }).catch(e=>{ if(summary) summary.textContent='load failed: '+e.message; });
}
function _weightParams(a1w,a2w,steps){
  const stepsNum = Number(steps||50000);
  const outDir = `whv2_weight_a1_${a1w}_a2_${a2w}_${stepsNum/1000}k`;
  try{
    if(typeof whV2CollectParams==='function') { const p=whV2CollectParams(); p.a1_pos_weight=Number(a1w); p.a2_pos_weight=Number(a2w); p.total_timesteps=stepsNum; p.out_dir=outDir; return p; }
  }catch(_){}
  return {panel:'wavehunter_v8_hs300_20040102_20260827.parquet', start_date:'2023-01-01', end_date:'2023-03-31', total_timesteps: stepsNum, window:20, top_k:20, lstm_hidden:64, n_steps:32, batch_size:32, seed:42, a1_pos_weight:Number(a1w), a2_pos_weight:Number(a2w), out_dir: outDir};
}
function startWeightSingle(){
  const a1=document.getElementById('weightA1Select')?.value||'60';
  const a2=document.getElementById('weightA2Select')?.value||'60';
  const steps=document.getElementById('weightStepsSelect')?.value||'50000';
  const body=_weightParams(a1,a2,steps);
  fetch('/api/wavehunter/v2/train/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status)})).then(d=>{ addLog('Weight single started a1='+a1+' a2='+a2+' -> '+d.out_name); loadWeightMatrix(); }).catch(e=>addLog('Weight start failed: '+e.message));
}
async function startWeightSweep(){
  if(_weightRunning){ addLog('Weight sweep already running'); return; }
  const steps=document.getElementById('weightStepsSelect')?.value||'50000';
  const gridA1=[25,40,60,80];
  const gridA2=[25,40,60,80];
  _weightRunning=true;
  for(const a1 of gridA1){ for(const a2 of gridA2){
    const st=await fetch('/api/wavehunter/v2/train/status').then(r=>r.json()).catch(()=>({running:false}));
    if(st.running){ addLog('wait train finish before next weight a1='+a1+' a2='+a2); await new Promise(r=>setTimeout(r,5000)); }
    const body=_weightParams(a1,a2,steps);
    try{ const d=await fetch('/api/wavehunter/v2/train/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status)})); addLog('Weight sweep started a1='+a1+' a2='+a2+' -> '+d.out_name); }catch(e){ addLog('Weight sweep start failed a1='+a1+' a2='+a2+': '+e.message); continue; }
    // wait finished
    await new Promise(res=>{
      const t=setInterval(async()=>{
        const s=await fetch('/api/wavehunter/v2/train/status').then(r=>r.json()).catch(()=>({running:false,status:'unknown'}));
        if(!s.running){ clearInterval(t); res(); }
      },5000);
    });
    await new Promise(r=>setTimeout(r,1500));
    loadWeightMatrix();
  }}
  _weightRunning=false;
  addLog('Weight sweep 4x4 finished');
}



// ═══ 矩阵统一设计：最佳参数发现（热力图 + 趋势 + 训练曲线联动）══════════════
let _matrixBestRow = null;
let _weightRowsCache = [];
let _ledgerRowsCache = [];

function _excessColor(v, vmin, vmax){
    if(v==null || isNaN(v)) return {bg:'#1a2a3a',fg:'#667788'};
    const t = vmax===vmin?0.5:(Number(v)-vmin)/(vmax-vmin);
    // red -> gray -> green
    if(t < 0.5){
        const k=t*2;
        const r=Math.round(180*(1-k)+80*k), g=Math.round(60*(1-k)+90*k), b=Math.round(60*(1-k)+90*k);
        return {bg:`rgb(${r},${g},${b})`, fg:'#e0e0e0'};
    } else {
        const k=(t-0.5)*2;
        const r=Math.round(80*(1-k)+40*k), g=Math.round(90*(1-k)+160*k), b=Math.round(90*(1-k)+80*k);
        return {bg:`rgb(${r},${g},${b})`, fg:'#e0e0e0'};
    }
}

function renderWeightBestSpotlight(rows){
    const el=document.getElementById('weightBestSpotlight');
    if(!el) return;
    const withExcess=rows.filter(r=>r.excessReturn!=null);
    if(!withExcess.length){
        el.innerHTML='🏆 最佳参数 Spotlight：'+(rows.length? '暂无超额账本，先跑模拟' : '暂无权重模型');
        _matrixBestRow=null; return;
    }
    withExcess.sort((a,b)=> Number(b.excessReturn)-Number(a.excessReturn));
    const best=withExcess[0];
    _matrixBestRow=best;
    const sharpe = best.sharpe!=null? Number(best.sharpe).toFixed(2):'—';
    const wr = best.winRate!=null? (Number(best.winRate)*100).toFixed(1)+'%' : '—';
    const pf = best.pf!=null? Number(best.pf).toFixed(2):'—';
    el.innerHTML=`<span class="spot-title">🏆 最佳参数</span>
        <span class="spot-metric spot-best" style="font-family:monospace">a1=${best.a1w} a2=${best.a2w}</span>
        <span class="spot-metric" style="color:#81c784">超额 ${Number(best.excessReturn).toFixed(2)}%</span>
        <span class="spot-metric">Sharpe ${sharpe}</span>
        <span class="spot-metric">WR ${wr}</span>
        <span class="spot-metric">PF ${pf}</span>
        <span style="margin-left:auto;display:flex;gap:8px">
            <button class="btn btn-sm btn-success" onclick="applyWeightBestToV2()">⚡ 一键应用到 🌊v2</button>
            <button class="btn btn-sm btn-primary" onclick="openWeightTrainChart('${best.name}')">📈 看训练曲线</button>
        </span>
        <span style="width:100%;font-size:11px;color:#8899aa;margin-top:6px">按 2023-04 样本外超额第一排序；F1≠超额，已验证。</span>`;
}

function renderWeightHeatmap(rows){
    const el=document.getElementById('weightHeatmap');
    if(!el) return;
    const a1s=[25,40,60,80], a2s=[25,40,60,80];
    const lookup={}; rows.forEach(r=>{ lookup[`${r.a1w}_${r.a2w}`]=r; });
    const vals=rows.map(r=>Number(r.excessReturn)).filter(v=>isFinite(v) && v!=null);
    const vmin = vals.length? Math.min(...vals): -5, vmax= vals.length? Math.max(...vals): 5;
    let h='<div class="heat-head"></div>' + a2s.map(a2=>`<div class="heat-head">a2=${a2}</div>`).join('');
    for(const a1 of a1s){
        h+=`<div class="heat-head" style="display:flex;align-items:center;justify-content:center">a1=${a1}</div>`;
        for(const a2 of a2s){
            const r=lookup[`${a1}_${a2}`];
            const v=r? Number(r.excessReturn): null;
            const c=_excessColor(v, vmin, vmax);
            const isBest = _matrixBestRow && String(_matrixBestRow.a1w)===String(a1) && String(_matrixBestRow.a2w)===String(a2);
            const hasData = !!r;
            const excessStr = v!=null? (v>0? '+'+v.toFixed(2) : v.toFixed(2))+'%' : '—';
            const p1 = hasData && r.a1p!=null? Number(r.a1p).toFixed(2):'—';
            const r1 = hasData && r.a1r!=null? Number(r.a1r).toFixed(2):'—';
            const p2 = hasData && r.a2p!=null? Number(r.a2p).toFixed(2):'—';
            const r2 = hasData && r.a2r!=null? Number(r.a2r).toFixed(2):'—';
            const sharpeStr = hasData && r.sharpe!=null? Number(r.sharpe).toFixed(2):'—';
            const tip = hasData? `模型: ${r.name} | 超额 ${excessStr} Sharpe ${sharpeStr} | P1 ${p1} R1 ${r1} | P2 ${p2} R2 ${r2} — 点击看训练曲线` : '无数据 — 点击无效';
            h+=`<div class="heat-cell ${isBest?'heat-best':''}" style="background:${c.bg};color:${c.fg}" onclick="selectWeightCell(${a1},${a2})"><div class="heat-label">${isBest? '★ 最佳': excessStr}</div><div class="heat-sub">P1 ${p1} R1 ${r1}</div><div class="heat-sub">P2 ${p2} R2 ${r2}</div><div class="heat-sub" style="font-size:9px;opacity:0.7">${hasData? r.name.slice(0,22): '—'}</div></div>`;
            // store tip as data attr for hover via title fallback handled by CSS tooltip
            // also set title via JS after render
        }
    }
    el.innerHTML=h;
    // set titles separately to avoid quote escaping issues
    try{
        const cells=el.querySelectorAll('.heat-cell');
        let idx=0;
        for(const a1 of a1s){ for(const a2 of a2s){
            const r=lookup[`${a1}_${a2}`];
            const cell=cells[idx++];
            if(!cell) continue;
            const v=r? Number(r.excessReturn): null;
            const hasData=!!r;
            const excessStr = v!=null? (v>0? '+'+v.toFixed(2) : v.toFixed(2))+'%' : '—';
            const p1 = hasData && r.a1p!=null? Number(r.a1p).toFixed(2):'—';
            const r1 = hasData && r.a1r!=null? Number(r.a1r).toFixed(2):'—';
            const p2 = hasData && r.a2p!=null? Number(r.a2p).toFixed(2):'—';
            const r2 = hasData && r.a2r!=null? Number(r.a2r).toFixed(2):'—';
            const sharpeStr = hasData && r.sharpe!=null? Number(r.sharpe).toFixed(2):'—';
            cell.title = hasData? `模型: ${r.name} | 超额 ${excessStr} Sharpe ${sharpeStr} | P1 ${p1} R1 ${r1} | P2 ${p2} R2 ${r2}` : '无数据';
        }}
    }catch(_){}
}

function selectWeightCell(a1,a2){
    const r=_weightRowsCache.find(x=> String(x.a1w)===String(a1) && String(x.a2w)===String(a2));
    if(r) openWeightTrainChart(r.name);
}

function renderWeightTrendChart(rows){
    const el=document.getElementById('weightTrendChart');
    if(!el || !window.echarts) return;
    const sorted = rows.slice().sort((a,b)=> Number(a.a1w)-Number(b.a1w) || Number(a.a2w)-Number(b.a2w));
    // 按 a1 分组看趋势：x = a1, y = 超额 (取同a1下最好的a2)
    const byA1={}; sorted.forEach(r=>{ if(r.excessReturn==null) return; const k=r.a1w; if(!byA1[k] || Number(r.excessReturn)>Number(byA1[k].excessReturn)) byA1[k]=r; });
    const xs=Object.keys(byA1).sort((a,b)=>a-b);
    const ys=xs.map(k=> Number(byA1[k].excessReturn));
    const chart=echarts.init(el); chart.clear();
    const themeBg='transparent';
    chart.setOption({
        backgroundColor: themeBg,
        textStyle:{color:'#e0e0e0'},
        grid:{left:'8%',right:'6%',top:'10%',bottom:'18%'},
        tooltip:{trigger:'axis', backgroundColor:'#1a2a3a', borderColor:'#4fc3f7', textStyle:{color:'#e0e0e0'}},
        xAxis:{type:'category', data: xs, name:'a1', nameTextStyle:{color:'#8899aa'}, axisLabel:{color:'#8899aa'}, axisLine:{lineStyle:{color:'#2a3a4a'}}},
        yAxis:{type:'value', name:'超额%', axisLabel:{color:'#8899aa'}, splitLine:{lineStyle:{color:'#1a2a3a'}}},
        series:[
            {type:'line', data: ys, smooth:true, symbol:'circle', symbolSize:6, lineStyle:{width:2, color:'#4fc3f7'}, itemStyle:{color:'#ffd54f'}, name:'最佳超额(a1趋势)'},
        ]
    });
    chart.resize();
    window._weightTrendChart=chart;
}

function applyWeightBestToV2(){
    if(!_matrixBestRow){ addLog('⚠ 暂无最佳参数'); return; }
    const a1=_matrixBestRow.a1w, a2=_matrixBestRow.a2w;
    // 写入 v2 表单：优先用 whv2 schema 动态表单，其次用旧 wh* id
    try{
        const trySet=(id,val)=>{ const el=document.getElementById(id); if(el){ el.value=String(val); el.dispatchEvent(new Event('change')); return true;} return false; };
        const hit = trySet('a1_pos_weight', a1) || trySet('whA1PosWeight', a1) || trySet('v2_a1_pos_weight', a1);
        const hit2= trySet('a2_pos_weight', a2) || trySet('whA2PosWeight', a2) || trySet('v2_a2_pos_weight', a2);
        if(!hit) addLog(`⚠ 未找到 a1 输入，尝试写入 localStorage a1=${a1}`);
        try{ localStorage.setItem('whv2_best_a1', String(a1)); localStorage.setItem('whv2_best_a2', String(a2)); }catch(_){}
    }catch(e){ addLog('⚠ 应用失败: '+e.message); }
    addLog(`⚡ 已应用最佳权重到 v2: a1=${a1} a2=${a2}（请到 🌊 主升浪猎手 v2 确认）`);
    if(typeof switchTab==='function') try{ switchTab('wavehunter_v2'); }catch(_){}
}

function openWeightTrainChart(modelName){
    const el=document.getElementById('weightTrainChart');
    if(!el) return;
    el.innerHTML='<div style="color:#4fc3f7;padding:20px;text-align:center">加载 '+modelName+' 训练曲线...</div>';
    // 优先 whv2 log，其次通用 metrics-detail
    fetch('/api/wavehunter/v2/train/log').then(r=>r.json()).then(d=>{
        // 如果当前 log 不是该模型，改走通用 metrics-detail
        const lines=d.lines||[];
        const isTarget = (d.status && d.status.out_name===modelName) || lines.join('').includes(modelName);
        if(!isTarget){
            return fetch('/api/train/metrics-detail/whv2/'+encodeURIComponent(modelName)).then(r=>r.json()).then(dd=>{
                if(dd.metrics && dd.metrics.length) renderGenericTrainChart('weightTrainChart', dd, modelName);
                else throw new Error('无训练指标');
            }).catch(()=>{ el.innerHTML='<div style="color:#ffb74d;padding:20px;text-align:center">该模型训练曲线暂不可用（仅权重矩阵训练有1k点日志）</div>'; });
        }
        // 解析 whv2 log 中的 1k点 A1/A2 曲线（简单文本解析，复用图表Tab逻辑：若有 metrics-detail则优先）
        return fetch('/api/train/metrics-detail/whv2/'+encodeURIComponent(modelName)).then(r=>r.json()).then(dd=>{
            if(dd.metrics && dd.metrics.length) renderGenericTrainChart('weightTrainChart', dd, modelName);
            else throw new Error('无指标');
        }).catch(()=>{
            // 回退：从 log 文本画简易折线
            renderWhv2LogChart('weightTrainChart', lines, modelName);
        });
    }).catch(e=>{ el.innerHTML='<div style="color:#f44336;padding:20px">加载失败: '+(e.message||e)+'</div>'; });
}

function renderGenericTrainChart(containerId, dd, modelName){
    const el=document.getElementById(containerId);
    if(!el || !window.echarts) return;
    const metrics=dd.metrics||[];
    const steps=metrics.map(m=> m.timestep ?? m.step ?? m.num_timesteps ?? 0);
    const extra=(m)=> m.extra||{};
    const getSeries=(key)=> metrics.map(m=> m[key] ?? (m.extra && m.extra[key]) ?? null);
    const loss=getSeries('train/loss').map((v,i)=> v ?? metrics[i].loss ?? extra(metrics[i]).loss ?? null);
    const a1acc=getSeries('a1_acc').map((v,i)=> v ?? extra(metrics[i]).a1_acc ?? null);
    const reward=getSeries('rollout/ep_rew_mean');
    const chart=echarts.init(el); chart.clear();
    chart.setOption({
        backgroundColor:'transparent', textStyle:{color:'#e0e0e0'},
        grid:{left:'8%',right:'6%',top:'10%',bottom:'14%'},
        tooltip:{trigger:'axis', backgroundColor:'#1a2a3a', borderColor:'#4fc3f7', textStyle:{color:'#e0e0e0'}},
        legend:{data:['loss','A1_acc','reward'], textStyle:{color:'#8899aa'}, top:0},
        xAxis:{type:'category', data: steps, axisLabel:{color:'#8899aa', fontSize:10}, axisLine:{lineStyle:{color:'#2a3a4a'}}},
        yAxis:{type:'value', axisLabel:{color:'#8899aa', fontSize:10}, splitLine:{lineStyle:{color:'#1a2a3a'}}},
        series:[
            {name:'loss', type:'line', data: loss, smooth:true, symbol:'none', lineStyle:{width:2, color:'#ff7043'}},
            {name:'A1_acc', type:'line', data: a1acc, smooth:true, symbol:'none', lineStyle:{width:2, color:'#66bb6a'}},
            {name:'reward', type:'line', data: reward, smooth:true, symbol:'none', lineStyle:{width:2, color:'#4fc3f7'}},
        ]
    });
    chart.resize();
    addLog(`📈 已加载训练曲线: ${modelName} (${metrics.length} 点)`);
}

function renderWhv2LogChart(containerId, lines, modelName){
    const el=document.getElementById(containerId);
    if(!el || !window.echarts) return;
    // 解析形如 "[wh] 1000 步 A1_acc 0.34 ..." 的行
    const pts=[];
    for(const line of lines){
        const m=line.match(/\b(\d+)\s*步[^0-9]*A1_acc\s*([0-9.]+).*A2_acc\s*([0-9.]+)/);
        if(m) pts.push({step: Number(m[1]), a1: Number(m[2]), a2: Number(m[3])});
    }
    if(!pts.length){ el.innerHTML='<div style="color:#667788;padding:20px;text-align:center">日志中无 A1/A2 曲线点</div>'; return; }
    const chart=echarts.init(el); chart.clear();
    chart.setOption({
        backgroundColor:'transparent', textStyle:{color:'#e0e0e0'},
        grid:{left:'8%',right:'6%',top:'10%',bottom:'14%'},
        tooltip:{trigger:'axis', backgroundColor:'#1a2a3a', borderColor:'#4fc3f7', textStyle:{color:'#e0e0e0'}},
        legend:{data:['A1_acc','A2_acc'], textStyle:{color:'#8899aa'}, top:0},
        xAxis:{type:'category', data: pts.map(p=>p.step), axisLabel:{color:'#8899aa', fontSize:10}, axisLine:{lineStyle:{color:'#2a3a4a'}}},
        yAxis:{type:'value', axisLabel:{color:'#8899aa'}, splitLine:{lineStyle:{color:'#1a2a3a'}}},
        series:[
            {name:'A1_acc', type:'line', data: pts.map(p=>p.a1), smooth:true, symbol:'none', lineStyle:{width:2, color:'#66bb6a'}},
            {name:'A2_acc', type:'line', data: pts.map(p=>p.a2), smooth:true, symbol:'none', lineStyle:{width:2, color:'#4fc3f7'}},
        ]
    });
    chart.resize();
}

function onLedgerSortChange(){
    const key=document.getElementById('ledgerSort')?.value||'excess';
    renderLedgerTableAndCharts(_ledgerRowsCache, key);
}

function renderLedgerTableAndCharts(rows, sortKey){
    const tbody=document.getElementById('mlTestBody');
    const totalEl=document.getElementById('mlTotalTests');
    const doneEl=document.getElementById('mlCompleted');
    const sharpeEl=document.getElementById('mlBestSharpe');
    const excessEl=document.getElementById('mlBestExcess');
    const hint=document.getElementById('ledgerBestHint');
    if(!rows || !rows.length){
        if(tbody) tbody.innerHTML='<tr><td colspan="10" style="text-align:center;color:#667788;padding:12px">暂无账本</td></tr>';
        return;
    }
    const keyMap={excess:'excess', sharpe:'sharpe', win_rate:'win_rate', total:'total'};
    const k=keyMap[sortKey]||'excess';
    const sorted=rows.slice().sort((a,b)=>{
        const av=a.metrics? Number(a.metrics[k] ?? a.metrics[k+'_return'] ?? a.metrics['sharpe_ratio'] ?? -Infinity) : -Infinity;
        const bv=b.metrics? Number(b.metrics[k] ?? b.metrics[k+'_return'] ?? b.metrics['sharpe_ratio'] ?? -Infinity) : -Infinity;
        return bv-av;
    });
    // 渲染表
    if(tbody){
        tbody.innerHTML=sorted.map((r,idx)=>{
            const m=r.metrics||{};
            const excess=m.excess ?? m.excess_return ?? m.excessReturn ?? '';
            const sharpe=m.sharpe ?? m.sharpe_ratio ?? '';
            const status=r.status||'done';
            const rowCls = idx===0? 'rank-gold':'';
            return `<tr class="${rowCls}" style="border-bottom:1px solid #1f3a52;cursor:pointer" onclick="selectLedgerRow('${r.test_id||r.run_id||r.id||''}')">
                <td style="padding:8px;font-family:monospace;font-size:11px">${(r.test_id||r.run_id||r.id||'').slice(0,22)}</td>
                <td style="padding:8px">${r.name||r.run_id||''}</td>
                <td style="padding:8px">${r.reward_type||m.reward_type||''}</td>
                <td style="padding:8px">${r.lr||m.lr||''}</td>
                <td style="padding:8px">${r.max_pos||m.max_pos||''}</td>
                <td style="padding:8px">${r.steps||m.steps||''}</td>
                <td style="padding:8px"><span class="status-badge ${status==='done'?'status-done':(status==='running'?'status-running':'status-idle')}">${status}</span></td>
                <td style="padding:8px;color:${Number(sharpe)>=1?'#81c784': Number(sharpe)<0?'#ef5350':'#e0e0e0'}">${sharpe!==''?Number(sharpe).toFixed(2):'—'}</td>
                <td style="padding:8px;color:${Number(excess)>=0?'#81c784':'#ef5350'}">${excess!==''?Number(excess).toFixed(2)+'%':'—'}</td>
                <td style="padding:8px"><button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); selectLedgerRow('${r.test_id||r.run_id||r.id||''}')">👁 看曲线</button></td>
            </tr>`;
        }).join('');
    }
    if(totalEl) totalEl.textContent=String(rows.length);
    if(doneEl) doneEl.textContent=String(rows.filter(r=>r.status==='done').length);
    const bestExcess = sorted[0]?.metrics ? Number(sorted[0].metrics.excess ?? sorted[0].metrics.excess_return ?? sorted[0].metrics.excessReturn ?? 0) : 0;
    const bestSharpe = sorted[0]?.metrics ? Number(sorted[0].metrics.sharpe ?? sorted[0].metrics.sharpe_ratio ?? 0) : 0;
    if(sharpeEl) sharpeEl.textContent = isFinite(bestSharpe)? bestSharpe.toFixed(2):'—';
    if(excessEl) excessEl.textContent = isFinite(bestExcess)? bestExcess.toFixed(2)+'%':'—';
    if(hint) hint.textContent = sorted[0]? `🏆 最佳：${(sorted[0].test_id||sorted[0].run_id||'').slice(0,22)} 超额 ${isFinite(bestExcess)?bestExcess.toFixed(2)+'%':'—'}` : '';
    // 散点图
    const scEl=document.getElementById('ledgerScatterChart');
    if(scEl && window.echarts){
        const chart=echarts.init(scEl); chart.clear();
        const pts=sorted.map(r=>{ const m=r.metrics||{}; return [Number(m.excess ?? m.excess_return ?? 0), Number(m.sharpe ?? m.sharpe_ratio ?? 0)]; }).filter(p=>isFinite(p[0])&&isFinite(p[1]));
        chart.setOption({
            backgroundColor:'transparent', textStyle:{color:'#e0e0e0'},
            grid:{left:'10%',right:'6%',top:'10%',bottom:'14%'},
            tooltip:{trigger:'item', backgroundColor:'#1a2a3a', borderColor:'#4fc3f7', textStyle:{color:'#e0e0e0'}},
            xAxis:{type:'value', name:'超额%', nameTextStyle:{color:'#8899aa'}, axisLabel:{color:'#8899aa'}, splitLine:{lineStyle:{color:'#1a2a3a'}}},
            yAxis:{type:'value', name:'Sharpe', nameTextStyle:{color:'#8899aa'}, axisLabel:{color:'#8899aa'}, splitLine:{lineStyle:{color:'#1a2a3a'}}},
            series:[{type:'scatter', data: pts, symbolSize:8, itemStyle:{color:'#4fc3f7'}}]
        });
        chart.resize();
    }
}

function selectLedgerRow(testId){
    if(!testId) return;
    const el=document.getElementById('ledgerTrainChart');
    if(!el) return;
    el.innerHTML='<div style="color:#4fc3f7;padding:20px;text-align:center">加载 '+testId+' 训练曲线...</div>';
    fetch('/api/train/metrics-detail/matrix/'+encodeURIComponent(testId)).then(r=>r.json()).then(dd=>{
        if(dd.metrics && dd.metrics.length) renderGenericTrainChart('ledgerTrainChart', dd, testId);
        else throw new Error('无指标');
    }).catch(()=>{ el.innerHTML='<div style="color:#667788;padding:20px;text-align:center">该账本无训练曲线（仅矩阵Run有）</div>'; });
    // 同时展开详情卡
    if(typeof mlShowDetail==='function') try{ mlShowDetail(testId); }catch(_){}
}

// 覆盖：增强 loadWeightMatrix 以驱动 Spotlight/热力图/趋势图
const _origLoadWeightMatrix = loadWeightMatrix;
loadWeightMatrix = function(){
    const summary=document.getElementById('weightMatrixSummary');
    const table=document.getElementById('weightMatrixTable');
    const badge=document.getElementById('weightBadge');
    const prog=document.getElementById('weightProgress');
    const eventsEl=document.getElementById('weightMatrixEvents');
    Promise.all([fetch('/api/wavehunter/v2/models').then(r=>r.json()), fetch('/api/wavehunter/v2/train/status').then(r=>r.json()), fetch('/api/sim/ledgers').then(r=>r.json())]).then(([m,st,led])=>{
        const models=(m.models||[]).slice();
        let rows=[];
        for(const mm of models){
            const hp = mm.hyperparams || mm.config || {};
            const ax = mm.aux_stats || {};
            const ledger = (led.ledgers||[]).find(l=>l.model===mm.name);
            const lm = ledger? ledger.metrics||{} : {};
            rows.push({
                name:mm.name, a1w: hp.a1_pos_weight ?? '', a2w: hp.a2_pos_weight ?? '',
                a1p: ax.a1_precision, a1r: ax.a1_recall, a2p: ax.a2_precision, a2r: ax.a2_recall,
                excessReturn: lm.excess ?? lm.excess_return ?? lm.excessReturn ?? null,
                sharpe: lm.sharpe ?? lm.sharpe_ratio ?? null,
                winRate: lm.win_rate ?? null, pf: lm.profit_factor ?? lm.pf ?? null,
                ledgerId: ledger? ledger.id : null, ledger: ledger
            });
        }
        rows.sort((a,b)=> (Number(a.a1w)||0)-(Number(b.a1w)||0) || (Number(a.a2w)||0)-(Number(b.a2w)||0) || a.name.localeCompare(b.name));
        _weightRowsCache = rows.slice();
        const trainState = st.running ? 'training '+(st.current||0)+'/'+(st.total||'?') : (st.status||'idle');
        if(summary) summary.textContent = 'Models: '+models.length+' | Train: '+trainState+' | Sweeps: '+rows.length+' | 按超额找最优';
        if(badge){ badge.textContent = st.running ? 'training' : (st.status==='finished'?'done':(st.status||'idle')); badge.className='status-badge '+(st.running?'status-running':(st.status==='finished'?'status-done':'status-idle')); }
        if(prog) prog.textContent = st.running ? ('current '+(st.current||0)+'/'+(st.total||'')+' '+(st.percent||0)+'%') : '';
        renderWeightBestSpotlight(rows);
        renderWeightHeatmap(rows);
        renderWeightTrendChart(rows);
        if(table){
            const withExcess = rows.slice().sort((a,b)=> Number(b.excessReturn ?? -Infinity) - Number(a.excessReturn ?? -Infinity));
            const bestName = withExcess[0] && withExcess[0].excessReturn!=null ? withExcess[0].name : null;
            let h='<table style="width:100%;border-collapse:collapse;font-size:11px"><tr style="color:#4fc3f7"><th style="padding:6px;text-align:left">Rank</th><th style="padding:6px">Model</th><th>a1w</th><th>a2w</th><th>P(A1)</th><th>R(A1)</th><th>P(A2)</th><th>R(A2)</th><th>超额</th><th>Sharpe</th><th>WR</th><th>操作</th></tr>';
            withExcess.forEach((r,idx)=>{
                const isBest = r.name===bestName;
                h+=`<tr class="${isBest?'rank-gold':''}" style="border-top:1px solid #1f3a52;cursor:pointer" onclick="openWeightTrainChart('${r.name}')">
                    <td style="padding:6px">${idx+1}${isBest?' 🏆':''}</td>
                    <td style="padding:6px;font-family:monospace;font-size:10px">${r.name}</td>
                    <td style="text-align:center">${r.a1w}</td><td style="text-align:center">${r.a2w}</td>
                    <td style="text-align:center">${r.a1p!=null?Number(r.a1p).toFixed(3):''}</td><td style="text-align:center">${r.a1r!=null?Number(r.a1r).toFixed(3):''}</td>
                    <td style="text-align:center">${r.a2p!=null?Number(r.a2p).toFixed(3):''}</td><td style="text-align:center">${r.a2r!=null?Number(r.a2r).toFixed(3):''}</td>
                    <td style="text-align:center;color:${Number(r.excessReturn)>=0?'#81c784':'#ef5350'}">${r.excessReturn!=null?Number(r.excessReturn).toFixed(2)+'%':'—'}</td>
                    <td style="text-align:center">${r.sharpe!=null?Number(r.sharpe).toFixed(2):'—'}</td>
                    <td style="text-align:center">${r.winRate!=null?(Number(r.winRate)*100).toFixed(1)+'%':'—'}</td>
                    <td style="text-align:center"><button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); openWeightTrainChart('${r.name}')">📈</button></td>
                </tr>`;
            });
            table.innerHTML = h+'</table>';
        }
        if(eventsEl) eventsEl.textContent = 'train cmd: '+(st.cmd||'').slice(0,300);
        if(st.running){ clearTimeout(_weightTimer); _weightTimer=setTimeout(loadWeightMatrix,5000); }
    }).catch(e=>{ if(summary) summary.textContent='load failed: '+e.message; });
};

// 增强账本刷新：缓存并联动图表
const _origMlRefresh = (typeof mlRefresh==='function') ? mlRefresh : null;
if(_origMlRefresh){
    const _wrappedMlRefresh = function(){
        return fetch('/api/matrix-ledger/list').then(r=>r.json()).then(d=>{
            const rows=(d.items||d.ledgers||d.rows||[]);
            _ledgerRowsCache=rows.slice();
            const key=document.getElementById('ledgerSort')?.value||'excess';
            renderLedgerTableAndCharts(_ledgerRowsCache, key);
            if(_origMlRefresh) return _origMlRefresh();
        }).catch(()=>{ if(_origMlRefresh) return _origMlRefresh(); });
    };
    // 不覆盖原 mlRefresh，改为在 ledger 子Tab切换时额外渲染
    window._renderLedgerCharts = ()=> renderLedgerTableAndCharts(_ledgerRowsCache, document.getElementById('ledgerSort')?.value||'excess');
}




// ═══ 通用扫参 · 任务集（选1参给范围 → 一键生成 N 组，亮标调参列，串行 train+sim，tqdm总进度，热力图+Sharpe函数曲线联动 v2/sim/图表） ═══
const SWEEP_RECOMMEND = {
  max_position_pct: '0.01,0.02,0.05,0.08',
  top_k: '10,20,30,50',
  learning_rate: '0.00005,0.0001,0.0002,0.0005',
  rebalance_days: '5,10,20,40',
  cost_bps: '5,10,15,20',
  lstm_hidden: '32,64,128',
  window: '10,20,30,60',
  a1_pos_weight: '25,40,60,80',
  a2_pos_weight: '25,40,60,80',
  batch_size: '16,32,64',
  n_steps: '16,32,64',
};
const SWEEP_TUNABLE = new Set(['max_position_pct','top_k','learning_rate','rebalance_days','cost_bps','lstm_hidden','window','a1_pos_weight','a2_pos_weight','batch_size','n_steps','target_kl','max_grad_norm','aux_lambda']);
let _sweepTaskset = null; // {paramKey, values:[], presetName, presetParams, tasks:[{idx,value,outName,status}], bestKey}
let _sweepRunning = false;

async function initSweepControls(){
  try{
    const sc=await fetch('/api/wavehunter/v2/schema').then(r=>r.json());
    const params=sc.parameters||[];
    const sel=document.getElementById('sweepParamKey');
    const presetSel=document.getElementById('sweepPresetSelect');
    if(sel){
      const opts=params.filter(p=> SWEEP_TUNABLE.has(p.key)).map(p=>{
        const rec=SWEEP_RECOMMEND[p.key] ? ` · 推荐 ${SWEEP_RECOMMEND[p.key]}` : '';
        return `<option value="${p.key}">${p.label} (${p.key}) — 默认 ${p.default}${rec}</option>`;
      }).join('');
      sel.innerHTML = opts || '<option value="max_position_pct">max_position_pct</option>';
      if(!sel.value) sel.value='max_position_pct';
      onSweepParamChange();
    }
    if(presetSel){
      const cfgs=await fetch('/api/wavehunter/v2/configs').then(r=>r.json()).then(d=>d.configs||[]).catch(()=>[]);
      presetSel.innerHTML = cfgs.map(c=>`<option value="${c.name}">${c.name}</option>`).join('') || '<option value="主升浪v2最优-40_60_10k-V8">主升浪v2最优-40_60_10k-V8</option>';
      if(cfgs.length) document.getElementById('sweepPresetName').textContent=cfgs[0].name;
    }
    // also init preset name display
    const curPreset=document.getElementById('sweepPresetSelect')?.value;
    if(curPreset) document.getElementById('sweepPresetName').textContent=curPreset;
  }catch(e){ console.error('initSweepControls', e); }
}
function onSweepParamChange(){
  const k=document.getElementById('sweepParamKey')?.value;
  const inp=document.getElementById('sweepRangeInput');
  if(!inp||!k) return;
  if(!inp.value || inp.dataset.auto!=='0'){
    inp.value = SWEEP_RECOMMEND[k] || '';
    inp.dataset.auto='1';
  }
  // hint
  const hint=document.getElementById('sweepTasksetHint');
  if(hint) hint.textContent = k ? `当前选中 ${k} · 推荐 ${SWEEP_RECOMMEND[k]||'—'} · 其余参数取预设 Other` : '';
}
function applySweepRecommend(){
  const k=document.getElementById('sweepParamKey')?.value;
  const inp=document.getElementById('sweepRangeInput');
  if(!k||!inp) return;
  inp.value = SWEEP_RECOMMEND[k] || '';
  inp.dataset.auto='1';
  addLog(`↺ 已填推荐范围 ${k}: ${inp.value}`);
}
async function loadSweepPreset(){
  const name=document.getElementById('sweepPresetSelect')?.value;
  if(!name) return;
  document.getElementById('sweepPresetName').textContent=name;
  addLog(`📥 扫参预设已切换: ${name}（Other params 取此预设）`);
}
function parseSweepRange(raw){
  if(!raw) return [];
  return raw.split(',').map(s=> s.trim()).filter(Boolean);
}
async function createSweepTaskset(){
  const paramKey=document.getElementById('sweepParamKey')?.value;
  const raw=document.getElementById('sweepRangeInput')?.value||'';
  const presetName=document.getElementById('sweepPresetSelect')?.value || '主升浪v2最优-40_60_10k-V8';
  if(!paramKey) { addLog('❌ 请先选择调参'); return; }
  const values=parseSweepRange(raw);
  if(!values.length){ addLog('❌ 范围为空'); return; }
  if(values.length>16){ addLog('❌ 单次最多16组'); return; }
  // load preset params
  let presetParams=null;
  try{
    const data=await fetch('/api/wavehunter/v2/configs').then(r=>r.json());
    const cfg=(data.configs||[]).find(c=> c.name===presetName);
    if(!cfg) throw new Error('预设不存在: '+presetName);
    presetParams=cfg.params;
  }catch(e){ addLog('❌ 载入预设失败: '+e.message); return; }
  // Build tasks: each value -> outName
  const tasks=values.map((v,i)=>{
    const safe=String(v).replace('.', 'p').replace(/[^0-9a-zA-Zp_-]/g,'');
    return {idx:i, value:v, outName:`whm_${paramKey}_${safe}_10k`, status:'pending'};
  });
  _sweepTaskset={paramKey, values, presetName, presetParams, tasks};
  // render preview table with highlight column
  renderSweepTasksetPreview();
  addLog(`➕ 已新建任务集 ${paramKey} × ${values.length}: [${values.join(', ')}] · Other 取 ${presetName}`);
  // persist for refresh
  try{ localStorage.setItem('whm_sweep_taskset', JSON.stringify(_sweepTaskset)); }catch(_){}
}
function renderSweepTasksetPreview(){
  const card=document.getElementById('sweepTasksetCard');
  const tableEl=document.getElementById('sweepTasksetTable');
  if(!card||!tableEl||!_sweepTaskset) return;
  card.style.display='';
  const {paramKey, tasks, presetName} = _sweepTaskset;
  const thBase=['#','outName','状态','超额','Sharpe','WR','P/R(若权重)','操作'];
  let h=`<div style="font-size:11px;color:#8899aa;margin-bottom:6px">调参 <b style="color:#ffd54f">${paramKey}</b> · 预设 ${presetName} · 亮标该列 — 点击行看训练曲线（复用图表Tab）</div>`;
  h+=`<table style="width:100%;border-collapse:collapse;font-size:11px"><tr style="color:#4fc3f7"><th style="padding:6px">#</th><th style="padding:6px">任务</th><th style="padding:6px;background:rgba(255,213,79,0.12);border:1px solid #ffd54f">${paramKey} ★</th><th style="padding:6px">状态</th><th style="padding:6px">超额</th><th style="padding:6px">Sharpe</th><th style="padding:6px">WR</th><th style="padding:6px">操作</th></tr>`;
  tasks.forEach((t,i)=>{
    const st=t.status;
    const badge = st==='done'?'status-done': st==='running'?'status-running': st==='failed'?'status-error':'status-idle';
    const excess = t.excess!=null? Number(t.excess).toFixed(2)+'%':'—';
    const sharpe = t.sharpe!=null? Number(t.sharpe).toFixed(2):'—';
    const wr = t.wr!=null? (Number(t.wr)*100).toFixed(1)+'%':'—';
    h+=`<tr style="border-top:1px solid #1f3a52;cursor:pointer" onclick="openWeightTrainChart('${t.outName}')">
      <td style="padding:6px;text-align:center">${i+1}</td>
      <td style="padding:6px;font-family:monospace;font-size:10px">${t.outName}</td>
      <td style="padding:6px;text-align:center;background:rgba(255,213,79,0.08);font-weight:bold;border-left:3px solid #ffd54f">${t.value}</td>
      <td style="padding:6px;text-align:center"><span class="status-badge ${badge}">${st}</span></td>
      <td style="padding:6px;text-align:center;color:${Number(t.excess)>=0?'#81c784':'#ef5350'}">${excess}</td>
      <td style="padding:6px;text-align:center">${sharpe}</td>
      <td style="padding:6px;text-align:center">${wr}</td>
      <td style="padding:6px;text-align:center"><button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); openWeightTrainChart('${t.outName}')">📈</button></td>
    </tr>`;
  });
  h+='</table>';
  tableEl.innerHTML=h;
  // also update total progress
  const done=tasks.filter(t=> t.status==='done').length;
  const total=tasks.length;
  const hint=document.getElementById('sweepTasksetHint');
  if(hint) hint.textContent = `任务集 ${paramKey}：${done}/${total} 完成 · ${presetName} Other`;
  document.getElementById('sweepTotalProgress').textContent = `${done}/${total} 完成`;
}
async function refreshSweepTaskset(){
  // restore from storage if empty
  if(!_sweepTaskset){
    try{ const raw=localStorage.getItem('whm_sweep_taskset'); if(raw) _sweepTaskset=JSON.parse(raw); }catch(_){}
  }
  if(!_sweepTaskset){ addLog('⚠ 无任务集，先新建'); return; }
  // enrich each task with ledger/model metrics if already exists
  try{
    const [mData, ledData]=await Promise.all([fetch('/api/wavehunter/v2/models').then(r=>r.json()), fetch('/api/sim/ledgers').then(r=>r.json())]);
    const ledgers=ledData.ledgers||[];
    _sweepTaskset.tasks.forEach(t=>{
      const mm=(mData.models||[]).find(x=> x.name===t.outName);
      const ld=ledgers.find(x=> x.model===t.outName);
      if(mm && mm.has_summary) t.status='done';
      else if(mm) t.status='done';
      if(ld){
        t.excess = ld.excess_return ?? ld.excess ?? null;
        t.sharpe = ld.sharpe ?? null;
        t.wr = ld.win_rate ?? null;
      }
    });
  }catch(_){}
  renderSweepTasksetPreview();
  // also refresh weight matrix visuals for this taskset
  try{
    // build rows for heatmap/trend from current taskset tasks that have ledger
    const rows=_sweepTaskset.tasks.filter(t=> t.excess!=null).map(t=> ({a1w:t.value, a2w:'', excessReturn:t.excess, sharpe:t.sharpe, winRate:t.wr, name:t.outName, a1p:null,a1r:null,a2p:null,a2r:null}));
    // reuse existing heatmap if taskset is weight-like, otherwise just update table already done
    if(_sweepTaskset.paramKey==='a1_pos_weight' || _sweepTaskset.paramKey==='a2_pos_weight'){
      // fallback to full weight matrix
      if(typeof loadWeightMatrix==='function') loadWeightMatrix();
    } else {
      // for generic param, render a 1D strip heatmap + trend over param values
      renderGenericSweepHeatmap(_sweepTaskset);
    }
  }catch(_){}
  addLog('🔄 任务集已刷新');
}
function renderGenericSweepHeatmap(taskset){
  const el=document.getElementById('weightHeatmap');
  const trendEl=document.getElementById('weightTrendChart');
  if(!el) return;
  const rows=taskset.tasks.slice();
  const vals=rows.map(r=> Number(r.excess)).filter(v=> isFinite(v));
  const vmin= vals.length? Math.min(...vals): -5, vmax= vals.length? Math.max(...vals): 5;
  // single row heatmap: columns = values
  let h='<div class="heat-head" style="grid-column: span 1"></div>';
  // header row: values
  rows.forEach(r=>{ h+=`<div class="heat-head" title="${taskset.paramKey}=${r.value}">${r.value}</div>`; });
  // need to adjust grid columns dynamically
  el.style.gridTemplateColumns = `80px repeat(${rows.length}, 1fr)`;
  // one data row
  h+=`<div class="heat-head" style="display:flex;align-items:center;justify-content:center">${taskset.paramKey}</div>`;
  rows.forEach(r=>{
    const v=r.excess!=null? Number(r.excess): null;
    const c=(typeof _excessColor==='function')? _excessColor(v, vmin, vmax): {bg:'#1a2a3a',fg:'#e0e0e0'};
    const label = v!=null? (v>0? '+'+v.toFixed(2): v.toFixed(2))+'%' : '—';
    const isBest = rows.filter(x=> x.excess!=null).sort((a,b)=> Number(b.excess)-Number(a.excess))[0]?.outName===r.outName && v!=null;
    h+=`<div class="heat-cell ${isBest?'heat-best':''}" style="background:${c.bg};color:${c.fg}" onclick="openWeightTrainChart('${r.outName}')"><div class="heat-label">${isBest?'★ '+label: label}</div><div class="heat-sub">${r.outName.slice(0,22)}</div></div>`;
  });
  el.innerHTML=h;
  // trend: param value -> excess/Sharpe dual line
  if(trendEl && window.echarts){
    const xs=rows.map(r=> String(r.value));
    const ysExcess=rows.map(r=> r.excess!=null? Number(r.excess): null);
    const ysSharpe=rows.map(r=> r.sharpe!=null? Number(r.sharpe): null);
    const chart=echarts.init(trendEl); chart.clear();
    chart.setOption({
      backgroundColor:'transparent', textStyle:{color:'#e0e0e0'},
      grid:{left:'8%',right:'8%',top:'10%',bottom:'18%'},
      tooltip:{trigger:'axis', backgroundColor:'#1a2a3a', borderColor:'#4fc3f7', textStyle:{color:'#e0e0e0'}},
      legend:{data:['超额%','Sharpe'], textStyle:{color:'#8899aa'}, top:0},
      xAxis:{type:'category', data: xs, name: taskset.paramKey, nameTextStyle:{color:'#8899aa'}, axisLabel:{color:'#8899aa'}, axisLine:{lineStyle:{color:'#2a3a4a'}}},
      yAxis:[
        {type:'value', name:'超额%', axisLabel:{color:'#81c784'}, splitLine:{lineStyle:{color:'#1a2a3a'}}},
        {type:'value', name:'Sharpe', axisLabel:{color:'#4fc3f7'}, splitLine:{show:false}}
      ],
      series:[
        {name:'超额%', type:'line', data: ysExcess, smooth:true, symbol:'circle', symbolSize:6, lineStyle:{width:2, color:'#81c784'}, yAxisIndex:0},
        {name:'Sharpe', type:'line', data: ysSharpe, smooth:true, symbol:'circle', symbolSize:6, lineStyle:{width:2, color:'#4fc3f7'}, yAxisIndex:1},
      ]
    });
    chart.resize();
  }
  // reset columns after for weight 4x4 case next
  setTimeout(()=>{ el.style.gridTemplateColumns=''; }, 0);
}
async function startSweepTaskset(){
  if(_sweepRunning){ addLog('⚠ 任务集已在运行'); return; }
  if(!_sweepTaskset){ addLog('❌ 请先新建任务集'); return; }
  _sweepRunning=true;
  document.getElementById('sweepTasksetBadge').textContent='运行中';
  document.getElementById('sweepTasksetBadge').className='status-badge status-running';
  const total=_sweepTaskset.tasks.length;
  for(let i=0;i<total;i++){
    const t=_sweepTaskset.tasks[i];
    // tqdm-style total progress log
    addLog(`⏳ [tqdm ${i+1}/${total}] ${t.outName} ${t.value} — 总进度 ${i}/${total}`);
    document.getElementById('sweepTotalProgress').textContent=`${i+1}/${total} ${t.outName}`;
    // check if already done (has model)
    try{
      const mData=await fetch('/api/wavehunter/v2/models').then(r=>r.json());
      if((mData.models||[]).some(m=> m.name===t.outName && m.has_ppo)){
        t.status='done'; renderSweepTasksetPreview(); addLog(`⏭ 已有模型跳过 ${t.outName}`); continue;
      }
    }catch(_){}
    // build body from preset + override one param
    let body=null;
    try{
      const preset=_sweepTaskset.presetParams;
      body={...preset, out_dir: t.outName};
      // coerce type per schema: numbers vs strings
      const rawVal=t.value;
      const numVal=Number(rawVal);
      body[_sweepTaskset.paramKey]= isNaN(numVal)? rawVal : numVal;
      // ensure required panel/start/end exist
      if(!body.panel) body.panel='wavehunter_v8_hs300_20040102_20260827.parquet';
      if(!body.start_date) body.start_date='2023-01-01';
      if(!body.end_date) body.end_date='2023-03-31';
      if(!body.total_timesteps) body.total_timesteps=10000;
    }catch(e){ addLog(`❌ 任务 ${t.outName} 参数组装失败: ${e.message}`); t.status='failed'; continue; }
    // start train via v2 API (single-instance lock)
    try{
      const d=await fetch('/api/wavehunter/v2/train/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=> r.ok? r.json(): r.json().then(e=>{throw new Error(e.detail||e.error||'HTTP '+r.status)}));
      addLog(`🚀 训练已启动 [${i+1}/${total}] ${d.out_name} ${_sweepTaskset.paramKey}=${t.value} — tqdm ${i+1}/${total}`);
    }catch(e){ addLog(`❌ 启动失败 ${t.outName}: ${e.message}`); t.status='failed'; renderSweepTasksetPreview(); continue; }
    t.status='running'; renderSweepTasksetPreview();
    // poll train until finished
    await new Promise(res=>{
      const tid=setInterval(async()=>{
        try{
          const s=await fetch('/api/wavehunter/v2/train/status').then(r=>r.json());
          document.getElementById('sweepTotalProgress').textContent=`${i+1}/${total} ${s.current||0}/${s.total||''} ${s.percent||0}%`;
          if(!s.running){ clearInterval(tid); res(); }
        }catch(_){}
      },4000);
    });
    // after train, auto sim 2023-04 OOS
    try{
      const simBody={model_dir: t.outName, panel: body.panel, start_date:'2023-04-01', end_date:'2023-04-30', initial_capital:100000, cost_bps: body.cost_bps ?? 15, slippage_bps:0, stop_loss_pct:15.0, max_position_pct: body.max_position_pct ?? 0.02, rebalance_days: Math.min(5, Number(body.rebalance_days)||5), top_k: body.top_k ?? 20, max_holding_days:120};
      const sd=await fetch('/api/sim/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(simBody)}).then(r=> r.ok? r.json(): r.json().then(e=>{throw new Error(e.detail||'HTTP '+r.status)}));
      addLog(`📊 模拟已启动 ${t.outName} → ${sd.task_id} (tqdm ${i+1}/${total})`);
      await new Promise(res=>{
        const tid=setInterval(async()=>{
          const s=await fetch('/api/sim/status').then(r=>r.json()).catch(()=>({running:false}));
          if(!s.running){ clearInterval(tid); res(); }
        },3000);
      });
      const res=await fetch('/api/sim/result').then(r=>r.json()).catch(()=>({}));
      const m=res.metrics||{};
      t.excess = m.excess_return ?? null;
      t.sharpe = m.sharpe ?? m.sharpe_ratio ?? null;
      t.wr = m.win_rate ?? null;
      t.status='done';
      addLog(`✅ 完成 [${i+1}/${total}] ${t.outName} 超额 ${t.excess!=null? Number(t.excess).toFixed(2)+'%':'—'} Sharpe ${t.sharpe!=null? Number(t.sharpe).toFixed(2):'—'} — tqdm ${i+1}/${total} done`);
    }catch(e){
      addLog(`⚠ 模拟失败 ${t.outName}: ${e.message}`);
      t.status='done';
    }
    renderSweepTasksetPreview();
    try{ localStorage.setItem('whm_sweep_taskset', JSON.stringify(_sweepTaskset)); }catch(_){}
    // refresh heatmap/trend
    try{ renderGenericSweepHeatmap(_sweepTaskset); }catch(_){}
  }
  _sweepRunning=false;
  document.getElementById('sweepTasksetBadge').textContent='完成';
  document.getElementById('sweepTasksetBadge').className='status-badge status-done';
  document.getElementById('sweepTotalProgress').textContent=`${total}/${total} 完成`;
  addLog(`🎉 任务集全部完成 ${total}/${total} — 热力图与 参数→Sharpe 曲线已刷新，可在图表Tab联动查看训练曲线`);
}
function stopSweepTaskset(){
  _sweepRunning=false;
  fetch('/api/wavehunter/v2/train/stop',{method:'POST'}).catch(()=>{});
  fetch('/api/sim/stop',{method:'POST'}).catch(()=>{});
  document.getElementById('sweepTasksetBadge').textContent='已停止';
  document.getElementById('sweepTasksetBadge').className='status-badge status-idle';
  addLog('⏹ 任务集已停止');
}
// auto-init on loadWeightMatrix
const _prevInitSweep = (typeof initSweepControls==='function')? null : null;
setTimeout(()=>{ try{ initSweepControls(); }catch(_){} }, 600);
