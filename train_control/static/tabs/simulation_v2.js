// 模拟交易 v2 Tab：只调用 /api/sim-v2/*，不触碰旧模拟交易 Tab。
(function () {
  function el(id) { return document.getElementById(id); }
  function log(msg) {
    if (typeof addLog === 'function') addLog(msg, 'simulation_v2');
    const box = el('simV2Log'); if (box) box.textContent += msg + '\n';
  }
  window.startSimulationV2 = function () {
    const model = el('simV2Model').value;
    if (!model) return log('❌ 请选择 v10 模型');
    const payload = {
      model_dir: model, panel: el('simV2Panel').value,
      initial_capital: Number(el('simV2Capital').value),
      start_date: el('simV2Start').value, end_date: el('simV2End').value,
      top_k: Number(el('simV2TopK').value), rebalance_days: Number(el('simV2Rebalance').value),
      cost_bps: Number(el('simV2Cost').value), slippage_bps: Number(el('simV2Slippage').value),
      stop_loss_pct: Number(el('simV2Stop').value), take_profit_pct: Number(el('simV2Take').value),
      a1_threshold: Number(el('simV2A1').value), a2_threshold: Number(el('simV2A2').value),
      peak_threshold: Number(el('simV2Peak').value), b1_threshold: Number(el('simV2B1').value),
      min_buy_score: Number(el('simV2MinScore').value)
    };
    const btn = el('simV2StartBtn'); btn.disabled = true; log('🚀 simulation-v2 提交: ' + model);
    fetch('/api/sim-v2/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})
      .then(async r => { const d=await r.json(); if(!r.ok) throw Error(d.detail||'HTTP '+r.status); return d; })
      .then(d => { log('✅ 已提交 task_id='+d.task_id); pollSimulationV2(); })
      .catch(e => { log('❌ v2 启动失败: '+e.message); btn.disabled=false; });
  };
  window.stopSimulationV2 = function () { fetch('/api/sim-v2/stop',{method:'POST'}).then(()=>log('⏹ v2 停止请求已发送')); };
  function pollSimulationV2() {
    fetch('/api/sim-v2/status').then(r=>r.json()).then(d=>{
      if (d.log_path) fetch('/api/sim-v2/log').then(r=>r.json()).then(x=>{ const box=el('simV2Log'); if(box) box.textContent=(x.lines||[]).join('\n'); });
      if (d.running) setTimeout(pollSimulationV2, 2000);
      else { const btn=el('simV2StartBtn'); if(btn) btn.disabled=false; log('🏁 v2 状态: '+d.status); }
    }).catch(e=>{log('❌ v2 状态失败: '+e.message); setTimeout(pollSimulationV2,5000);});
  }
  window.loadSimulationV2Models = function () {
    fetch('/api/wavehunter/v2/models').then(r=>r.json()).then(d=>{
      const s=el('simV2Model'); if(!s)return;
      s.innerHTML='<option value="">-- 选择 v10 模型 --</option>'+(d.models||[]).filter(x=>x.name&&x.name.includes('v10')).map(x=>`<option value="${x.name}">${x.name}</option>`).join('');
    });
    fetch('/api/panels').then(r=>r.json()).then(d=>{const s=el('simV2Panel');if(s)s.innerHTML=(d.panels||[]).filter(x=>x.name.includes('wavehunter_v10') || x.name.includes('wavehunter_v9')).map(x=>`<option value="${x.name}">${x.name}</option>`).join('');});
  };
})();
