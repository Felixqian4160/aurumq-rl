// v10.1 独立模拟入口，使用独立 /api/v10_1/simulation/*。
(function(){
  const $=id=>document.getElementById(id);
  function log(x){if(window.addLog)window.addLog(x,'v10.1');const b=$('simV2Log');if(b)b.textContent+=x+'\n';}
  function payload(){return {model_dir:$('simV2Model')?.value||'',panel:$('simV2Panel')?.value||'',initial_capital:Number($('simV2Capital')?.value||100000),start_date:$('simV2Start')?.value||'2025-01-01',end_date:$('simV2End')?.value||'2025-12-31',top_k:Number($('simV2TopK')?.value||20),rebalance_days:Number($('simV2Rebalance')?.value||20),cost_bps:Number($('simV2Cost')?.value||15),slippage_bps:Number($('simV2Slippage')?.value||10),max_position_pct:.05,min_buy_score:Number($('simV2MinScore')?.value||.55),calibration_start_date:'2023-01-01',calibration_end_date:'2023-12-29',a1_threshold:.55,a2_threshold:.55,peak_threshold:.5,b1_threshold:.55,portfolio_audit_version:'v10.1-sim-v3'}}
  let task='';
  window.startV10_1Simulation=function(){const p=payload();if(!p.model_dir)return log('❌ 请选择 whv10_1 模型');if(!p.panel.startsWith('wavehunter_v10_1_'))return log('❌ 请选择 v10.1 面板');const b=$('simV2StartBtn');if(b)b.disabled=true;log('🚀 v10.1 模拟提交');apiFetch('/api/v10_1/simulation/start',{method:'POST',body:p}).then(d=>{task=d.task_id;log('✅ task_id='+task);poll()}).catch(e=>{log('❌ '+e.message);if(b)b.disabled=false})}
  function poll(){apiFetch('/api/v10_1/simulation/status?task_id='+encodeURIComponent(task)).then(d=>{if(d.running)setTimeout(poll,2000);else{log('🏁 v10.1 状态='+d.status);if(d.metrics)log(JSON.stringify(d.metrics));const b=$('simV2StartBtn');if(b)b.disabled=false}}).catch(e=>{log('❌ 状态失败 '+e.message);setTimeout(poll,5000)})}
  window.stopV10_1Simulation=function(){if(task)apiFetch('/api/v10_1/simulation/stop?task_id='+encodeURIComponent(task),{method:'POST'}).then(d=>log('⏹ '+d.status))};
})();
