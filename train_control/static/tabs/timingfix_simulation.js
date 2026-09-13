// v10_timingfix 训练/模拟 Tab — 前端只提交参数、轮询持久化状态、显示真实业务结果。
(function () {
  const $ = id => document.getElementById(id);
  const log = (msg, source) => {
    try { window.addLog && window.addLog(msg, source || 'timingfix'); }
    catch (_) { const box = $('timingfixLog'); if (box) box.textContent += msg + '\n'; }
  };

  // ── 通用: 防止重复提交 ──
  const _busy = new Set();
  function _enter(label, fn) {
    if (_busy.has(label)) return;
    _busy.add(label);
    return Promise.resolve().then(fn).finally(() => _busy.delete(label));
  }

  // ── 训练 ──
  function _trainPayload() {
    return {
      panel: $('tfTrainPanel').value,
      start_date: $('tfTrainStart').value,
      end_date: $('tfTrainEnd').value,
      out_dir: $('tfTrainOut').value,
      total_timesteps: Number($('tfTrainSteps').value),
      seed: Number($('tfTrainSeed').value),
      window: Number($('tfTrainWindow').value),
      top_k: Number($('tfTrainTopK').value),
      max_position_pct: Number($('tfTrainMaxPos').value),
      cost_bps: Number($('tfTrainCost').value),
      slippage_bps: Number($('tfTrainSlippage').value),
      stop_loss_pct: Number($('tfTrainStop').value),
      cooldown_days: Number($('tfTrainCooldown').value),
      lstm_hidden: Number($('tfTrainLstmHidden').value),
      lstm_layers: Number($('tfTrainLstmLayers').value),
      n_steps: Number($('tfTrainNSteps').value),
      batch_size: Number($('tfTrainBatchSize').value),
      learning_rate: Number($('tfTrainLr').value),
    };
  }

  function _renderTrainingStatus(d) {
    const status = $('tfTrainStatus');
    if (status) {
      const running = !!d.running;
      const phase = d.status || 'idle';
      status.textContent = running ? `${phase} · running` : phase;
      status.className = 'status-badge ' + (running ? 'status-running'
        : (phase === 'failed' ? 'status-failed'
        : (phase === 'finished' || phase === 'done' ? 'status-done' : 'status-idle')));
    }
    const btn = $('tfTrainBtn'); if (btn) btn.disabled = !!d.running || _busy.has('train_start');
    const stopBtn = $('tfTrainStopBtn'); if (stopBtn) stopBtn.disabled = !d.running;
    if (d.error) log('❌ 训练错误: ' + d.error, 'timingfix');
  }

  let _trainTaskId = '';
  function _pollTraining() {
    const url = _trainTaskId
      ? '/api/timingfix/training/status?task_id=' + encodeURIComponent(_trainTaskId)
      : '/api/timingfix/training/status';
    apiFetch(url).then(d => {
      if (d.task_id) _trainTaskId = d.task_id;
      _renderTrainingStatus(d);
      if (d.running) setTimeout(_pollTraining, 2000);
    }).catch(e => {
      log('❌ 训练状态读取失败: ' + e.message, 'timingfix');
      setTimeout(_pollTraining, 5000);
    });
  }

  function _pollTrainingLog() {
    const url = _trainTaskId
      ? '/api/timingfix/training/log?task_id=' + encodeURIComponent(_trainTaskId)
      : '/api/timingfix/training/log';
    apiFetch(url).then(d => {
      const box = $('tfTrainLog');
      if (box) box.textContent = (d.lines || []).slice(-200).join('\n');
      if ((_trainTaskId && d.status && d.status.running) || (d.status && d.status.running)) {
        setTimeout(_pollTrainingLog, 3000);
      }
    }).catch(() => setTimeout(_pollTrainingLog, 5000));
  }

  window.startTimingFixTraining = function () {
    return _enter('train_start', () => {
      const p = _trainPayload();
      if (!p.panel) return log('❌ 请选择训练面板', 'timingfix');
      if (!p.out_dir) return log('❌ 请填写训练输出目录', 'timingfix');
      if (!p.start_date || !p.end_date) return log('❌ 请填写训练起止日期', 'timingfix');
      const btn = $('tfTrainBtn'); if (btn) btn.disabled = true;
      log('🚀 TimingFix 训练提交: ' + p.out_dir, 'timingfix');
      apiFetch('/api/timingfix/training/start', {method: 'POST', body: p})
        .then(d => {
          if (!d.task_id) throw new Error('训练响应缺少 task_id');
          _trainTaskId = d.task_id;
          log('✅ 训练已提交 task_id=' + d.task_id, 'timingfix');
          _renderTrainingStatus(d);
          _pollTraining();
          _pollTrainingLog();
        })
        .catch(e => {
          log('❌ 训练启动失败: ' + e.message, 'timingfix');
          if (btn) btn.disabled = false;
        });
    });
  };

  window.stopTimingFixTraining = function () {
    if (!_trainTaskId) return log('❌ 没有运行中的训练任务', 'timingfix');
    apiFetch('/api/timingfix/training/stop?task_id=' + encodeURIComponent(_trainTaskId), {method: 'POST'})
      .then(d => log('⏹ 训练停止请求已发送: ' + d.status, 'timingfix'))
      .catch(e => log('❌ 训练停止失败: ' + e.message, 'timingfix'));
  };

    // v10.1 independent training/smoke card uses the visible TimingFix area.
    let _v101TrainTaskId = '';
    function _v101Payload(){
      return {
        panel: $('tfTrainPanel')?.value || '', start_date: $('tfTrainStart')?.value || '2023-01-01',
        end_date: $('tfTrainEnd')?.value || '2023-12-31', out_dir: $('tfTrainOut')?.value || 'models/whv10_1_csi500_smoke_100',
        total_timesteps: Number($('tfTrainSteps')?.value || 100), seed: Number($('tfTrainSeed')?.value || 42),
        window: Number($('tfTrainWindow')?.value || 20), top_k: Number($('tfTrainTopK')?.value || 20),
        max_position_pct: Number($('tfTrainMaxPos')?.value || 0.02), cost_bps: Number($('tfTrainCost')?.value || 15),
        stop_loss_pct: Number($('tfTrainStop')?.value || 0), cooldown_days: Number($('tfTrainCooldown')?.value || 5),
        lstm_hidden: Number($('tfTrainLstmHidden')?.value || 64), lstm_layers: Number($('tfTrainLstmLayers')?.value || 1),
        n_steps: Math.min(32, Number($('tfTrainNSteps')?.value || 32)), batch_size: Math.min(32, Number($('tfTrainBatchSize')?.value || 32)),
        learning_rate: Number($('tfTrainLr')?.value || 0.00003)
      };
    }
    function _v101Poll(){
      const u=_v101TrainTaskId?'/api/v10_1/training/status?task_id='+encodeURIComponent(_v101TrainTaskId):'/api/v10_1/training/status';
      apiFetch(u).then(d=>{if(d.task_id)_v101TrainTaskId=d.task_id; if($('tfTrainStatus')){$('tfTrainStatus').textContent=(d.running?'running · ':'')+(d.status||'idle');$('tfTrainStatus').className='status-badge '+(d.running?'status-running':(d.status==='done'?'status-done':(d.status==='failed'?'status-failed':'status-idle')))} if(d.running)setTimeout(_v101Poll,2000)}).catch(()=>setTimeout(_v101Poll,5000));
    }
    window.startV10_1Training=function(){const p=_v101Payload();if(!p.panel.startsWith('wavehunter_v10_1_'))return log('❌ v10.1 训练必须选择 v10.1 面板','timingfix');p.out_dir=p.out_dir.startsWith('models/whv10_1_')?p.out_dir:'models/whv10_1_'+p.out_dir.replace(/^models\//,'');log('🚀 v10.1 训练提交: '+p.out_dir,'v10.1');apiFetch('/api/v10_1/training/start',{method:'POST',body:p}).then(d=>{_v101TrainTaskId=d.task_id;log('✅ v10.1 task_id='+d.task_id,'v10.1');_v101Poll()}).catch(e=>log('❌ v10.1 训练失败: '+e.message,'v10.1'))};
    window.stopV10_1Training=function(){if(!_v101TrainTaskId)return;apiFetch('/api/v10_1/training/stop?task_id='+encodeURIComponent(_v101TrainTaskId),{method:'POST'}).then(d=>log('⏹ v10.1 stop='+d.status,'v10.1'))};

  function _simPayload() {
    return {
      model_dir: $('tfModel').value,
      panel: $('tfPanel').value,
      initial_capital: Number($('tfCapital').value),
      start_date: $('tfStart').value,
      end_date: $('tfEnd').value,
      top_k: Number($('tfTopK').value),
      cost_bps: Number($('tfCost').value),
      slippage_bps: Number($('tfSlippage').value),
      stop_loss_pct: Number($('tfStop').value),
      cooldown_days: Number($('tfCooldown').value),
      a1_threshold: Number($('tfA1').value),
      a2_threshold: Number($('tfA2').value),
      peak_threshold: Number($('tfPeak').value),
      b1_threshold: Number($('tfB1').value),
    };
  }

  function _renderSimulationStatus(d) {
    const status = $('tfStatus');
    if (status) {
      const running = !!d.running;
      const phase = d.status || 'idle';
      status.textContent = running ? `${phase} · running` : phase;
      status.className = 'status-badge ' + (running ? 'status-running'
        : (phase === 'failed' ? 'status-failed'
        : (phase === 'finished' || phase === 'done' ? 'status-done' : 'status-idle')));
    }
    const result = $('tfResult');
    if (result) {
      if (d.metrics) result.textContent = JSON.stringify(d.metrics, null, 2);
      else if (d.error) result.textContent = '❌ ' + d.error;
      else result.textContent = d.ledger_file || '—';
    }
    const ledger = $('tfLedger');
    if (ledger && d.ledger_file) ledger.textContent = d.ledger_file;
    const btn = $('tfStartBtn'); if (btn) btn.disabled = !!d.running || _busy.has('sim_start');
    const stopBtn = $('tfStopBtn'); if (stopBtn) stopBtn.disabled = !d.running;
    if (d.error) log('❌ 模拟错误: ' + d.error, 'timingfix');
  }

  let _simTaskId = '';
  function _pollSimulation() {
    const url = _simTaskId
      ? '/api/timingfix/simulation/status?task_id=' + encodeURIComponent(_simTaskId)
      : '/api/timingfix/simulation/status';
    apiFetch(url).then(d => {
      if (d.task_id) _simTaskId = d.task_id;
      _renderSimulationStatus(d);
      if (d.running) setTimeout(_pollSimulation, 2000);
    }).catch(e => {
      log('❌ 模拟状态读取失败: ' + e.message, 'timingfix');
      setTimeout(_pollSimulation, 5000);
    });
  }

  function _pollSimulationLog() {
    const url = _simTaskId
      ? '/api/timingfix/simulation/log?task_id=' + encodeURIComponent(_simTaskId)
      : '/api/timingfix/simulation/log';
    apiFetch(url).then(d => {
      const box = $('timingfixLog');
      if (box) box.textContent = (d.lines || []).slice(-200).join('\n');
      if (d.status && d.status.running) setTimeout(_pollSimulationLog, 3000);
    }).catch(() => setTimeout(_pollSimulationLog, 5000));
  }

  window.startTimingFixSimulation = function () {
    return _enter('sim_start', () => {
      const p = _simPayload();
      if (!p.model_dir) return log('❌ 请选择 timingfix 模型', 'timingfix');
      if (!p.panel) return log('❌ 请选择模拟面板', 'timingfix');
      const btn = $('tfStartBtn'); if (btn) btn.disabled = true;
      log('🚀 v10_timingfix 模拟提交: ' + p.model_dir, 'timingfix');
      apiFetch('/api/timingfix/simulation/start', {method: 'POST', body: p})
        .then(d => {
          if (!d.task_id) throw new Error('模拟响应缺少 task_id');
          _simTaskId = d.task_id;
          log('✅ 模拟已提交 task_id=' + d.task_id, 'timingfix');
          _renderSimulationStatus(d);
          _pollSimulation();
          _pollSimulationLog();
        })
        .catch(e => {
          log('❌ 模拟启动失败: ' + e.message, 'timingfix');
          if (btn) btn.disabled = false;
        });
    });
  };

  window.stopTimingFixSimulation = function () {
    if (!_simTaskId) return log('❌ 没有运行中的模拟任务', 'timingfix');
    apiFetch('/api/timingfix/simulation/stop?task_id=' + encodeURIComponent(_simTaskId), {method: 'POST'})
      .then(d => log('⏹ 模拟停止请求已发送: ' + d.status, 'timingfix'))
      .catch(e => log('❌ 模拟停止失败: ' + e.message, 'timingfix'));
  };

  // ── 加载模型/面板下拉选项 ──
  window.loadTimingFixOptions = function () {
    Promise.all([
      apiFetch('/api/train/models').catch(() => ({models: []})),
      apiFetch('/api/panels').catch(() => ({panels: []})),
    ]).then(([models, panels]) => {
      const m = $('tfModel'), p = $('tfPanel');
      if (m) m.innerHTML = '<option value="">-- 选择 timingfix 模型 --</option>' +
        (models.models || []).filter(x => x.name && x.name.includes('timingfix') && x.has_onnx && x.has_final_zip)
          .map(x => `<option value="${x.name}">${x.name}</option>`).join('');
      if (p) p.innerHTML = '<option value="">-- 选择面板 --</option>' +
        (panels.panels || []).filter(x => x.name && x.name.includes('wavehunter_v10'))
          .map(x => `<option value="${x.name}">${x.name}</option>`).join('');
      const tp = $('tfTrainPanel');
      if (tp) tp.innerHTML = (panels.panels || [])
        .filter(x => x.name && x.name.includes('wavehunter_v10'))
        .map(x => `<option value="${x.name}">${x.name}</option>`).join('');
      _pollSimulation();
      _pollTraining();
    }).catch(e => log('❌ 加载 timingfix 选项失败: ' + e.message, 'timingfix'));
  };
})();
