// tabs/p22c.js — 🌊 Phase 22C 主升浪 (P22C 已隐藏，仅保留代码)
// 从 index.html inline JS 抽离；依赖全局 fetch/addLog/toast
// Phase 22C 主升浪
// ══════════════════════════════════════════════════

function loadP22cPanels() {
    fetch('/api/train/panels').then(r=>r.json()).then(d => {
        const sel = document.getElementById('p22cPanel');
        sel.innerHTML = (d.panels || []).map(p =>
            `<option value="${p.name}" ${p.name.includes('sse50') ? 'selected' : ''}>${p.name} (${p.size_gb} GB)</option>`
        ).join('');
    });
}

function loadP22cRuns() {
    // 用统一模型列表渲染已训练模型区
    loadModels('p22cRunList');
    // 评估下拉框单独用 /api/p22c/runs
    fetch('/api/p22c/runs').then(r=>r.json()).then(d => {
        const sel = document.getElementById('p22cEvalRun');
        if (!d.runs || d.runs.length === 0) {
            sel.innerHTML = '<option value="">-- 无 Runs --</option>';
            return;
        }
        sel.innerHTML = '<option value="">-- 选择 Run --</option>' +
            d.runs.map(m => `<option value="${m.name}">${m.name}</option>`).join('');
    }).catch(() => {});
}

function cleanP22cCheckpoints(name, btn) {
    if (!confirm(`确定清理 ${name} 的 Checkpoints？\n\n训练过程保存的中间检查点将被删除。\nRun 本身（ppo_final.zip / 指标）会保留。`)) return;
    btn.disabled = true; btn.textContent = '⏳';
    fetch(`/api/p22c/runs/${encodeURIComponent(name)}/clean-checkpoints`, { method: 'POST' })
    .then(r => r.json())
    .then(d => {
        if (d.status === 'cleaned') {
            addLog(`🧹 已清理 ${name} Checkpoints: ${d.cleaned_mb} MB 释放`);
            loadP22cRuns();
        } else if (d.status === 'already_clean') {
            addLog(`✅ ${name} 已是干净状态`);
            loadP22cRuns();
        } else {
            throw new Error(d.detail || '清理失败');
        }
    })
    .catch(e => {
        btn.disabled = false; btn.textContent = '🧹 清理';
        addLog('❌ ' + e.message);
    });
}

function deleteP22cRun(name, btn) {
    if (!confirm(`确定删除 Run ${name}？\n\n该目录所有文件（含 checkpoints / 评估结果）将被永久删除！`)) return;
    if (btn) { btn.disabled = true; btn.textContent = '⏳'; }
    fetch(`/api/p22c/runs/${encodeURIComponent(name)}`, { method: 'DELETE' })
    .then(r => r.json())
    .then(d => {
        if (d.status === 'deleted') {
            addLog(`🗑 已删除 Run ${name} (${d.size_mb} MB)`);
            loadP22cRuns();
            if (typeof loadMLModels === 'function') loadMLModels();
        } else {
            throw new Error(d.detail || '删除失败');
        }
    })
    .catch(e => {
        if (btn) { btn.disabled = false; btn.textContent = '🗑'; }
        addLog('❌ 删除 Run 失败: ' + e.message);
    });
}

let _p22cLastTrainRunning = false;
let _p22cLastEvalRunning = false;

function pollP22cTrainStatus() {
    fetch('/api/p22c/train/status').then(r=>r.json()).then(d => {
        const badge = document.getElementById('p22cTrainBadge');
        const btnStart = document.getElementById('btnP22cTrainStart');
        const btnStop = document.getElementById('btnP22cTrainStop');
        if (d.running) {
            badge.className = 'status-badge status-running';
            badge.textContent = '运行中';
            btnStart.disabled = true;
            btnStop.disabled = false;
            document.getElementById('p22cTrainProgress').textContent = '⏳ 训练进行中，查看日志...';
            _p22cLastTrainRunning = true;
        } else {
            badge.className = 'status-badge status-idle';
            badge.textContent = '空闲';
            btnStart.disabled = false;
            btnStop.disabled = true;
            document.getElementById('p22cTrainProgress').textContent = '';
            // 训练刚结束：刷新模型列表 + P22C runs，等2秒让 finish log 到达
            if (_p22cLastTrainRunning) {
                _p22cLastTrainRunning = false;
                setTimeout(() => {
                    loadP22cRuns();
                    refreshChartModels();
                    loadSimModels();
                    addLog('📂 训练完成，已刷新模型列表');
                }, 2000);
            }
        }
    }).catch(() => {});
}
setInterval(pollP22cTrainStatus, 2000);
pollP22cTrainStatus();

function pollP22cEvalStatus() {
    fetch('/api/p22c/eval/status').then(r=>r.json()).then(d => {
        const badge = document.getElementById('p22cEvalBadge');
        const btnStart = document.getElementById('btnP22cEvalStart');
        const btnStop = document.getElementById('btnP22cEvalStop');
        if (d.running) {
            badge.className = 'status-badge status-running';
            badge.textContent = '评估中';
            btnStart.disabled = true;
            btnStop.disabled = false;
            _p22cLastEvalRunning = true;
        } else {
            badge.className = 'status-badge status-idle';
            badge.textContent = '空闲';
            btnStart.disabled = false;
            btnStop.disabled = true;
            if (_p22cLastEvalRunning) {
                _p22cLastEvalRunning = false;
            }
        }
    }).catch(() => {});
}
setInterval(pollP22cEvalStatus, 2000);
pollP22cEvalStatus();

function startP22cTrain() {
    const body = getP22cParams();
    fetch('/api/p22c/train/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    }).then(r => {
        if (!r.ok) return r.json().then(e => { throw new Error(e.detail); });
        addLog(`🌊 Phase 22C 训练已提交 (seed=${body.seed})`);
        loadP22cRuns();
    }).catch(e => addLog('❌ ' + e.message));
}

function stopP22cTrain() {
    fetch('/api/p22c/train/stop', {method: 'POST'})
        .then(() => addLog('⏹ 正在停止 Phase 22C 训练...'))
        .catch(e => addLog('❌ ' + e.message));
}

function startP22cEval() {
    const runName = document.getElementById('p22cEvalRun').value;
    if (!runName) { addLog('❌ 请选择一个 Run'); return; }
    const body = {
        run_dir: 'runs/' + runName,
        val_start: document.getElementById('p22cValStart').value,
        val_end: document.getElementById('p22cValEnd').value,
        top_k: parseInt(document.getElementById('p22cEvalTopK').value) || 5,
    };
    fetch('/api/p22c/eval/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    }).then(r => {
        if (!r.ok) return r.json().then(e => { throw new Error(e.detail); });
        addLog('📊 Phase 22C 评估已提交: ' + runName);
        // 轮询评估结果
        setTimeout(pollP22cEvalResult, 5000);
    }).catch(e => addLog('❌ ' + e.message));
}

function stopP22cEval() {
    fetch('/api/p22c/eval/stop', {method: 'POST'})
        .then(() => addLog('⏹ 正在停止评估...'))
        .catch(e => addLog('❌ ' + e.message));
}

function pollP22cEvalResult() {
    fetch('/api/p22c/eval/status').then(r=>r.json()).then(d => {
        if (d.running) {
            setTimeout(pollP22cEvalResult, 3000);
            return;
        }
        setTimeout(() => {
            fetch('/api/p22c/eval/result').then(r=>r.json()).then(d => {
                const rows = d.rows || d.eval_results || [];
                if (rows.length > 0) {
                    renderP22cEvalResults({rows});
                    addLog('📊 评估结果已加载');
                    loadP22cRuns();
                } else if (d.status === 'failed' || d.error) {
                    addLog('❌ 评估失败: ' + (d.error || d.status));
                } else {
                    addLog('⚠ 评估结束但没有结果数据');
                }
            }).catch(e => addLog('❌ 读取评估结果失败: ' + e.message));
        }, 2000);
    }).catch(e => addLog('❌ 读取评估状态失败: ' + e.message));
}

function renderP22cEvalResults(data) {
    const rows = data.rows || [];
    if (rows.length === 0) return;

    // 取最佳结果（最高 eval_score）
    const best = rows.reduce((a, b) => (a.eval_score || 0) > (b.eval_score || 0) ? a : b);

    // 填 KPI
    document.getElementById('p22cEvalResults').style.display = 'block';
    document.getElementById('p22cKpiHit').textContent = ((best.main_wave_hit_rate || 0) * 100).toFixed(2) + '%';
    document.getElementById('p22cKpiWin').textContent = ((best.basic_win_rate || 0) * 100).toFixed(1) + '%';
    const hold = (best.avg_hold_return || 0) * 100;
    document.getElementById('p22cKpiHold').textContent = (hold >= 0 ? '+' : '') + hold.toFixed(2) + '%';
    document.getElementById('p22cKpiHold').style.color = hold >= 0 ? '#f44336' : '#4caf50';
    document.getElementById('p22cKpiDD').textContent = ((best.avg_max_drawdown || 0) * 100).toFixed(2) + '%';
    document.getElementById('p22cKpiPayoff').textContent = (best.payoff_ratio || 0).toFixed(2);
    document.getElementById('p22cKpiScore').textContent = (best.eval_score || 0).toFixed(4);

    // 渲染图表
    document.getElementById('p22cEvalChartContainer').style.display = 'block';

    const theme = {
        backgroundColor: 'transparent',
        textStyle: { color: '#e0e0e0' },
        grid: { left: '12%', right: '8%', top: '14%', bottom: '16%' },
        tooltip: { trigger: 'axis', backgroundColor: '#1a2a3a', borderColor: '#4fc3f7', textStyle: { color: '#e0e0e0' } },
    };

    // 图1: 命中率 vs 胜率
    const topKs = rows.map(r => 'Top-' + r.top_k);
    const hitRates = rows.map(r => (r.main_wave_hit_rate || 0) * 100);
    const winRates = rows.map(r => (r.basic_win_rate || 0) * 100);
    const c1 = echarts.init(document.getElementById('p22cChartHitWin'));
    c1.setOption({
        ...theme,
        legend: { data: ['命中率%', '胜率%'], textStyle: { color: '#e0e0e0' }, top: 0 },
        xAxis: { type: 'category', data: topKs, axisLabel: { color: '#8899aa' }, axisLine: { lineStyle: { color: '#2a3a4a' } } },
        yAxis: { type: 'value', name: '%', axisLabel: { color: '#8899aa' }, splitLine: { lineStyle: { color: '#1a2a3a' } } },
        series: [
            { name: '命中率%', type: 'bar', data: hitRates, itemStyle: { color: '#4fc3f7' }, barWidth: '30%' },
            { name: '胜率%', type: 'bar', data: winRates, itemStyle: { color: '#f44336' }, barWidth: '30%' },
        ],
    });
    c1.resize();

    // 图2: 持仓收益 vs 最大回撤
    const holdRets = rows.map(r => (r.avg_hold_return || 0) * 100);
    const drawdowns = rows.map(r => (r.avg_max_drawdown || 0) * 100);
    const c2 = echarts.init(document.getElementById('p22cChartRetDD'));
    c2.setOption({
        ...theme,
        legend: { data: ['平均持仓收益%', '平均最大回撤%'], textStyle: { color: '#e0e0e0' }, top: 0 },
        xAxis: { type: 'category', data: topKs, axisLabel: { color: '#8899aa' }, axisLine: { lineStyle: { color: '#2a3a4a' } } },
        yAxis: { type: 'value', name: '%', axisLabel: { color: '#8899aa' }, splitLine: { lineStyle: { color: '#1a2a3a' } } },
        series: [
            { name: '平均持仓收益%', type: 'bar', data: holdRets, itemStyle: { color: holdRets.some(v => v >= 0) ? '#f44336' : '#4caf50' }, barWidth: '30%' },
            { name: '平均最大回撤%', type: 'bar', data: drawdowns.map(v => -Math.abs(v)), itemStyle: { color: '#4caf50' }, barWidth: '30%' },
        ],
    });
    c2.resize();

    // 图3: 综合评分
    const scores = rows.map(r => r.eval_score || 0);
    const c3 = echarts.init(document.getElementById('p22cChartScore'));
    c3.setOption({
        ...theme,
        xAxis: { type: 'category', data: topKs, axisLabel: { color: '#8899aa' }, axisLine: { lineStyle: { color: '#2a3a4a' } } },
        yAxis: { type: 'value', name: 'eval_score', axisLabel: { color: '#8899aa' }, splitLine: { lineStyle: { color: '#1a2a3a' } } },
        series: [{ type: 'bar', data: scores, itemStyle: { color: '#ffcc02' }, barWidth: '40%',
            label: { show: true, position: 'top', color: '#ffcc02', formatter: '{c}' } }],
    });
    c3.resize();
}

loadP22cPanels();
loadP22cRuns();
loadP22cConfigs();

// ══════════════════════════════════════════════════
// Phase 22C 参数保存/加载
// ══════════════════════════════════════════════════

function getP22cParams() {
    return {
        panel: document.getElementById('p22cPanel').value,
        total_timesteps: parseInt(document.getElementById('p22cSteps').value) || 200000,
        start_date: document.getElementById('p22cStartDate').value,
        end_date: document.getElementById('p22cEndDate').value,
        n_envs: parseInt(document.getElementById('p22cEnvs').value) || 16,
        top_k: parseInt(document.getElementById('p22cTopK').value) || 3,
        seed: parseInt(document.getElementById('p22cSeed').value) || Math.floor(Math.random() * 100000),
        forward_period: parseInt(document.getElementById('p22cFwd').value) || 10,
        // PPO 超参数
        learning_rate: parseFloat(document.getElementById('p22cLR').value) || 0.00003,
        n_steps: parseInt(document.getElementById('p22cNSteps').value) || 256,
        n_epochs: parseInt(document.getElementById('p22cNEpochs').value) || 10,
        batch_size: parseInt(document.getElementById('p22cBatchSize').value) || 512,
        // 网络架构
        encoder_hidden: document.getElementById('p22cEncHidden').value || '256,128,64',
        encoder_out_dim: parseInt(document.getElementById('p22cEncOutDim').value) || 64,
        // main-wave 配置
        mwl_hold_window: parseInt(document.getElementById('p22cHoldWindow').value) || 5,
        mwl_vol_window: parseInt(document.getElementById('p22cVolWindow').value) || 20,
        mwl_sigma_multiplier: parseFloat(document.getElementById('p22cSigma').value) || 2.0,
        mwl_absolute_threshold: parseFloat(document.getElementById('p22cAbsThresh').value) || 0.06,
        mwl_amount_ma_min: parseFloat(document.getElementById('p22cAmountMin').value) || 1e8,
    };
}

function setP22cParams(p) {
    if (p.panel) document.getElementById('p22cPanel').value = p.panel;
    if (p.total_timesteps) document.getElementById('p22cSteps').value = p.total_timesteps;
    if (p.start_date) document.getElementById('p22cStartDate').value = p.start_date;
    if (p.end_date) document.getElementById('p22cEndDate').value = p.end_date;
    if (p.n_envs) document.getElementById('p22cEnvs').value = p.n_envs;
    if (p.top_k) document.getElementById('p22cTopK').value = p.top_k;
    if (p.seed !== undefined) document.getElementById('p22cSeed').value = p.seed;
    if (p.forward_period) document.getElementById('p22cFwd').value = p.forward_period;
    // PPO 超参数
    if (p.learning_rate !== undefined) document.getElementById('p22cLR').value = p.learning_rate;
    if (p.n_steps !== undefined) document.getElementById('p22cNSteps').value = p.n_steps;
    if (p.n_epochs !== undefined) document.getElementById('p22cNEpochs').value = p.n_epochs;
    if (p.batch_size !== undefined) document.getElementById('p22cBatchSize').value = p.batch_size;
    // 网络架构
    if (p.encoder_hidden) document.getElementById('p22cEncHidden').value = p.encoder_hidden;
    if (p.encoder_out_dim !== undefined) document.getElementById('p22cEncOutDim').value = p.encoder_out_dim;
    // main-wave 配置
    if (p.mwl_hold_window) document.getElementById('p22cHoldWindow').value = p.mwl_hold_window;
    if (p.mwl_vol_window) document.getElementById('p22cVolWindow').value = p.mwl_vol_window;
    if (p.mwl_sigma_multiplier !== undefined) document.getElementById('p22cSigma').value = p.mwl_sigma_multiplier;
    if (p.mwl_absolute_threshold !== undefined) document.getElementById('p22cAbsThresh').value = p.mwl_absolute_threshold;
    if (p.mwl_amount_ma_min !== undefined) document.getElementById('p22cAmountMin').value = p.mwl_amount_ma_min;
}

function resetP22cConfig() {
    setP22cParams({
        total_timesteps: 200000,
        n_envs: 16,
        top_k: 3,
        seed: 42,
        forward_period: 10,
        learning_rate: 0.00003,
        n_steps: 256,
        n_epochs: 10,
        batch_size: 512,
        encoder_hidden: '256,128,64',
        encoder_out_dim: 64,
        mwl_hold_window: 5,
        mwl_vol_window: 20,
        mwl_sigma_multiplier: 2.0,
        mwl_absolute_threshold: 0.06,
        mwl_amount_ma_min: 1e8,
    });
    addLog('↺ P22C 参数已重置为默认');
}

function loadP22cConfigs() {
    fetch('/api/p22c/configs').then(r=>r.json()).then(d => {
        const sel = document.getElementById('p22cSavedConfigs');
        sel.innerHTML = '<option value="">-- 已保存参数 --</option>' +
            (d.configs||[]).map(c =>
                `<option value="${c.name}">${c.name} (${c.saved_at})</option>`
            ).join('');
    });
}

function loadP22cConfig(name) {
    if (!name) return;
    fetch('/api/p22c/configs').then(r=>r.json()).then(d => {
        const cfg = (d.configs||[]).find(c => c.name === name);
        if (cfg && cfg.params) {
            setP22cParams(cfg.params);
            addLog(`📂 已加载 Phase 22C 参数: ${name}`);
        }
    });
}

function saveP22cConfig() {
    const name = prompt('参数名称:', '');
    if (!name) return;
    const params = getP22cParams();
    fetch('/api/p22c/configs/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, params}),
    }).then(r => r.json()).then(d => {
        addLog(`💾 Phase 22C 参数已保存: ${name}`);
        loadP22cConfigs();
    });
}

function deleteP22cConfig() {
    const sel = document.getElementById('p22cSavedConfigs');
    const name = sel.value;
    if (!name) { addLog('⚠ 请先选择要删除的参数'); return; }
    if (!confirm(`确认删除参数 "${name}"?`)) return;
    fetch(`/api/p22c/configs/${encodeURIComponent(name)}`, {method: 'DELETE'})
        .then(r => r.json()).then(d => {
            addLog(`🗑 已删除 Phase 22C 参数: ${name}`);
            loadP22cConfigs();
        });
}

// ══════════════════════════════════════════════════
