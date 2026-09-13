// tabs/knn_joint.js — 🧬 联合训练 + KNN+LSTM (KNN+LSTM 已隐藏，仅保留代码)
// 从 index.html inline JS 抽离；依赖全局 fetch/addLog
// KNN + LSTM Tab
// ══════════════════════════════════════════════════

const KNL_PRESET_KEY = 'aurumq_knl_presets_v1';

// 内置预设（HS300 正式收敛训练配置，任何浏览器可直接选择；用户自定义预设优先级更高）
const BUILTIN_KNL_PRESETS = {
    'HS300-最优AUC': {
        panel: 'factor_panel_hs300_20040102_20260731.parquet',
        window: '20', k_neighbors: '20',
        lstm_hidden: '384', lstm_layers: '2', epochs: '80',
        lstm_weight: '0.6', knn_weight: '0.4', confidence: '0.55',
        train_end: '2024-12-31', test_start: '2025-01-01', device: 'cuda',
        forward_period: '5',  // 5天窗口降噪
        // weight_decay=1e-4, patience=15 在后端 CLI 参数中固定
    },
    'HS300-正式收敛': {
        panel: 'factor_panel_hs300_20040102_20260731.parquet',
        window: '20', k_neighbors: '20',
        lstm_hidden: '384', lstm_layers: '2', epochs: '50',
        lstm_weight: '0.6', knn_weight: '0.4', confidence: '0.55',
        train_end: '2024-12-31', test_start: '2025-01-01', device: 'cuda',
    },
    'HS300-快速验证': {
        panel: 'factor_panel_hs300_20040102_20260731.parquet',
        window: '20', k_neighbors: '20',
        lstm_hidden: '128', lstm_layers: '2', epochs: '3',
        lstm_weight: '0.6', knn_weight: '0.4', confidence: '0.55',
        train_end: '2024-12-31', test_start: '2025-01-01', device: 'cuda',
    },
};

function knlGetPresets() {
    try { return Object.assign({}, BUILTIN_KNL_PRESETS, JSON.parse(localStorage.getItem(KNL_PRESET_KEY) || '{}')); }
    catch(e) { return Object.assign({}, BUILTIN_KNL_PRESETS); }
}
function knlSetPresets(p) { localStorage.setItem(KNL_PRESET_KEY, JSON.stringify(p)); }

function knlRefreshPresetSelect() {
    const sel = document.getElementById('knlPresetSelect');
    const presets = knlGetPresets();
    sel.innerHTML = '<option value="">-- 选择预设 --</option>' +
        Object.keys(presets).sort().map(n => `<option value="${n}">${n}</option>`).join('');
}

function knlGetCurrentParams() {
    return {
        panel: document.getElementById('knlPanel').value,
        window: document.getElementById('knlWindow').value,
        forward_period: document.getElementById('knlForwardPeriod')?.value || '1',
        k_neighbors: document.getElementById('knlK').value,
        lstm_hidden: document.getElementById('knlHidden').value,
        lstm_layers: document.getElementById('knlLayers').value,
        epochs: document.getElementById('knlEpochs').value,
        lstm_weight: document.getElementById('knlLstmW').value,
        knn_weight: document.getElementById('knlKnnW').value,
        confidence: document.getElementById('knlConf').value,
        forward_period: parseInt(document.getElementById('knlForwardPeriod')?.value) || 1,
        train_end: document.getElementById('knlTrainEnd').value,
        test_start: document.getElementById('knlTestStart').value,
        device: document.getElementById('knlDevice').value,
    };
}

function knlSetParams(p) {
    if (p.panel) document.getElementById('knlPanel').value = p.panel;
    if (p.window) document.getElementById('knlWindow').value = p.window;
    if (p.forward_period) document.getElementById('knlForwardPeriod').value = p.forward_period;
    if (p.k_neighbors) document.getElementById('knlK').value = p.k_neighbors;
    if (p.lstm_hidden) document.getElementById('knlHidden').value = p.lstm_hidden;
    if (p.lstm_layers) document.getElementById('knlLayers').value = p.lstm_layers;
    if (p.epochs) document.getElementById('knlEpochs').value = p.epochs;
    if (p.lstm_weight) document.getElementById('knlLstmW').value = p.lstm_weight;
    if (p.knn_weight) document.getElementById('knlKnnW').value = p.knn_weight;
    if (p.confidence) document.getElementById('knlConf').value = p.confidence;
    if (p.train_end) document.getElementById('knlTrainEnd').value = p.train_end;
    if (p.test_start) document.getElementById('knlTestStart').value = p.test_start;
    if (p.device) document.getElementById('knlDevice').value = p.device;
}

function knlSavePreset() {
    const name = document.getElementById('knlPresetName').value.trim();
    if (!name) { alert('请输入预设名'); return; }
    const presets = knlGetPresets();
    presets[name] = knlGetCurrentParams();
    knlSetPresets(presets);
    knlRefreshPresetSelect();
    document.getElementById('knlPresetSelect').value = name;
    addKnlLog('💾 预设已保存: ' + name);
}

function knlLoadPreset(name) {
    if (!name) return;
    const presets = knlGetPresets();
    if (presets[name]) {
        knlSetParams(presets[name]);
        addKnlLog('📋 已加载预设: ' + name);
    }
}

function knlDeletePreset() {
    const sel = document.getElementById('knlPresetSelect');
    const name = sel.value;
    if (!name) { alert('请选择要删除的预设'); return; }
    if (!confirm('确认删除预设 ' + name + '?')) return;
    const presets = knlGetPresets();
    delete presets[name];
    knlSetPresets(presets);
    knlRefreshPresetSelect();
    addKnlLog('🗑 预设已删除: ' + name);
}

function knlResetPreset() {
    const defaults = {
        panel: '',
        window: '20',
        forward_period: '5',
        k_neighbors: '20',
        lstm_hidden: '128',
        lstm_layers: '2',
        epochs: '50',
        batch_size: '2048',
        lstm_weight: '0.6',
        knn_weight: '0.4',
        confidence: '0.55',
        train_end: '2024-12-31',
        test_start: '2025-01-01',
        device: 'cuda',
    };
    knlSetParams(defaults);
    addKnlLog('↺ 已重置为默认参数');
}

function knlModeSwitch() {
    const mode = document.querySelector('input[name="knlMode"]:checked')?.value || 'single';
    document.getElementById('knlRollParams').style.display = mode === 'roll' ? 'block' : 'none';
}

function knnLstmStart() {
    const mode = document.querySelector('input[name="knlMode"]:checked')?.value || 'single';
    document.getElementById('knlTrainBadge').className = 'status-badge status-running';
    document.getElementById('knlTrainBadge').textContent = '训练中';
    document.getElementById('knlStartBtn').disabled = true;
    document.getElementById('knlStopBtn').disabled = false;

    if (mode === 'roll') {
        // ── 滚动分段训练（20年逐段 + 续训）──
        const params = {
            panel: document.getElementById('knlPanel').value,
            segments: document.getElementById('knlSegments').value,
            resume_at: parseInt(document.getElementById('knlRollResumeAt').value) || 0,
            test_start: document.getElementById('knlTestStart').value,
            window: parseInt(document.getElementById('knlWindow').value) || 20,
            k_neighbors: parseInt(document.getElementById('knlK').value) || 20,
            lstm_hidden: parseInt(document.getElementById('knlHidden').value) || 384,
            lstm_layers: parseInt(document.getElementById('knlLayers').value) || 2,
            epochs: parseInt(document.getElementById('knlRollEpochs').value) || 15,
            batch_size: parseInt(document.getElementById('knlBatchSize')?.value) || 2048,
            max_samples_per_stock: parseInt(document.getElementById('knlRollMaxSamples').value) || 2000,
            lstm_weight: parseFloat(document.getElementById('knlLstmW').value) || 0.6,
            knn_weight: parseFloat(document.getElementById('knlKnnW').value) || 0.4,
            confidence: parseFloat(document.getElementById('knlConf').value) || 0.55,
            device: document.getElementById('knlDevice').value,
        };
        fetch('/api/knn_lstm/incremental/train', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(params),
        }).then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); })).then(d => {
            if (d.error || d.detail) { addKnlLog('❌ ' + (d.error || d.detail)); knnLstmResetBadge(); return; }
            addKnlLog('🌊 滚动分段训练已启动: ' + d.out_dir);
            pollKnnLstmStatus();
        }).catch(e => { addKnlLog('❌ ' + e.message); knnLstmResetBadge(); });
        return;
    }

    // ── 单段训练 ──
    const params = {
        panel: document.getElementById('knlPanel').value,
        window: parseInt(document.getElementById('knlWindow').value) || 20,
        k_neighbors: parseInt(document.getElementById('knlK').value) || 20,
        lstm_hidden: parseInt(document.getElementById('knlHidden').value) || 128,
        lstm_layers: parseInt(document.getElementById('knlLayers').value) || 2,
        epochs: parseInt(document.getElementById('knlEpochs').value) || 50,
        batch_size: parseInt(document.getElementById('knlBatchSize')?.value) || 2048,
        num_workers: 3,
        max_samples_per_stock: 2000,
        lstm_weight: parseFloat(document.getElementById('knlLstmW').value) || 0.6,
        knn_weight: parseFloat(document.getElementById('knlKnnW').value) || 0.4,
        confidence: parseFloat(document.getElementById('knlConf').value) || 0.55,
        train_end: document.getElementById('knlTrainEnd').value,
        test_start: document.getElementById('knlTestStart').value,
        device: document.getElementById('knlDevice').value,
    };
    fetch('/api/knn_lstm/train', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
    }).then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); })).then(d => {
        if (d.error || d.detail) { addKnlLog('❌ ' + (d.error || d.detail)); knnLstmResetBadge(); return; }
            addKnlLog('🔬 训练已启动: ' + d.out_dir);
            pollKnnLstmStatus();
    }).catch(e => { addKnlLog('❌ ' + e.message); knnLstmResetBadge(); });
}

function knnLstmStop() {
    fetch('/api/knn_lstm/train/stop', {method: 'POST'})
        .then(r => r.json())
        .then(d => { addKnlLog('⏹ 已停止'); knnLstmResetBadge(); })
        .catch(e => addKnlLog('❌ ' + e.message));
}

function pollKnnLstmStatus() {
    fetch('/api/knn_lstm/train/status').then(r => r.json()).then(d => {
        const badge = document.getElementById('knlTrainBadge');
        const progress = document.getElementById('knlProgress');
        if (d.running) {
            badge.className = 'status-badge status-running';
            badge.textContent = '训练中';
            progress.textContent = d.progress || '训练进行中，查看日志';
            setTimeout(pollKnnLstmStatus, 2000);
        } else {
            knnLstmResetBadge();
            progress.textContent = d.status === 'failed' ? ('失败: ' + (d.error || '未知错误')) : '';
            loadKnnLstmModels();
        }
    }).catch(e => addKnlLog('❌ 读取训练状态失败: ' + e.message));
}

function knnLstmResetBadge() {
    document.getElementById('knlTrainBadge').className = 'status-badge status-idle';
    document.getElementById('knlTrainBadge').textContent = '空闲';
    document.getElementById('knlStartBtn').disabled = false;
    document.getElementById('knlStopBtn').disabled = true;
}

function addKnlLog(msg) { addLog(msg, 'knn_lstm'); }

// ══════════════════════════════════════════════════
// ══════════════════════════════════════════════════

const KJ_PRESET_KEY = 'aurumq_knn_joint_presets_v1';

// 内置预设（写死在代码里，任何人打开页面都有）— 2026-08-11 参数搜索最优结果
const KJ_BUILTIN_PRESETS = {
    '🏆最优-夏普1.306': {
        panel: '',
        start_date: '2024-01-01',
        end_date: '2024-06-30',
        total_timesteps: '200000',
        top_k: '20',
        forward_period: '10',
        window: '10',
        lstm_hidden: '128',
        lstm_layers: '1',
        n_envs: '1',
        learning_rate: '0.0001',
        n_factors: '0',
        cost_bps: '30',
        max_position_pct: '0.02',
        max_industry_pct: '0.30',
        reward_type: 'return',
        seed: '42',
        max_grad_norm: '1.0',
        n_steps: '128',
        batch_size: '64',
        n_epochs: '10',
        rebalance_days: '20',
        use_knn: true,
        knn_k: '20',
        knn_ref_lookback: '750',
        knn_max_ref_days: '12',
        target_kl: '0.02',
        learning_rate_schedule: 'constant',
    },
    '融合-300k低回撤': {
        panel: '',
        start_date: '2024-01-01',
        end_date: '2024-06-30',
        total_timesteps: '300000',
        top_k: '20',
        forward_period: '10',
        window: '10',
        lstm_hidden: '128',
        lstm_layers: '1',
        n_envs: '1',
        learning_rate: '0.0001',
        n_factors: '0',
        cost_bps: '30',
        max_position_pct: '0.05',
        max_industry_pct: '0.30',
        reward_type: 'return',
        seed: '42',
        max_grad_norm: '1.0',
        n_steps: '128',
        batch_size: '64',
        n_epochs: '10',
        rebalance_days: '20',
        use_knn: true,
        knn_k: '20',
        knn_ref_lookback: '750',
        knn_max_ref_days: '12',
        target_kl: '0.02',
        learning_rate_schedule: 'constant',
    },
    '融合-100k快速验证': {
        panel: '',
        start_date: '2024-01-01',
        end_date: '2024-06-30',
        total_timesteps: '100000',
        top_k: '20',
        forward_period: '10',
        window: '10',
        lstm_hidden: '128',
        lstm_layers: '1',
        n_envs: '1',
        learning_rate: '0.0001',
        n_factors: '0',
        cost_bps: '30',
        max_position_pct: '0.05',
        max_industry_pct: '0.30',
        reward_type: 'return',
        seed: '42',
        max_grad_norm: '1.0',
        n_steps: '128',
        batch_size: '64',
        n_epochs: '10',
        rebalance_days: '20',
        use_knn: true,
        knn_k: '20',
        knn_ref_lookback: '750',
        knn_max_ref_days: '12',
        target_kl: '0.02',
        learning_rate_schedule: 'constant',
    },
};

function kjGetPresets() {
    try { return Object.assign({}, KJ_BUILTIN_PRESETS, JSON.parse(localStorage.getItem(KJ_PRESET_KEY) || '{}')); }
    catch(e) { return Object.assign({}, KJ_BUILTIN_PRESETS); }
}
function kjSetPresets(p) { localStorage.setItem(KJ_PRESET_KEY, JSON.stringify(p)); }
function kjRefreshPresetSelect() {
    const sel = document.getElementById('kjPresetSelect');
    if (!sel) return;
    const presets = kjGetPresets();
    sel.innerHTML = '<option value="">-- 选择预设 --</option>' + Object.keys(presets).sort().map(n => '<option value="'+n+'">'+n+'</option>').join('');
}
function kjGetCurrentParams() {
    return {
        panel: document.getElementById('kjPanel').value,
        start_date: document.getElementById('kjStartDate').value,
        end_date: document.getElementById('kjEndDate').value,
        total_timesteps: document.getElementById('kjSteps').value,
        top_k: document.getElementById('kjTopK').value,
        forward_period: document.getElementById('kjForwardPeriod').value,
        window: document.getElementById('kjWindow').value,
        lstm_hidden: document.getElementById('kjHidden').value,
        lstm_layers: document.getElementById('kjLayers').value,
        n_envs: document.getElementById('kjEnvs').value,
        learning_rate: document.getElementById('kjLR').value,
        n_factors: document.getElementById('kjFactors').value,
        cost_bps: document.getElementById('kjCostBps').value,
        max_position_pct: document.getElementById('kjMaxPos').value,
        max_industry_pct: document.getElementById('kjMaxInd').value,
        reward_type: document.getElementById('kjRewardType').value,
        seed: document.getElementById('kjSeed').value,
        max_grad_norm: document.getElementById('kjGradNorm').value,
        n_steps: document.getElementById('kjNSteps').value,
        batch_size: document.getElementById('kjBatchSize').value,
        n_epochs: document.getElementById('kjNEpochs').value,
        rebalance_days: document.getElementById('kjRebalanceDays').value,
        feature_group_weights_json: document.getElementById('kjFundW').value,
        use_knn: document.getElementById('kjUseKnn')?.value || '1',
        knn_k: document.getElementById('kjKnnK')?.value || '20',
        knn_ref_lookback: document.getElementById('kjKnnRef')?.value || '240',
        knn_max_ref_days: document.getElementById('kjKnnMaxRef')?.value || '12',
        target_kl: document.getElementById('kjTargetKl')?.value || '0.02',
    };
}
function kjSetCurrentParams(p) {
    if (p.panel !== undefined) document.getElementById('kjPanel').value = p.panel;
    if (p.start_date !== undefined) document.getElementById('kjStartDate').value = p.start_date;
    if (p.end_date !== undefined) document.getElementById('kjEndDate').value = p.end_date;
    if (p.total_timesteps !== undefined) document.getElementById('kjSteps').value = p.total_timesteps;
    if (p.top_k !== undefined) document.getElementById('kjTopK').value = p.top_k;
    if (p.forward_period !== undefined) document.getElementById('kjForwardPeriod').value = p.forward_period;
    if (p.window !== undefined) document.getElementById('kjWindow').value = p.window;
    if (p.lstm_hidden !== undefined) document.getElementById('kjHidden').value = p.lstm_hidden;
    if (p.lstm_layers !== undefined) document.getElementById('kjLayers').value = p.lstm_layers;
    if (p.n_envs !== undefined) document.getElementById('kjEnvs').value = p.n_envs;
    if (p.learning_rate !== undefined) document.getElementById('kjLR').value = p.learning_rate;
    if (p.n_factors !== undefined) document.getElementById('kjFactors').value = p.n_factors;
    if (p.cost_bps !== undefined) document.getElementById('kjCostBps').value = p.cost_bps;
    if (p.max_position_pct !== undefined) document.getElementById('kjMaxPos').value = p.max_position_pct;
    if (p.max_industry_pct !== undefined) document.getElementById('kjMaxInd').value = p.max_industry_pct;
    if (p.reward_type !== undefined) document.getElementById('kjRewardType').value = p.reward_type;
    if (p.seed !== undefined) document.getElementById('kjSeed').value = p.seed;
    if (p.max_grad_norm !== undefined) document.getElementById('kjGradNorm').value = p.max_grad_norm;
    if (p.n_steps !== undefined) document.getElementById('kjNSteps').value = p.n_steps;
    if (p.batch_size !== undefined) document.getElementById('kjBatchSize').value = p.batch_size;
    if (p.n_epochs !== undefined) document.getElementById('kjNEpochs').value = p.n_epochs;
    if (p.rebalance_days !== undefined) document.getElementById('kjRebalanceDays').value = p.rebalance_days;
    if (p.feature_group_weights_json !== undefined) document.getElementById('kjFundW').value = p.feature_group_weights_json;
    if (p.use_knn !== undefined) document.getElementById('kjUseKnn').value = String(p.use_knn);
    if (p.knn_k !== undefined) document.getElementById('kjKnnK').value = p.knn_k;
    if (p.knn_ref_lookback !== undefined) document.getElementById('kjKnnRef').value = p.knn_ref_lookback;
    if (p.knn_max_ref_days !== undefined) document.getElementById('kjKnnMaxRef').value = p.knn_max_ref_days;
    if (p.target_kl !== undefined && document.getElementById('kjTargetKl')) document.getElementById('kjTargetKl').value = p.target_kl;
}
function kjSavePreset() {
    const name = document.getElementById('kjPresetName').value.trim();
    if (!name) { alert('请输入预设名'); return; }
    const presets = kjGetPresets();
    presets[name] = kjGetCurrentParams();
    kjSetPresets(presets);
    kjRefreshPresetSelect();
    document.getElementById('kjPresetSelect').value = name;
    addKjLog('💾 预设已保存: ' + name);
}
function kjLoadPreset(name) {
    if (!name) return;
    const presets = kjGetPresets();
    if (!presets[name]) return;
    kjSetCurrentParams(presets[name]);
    // 内置预设 panel 为空 → 自动填入当前下拉框第一个可用面板
    if (!presets[name].panel) {
        const sel = document.getElementById('kjPanel');
        if (sel && sel.options.length > 0) sel.value = sel.options[0].value;
    }
    addKjLog('📂 已加载预设: ' + name);
}
function kjDeletePreset() {
    const sel = document.getElementById('kjPresetSelect');
    const name = sel.value;
    if (!name) return;
    if (KJ_BUILTIN_PRESETS[name]) {
        addKjLog('⛔ 内置预设不可删除: ' + name);
        return;
    }
    const presets = kjGetPresets();
    delete presets[name];
    kjSetPresets(presets);
    kjRefreshPresetSelect();
    addKjLog('🗑 预设已删除: ' + name);
}
function kjResetPreset() {
    const defaults = {
        panel: '',
        start_date: '2024-01-01',
        end_date: '2024-06-30',
        total_timesteps: '2000000',
        top_k: '20',
        forward_period: '10',
        window: '20',
        lstm_hidden: '128',
        lstm_layers: '1',
        n_envs: '1',
        learning_rate: '0.0003',
        n_factors: '0',
        cost_bps: '30',
        max_position_pct: '0.05',
        max_industry_pct: '0.30',
        reward_type: 'return',
        seed: '42',
        max_grad_norm: '0.5',
        n_steps: '128',
        batch_size: '0',
        n_epochs: '0',
        knn_ref_lookback: '750',
        knn_max_ref_days: '12',
        target_kl: '0.02',
    };
    kjSetCurrentParams(defaults);
    addKjLog('↺ 已重置为默认参数');
}

// 🧬 联合训练 (LSTM+PPO 端到端)
// ══════════════════════════════════════════════════
function kjStart() {
    document.getElementById('kjTrainBadge').className = 'status-badge status-running';
    document.getElementById('kjTrainBadge').textContent = '训练中';
    document.getElementById('kjStartBtn').disabled = true;
    document.getElementById('kjStopBtn').disabled = false;
    const params = {
        panel: document.getElementById('kjPanel').value,
        start_date: document.getElementById('kjStartDate').value,
        end_date: document.getElementById('kjEndDate').value,
        total_timesteps: parseInt(document.getElementById('kjSteps').value) || 2000000,
        top_k: parseInt(document.getElementById('kjTopK').value) || 20,
        forward_period: parseInt(document.getElementById('kjForwardPeriod').value) || 10,
        window: parseInt(document.getElementById('kjWindow').value) || 10,
        lstm_hidden: parseInt(document.getElementById('kjHidden').value) || 128,
        lstm_layers: parseInt(document.getElementById('kjLayers').value) || 1,
        n_envs: parseInt(document.getElementById('kjEnvs').value) || 1,
        learning_rate: parseFloat(document.getElementById('kjLR').value) || 0.0003,
        n_factors: parseInt(document.getElementById('kjFactors').value) || null,
        cost_bps: parseFloat(document.getElementById('kjCostBps').value) || 30,
        max_position_pct: parseFloat(document.getElementById('kjMaxPos').value) || 0.05,
        max_industry_pct: parseFloat(document.getElementById('kjMaxInd').value) || 0.30,
        reward_type: document.getElementById('kjRewardType').value || 'return',
        seed: parseInt(document.getElementById('kjSeed').value) || 42,
        max_grad_norm: parseFloat(document.getElementById('kjGradNorm').value) || 0.5,
        n_steps: parseInt(document.getElementById('kjNSteps').value) || 128,
        batch_size: parseInt(document.getElementById('kjBatchSize').value) || null,
        n_epochs: parseInt(document.getElementById('kjNEpochs').value) || null,
        rebalance_days: parseInt(document.getElementById('kjRebalanceDays').value) || 20,
        feature_group_weights_json: document.getElementById('kjFundW').value.trim() || '',
        use_knn: document.getElementById('kjUseKnn')?.value === '1',
        knn_k: parseInt(document.getElementById('kjKnnK')?.value) || 20,
        knn_ref_lookback: parseInt(document.getElementById('kjKnnRef')?.value) || 240,
        knn_max_ref_days: parseInt(document.getElementById('kjKnnMaxRef')?.value) || 12,
        target_kl: parseFloat(document.getElementById('kjTargetKl')?.value) || null,
    };
    fetch('/api/kj/train', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
    }).then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); })).then(d => {
        if (d.error || d.detail) { addKjLog('❌ ' + (d.error || d.detail)); kjResetBadge(); return; }
        addKjLog('🚀 联合训练已启动: ' + (d.out_dir || d.model || '任务已提交'));
        pollKjStatus();
    }).catch(e => { addKjLog('❌ ' + e.message); kjResetBadge(); });
}
function kjStop() {
    fetch('/api/kj/train/stop', {method: 'POST'}).then(r => r.json())
        .then(d => {
            addKjLog('⏹ 已停止');
            /* SSE unified - no separate close needed */
            kjResetBadge();
        })
        .catch(e => addKjLog('❌ ' + e.message));
}
function kjResetBadge() {
    document.getElementById('kjTrainBadge').className = 'status-badge status-idle';
    document.getElementById('kjTrainBadge').textContent = '空闲';
    document.getElementById('kjStartBtn').disabled = false;
    document.getElementById('kjStopBtn').disabled = true;
}
function pollKjStatus() {
    fetch('/api/kj/train/status').then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); })).then(d => {
        const el = document.getElementById('kjProgress');
        if (d.running) {
            document.getElementById('kjTrainBadge').className = 'status-badge status-running';
            document.getElementById('kjTrainBadge').textContent = '训练中';
            document.getElementById('kjStartBtn').disabled = true;
            document.getElementById('kjStopBtn').disabled = false;
            if (el && d.progress) el.textContent = d.progress;
            setTimeout(pollKjStatus, 5000);
        } else {
            kjResetBadge();
            /* SSE unified */
            if (el) el.textContent = '';
            loadJointModels();
        }
    }).catch(e => { addKjLog('❌ 读取训练状态失败: ' + e.message); setTimeout(pollKjStatus, 10000); });
}

function addKjLog(msg) { addLog(msg, 'kj'); }
// kj 别名（KNN+LSTM+PPO 融合 Tab 的日志）
function loadKjModels() {
    // 联合训练、KNN+LSTM 和各个联合 Tab 统一读取同一个列表。
    fetch('/api/joint/models').then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); })).then(d => {
        const el = document.getElementById('kjModelList');
        if (!d.models || !d.models.length) {
            el.innerHTML = '<span style="color:#667788;font-size:12px">暂无融合模型</span>';
            return;
        }
        el.innerHTML = d.models.map(m => {
            const meta = m.meta || {};
            let tagHtml = '';
            if (m.status === 'running') tagHtml = '<span style="color:#ff9800;font-size:10px;background:#3a2a00;padding:1px 6px;border-radius:8px;margin-left:6px">🏃 训练中</span>';
            else if (m.status === 'done') tagHtml = '<span style="color:#4caf50;font-size:10px;background:#0d2b1a;padding:1px 6px;border-radius:8px;margin-left:6px">✅ 完成</span>';
            else if (m.status === 'interrupted') tagHtml = '<span style="color:#f44336;font-size:10px;background:#3a0d0d;padding:1px 6px;border-radius:8px;margin-left:6px">⚠️ 中断</span>';
            return `<div class="file-item" style="padding:8px;border-bottom:1px solid #1f2a3a">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-size:13px;color:#e0e0e0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${m.name}">${m.name}${tagHtml}</span>
                    <button class="btn btn-sm btn-danger" onclick="kjDeleteModel('${m.name}')" style="padding:2px 8px;font-size:10px">🗑</button>
                </div>
                <div style="font-size:11px;color:#8899aa;margin-top:4px">
                    ${m.has_onnx ? '✅' : '❌'} ONNX | ${m.size_mb}MB | window=${meta.window || '?'} hidden=${meta.hidden || '?'}
                </div>
            </div>`;
        }).join('');
    }).catch(e => { document.getElementById('kjModelList').innerHTML = '<span style="color:#f44336;font-size:12px">加载失败</span>'; addKjLog('❌ 联合模型列表加载失败: ' + e.message); });
}
function kjDeleteModel(name) {
    if (!confirm('删除联合模型 ' + name + '？')) return;
    fetch('/api/kj/models/' + encodeURIComponent(name), {method: 'DELETE'})
        .then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); }))
        .then(d => { if (d.error || d.detail) throw new Error(d.error || d.detail); loadKjModels(); addKjLog('🗑 已删除: ' + name); })
        .catch(e => addKjLog('❌ ' + e.message));
}
function loadKjPanels() {
    fetch('/api/panels').then(r => r.json()).then(d => {
        const sel = document.getElementById('kjPanel');
        sel.innerHTML = (d.panels || []).map(p =>
            `<option value="${p.name}">${p.name} (${p.size_gb} GB)</option>`
        ).join('') || '<option value="">无面板</option>';
    }).catch(() => {});
}


function loadKnnLstmModels() {
    // 与联合训练 Tab 共用统一模型注册表，避免两个列表各扫各的目录。
    fetch('/api/joint/models').then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); })).then(d => {
        const el = document.getElementById('knlModelList');
        const items = (d.models || []).filter(m => m.name.startsWith('joint_') || m.name.startsWith('knn_lstm') || m.name.startsWith('real_test_'));
        if (!items.length) {
            el.innerHTML = '<span style="color:#667788;font-size:12px">暂无已训练的模型</span>';
            return;
        }
        el.innerHTML = items.map(m => {
            const acc = m.metrics?.accuracy || 0;
            const auc = m.metrics?.auc || 0;
            const f1 = m.metrics?.f1 || 0;
            const lstm_acc = m.lstm_only?.accuracy || 0;
            const knn_acc = m.knn_only?.accuracy || 0;
            const sig = m.metrics?.signal_long || 0;
            const cfg = m.config || {};
            const desc = [
                cfg.window ? `窗口${cfg.window}d` : '',
                cfg.k_neighbors ? `K=${cfg.k_neighbors}` : '',
                cfg.lstm_hidden ? `LSTM${cfg.lstm_hidden}` : '',
                cfg.epochs ? `${cfg.epochs}轮` : '',
            ].filter(Boolean).join(' · ');
            return `
            <div class="file-item" style="display:flex;align-items:flex-start;gap:8px;padding:6px 10px;border-bottom:1px solid #1f3a52">
                <div style="flex:1;min-width:0">
                    <div class="file-name" style="font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${m.name}">${m.name}</div>
                    <div class="file-size" style="font-size:11px;color:#8899aa">
                        💾 ${m.size_mb}MB · ${desc}
                    </div>
                    <div style="display:flex;gap:12px;margin-top:4px;font-size:11px">
                        <span>融合 <b style="color:${acc>0.52?'#f44336':'#4caf50'}">${(acc*100).toFixed(1)}%</b></span>
                        <span>AUC <b style="color:#4fc3f7">${auc.toFixed(3)}</b></span>
                        <span>F1 <b>${f1.toFixed(3)}</b></span>
                        <span style="color:#667788">LSTM ${(lstm_acc*100).toFixed(1)}%</span>
                        <span style="color:#667788">KNN ${(knn_acc*100).toFixed(1)}%</span>
                        <span style="color:#667788">信号${sig}</span>
                    </div>
                </div>
                <button onclick="knnLstmDelete('${m.name}', this)"
                    style="background:#c62828;color:#fff;border:none;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px;flex-shrink:0">
                    🗑
                </button>
            </div>`;
        }).join('');
    });
}

function knnLstmDelete(name) {
    if (!confirm('确认删除 ' + name + '?')) return;
    fetch('/api/knn_lstm/models/' + name, {method: 'DELETE'})
        .then(r => r.json())
        .then(d => { addKnlLog('🗑 已删除: ' + name); loadKnnLstmModels(); })
        .catch(e => addKnlLog('❌ ' + e.message));
}

// 切换到 KNN+LSTM / 联合训练 tab 时加载模型列表
function _wrapSwitchTab(){
  if(typeof switchTab==='undefined'){ setTimeout(_wrapSwitchTab, 100); return; }
  const _origSwitchTab = switchTab;
  switchTab = function(id) {
    _origSwitchTab(id);
    if (id === 'knn_lstm' && !isHiddenTab('knn_lstm')) { loadKnnLstmModels(); knlRefreshPresetSelect(); }
    if (id === 'ml' && !isHiddenTab('ml')) { mlRefreshPresetSelect(); }
    if (id === 'knn_joint') { loadKjModels(); loadKjPanels(); kjRefreshPresetSelect(); }
    if (id === 'walkforward') { loadWfPanels(); }
    if (id === 'matrix') { loadMtxPanels(); }
    if (id === 'data_mgmt') { loadRecentStructuredLogs(true); dmRefreshCache(); }
  };
}
_wrapSwitchTab();

// 页面初始化时预加载面板列表（defer until walkforward module ready）
function _initKnlPanels(){ try{ if(typeof loadWfPanels==='function') loadWfPanels(); }catch(e){} try{ if(typeof loadMtxPanels==='function') loadMtxPanels(); }catch(e){} }
if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', _initKnlPanels); else setTimeout(_initKnlPanels, 200);
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) loadRecentStructuredLogs(true);
});
window.addEventListener('focus', () => loadRecentStructuredLogs(true));
// 长训练监控页刷新后立即恢复矩阵状态；不启动任务，只读取状态。
loadWhV2Matrix();

// ═══════════════════════════════════════════════════════════════
// 市场状态识别 Tab
// ═══════════════════════════════════════════════════════════════

function loadRegime() {
    fetch('/api/market/regime').then(r=>r.json()).then(d => {
        if (d.error) { addLog('❌ ' + d.error); return; }
        renderRegimeChart(d);
        renderRegimeScoreChart(d);
        renderRegimePeriods(d.periods);
        renderRegimeTrainPlan(d.train_plan);
        addLog('📊 市场状态数据已加载 (v2 6维度)');
    });
}

function renderRegimeChart(d) {
    const chart = echarts.init(document.getElementById('regimeChart'));
    // 用regimes生成markArea
    const markAreas = [];
    let i = 0;
    while (i < d.regimes.length) {
        const r = d.regimes[i];
        if (r === 'bull' || r === 'bear') {
            const start = d.dates[i];
            let j = i;
            while (j < d.regimes.length && d.regimes[j] === r) j++;
            const end = d.dates[j - 1];
            const days = j - i;
            if (days >= 30) {
                markAreas.push([{
                    xAxis: start,
                    itemStyle: {color: r === 'bull' ? 'rgba(76,175,80,0.18)' : 'rgba(244,67,54,0.18)'}
                }, {xAxis: end}]);
            }
            i = j;
        } else { i++; }
    }
    chart.setOption({
        tooltip: {trigger: 'axis'},
        legend: {data: ['沪深300'], textStyle: {color: '#ccc'}},
        xAxis: {type: 'category', data: d.dates, axisLabel: {color: '#999', fontSize: 10}},
        yAxis: {type: 'value', name: '价格', axisLabel: {color: '#999'}, splitLine: {lineStyle: {color: '#1a2a3a'}}},
        series: [{
            name: '沪深300', type: 'line', data: d.closes,
            lineStyle: {color: '#4fc3f7', width: 1.5}, itemStyle: {color: '#4fc3f7'},
            showSymbol: false,
            markArea: {silent: true, data: markAreas},
        }],
        dataZoom: [{type: 'inside'}, {type: 'slider'}],
        grid: {left: 60, right: 20, top: 40, bottom: 60},
        backgroundColor: 'transparent',
    });
}

function renderRegimeScoreChart(d) {
    const chart = echarts.init(document.getElementById('regimeScoreChart'));
    const labels = d.dim_labels || {};
    const dimColors = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#00bcd4', '#f44336'];
    const dimKeys = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6'];
    const dimLabels = ['均线排列', 'ADX+RSI', '量价配合', '波动率', '市场广度', '动量'];

    const series = [{
        name: '⭐ 综合得分', type: 'line', data: d.scores,
        lineStyle: {color: '#fff', width: 2, type: 'dotted'}, showSymbol: false, z: 10,
    }];
    dimKeys.forEach((k, i) => {
        if (d.dimensions && d.dimensions[k]) {
            series.push({
                name: dimLabels[i], type: 'line', data: d.dimensions[k],
                lineStyle: {color: dimColors[i], width: 1}, showSymbol: false,
            });
        }
    });

    chart.setOption({
        tooltip: {trigger: 'axis'},
        legend: {data: series.map(s => s.name), textStyle: {color: '#ccc', fontSize: 10}},
        xAxis: {type: 'category', data: d.dates, axisLabel: {color: '#999', fontSize: 10}},
        yAxis: {type: 'value', min: -1, max: 1, axisLabel: {color: '#999'}, splitLine: {lineStyle: {color: '#1a2a3a'}}},
        series: series,
        dataZoom: [{type: 'inside'}, {type: 'slider'}],
        grid: {left: 50, right: 20, top: 30, bottom: 50},
        backgroundColor: 'transparent',
    });
}

function renderRegimePeriods(periods) {
    const el = document.getElementById('regimePeriods');
    let html = '<table style="width:100%;border-collapse:collapse;font-size:12px">';
    html += '<tr style="color:#4fc3f7;border-bottom:1px solid #2a3a4a"><th style="padding:4px">状态</th><th>起始</th><th>结束</th><th>天数</th><th>涨跌</th><th>均分</th></tr>';
    const colors = {'bull': '#4caf50', 'bear': '#f44336', 'oscillation': '#ff9800'};
    const icons = {'bull': '🟢', 'bear': '🔴', 'oscillation': '🟡'};
    const names = {'bull': '牛市', 'bear': '熊市', 'oscillation': '震荡市'};
    for (const p of (periods || [])) {
        if (p.days >= 30) {
            const chgColor = p.return >= 0 ? '#f44336' : '#4caf50';
            html += `<tr style="border-bottom:1px solid #0d1b2a;color:${colors[p.regime]||'#ccc'}">`;
            html += `<td style="padding:4px 6px">${icons[p.regime]||'⚪'} ${names[p.regime]||p.regime}</td>`;
            html += `<td>${p.start}</td><td>${p.end}</td><td>${p.days}</td>`;
            html += `<td style="color:${chgColor}">${p.return >= 0 ? '+' : ''}${p.return}%</td>`;
            html += `<td>${p.avg_score ? p.avg_score.toFixed(3) : '-'}</td>`;
            html += '</tr>';
        }
    }
    html += '</table>';
    el.innerHTML = html;
}

function renderRegimeTrainPlan(plan) {
    const el = document.getElementById('regimeTrainPlan');
    if (!plan || plan.length === 0) { el.innerHTML = '<p style="color:#888">暂无训练方案</p>'; return; }
    const names = {'bull': '🟢 牛市', 'bear': '🔴 熊市', 'oscillation': '🟡 震荡市'};
    const bgColors = {'bull': '#4caf50', 'bear': '#f44336', 'oscillation': '#ff9800'};
    let html = '<table style="width:100%;border-collapse:collapse;font-size:12px">';
    html += '<tr style="color:#4fc3f7;border-bottom:1px solid #2a3a4a"><th>市场状态</th><th>训练区间</th><th>合计天数</th><th>操作</th></tr>';
    for (const p of plan) {
        const intervals = (p.train || []).map(t => '<div style="font-size:11px;color:#aaa">' + t + '</div>').join('');
        html += `<tr style="border-bottom:1px solid #0d1b2a">`;
        html += `<td style="padding:4px 6px;color:${bgColors[p.regime]||'#ccc'};font-weight:bold">${names[p.regime]||p.regime}</td>`;
        html += `<td>${intervals}</td><td>${p.total_days}天</td>`;
        html += `<td><button onclick="startRegimeTraining('${p.regime}')" style="background:${bgColors[p.regime]||'#555'};color:#fff;border:none;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:11px">🚀 训练</button></td>`;
        html += '</tr>';
    }
    html += '</table>';
    el.innerHTML = html;
}

function startRegimeTraining(regime) {
    alert('训练功能待实现: ' + regime);
}

// 市场状态数据改为懒加载（切到 regime Tab 时才加载，避免 ECharts 在隐藏容器上初始化）
// loadRegime(); ← 已移至 switchTab('regime') 懒加载

// ══════════════════════════════════════════════════
