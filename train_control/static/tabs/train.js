/** tabs/train.js — 训练 Tab（参数/预设/启停/模型管理）
 * 依赖全局: fetch / addLog / echarts
 * 由 index.html <script src="/static/tabs/train.js"> 加载
 * 负责 #tab-train 区域；状态走 /api/train/*
 */
const TRAIN_PARAMS = [
    // 基础
    {key:'panel', en:'Factor Panel', label:'📊 因子面板', type:'select', default:'',
     options:'panels',
     desc:'选择用于训练的因子面板文件，面板越大训练效果越好但耗时更长',
     range_desc:''},
    {key:'algorithm', en:'Algorithm', label:'🧠 算法', type:'select', default:'PPO',
     options:[{v:'PPO',t:'PPO (推荐)'},{v:'A2C',t:'A2C'},{v:'SAC',t:'SAC'}],
     desc:'PPO=最稳定主流算法 / A2C=更快但波动大 / SAC=适合连续动作空间',
     range_desc:''},
    {key:'total_timesteps', en:'Total Steps', label:'总训练步数', type:'int', default:2000000, min:10000, max:10000000,
     desc:'训练总迭代步数。10万步≈30分钟，100万步≈6小时。值越大模型收敛越好但耗时越长',
     range_desc:'↓小=快速验证(10万步) · ↑大=充分收敛(50万~200万步) · 推荐: 100万步'},

    // 数据
    {key:'start_date', en:'Start Date', label:'开始日期', type:'str', default:'2020-01-01',
     desc:'训练数据起始日。建议用近2-3年数据，太长的历史数据可能包含过时市场模式',
     range_desc:'↓近=贴近当前市场 · ↑远=涵盖更多周期 · 推荐: 2020-01-01'},
    {key:'end_date', en:'End Date', label:'结束日期', type:'str', default:'2025-12-31',
     desc:'训练数据截止日。留空自动取面板最大日期',
     range_desc:'留空=取到最新'},
    {key:'universe_filter', en:'Stock Filter', label:'股票池筛选', type:'select', default:'main_board_non_st',
     options:[{v:'main_board_non_st',t:'主板非ST (默认)'},{v:'all_a',t:'全A股'},{v:'hs300',t:'沪深300'},{v:'zz500',t:'中证500'}],
     desc:'因子面板已选定股票池，此选项在面板基础上再过滤。主板非ST最干净',
     range_desc:'主板非ST=过滤ST/退市风险股 · 全A股=样本最多含ST'},

    // 环境
    {key:'n_envs', en:'Parallel Envs', label:'并行环境数', type:'int', default:16, min:1, max:16,
     desc:'同时运行的模拟环境数量。RTX 3060 12GB 建议 4-6，4070 建议 8-16',
     range_desc:'↓小=省显存(2~4) · ↑大=高吞吐(12~16) · RTX3060推荐: 6 | RTX4070推荐: 16'},
    {key:'n_factors', en:'Num Factors', label:'因子数量', type:'int', default:298, min:0, max:298,
     desc:'0=使用面板全部因子。减少可加速训练但可能降低表现',
     range_desc:'↓小=快速训练(32~64) · ↑大=信息更全(128~298) · 推荐: 298(全部)'},
    {key:'top_k', en:'Top-K Stocks', label:'选股数量', type:'int', default:30, min:5, max:100,
     desc:'每步选多少只股票建仓。10-30为合理范围，越大越分散',
     range_desc:'↓小=集中持仓(10~20) · ↑大=分散风险(30~50) · 推荐: 30'},

    // 收益与成本
    {key:'forward_period', en:'Hold Days', label:'收益窗口(天)', type:'int', default:20, min:1, max:60,
     desc:'持有多天后计算收益。10≈两周，20≈一个月。与实际持仓周期对齐',
     range_desc:'↓短=短线(5~10天) · ↑长=中长线(20~40天) · 推荐: 20'},
    {key:'cost_bps', en:'Cost (bps)', label:'交易成本(bps)', type:'float', default:30.0, min:5, max:100,
     desc:'单边交易成本(基点)。A股实际约20-30bps（含印花税+滑点）',
     range_desc:'↓小=乐观假设(10~20) · ↑大=保守假设(30~50) · 推荐: 30'},

    // 优化器
    {key:'learning_rate', en:'Learning Rate', label:'学习率', type:'float', default:0.0003, min:0.00001, max:0.01, step:0.00001,
     desc:'模型更新步长。3e-4是PPO默认值，太大训练不稳定，太小收敛慢',
     range_desc:'↓小=稳定但慢(1e-4) · ↑大=快但可能震荡(1e-3) · 推荐: 3e-4'},
    {key:'target_kl', en:'KL Target', label:'KL阈值', type:'float', default:0.05, min:0.0, max:0.5, step:0.01,
     desc:'PPO KL散度裁剪阈值。0.05防一步更新太大导致崩溃。0=不限制',
     range_desc:'↓小=保守更新(0.01) · ↑大=激进更新(0.1) · 推荐: 0.05'},
    {key:'seed', en:'Random Seed', label:'随机种子', type:'str', default:'随机',
     desc:'随机种子。"随机"=每次不同，固定值可复现训练结果',
     range_desc:'固定=可复现 · 随机=探索不同结果'},

    // 高级
    {key:'policy_net', en:'Network Size', label:'🧠 策略网络', type:'select', default:'[64,64] (默认)', resume_hide:true,
     options:[
       {v:'',t:'[64,64] (默认 - CPU友好)'},
       {v:'{"net_arch":[256,256]}',t:'[256,256] (GPU轻量)'},
       {v:'{"net_arch":[512,512]}',t:'[512,512] (GPU适中 - 推荐)'},
       {v:'{"net_arch":[1024,1024]}',t:'[1024,1024] (GPU重度 - RTX4070+)'},
       {v:'{"net_arch":[1024,512]}',t:'[1024,512] (非对称)'},
     ],
     desc:'MLP网络层宽度。越大GPU利用率越高但收敛更难。[64,64]≈CPU负载, [512,512]≈GPU负载',
     range_desc:'↓小=CPU轻量(默认) · ↑大=GPU满载 · 推荐: [512,512] 吃GPU | 留空=CPU'},
    {key:'n_steps', en:'Rollout Steps', label:'Rollout步数', type:'int', default:0, min:0, max:8192,
     desc:'每轮每环境收集多少步再更新。0=PPO默认2048。增大可降低更新频率',
     range_desc:'↓小=高频更新(1024) · ↑大=低频稳定(4096) · 推荐: 0(默认2048)'},
    {key:'batch_size', en:'Batch Size', label:'批大小', type:'int', default:0, min:0, max:4096,
     desc:'SGD mini-batch大小。0=PPO默认64。增大提升GPU利用率但吃显存',
     range_desc:'↓小=训练快(64) · ↑大=GPU占满(512~2048) · 推荐: 0(默认64)'},
];

function renderTrainParams() {
    const grid = document.getElementById('trainParamGrid');
    if(!grid) return;
    const _loadPanelOptions = window.loadPanelOptions || function(){};
    const isResume = document.querySelector('input[name="trainMode"]:checked')?.value === 'resume';
    grid.innerHTML = TRAIN_PARAMS.filter(p => !isResume || !p.resume_hide).map(p => {
        let inp = '';
        if (p.type === 'select') {
            let opts = '';
            if (p.options === 'panels') {
                // 异步加载，先放占位
                opts = '<option value="">加载面板列表中...</option>';
                setTimeout(_loadPanelOptions, 100);
            } else if (Array.isArray(p.options)) {
                opts = p.options.map(o => `<option value="${o.v.replace(/"/g, '&quot;')}" ${o.v===p.default?'selected':''}>${o.t}</option>`).join('');
            }
            inp = `<select id="tp_${p.key}">${opts}</select>`;
        } else if (p.type === 'int' || p.type === 'float') {
            inp = `<input type="number" id="tp_${p.key}" value="${p.default}" min="${p.min||0}" max="${p.max||99999}" step="${p.step||(p.type==='float'?0.0001:1)}" style="width:100px">`;
        } else {
            inp = `<input type="text" id="tp_${p.key}" value="${p.default}" style="width:120px">`;
        }
        let rangeHint = (p.min !== undefined && p.max !== undefined) ? `<span class="phint">范围: ${p.min}~${p.max}</span>` : '';
        let enTag = p.en ? `<span class="pen">${p.en}</span>` : '';
        let rangeDesc = p.range_desc ? `<br><span class="prange">${p.range_desc}</span>` : '';
        return `<div class="param-item">
            <div class="plabel">${p.label} ${enTag}</div>
            <div class="pdesc">${p.desc}${rangeDesc}</div>
            <div class="pinput">${inp} ${rangeHint}</div>
        </div>`;
    }).join('');
}

function onTrainModeChange() {
    const isResume = document.querySelector('input[name="trainMode"]:checked')?.value === 'resume';
    document.getElementById('resumeModelGroup').style.display = isResume ? 'inline-flex' : 'none';
    document.getElementById('resumeHint').textContent = isResume
        ? '续训: 加载已有模型的 ppo_final.zip 继续训练，保留网络结构/权重'
        : '新训练: 从零开始训练模型';
    renderTrainParams();
    if (isResume) refreshResumeModels();
}

function refreshResumeModels() {
    apiFetch('/api/train/models').then(d => {
        const sel = document.getElementById('resumeModelSelect');
        sel.innerHTML = '<option value="">-- 请选择续训模型 --</option>';
        if (d.models) {
            d.models.forEach(m => {
                if (!m.has_final_zip) return;  // 没有 zip 无法续训
                sel.innerHTML += `<option value="${m.path}/ppo_final.zip">${m.name} (${m.n_stocks||'?'}只, ${m.total_timesteps||'?'}步, ${m.modified})</option>`;
            });
        }
        if (sel.options.length === 1) {
            sel.innerHTML = '<option value="">没有可选续训模型 (需含 ppo_final.zip)</option>';
        }
    }).catch(() => {});
}

function getTrainParams() {
    const r = {};
    TRAIN_PARAMS.forEach(p => {
        const el = document.getElementById('tp_' + p.key);
        if (!el) return;
        if (p.type === 'int') r[p.key] = parseInt(el.value) || 0;
        else if (p.type === 'float') r[p.key] = parseFloat(el.value) || 0;
        else r[p.key] = el.value;
    });
    return r;
}

function getPresets() {
    try { return JSON.parse(localStorage.getItem(PRESET_KEY) || '{}'); }
    catch(e) { return {}; }
}

function setPresets(p) { localStorage.setItem(PRESET_KEY, JSON.stringify(p)); }

function refreshPresetSelect() {
    const sel = document.getElementById('presetSelect');
    if(!sel) return;
    const presets = getPresets();
    sel.innerHTML = '<option value="">-- 选择预设 --</option>' +
        Object.keys(presets).sort().map(n => `<option value="${n}">${n}</option>`).join('');
}

function getCurrentParams() {
    const out = {};
    TRAIN_PARAMS.forEach(p => {
        const el = document.getElementById('tp_' + p.key);
        if (el) out[p.key] = el.value;
    });
    return out;
}

function setCurrentParams(params) {
    TRAIN_PARAMS.forEach(p => {
        const el = document.getElementById('tp_' + p.key);
        if (el && params[p.key] !== undefined) el.value = params[p.key];
    });
}

function savePreset() {
    const name = document.getElementById('presetName').value.trim();
    if (!name) { alert('请输入预设名'); return; }
    const presets = getPresets();
    presets[name] = getCurrentParams();
    setPresets(presets);
    refreshPresetSelect();
    document.getElementById('presetSelect').value = name;
    toast('💾 预设已保存: ' + name);
    addLog(`💾 预设已保存: ${name}`);
}

function loadPreset(name) {
    if (!name) return;
    const presets = getPresets();
    if (!presets[name]) { toast('❌ 预设不存在'); return; }
    setCurrentParams(presets[name]);
    toast('📂 已加载预设: ' + name);
    addLog(`📂 已加载预设: ${name}`);
}

function deletePreset() {
    const name = document.getElementById('presetSelect').value;
    if (!name) { alert('请先选择要删除的预设'); return; }
    if (!confirm(`删除预设 ${name}？`)) return;
    const presets = getPresets();
    delete presets[name];
    setPresets(presets);
    refreshPresetSelect();
    toast('🗑 已删除预设: ' + name);
    addLog(`🗑 已删除预设: ${name}`);
}

function resetPreset() {
    const defaults = {};
    TRAIN_PARAMS.forEach(p => defaults[p.key] = String(p.default));
    setCurrentParams(defaults);
    toast('↺ 已重置为默认参数');
    addLog('↺ 已重置为默认参数');
}

function refreshChartModels() {
    apiFetch('/api/all/models').then(d => {
        const sel = document.getElementById('chartModelSelect');
        sel.innerHTML = '<option value="">-- 请选择 --</option>' +
            (d.models || []).map(m => {
                // value = source:name
                const v = m.source + ':' + m.name;
                const steps = m.total_timesteps ? (m.total_timesteps/1e6).toFixed(1)+'M' : '?';
                const stocks = m.n_stocks || '?';
                return `<option value="${v}">${m.display_name} (${steps}步, ${stocks}只, ${m.modified})</option>`;
            }).join('');
    }).catch(e => { addLog('❌ 图表模型列表加载失败: ' + e.message); });
}

function pollTrainStatus() {
    apiFetch('/api/train/status').then(d => {
        const badge = document.getElementById('trainBadge');
        const btnStart = document.getElementById('btnTrainStart');
        const btnStop = document.getElementById('btnTrainStop');
        if (d.running) {
            badge.className = 'status-badge status-running';
            badge.textContent = `运行中 (${d.algorithm || '?'})`;
            btnStart.disabled = true;
            btnStop.disabled = false;
            // P2 align: 显示面板 + 步数信息
            const panelShort = (d.panel || '').replace('factor_panel_','').replace('.parquet','');
            const steps = d.total_timesteps ? `${(d.total_timesteps/1e6).toFixed(1)}M 步` : '';
            const info = [panelShort, steps].filter(Boolean).join(' · ');
            document.getElementById('trainProgress').textContent = info
                ? `⏳ ${info} — 查看日志...`
                : '⏳ 训练进行中，查看日志...';
        } else {
            badge.className = 'status-badge status-idle';
            badge.textContent = '空闲';
            btnStart.disabled = false;
            btnStop.disabled = true;
            const progress = document.getElementById('trainProgress');
            if (progress) progress.textContent = '';
        }
    }).catch(e => addLog('❌ 读取训练状态失败: ' + e.message));
}

function startTraining() {
    const params = getTrainParams();
    // 随机种子处理
    if (params.seed === '' || params.seed === '随机') {
        params.seed = Math.floor(Math.random() * 99999) + 1;
    } else {
        params.seed = parseInt(params.seed) || Math.floor(Math.random() * 99999) + 1;
    }
    // 策略网络 → policy_kwargs_json
    params.policy_kwargs_json = params.policy_net || '';
    delete params.policy_net;
    // 续训模式
    const isResume = document.querySelector('input[name="trainMode"]:checked')?.value === 'resume';
    if (isResume) {
        const sel = document.getElementById('resumeModelSelect');
        params.resume_from = sel.value;
        if (!params.resume_from) { addLog('❌ 请选择续训模型'); return; }
        // 显示日志
        addLog('🔄 续训模式: 从 ' + sel.options[sel.selectedIndex].text + ' 继续训练');
    } else {
        params.resume_from = '';
    }

    apiFetch('/api/train/start', {
        method: 'POST',
        body: params,
    }).then(d => {
        if (d.error || d.detail) throw new Error(d.error || d.detail);
        addLog('🚀 训练任务已提交: ' + (d.out_dir || d.model || '任务已提交'));
        document.getElementById('trainProgress').textContent = '已提交, 等待开始...';
        setTimeout(loadModels, 3000);
    }).catch(e => {
        addLog('❌ ' + e.message);
    });
}

function stopTraining() {
    apiFetch('/api/train/stop', {method: 'POST'})
        .then(() => addLog('⏹ 正在停止训练...'))
        .catch(e => addLog('❌ ' + e.message));
}

function loadModels(targetId) {
    const listId = targetId || 'modelList';
    apiFetch('/api/all/models').then(d => {
        const list = document.getElementById(listId);
        if (!d.models || d.models.length === 0) {
            list.innerHTML = '<span style="color:#667788;font-size:12px">暂无已训练的模型</span>';
            return;
        }
        // 批量操作工具栏
        const toolbar = `<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:8px;padding:6px;background:#0f1f2e;border:1px solid #1f3a52;border-radius:6px">
            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;color:#4fc3f7;font-size:12px"><input type="checkbox" id="modelSelectAll" onchange="toggleAllModels(this.checked)"> 全选</label>
            <span style="color:#667788;font-size:11px" id="modelSelectedCount">已选 0</span>
            <button class="btn btn-sm" style="background:#c62828;color:#fff;padding:2px 10px" onclick="batchDeleteModels()">🗑 批量删除</button>
            <button class="btn btn-sm" style="background:#1a2a3a;color:#8899aa;border:1px solid #2a3a4a;padding:2px 10px" onclick="loadModels()">🔄 刷新</button>
        </div>`;
        list.innerHTML = toolbar + d.models.map(m => {
            const stocks = m.n_stocks ? `${m.n_stocks} 只` : '? 只';
            const steps = m.total_timesteps ? `${m.total_timesteps.toLocaleString()} 步` : '? 步';
            const tags = [
                m.has_onnx ? '✅ONNX' : '',
                m.has_metrics ? '✅指标' : (m.has_training_summary ? '✅训练摘要' : ''),
                m.has_final_zip ? '✅可续训' : '',
                m.source === 'runs' ? '🌊' : '',
            ].filter(Boolean).join(' ');
            // Checkpoint 信息
            let ckptHtml = '';
            if (m.has_checkpoints && m.has_final_zip) {
                ckptHtml = `<div style="margin-top:4px;font-size:11px;color:#ff9800">
                    📦 Checkpoints: ${m.checkpoints_size_mb} MB
                    <button onclick="cleanCheckpoints('${m.name}', this)"
                        style="background:#1a2a3a;color:#ff9800;border:1px solid #ff9800;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:10px;margin-left:6px">
                        🧹 清理
                    </button>
                </div>`;
            }
            const sizeInfo = m.has_checkpoints
                ? `${m.size_no_ckpt_mb} MB (含 Checkpoints: ${m.size_mb} MB)`
                : `${m.size_mb} MB`;
            // runs 用 p22c 删除 API；models 用 train 删除 API
            const delFn = m.source === 'runs' ? 'deleteP22cRun' : 'deleteModel';
            return `
            <div class="file-item" style="display:flex;align-items:flex-start;gap:8px;padding:6px 10px;border-bottom:1px solid #1f3a52">
                <input type="checkbox" class="model-checkbox" value="${m.name}" data-source="${m.source}" onchange="updateModelSelectedCount()" style="margin-top:4px;flex-shrink:0">
                <div style="flex:1;min-width:0">
                    <div class="file-name" style="font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${m.name}">${m.display_name}</div>
                    <div class="file-size" style="font-size:11px;color:#8899aa">
                        📅 ${m.modified} · 💾 ${sizeInfo}<br>
                        🏢 ${stocks} · 🚶 ${steps} · ${tags}
                    </div>
                    ${ckptHtml}
                </div>
                <button onclick="${delFn}('${m.name}', this)"
                    style="background:#c62828;color:#fff;border:none;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px;flex-shrink:0">
                    🗑
                </button>
            </div>`;
        }).join('');
    }).catch(e => {
        const list = document.getElementById(listId);
        if (list) list.innerHTML = '<span style="color:#f44336;font-size:12px">加载失败</span>';
        addLog('❌ 模型列表加载失败: ' + e.message);
    });
}

function toggleAllModels(checked){
    document.querySelectorAll('#modelList .model-checkbox').forEach(cb=> cb.checked=checked);
    updateModelSelectedCount();
}
function updateModelSelectedCount(){
    const n=document.querySelectorAll('#modelList .model-checkbox:checked').length;
    const el=document.getElementById('modelSelectedCount');
    if(el) el.textContent='已选 '+n;
    const all=document.querySelectorAll('#modelList .model-checkbox');
    const selAll=document.getElementById('modelSelectAll');
    if(selAll) selAll.checked = all.length>0 && n===all.length;
}
async function batchDeleteModels(){
    const checked=[...document.querySelectorAll('#modelList .model-checkbox:checked')];
    if(checked.length===0){ toast('请选择要删除的模型'); return; }
    const names=checked.map(c=> c.value);
    const preview=names.slice(0,8).join('\n') + (names.length>8 ? '\n...等'+names.length+'个' : '');
    if(!confirm(`确定批量删除 ${names.length} 个模型？\n\n${preview}\n\n该操作不可撤销！`)) return;
    addLog(`🗑 批量删除 ${names.length} 个模型开始...`);
    let ok=0, fail=0;
    for(const cb of checked){
        const name=cb.value;
        const source=cb.dataset.source;
        const isRun = source==='runs';
        const url = isRun ? `/api/p22c/models/${encodeURIComponent(name)}` : `/api/train/models/${encodeURIComponent(name)}`;
        // fallback: 波 hunter v2 模型走 /api/wavehunter/v2/models
        const urls = isRun ? [url] : [url, `/api/wavehunter/v2/models/${encodeURIComponent(name)}`, `/api/wavehunter/models/${encodeURIComponent(name)}`];
        let deleted=false;
        for(const u of urls){
            try{
                const r=await fetch(u, {method:'DELETE'});
                const d=await r.json();
                if(r.ok && (d.status==='deleted' || d.status==='ok')){ deleted=true; break; }
                if(r.status===404) continue;
            }catch(e){}
        }
        if(deleted){ ok++; } else { fail++; addLog(`❌ 删除失败: ${name}`); }
    }
    addLog(`✅ 批量删除完成: 成功 ${ok} / 失败 ${fail}`);
    toast(`批量删除完成: 成功 ${ok} / 失败 ${fail}`);
    loadModels();
}

function deleteModel(name, btn) {
    if (!confirm(`确定删除模型 ${name}？\n\n该目录所有文件（含 ONNX/checkpoint/tb_logs）将被永久删除！`)) return;
    btn.disabled = true;
    btn.textContent = '⏳';
    fetch(`/api/train/models/${encodeURIComponent(name)}`, { method: 'DELETE' })
    .then(r => r.json())
    .then(d => {
        if (d.status === 'deleted') {
            addLog(`🗑 已删除模型 ${name} (${d.size_mb} MB)`);
            loadModels();
        } else {
            throw new Error(d.detail || '删除失败');
        }
    })
    .catch(e => {
        btn.disabled = false; btn.textContent = '🗑';
        addLog('❌ 删除模型失败: ' + e.message, 'trainLogPanel', 'trainLogCount');
    });
}
