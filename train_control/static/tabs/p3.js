// tabs/p3.js — 📦 P3 数据包构建 Tab
// 从 index.html inline JS 抽离；依赖全局 fetch/addLog
// P3 数据包构建
// ══════════════════════════════════════════════════
let _p3PollTimer = null;
let _p3SelectedPool = 'hs300'; // 默认选中沪深300

const P3_POOL_MAP = {
    'sse50': {file: '', dir: 'p3_sse50', pool: 'sse50'},
    'hs300': {file: '', dir: 'p3_hs300', pool: 'hs300'},
    'csi500': {file: '', dir: 'p3_csi500', pool: 'csi500'},
    'cs800': {file: '', dir: 'p3_cs800', pool: 'cs800'},
};

function selectP3Pool(pool) {
    _p3SelectedPool = pool;
    // 同步select下拉框
    const sel = document.getElementById('p3PoolSelect');
    if (sel) sel.value = pool;
}

function startP3Build() {
    const pool = P3_POOL_MAP[_p3SelectedPool];
    const fwd = document.getElementById('p3ForwardPeriod').value;
    const startDate = document.getElementById('p3StartDate').value;
    const endDate = document.getElementById('p3EndDate').value;

    document.getElementById('p3BuildBtn').disabled = true;
    document.getElementById('p3BuildBtn').textContent = '⏳ 构建中...';
    document.getElementById('p3BuildBadge').className = 'status-badge status-running';
    document.getElementById('p3BuildBadge').textContent = '构建中';
    addLog(`📦 开始构建 P3 数据包: ${_p3SelectedPool} (${startDate} ~ ${endDate})`);

    const params = new URLSearchParams({
        factor_panel: pool.file || pool.pool || _p3SelectedPool,
        out_dir: pool.dir,
        forward_period: fwd,
        start_date: startDate,
        end_date: endDate
    });
    fetch(`/api/p3/build/start?${params}`, {method: 'POST'})
        .then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); }))
        .then(d => {
            if (d.error || d.detail) throw new Error(d.error || d.detail);
            addLog(`📦 构建任务已启动: ${d.task_id || d.out_dir || _p3SelectedPool}`);
            _pollP3Build();
        })
        .catch(e => {
            addLog(`❌ 启动失败: ${e.message}`);
            document.getElementById('p3BuildBtn').disabled = false;
            document.getElementById('p3BuildBtn').textContent = '▶ 开始构建';
            document.getElementById('p3BuildBadge').className = 'status-badge status-idle';
            document.getElementById('p3BuildBadge').textContent = '空闲';
        });
}

function _pollP3Build() {
    if (_p3PollTimer) clearInterval(_p3PollTimer);
    _p3PollTimer = setInterval(() => {
        fetch('/api/p3/build/status').then(r => r.json()).then(d => {
            if (d.output) {
                const line = String(d.output).split('\\n').filter(Boolean).slice(-1)[0];
                if (line) addLog('📦 P3进度: ' + line);
            }
            if (!d.running && d.result) {
                clearInterval(_p3PollTimer);
                _p3PollTimer = null;
                document.getElementById('p3BuildBtn').disabled = false;
                document.getElementById('p3BuildBtn').textContent = '▶ 开始构建';

                if (d.result.status === 'ok') {
                    document.getElementById('p3BuildBadge').className = 'status-badge status-done';
                    document.getElementById('p3BuildBadge').textContent = '完成';
                    const m = d.output.match(/"n_features":\s*(\d+)[^}]*"n_stocks":\s*(\d+)[^}]*"n_dates":\s*(\d+)/);
                    if (m) {
                        addLog(`✅ P3 数据包构建完成: ${m[1]}因子 / ${m[2]}只 / ${m[3]}天`);
                    }
                    loadP3Bundles();
                } else {
                    document.getElementById('p3BuildBadge').className = 'status-badge status-error';
                    document.getElementById('p3BuildBadge').textContent = '失败';
                    addLog(`❌ P3 构建失败: ${d.result.error || d.result.detail || JSON.stringify(d.result)}`);
                }
            }
        });
    }, 3000);
}

function loadP3Bundles() {
    fetch('/api/p3/bundles').then(r => r.json()).then(d => {
        const el = document.getElementById('p3BundleList');
        if (!d.bundles || d.bundles.length === 0) {
            el.innerHTML = '<span style="color:#667788;font-size:12px">暂无已构建的数据包</span>';
            return;
        }
        el.innerHTML = d.bundles.map(b => {
            const dr = b.date_range || [];
            const dataSpan = dr.length >= 2 ? dr[0].slice(0,10) + ' ~ ' + dr[1].slice(0,10) : '';
            const sp = b.splits || {};
            const trainSpan = sp.train ? sp.train[0] + ' ~ ' + sp.train[1] : '';
            return `<div class="file-item" style="display:flex;align-items:center;gap:8px;padding:6px 10px">
                <div style="flex:1">
                    <div class="file-name">${b.name}</div>
                    <div class="file-size">${b.size_gb} GB · ${b.modified}</div>
                    ${dataSpan ? `<div style="color:#667788;font-size:10px">数据: ${dataSpan}</div>` : ''}
                    ${trainSpan ? `<div style="color:#667788;font-size:10px">训练: ${trainSpan}</div>` : ''}
                </div>
                <button onclick="deleteP3Bundle('${b.name}', this)"
                    style="background:#c62828;color:#fff;border:none;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:12px">
                    🗑 删除
                </button>
            </div>`;
        }).join('');
    }).catch(e => {
        document.getElementById('p3BundleList').innerHTML = '<span style="color:#f44336;font-size:12px">加载失败</span>';
        addLog('❌ P3数据包列表加载失败: ' + e.message);
    });
}

function deleteP3Bundle(name, btn) {
    if (!confirm(`确定要删除 ${name} 吗？\n\n此操作不可撤销！`)) return;
    btn.disabled = true;
    btn.textContent = '⏳ 删除中...';
    fetch(`/api/p3/bundles/${encodeURIComponent(name)}`, { method: 'DELETE' })
    .then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); }))
    .then(d => {
        addLog(`🗑 已删除: ${name}`);
        loadP3Bundles();
    })
    .catch(e => {
        addLog(`❌ 删除失败: ${e.message}`);
        btn.disabled = false;
        btn.textContent = '🗑 删除';
    });
}

// 初始化
selectP3Pool('hs300');
loadP3Bundles();

// ═══════════════════════════════════════════════════════════════
