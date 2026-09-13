/** tabs/panel.js — 因子面板 Tab（v1/v2/v8 构建 + 已生成列表）
 * 依赖全局: fetch / addLog
 * 由 index.html <script src="/static/tabs/panel.js"> 加载
 * 负责 #tab-panel 区域；状态走 /api/build/* + /api/wavehunter/panel/build/* + /api/panels
 */
function panelSetDateAll() {
    document.getElementById('startDate').value='20040102';
    const today = new Date();
    document.getElementById('endDate').value = today.getFullYear().toString() +
        String(today.getMonth()+1).padStart(2,'0') + String(today.getDate()).padStart(2,'0');
    const btn = event.target;
    const orig = btn.textContent;
    btn.textContent = '✅ 已设置';
    btn.style.background = '#2e7d32';
    setTimeout(() => { btn.textContent = orig; btn.style.background = ''; }, 1500);
}

function startBuild() {
    const pool = document.getElementById('poolSelect').value;
    const start = document.getElementById('startDate').value.trim() || '20040102';
    const end = document.getElementById('endDate').value.trim();
    const includeFundamental = document.getElementById('includeFundamental').checked;
    const btn = document.getElementById('btnStart');
    if (btn) btn.disabled = true;
    apiFetch('/api/build/start', {
        method: 'POST',
        body: {pool, start_date: start, end_date: end, include_fundamental: includeFundamental},
    }).then(d => {
        addLog('🚀 构建任务已提交' + (includeFundamental ? ' [含基本面]' : ' [纯技术]'));
        setTimeout(loadPanels, 1000);
    }).catch(e => {
        addLog('❌ 面板构建失败: ' + e.message);
        if (btn) btn.disabled = false;
    });
}

function stopBuild() {
    apiFetch('/api/panel/build/stop', {method: 'POST'}).then(d => {
        addLog('⏹ 面板构建已停止: ' + ((d.stopped || []).join(', ') || '无活动任务'), 'panel_build');
        ['statusBadge','v2BuildBadge','v8BuildBadge'].forEach(id => {
            const el = document.getElementById(id);
            if (el) { el.textContent = '空闲'; el.className = 'status-badge status-idle'; }
        });
        ['btnStart','v2BuildBtn','v8BuildBtn','btnStopBuild'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.disabled = false;
        });
    }).catch(e => addLog('❌ 停止失败: ' + e.message, 'panel_build'));
}

let _v8BuildPollTimer = null;
function startV8Build() {
    const btn = document.getElementById('v8BuildBtn'), badge = document.getElementById('v8BuildBadge');
    const pool = document.getElementById('poolSelect').value, start_date = document.getElementById('startDate').value.trim() || '20040102';
    let end_date = document.getElementById('endDate').value.trim();
    if (!end_date) {
        const t = new Date(); end_date = t.getFullYear() + String(t.getMonth() + 1).padStart(2, '0') + String(t.getDate()).padStart(2, '0');
    }
    _v8BuildLogIdx = 0;
    btn.disabled = true; badge.textContent = '提交中'; badge.className = 'status-badge status-running';
    apiFetch('/api/wavehunter/v8/panel/build/start', {
        method: 'POST',
        body: {pool, start_date, end_date},
    }).then(d => {
        addLog('🚀 v8 因子面板构建已启动: ' + pool, 'panel_build');
        pollV8Build();
    }).catch(e => {
        badge.textContent = '失败'; badge.className = 'status-badge status-failed';
        btn.disabled = false;
        addLog('❌ v8 面板构建失败: ' + e.message, 'panel_build');
    });
}
function pollV8Build() {
    clearTimeout(_v8BuildPollTimer);
    apiFetch('/api/wavehunter/v8/panel/build/status').then(d => {
        const badge = document.getElementById('v8BuildBadge'), btn = document.getElementById('v8BuildBtn');
        appendBuildRuntimeLogs(d, 'v8面板构建');
        const logs = d.log || [];
        if (d.running) {
            badge.textContent = '构建中'; badge.className = 'status-badge status-running';
            _v8BuildPollTimer = setTimeout(pollV8Build, 2000);
        } else {
            const ok = d.status === 'done' && d.result;
            badge.textContent = ok ? '完成' : (d.status === 'failed' ? '失败' : '空闲');
            badge.className = 'status-badge ' + (ok ? 'status-done' : (d.status === 'failed' ? 'status-failed' : 'status-idle'));
            btn.disabled = false;
            if (ok) loadPanels();
        }
    }).catch(e => {
        addLog('❌ v8 状态读取失败: ' + e.message, 'panel_build');
        _v8BuildPollTimer = setTimeout(pollV8Build, 5000);
    });
}

let _v2BuildPollTimer = null;
let _v2BuildTaskId = null;
let _v2BuildSeenLogs = new Set();
function startV2Build() {
    const badge = document.getElementById('v2BuildBadge');
    const btn = document.getElementById('v2BuildBtn');
    badge.className = 'status-badge status-running';
    badge.textContent = '构建中';
    btn.disabled = true;
    apiFetch('/api/build/v2', {
        method: 'POST',
        body: {
            pool: document.getElementById('poolSelect').value,
            start_date: document.getElementById('startDate').value,
            end_date: document.getElementById('endDate').value,
            include_fundamental: document.getElementById('includeFundamental').checked,
            version: 'v2',
            threshold: 0.6,
        },
    }).then(d => {
        if (!d.task_id) throw new Error('v2 启动响应缺少 task_id');
        _v2BuildTaskId = d.task_id;
        addLog('🚀 v2 面板构建已提交: ' + d.task_id, 'panel_build');
        pollV2BuildStatus();
    }).catch(e => {
        badge.textContent = '失败'; badge.className = 'status-badge status-failed'; btn.disabled = false;
        addLog('❌ v2 构建失败: ' + e.message, 'panel_build');
    });
}
let _v10BuildPollTimer = null;
let _v10BuildDone = false;
function startV10Build() {
    const btn = document.getElementById('v10BuildBtn'), badge = document.getElementById('v10BuildBadge'), logEl = document.getElementById('v10BuildLog');
    const pool = document.getElementById('poolSelect').value, start_date = document.getElementById('startDate').value.trim() || '20040102';
    let end_date = document.getElementById('endDate').value.trim();
    if (!end_date) { const t = new Date(); end_date = t.getFullYear() + String(t.getMonth() + 1).padStart(2, '0') + String(t.getDate()).padStart(2, '0'); }
    btn.disabled = true; badge.textContent = '提交中'; badge.className = 'status-badge status-running';
    if (logEl) logEl.textContent = ''; _v10BuildDone = false; window._v10BuildLogIdx = 0;
    apiFetch('/api/wavehunter/v10/panel/build/start', {
        method: 'POST',
        body: {pool, start_date, end_date},
    }).then(d => {
        addLog('🚀 v10 独立面板构建已启动: ' + pool, 'panel_build');
        if (logEl) logEl.textContent = '已提交...';
        pollV10Build();
    }).catch(e => {
        badge.textContent = '失败'; badge.className = 'status-badge status-failed'; btn.disabled = false;
        if (logEl) logEl.textContent = '❌ ' + e.message;
        addLog('❌ v10 构建失败: ' + e.message, 'panel_build');
    });
}
function pollV10Build() {
    clearTimeout(_v10BuildPollTimer); if (_v10BuildDone) return;
    apiFetch('/api/wavehunter/v10/panel/build/status').then(d => {
        const btn = document.getElementById('v10BuildBtn'), badge = document.getElementById('v10BuildBadge'), logEl = document.getElementById('v10BuildLog');
        if (d.log && logEl) logEl.textContent = d.log.slice(-6).join('\n');
        if (d.log) d.log.forEach((line, i) => {
            if (line && i >= (window._v10BuildLogIdx || 0)) addLog('[v10独立面板] ' + line, 'panel_build');
        });
        window._v10BuildLogIdx = (d.log || []).length;
        if (d.running) {
            badge.textContent = '构建中'; badge.className = 'status-badge status-running';
            _v10BuildPollTimer = setTimeout(pollV10Build, 2000);
        } else {
            const ok = d.status === 'done' && d.result;
            badge.textContent = ok ? '完成' : (d.status === 'failed' ? '失败' : '空闲');
            badge.className = 'status-badge ' + (ok ? 'status-done' : (d.status === 'failed' ? 'status-failed' : 'status-idle'));
            btn.disabled = false; _v10BuildDone = true;
            if (ok) { addLog('✅ v10 独立面板构建完成', 'panel_build'); loadPanels(); }
        }
    }).catch(e => {
        addLog('❌ v10 状态失败: ' + e.message, 'panel_build');
        _v10BuildPollTimer = setTimeout(pollV10Build, 5000);
    });
}


let _v10_1BuildPollTimer = null;
let _v10_1BuildDone = false;
function startV10_1Build() {
    const btn = document.getElementById('v10_1BuildBtn'), badge = document.getElementById('v10_1BuildBadge'), logEl = document.getElementById('v10_1BuildLog');
    const pool = document.getElementById('poolSelect').value;
    const start_date = document.getElementById('startDate').value.trim() || '20040102';
    let end_date = document.getElementById('endDate').value.trim();
    if (!end_date) { const t = new Date(); end_date = t.getFullYear() + String(t.getMonth() + 1).padStart(2, '0') + String(t.getDate()).padStart(2, '0'); }
    btn.disabled = true; badge.textContent = '提交中'; badge.className = 'status-badge status-running';
    if (logEl) logEl.textContent = ''; _v10_1BuildDone = false; window._v10_1BuildLogIdx = 0;
    apiFetch('/api/wavehunter/v10.1/panel/build/start', {method:'POST', body:{pool, start_date, end_date}}).then(d => {
        addLog('🚀 v10.1 独立面板构建已启动: ' + pool, 'panel_build');
        if (logEl) logEl.textContent = '已提交...';
        pollV10_1Build();
    }).catch(e => {
        badge.textContent = '失败'; badge.className = 'status-badge status-failed'; btn.disabled = false;
        if (logEl) logEl.textContent = '❌ ' + e.message;
        addLog('❌ v10.1 构建失败: ' + e.message, 'panel_build');
    });
}
function pollV10_1Build() {
    clearTimeout(_v10_1BuildPollTimer); if (_v10_1BuildDone) return;
    apiFetch('/api/wavehunter/v10.1/panel/build/status').then(d => {
        const btn = document.getElementById('v10_1BuildBtn'), badge = document.getElementById('v10_1BuildBadge'), logEl = document.getElementById('v10_1BuildLog');
        if (d.log && logEl) logEl.textContent = d.log.slice(-6).join('\n');
        if (d.log) d.log.forEach((line, i) => { if (line && i >= (window._v10_1BuildLogIdx || 0)) addLog('[v10.1面板] ' + line, 'panel_build'); });
        window._v10_1BuildLogIdx = (d.log || []).length;
        if (d.running) {
            badge.textContent = '构建中'; badge.className = 'status-badge status-running';
            _v10_1BuildPollTimer = setTimeout(pollV10_1Build, 2000);
        } else {
            const ok = ['done','completed','finished'].includes(d.status) && d.result;
            badge.textContent = ok ? '完成' : (d.status === 'failed' ? '失败' : '空闲');
            badge.className = 'status-badge ' + (ok ? 'status-done' : (d.status === 'failed' ? 'status-failed' : 'status-idle'));
            btn.disabled = false; _v10_1BuildDone = true;
            if (ok) { addLog('✅ v10.1 独立面板构建完成', 'panel_build'); loadPanels(); }
        }
    }).catch(e => { addLog('❌ v10.1 状态失败: ' + e.message, 'panel_build'); _v10_1BuildPollTimer = setTimeout(pollV10_1Build, 5000); });
}

let _v9BuildDone = false;
function startV9Build() {
    const btn = document.getElementById('v9BuildBtn'), badge = document.getElementById('v9BuildBadge');
    const pool = document.getElementById('poolSelect').value, start_date = document.getElementById('startDate').value.trim() || '20040102';
    let end_date = document.getElementById('endDate').value.trim();
    if (!end_date) { const t = new Date(); end_date = t.getFullYear() + String(t.getMonth() + 1).padStart(2, '0') + String(t.getDate()).padStart(2, '0'); }
    btn.disabled = true; badge.textContent = '提交中'; badge.className = 'status-badge status-running';
    const logEl = document.getElementById('v9BuildLog'); if (logEl) logEl.textContent = '';
    window._v9BuildLogIdx = 0; _v9BuildDone = false;
    apiFetch('/api/wavehunter/v9/panel/build/start', {
        method: 'POST',
        body: {pool, start_date, end_date},
    }).then(d => {
        addLog('🚀 v9 面板构建启动: ' + pool, 'panel_build');
        if (logEl) logEl.textContent = '已提交...';
        pollV9Build();
    }).catch(e => {
        badge.textContent = '失败'; badge.className = 'status-badge status-failed'; btn.disabled = false;
        if (logEl) logEl.textContent = '❌ ' + e.message;
        addLog('❌ v9 构建失败: ' + e.message, 'panel_build');
    });
}
function pollV9Build() {
    clearTimeout(_v9BuildPollTimer);
    if (_v9BuildDone) return;
    apiFetch('/api/wavehunter/v9/panel/build/status').then(d => {
        const badge = document.getElementById('v9BuildBadge'), btn = document.getElementById('v9BuildBtn'), logEl = document.getElementById('v9BuildLog');
        if (d.log && logEl) logEl.textContent = d.log.slice(-6).join('\n');
        if (d.log) d.log.forEach((line, i) => {
            if (line && i >= (window._v9BuildLogIdx || 0)) addLog('[v9面板] ' + line, 'panel_build');
        });
        window._v9BuildLogIdx = (d.log || []).length;
        if (d.running) {
            badge.textContent = '构建中'; badge.className = 'status-badge status-running';
            _v9BuildPollTimer = setTimeout(pollV9Build, 2000);
        } else {
            const ok = d.status === 'done' && d.result;
            badge.textContent = ok ? '完成' : (d.status === 'failed' ? '失败' : '空闲');
            badge.className = 'status-badge ' + (ok ? 'status-done' : (d.status === 'failed' ? 'status-failed' : 'status-idle'));
            btn.disabled = false; _v9BuildDone = true;
            if (ok) { addLog('✅ v9 构建完成', 'panel_build'); loadPanels(); }
        }
    }).catch(e => {
        addLog('❌ v9 状态失败: ' + e.message, 'panel_build');
        _v9BuildPollTimer = setTimeout(pollV9Build, 5000);
    });
}
function pollV2BuildStatus() {
    if (_v2BuildPollTimer) clearTimeout(_v2BuildPollTimer);
    const url = '/api/build/v2/status' + (_v2BuildTaskId ? '?task_id=' + encodeURIComponent(_v2BuildTaskId) : '');
    const poll = () => apiFetch(url).then(d => {
            const lines = Array.isArray(d.log) ? d.log : [];
            lines.forEach((line, index) => {
                const key = (d.task_id || _v2BuildTaskId) + '|' + index + '|' + line;
                if (line && !_v2BuildSeenLogs.has(key)) {
                    addLog('[v2面板] ' + line, 'panel_build');
                    _v2BuildSeenLogs.add(key);
                }
            });
            if (!d.running) {
                _v2BuildPollTimer = null;
                v2BuildReset();
                if (d.result) {
                    if (d.result.status === 'ok') addLog('✅ v2 面板构建完成', 'panel_build');
                    else addLog('❌ v2 构建失败: ' + (d.result.error || JSON.stringify(d.result)), 'panel_build');
                    loadPanels();
                }
            } else {
                _v2BuildPollTimer = setTimeout(poll, 2000);
            }
        }).catch(e => { addLog('❌ v2 状态读取失败: ' + e.message, 'panel_build'); _v2BuildPollTimer = setTimeout(poll, 5000); });
    poll();
}
function v2BuildReset(){
    const badge=document.getElementById('v2BuildBadge'), btn=document.getElementById('v2BuildBtn');
    if(badge){badge.textContent='空闲';badge.className='status-badge status-idle';}
    if(btn) btn.disabled=false;
    _v2BuildTaskId=null; _v2BuildSeenLogs=new Set();
}

function loadPanels() {
    apiFetch('/api/panels').then(d => {
        const list = document.getElementById('panelList');
        if (!d.panels || d.panels.length === 0) {
            list.innerHTML = '<span style="color:#667788;font-size:12px">暂无已生成的面板</span>';
            return;
        }
        const panelHtml = d.panels.map(p => {
            const isV9 = p.name.includes('wavehunter_v9');
            const isV101 = p.name.includes('wavehunter_v10_1_');
            const isV10 = p.name.includes('wavehunter_v10') && !isV101;
            let statsHtml = '';
            if (isV9 || isV10 || isV101) {
                apiFetch(`/api/panels/${encodeURIComponent(p.name)}/stats`).then(stats => {
                    const el = document.getElementById(`stats-${p.name}`);
                    if (!el || !stats) return;
                    let inner = '';
                    if (stats.a1_density !== undefined) {
                        inner = `<span style="color:#4caf50">A1:${stats.a1_density}%</span> <span style="color:#2196f3">A2:${stats.a2_density}%</span> <span style="color:#ff9800">比:${stats.a1_a2_ratio}</span>`;
                    } else if (stats.v10_labels) {
                        const a1 = stats.v10_labels.a1_point || {};
                        const a2 = stats.v10_labels.a2_interval || {};
                        inner = `<span style="color:#4caf50">A1:${a1.density_pct ?? 0}%</span> <span style="color:#2196f3">A2:${a2.density_pct ?? 0}%</span>`;
                    }
                    el.innerHTML = inner || `<span style="color:#667788">${stats.columns} 列 · ${stats.total_rows} 行</span>`;
                }).catch(() => {});
                statsHtml = `<div id="stats-${p.name}" style="font-size:11px;color:#667788">加载中...</div>`;
            }
            return `<div class="file-item" style="display:flex;align-items:center;gap:8px;padding:6px 10px">
                <div style="flex:1">
                    <div class="file-name">${p.name}</div>
                    <div class="file-size">${p.size_gb} GB · ${p.modified}</div>
                    ${statsHtml}
                </div>
                <button onclick="deletePanel('${p.name}', this)" class="btn btn-sm" style="background:#7a3d3d;color:#fff">🗑 删除</button>
            </div>`;
        }).join('');
        list.innerHTML = panelHtml;
    }).catch(e => { const el = document.getElementById('panelList'); if (el) el.innerHTML = '<span style="color:#f44336;font-size:12px">❌ ' + e.message + '</span>'; });
}
function deletePanel(name, btn) {
    if (!confirm(`确定要删除 ${name} 吗？\n\n此操作不可撤销！`)) return;
    btn.disabled = true;
    btn.textContent = '⏳ 删除中...';
    addLog(`🗑 正在删除 ${name}...`, 'panel_ops');
    apiFetch(`/api/panels/${encodeURIComponent(name)}`, {method: 'DELETE'})
    .then(d => {
        if (d.status === 'deleted') {
            const statsNote = d.removed_stats && d.removed_stats.length
                ? ` (同步清理 ${d.removed_stats.length} 个 stats: ${d.removed_stats.join(', ')})`
                : '';
            addLog(`✅ 已删除 ${name} (${d.size_mb} MB)${statsNote}`, 'panel_ops');
            loadPanels();
        } else {
            throw new Error(d.detail || '删除失败');
        }
    })
    .catch(e => {
        btn.disabled = false; btn.textContent = '🗑 删除';
        addLog(`❌ 删除失败 ${name}: ${e.message}`, 'panel_ops');
    });
}
loadPanels();
