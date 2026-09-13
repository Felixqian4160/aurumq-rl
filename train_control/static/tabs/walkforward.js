// tabs/walkforward.js — 📊 Walk-forward 滚动验证
// 从 index.html inline JS 抽离；依赖全局 fetch/addLog
// Walk-forward 滚动验证
// ══════════════════════════════════════════════════
function wfStart() {
    const params = {
        panel: document.getElementById('wfPanel').value,
        steps: parseInt(document.getElementById('wfSteps').value) || 100000,
        train_years: parseInt(document.getElementById('wfTrainYears').value) || 2,
        val_years: parseInt(document.getElementById('wfValYears').value) || 1,
    };
    document.getElementById('wfBadge').className = 'status-badge status-running';
    document.getElementById('wfBadge').textContent = '运行中';
    document.getElementById('wfStartBtn').disabled = true;
    document.getElementById('wfStopBtn').disabled = false;
    document.getElementById('wfResults').style.display = 'none';
    fetch('/api/walkforward/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
    }).then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); })).then(d => {
        if (d.error || d.detail) { addLog('❌ ' + (d.error || d.detail)); wfResetBadge(); return; }
        addLog('🚀 Walk-forward 已启动: ' + (d.total_windows || d.windows || '窗口计算中'));
        pollWfStatus();
    }).catch(e => { addLog('❌ ' + e.message); wfResetBadge(); });
}
function wfStop() {
    fetch('/api/walkforward/stop', {method: 'POST'}).then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); })).then(() => {
        addLog('⏹ Walk-forward 已停止');
        wfResetBadge();
    }).catch(e => addLog('❌ ' + e.message));
}
function wfResetBadge() {
    document.getElementById('wfBadge').className = 'status-badge status-idle';
    document.getElementById('wfBadge').textContent = '空闲';
    document.getElementById('wfStartBtn').disabled = false;
    document.getElementById('wfStopBtn').disabled = true;
}
function pollWfStatus() {
    fetch('/api/walkforward/status').then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); })).then(d => {
        if (d.running) {
            document.getElementById('wfProgress').textContent = d.progress || '';
            setTimeout(pollWfStatus, 5000);
        } else {
            wfResetBadge();
            if (d.completed) loadWfResults();
        }
    }).catch(e => { addLog('❌ Walk-forward状态读取失败: ' + e.message); setTimeout(pollWfStatus, 10000); });
}
function loadWfResults() {
    fetch('/api/walkforward/result').then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); })).then(d => {
        if (!d.windows || d.windows.length === 0) {
            addLog('⚠ Walk-forward已结束，但没有窗口结果');
            return;
        }
        document.getElementById('wfResults').style.display = 'block';
        // 表格
        let html = '<table style="width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed">';
        html += '<tr style="background:#1a2a3a;color:#4fc3f7"><th style="padding:6px;text-align:center">窗口</th><th>训练期</th><th>验证期</th><th>Sharpe</th><th>超额</th><th>回撤</th></tr>';
        d.windows.forEach((w, i) => {
            const sharpeColor = w.sharpe > 0 ? '#f44336' : '#4caf50';
            html += `<tr style="border-bottom:1px solid #1f3a52">
                <td style="padding:4px 6px;text-align:center">${i+1}</td>
                <td style="padding:4px 6px;text-align:center">${w.train_start}~${w.train_end}</td>
                <td style="padding:4px 6px;text-align:center">${w.val_start}~${w.val_end}</td>
                <td style="padding:4px 6px;text-align:center;color:${sharpeColor};font-weight:bold">${w.sharpe.toFixed(3)}</td>
                <td style="padding:4px 6px;text-align:center;color:${w.excess > 0 ? '#f44336' : '#4caf50'}">${w.excess.toFixed(1)}%</td>
                <td style="padding:4px 6px;text-align:center;color:#4caf50">${w.max_dd.toFixed(1)}%</td>
            </tr>`;
        });
        html += '</table>';
        document.getElementById('wfTable').innerHTML = html;
        // 统计
        const avgSharpe = d.windows.reduce((s, w) => s + w.sharpe, 0) / d.windows.length;
        const posCount = d.windows.filter(w => w.sharpe > 0).length;
        const avgExcess = d.windows.reduce((s, w) => s + w.excess, 0) / d.windows.length;
        const verdict = avgSharpe < 0 ? '❌ 严重过拟合' : avgSharpe < 0.5 ? '⚠ 泛化一般' : '✅ 泛化良好';
        const verdictColor = avgSharpe < 0 ? '#f44336' : avgSharpe < 0.5 ? '#ff9800' : '#4caf50';
        document.getElementById('wfSummary').innerHTML = `
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">
                <div class="card" style="text-align:center;padding:12px">
                    <div style="font-size:10px;color:#667788">平均 Sharpe</div>
                    <div style="font-size:22px;font-weight:bold;color:${avgSharpe > 0 ? '#f44336' : '#4caf50'}">${avgSharpe.toFixed(3)}</div>
                </div>
                <div class="card" style="text-align:center;padding:12px">
                    <div style="font-size:10px;color:#667788">正 Sharpe 窗口</div>
                    <div style="font-size:22px;font-weight:bold;color:#4fc3f7">${posCount}/${d.windows.length}</div>
                </div>
                <div class="card" style="text-align:center;padding:12px">
                    <div style="font-size:10px;color:#667788">平均超额</div>
                    <div style="font-size:22px;font-weight:bold;color:${avgExcess > 0 ? '#f44336' : '#4caf50'}">${avgExcess.toFixed(1)}%</div>
                </div>
                <div class="card" style="text-align:center;padding:12px">
                    <div style="font-size:10px;color:#667788">过拟合判断</div>
                    <div style="font-size:18px;font-weight:bold;color:${verdictColor}">${verdict}</div>
                </div>
            </div>
        `;
        // 图表
        setTimeout(() => {
            const chart = echarts.init(document.getElementById('wfChart'));
            chart.setOption({
                backgroundColor: 'transparent',
                textStyle: { color: '#e0e0e0' },
                grid: { left: '8%', right: '6%', top: '10%', bottom: '14%' },
                xAxis: { type: 'category', data: d.windows.map((w, i) => 'W' + (i+1)), axisLabel: { color: '#8899aa' }, axisLine: { lineStyle: { color: '#2a3a4a' } } },
                yAxis: { type: 'value', axisLabel: { color: '#8899aa' }, splitLine: { lineStyle: { color: '#1a2a3a' } } },
                tooltip: { trigger: 'axis', backgroundColor: '#1a2a3a', borderColor: '#4fc3f7', textStyle: { color: '#e0e0e0' } },
                series: [
                    { type: 'bar', data: d.windows.map(w => w.sharpe), name: 'Sharpe', itemStyle: { color: p => p.value > 0 ? '#f44336' : '#4caf50' } },
                    { type: 'line', data: d.windows.map(w => w.sharpe), name: '趋势', smooth: true, symbol: 'none', lineStyle: { color: '#4fc3f7', width: 2 } }
                ],
            });
            chart.resize();
        }, 100);
        addLog(`📊 Walk-forward 完成: 平均 Sharpe ${avgSharpe.toFixed(3)}, ${posCount}/${d.windows.length} 正窗口`);
    }).catch(e => addLog('❌ 加载结果失败: ' + e.message));
}
function loadWfPanels() {
    fetch('/api/panels').then(r => r.json()).then(d => {
        const sel = document.getElementById('wfPanel');
        if (!sel) return;
        sel.innerHTML = (d.panels || []).map(p => `<option value="${p.name}">${p.name} (${p.size_gb} GB)</option>`).join('') || '<option value="">无面板</option>';
    }).catch(() => {});
}

// ══════════════════════════════════════════════════
// 参数矩阵扫描
// ══════════════════════════════════════════════════
function mtxStart() {
    const params = {
        panel: document.getElementById('mtxPanel').value,
        steps: parseInt(document.getElementById('mtxSteps').value) || 100000,
        max_position_pcts: document.getElementById('mtxMaxPos').value.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v)),
        top_ks: document.getElementById('mtxTopK').value.split(',').map(s => parseInt(s.trim())).filter(v => !isNaN(v)),
        rebalance_days_list: document.getElementById('mtxRebalance').value.split(',').map(s => parseInt(s.trim())).filter(v => !isNaN(v)),
        lstm_hiddens: document.getElementById('mtxHidden').value.split(',').map(s => parseInt(s.trim())).filter(v => !isNaN(v)),
    };
    const total = (params.max_position_pcts.length || 1) * (params.top_ks.length || 1) * (params.rebalance_days_list.length || 1) * (params.lstm_hiddens.length || 1);
    document.getElementById('mtxBadge').className = 'status-badge status-running';
    document.getElementById('mtxBadge').textContent = '扫描中';
    document.getElementById('mtxStartBtn').disabled = true;
    document.getElementById('mtxStopBtn').disabled = false;
    document.getElementById('mtxResults').style.display = 'none';
    fetch('/api/matrix/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
    }).then(r => r.json()).then(d => {
        if (d.error) { addLog('❌ ' + d.error); mtxResetBadge(); return; }
        addLog('🚀 参数矩阵已启动: ' + d.total + ' 个配置');
        pollMtxStatus();
    }).catch(e => { addLog('❌ ' + e.message); mtxResetBadge(); });
}
function mtxStop() {
    fetch('/api/matrix/stop', {method: 'POST'}).then(() => {
        addLog('⏹ 参数矩阵已停止');
        mtxResetBadge();
    }).catch(e => addLog('❌ ' + e.message));
}
function mtxResetBadge() {
    document.getElementById('mtxBadge').className = 'status-badge status-idle';
    document.getElementById('mtxBadge').textContent = '空闲';
    document.getElementById('mtxStartBtn').disabled = false;
    document.getElementById('mtxStopBtn').disabled = true;
}
function pollMtxStatus() {
    fetch('/api/matrix/status').then(r => r.json()).then(d => {
        if (d.running) {
            document.getElementById('mtxProgress').textContent = d.progress || '';
            setTimeout(pollMtxStatus, 5000);
        } else {
            mtxResetBadge();
            if (d.completed) loadMtxResults();
        }
    }).catch(() => setTimeout(pollMtxStatus, 10000));
}
function loadMtxResults() {
    fetch('/api/matrix/result').then(r => r.json()).then(d => {
        if (!d.results || d.results.length === 0) return;
        document.getElementById('mtxResults').style.display = 'block';
        // 表格（按 Sharpe 降序）
        const sorted = [...d.results].sort((a, b) => b.sharpe - a.sharpe);
        let html = '<table style="width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed">';
        html += '<tr style="background:#1a2a3a;color:#4fc3f7"><th style="padding:6px">MaxPos</th><th>TopK</th><th>Rebalance</th><th>Hidden</th><th>Sharpe</th><th>超额</th><th>回撤</th></tr>';
        sorted.forEach(r => {
            html += `<tr style="border-bottom:1px solid #1f3a52">
                <td style="padding:4px 6px;text-align:center">${r.max_position_pct}</td>
                <td style="padding:4px 6px;text-align:center">${r.top_k}</td>
                <td style="padding:4px 6px;text-align:center">${r.rebalance_days}</td>
                <td style="padding:4px 6px;text-align:center">${r.lstm_hidden}</td>
                <td style="padding:4px 6px;text-align:center;color:${r.sharpe > 0 ? '#f44336' : '#4caf50'};font-weight:bold">${r.sharpe.toFixed(3)}</td>
                <td style="padding:4px 6px;text-align:center;color:${r.excess > 0 ? '#f44336' : '#4caf50'}">${r.excess.toFixed(1)}%</td>
                <td style="padding:4px 6px;text-align:center;color:#4caf50">${r.max_dd.toFixed(1)}%</td>
            </tr>`;
        });
        html += '</table>';
        document.getElementById('mtxTable').innerHTML = html;
        addLog(`📊 参数矩阵完成: ${d.results.length} 个配置, 最优 Sharpe ${sorted[0]?.sharpe.toFixed(3) || 'N/A'}`);
    }).catch(e => addLog('❌ 加载结果失败: ' + e.message));
}
function loadMtxPanels() {
    fetch('/api/panels').then(r => r.json()).then(d => {
        const sel = document.getElementById('mtxPanel');
        if (!sel) return;
        sel.innerHTML = (d.panels || []).map(p => `<option value="${p.name}">${p.name} (${p.size_gb} GB)</option>`).join('') || '<option value="">无面板</option>';
    }).catch(() => {});
}

// ══════════════════════════════════════════════════
