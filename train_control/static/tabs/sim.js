/** tabs/sim.js — 模拟交易 Tab
 * 依赖全局: fetch / addLog / echarts (renderSimNavChart)
 * 由 index.html <script src="/static/tabs/sim.js"> 加载
 * 负责 #tab-sim 区域；状态走 /api/sim/*
 */
function loadSimModels() {
    apiFetch('/api/all/models').then(d => {
        const sel = document.getElementById('simModel');
        const val = sel.value;
        sel.innerHTML = '<option value="">-- 请选择模型 --</option>' +
            (d.models || []).filter(m => m.has_onnx).map(m => {
                const v = m.source + ':' + m.name;
                const steps = m.total_timesteps ? (m.total_timesteps / 1e6).toFixed(1) + 'M步' : '?';
                return `<option value="${v}">${m.display_name} (${steps})</option>`;
            }).join('');
        if (val) {
            const opt = sel.querySelector(`option[value="${val}"]`);
            if (opt) sel.value = val;
        }
        window._simModels = d.models || [];
        if (!val && sel.value) onSimModelChange();
    }).catch(e => addLog('❌ 加载模型失败: ' + e.message));
}

function onSimModelChange() {
    const sel = document.getElementById('simModel');
    const v = sel.value;
    if (!v || !window._simModels) return;
    const [src, name] = v.split(':');
    const m = (window._simModels || []).find(x => x.source === src && x.name === name);
    if (!m) return;
    const p = {};
    if (m.rebalance_days !== undefined && m.rebalance_days !== null) p.rebalance_days = m.rebalance_days;
    if (m.cost_bps !== undefined && m.cost_bps !== null) p.cost_bps = m.cost_bps;
    if (m.max_position_pct !== undefined && m.max_position_pct !== null) p.max_position_pct = m.max_position_pct;
    if (m.max_holding_days !== undefined && m.max_holding_days !== null) p.max_holding_days = m.max_holding_days;
    if (m.top_k) p.top_k = m.top_k;
    setSimParams(p);
    addLog(`📋 已载入模型 ${name} 训练参数: rebalance=${p.rebalance_days ?? '-'}天 cost=${p.cost_bps ?? '-'}bps max_pos=${p.max_position_pct ?? '-'} max_hold=${p.max_holding_days ?? '-'}天`);
}

function loadSimPanels() {
    apiFetch('/api/train/panels').then(d => {
        const sel = document.getElementById('simPanel');
        if (!sel) return;
        sel.innerHTML = (d.panels || []).map(p =>
            `<option value="${p.name}">${p.name} (${p.size_gb} GB)</option>`
        ).join('');
    }).catch(e => addLog('❌ 加载面板失败: ' + e.message));
}
function _initSimTab() {
    try { loadSimModels(); } catch (e) {}
    try { loadSimPanels(); } catch (e) {}
    try { loadSimConfigs(); } catch (e) {}
    try { loadSimLedgers(); } catch (e) {}
}
if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', _initSimTab); else setTimeout(_initSimTab, 0);

function getSimParams() {
    return {
        model_dir: document.getElementById('simModel').value,
        panel: document.getElementById('simPanel').value,
        initial_capital: parseFloat(document.getElementById('simCapital').value) || 100000,
        start_date: document.getElementById('simStartDate').value || '2024-01-01',
        end_date: document.getElementById('simEndDate').value || '',
        top_k: parseInt(document.getElementById('simTopK').value) || 30,
        rebalance_days: (v=document.getElementById('simRebalance').value) ? parseInt(v) : -1,
        cost_bps: (v=document.getElementById('simCostBps').value) ? parseFloat(v) : -1,
        slippage_bps: (v=document.getElementById('simSlippage').value) ? parseFloat(v) : 10,
        stop_loss_pct: (v=document.getElementById('simStopLoss').value) ? parseFloat(v) : 8,
        max_position_pct: (v=document.getElementById('simMaxPos').value) ? parseFloat(v) : -1,
        max_holding_days: (v=document.getElementById('simMaxHold').value) ? parseInt(v) : -1,
    };
}

function setSimParams(p) {
    if (p.model_dir) document.getElementById('simModel').value = p.model_dir;
    if (p.panel) document.getElementById('simPanel').value = p.panel;
    if (p.initial_capital) document.getElementById('simCapital').value = p.initial_capital;
    if (p.start_date) document.getElementById('simStartDate').value = p.start_date;
    if (p.end_date !== undefined) document.getElementById('simEndDate').value = p.end_date;
    if (p.top_k) document.getElementById('simTopK').value = p.top_k;
    if (p.rebalance_days !== undefined) document.getElementById('simRebalance').value = p.rebalance_days;
    if (p.cost_bps !== undefined) document.getElementById('simCostBps').value = p.cost_bps;
    if (p.slippage_bps !== undefined) document.getElementById('simSlippage').value = p.slippage_bps;
    if (p.stop_loss_pct !== undefined) document.getElementById('simStopLoss').value = p.stop_loss_pct;
    if (p.max_position_pct !== undefined) document.getElementById('simMaxPos').value = p.max_position_pct;
    if (p.max_holding_days !== undefined) document.getElementById('simMaxHold').value = p.max_holding_days;
}

function resetSimConfig() {
    setSimParams({
        initial_capital: 100000,
        start_date: '2024-01-01',
        end_date: '',
        top_k: 30,
        rebalance_days: -1,
        cost_bps: -1,
        slippage_bps: 10,
        stop_loss_pct: 8,
        max_position_pct: -1,
        max_holding_days: -1,
    });
    addLog('↺ 模拟交易参数已重置为默认');
}

function loadSimConfigs() {
    apiFetch('/api/sim/configs').then(d => {
        const sel = document.getElementById('simSavedConfigs');
        if (!sel) return;
        sel.innerHTML = '<option value="">-- 已保存参数 --</option>' +
            (d.configs || []).map(c =>
                `<option value="${c.name}">${c.name} (${c.saved_at})</option>`
            ).join('');
    }).catch(e => addLog('❌ 加载已保存参数失败: ' + e.message));
}

function loadSimConfig(name) {
    if (!name) return;
    apiFetch('/api/sim/configs').then(d => {
        const cfg = (d.configs || []).find(c => c.name === name);
        if (cfg && cfg.params) {
            setSimParams(cfg.params);
            addLog(`📂 已加载参数: ${name}`);
        } else {
            addLog(`⚠ 配置 ${name} 不存在或无 params 字段`);
        }
    }).catch(e => addLog('❌ 加载配置失败: ' + e.message));
}

function saveSimConfig() {
    const name = prompt('参数名称:', '');
    if (!name) return;
    const params = getSimParams();
    apiFetch('/api/sim/configs/save', {method: 'POST', body: {name, params}})
        .then(d => {
            addLog(`💾 参数已保存: ${name}`);
            loadSimConfigs();
        })
        .catch(e => addLog('❌ 参数保存失败: ' + e.message));
}

function deleteSimConfig() {
    const sel = document.getElementById('simSavedConfigs');
    const name = sel ? sel.value : '';
    if (!name) { addLog('⚠ 请先选择要删除的参数'); return; }
    if (!confirm(`确认删除参数 "${name}"?`)) return;
    apiFetch(`/api/sim/configs/${encodeURIComponent(name)}`, {method: 'DELETE'})
        .then(d => {
            addLog(`🗑 已删除参数: ${name}`);
            loadSimConfigs();
        })
        .catch(e => addLog('❌ 参数删除失败: ' + e.message));
}

let _simTaskId = '';
let _simPollTimer = null;

function _renderSimStatus(d) {
    const badge = document.getElementById('simBadge');
    const startBtn = document.getElementById('btnSimStart');
    const stopBtn = document.getElementById('btnSimStop');
    const progress = document.getElementById('simProgress');
    if (progress) progress.textContent = d.progress || '';
    const running = !!d.running;
    if (badge) {
        const phase = d.status || 'idle';
        badge.className = 'status-badge ' + (running
            ? 'status-running'
            : ((phase === 'done' || phase === 'finished') ? 'status-done'
            : (phase === 'failed' ? 'status-failed'
            : (phase === 'stopped' ? 'status-failed' : 'status-idle'))));
        badge.textContent = running ? '运行中'
            : ((phase === 'done' || phase === 'finished') ? '完成'
            : (phase === 'failed' ? '失败'
            : (phase === 'stopped' ? '已停止' : phase)));
    }
    if (startBtn) startBtn.disabled = running;
    if (stopBtn) stopBtn.disabled = !running;
    if (d.error) addLog('❌ 模拟错误: ' + d.error);
}

function startSim() {
    const params = getSimParams();
    if (!params.model_dir) { addLog('❌ 请选择模型'); return; }
    const startBtn = document.getElementById('btnSimStart');
    const stopBtn = document.getElementById('btnSimStop');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
    const results = document.getElementById('simResults');
    if (results) results.style.display = 'none';
    addLog(`🚀 模拟交易启动: ${params.model_dir} | ¥${params.initial_capital.toLocaleString()}`);
    apiFetch('/api/sim/start', {method: 'POST', body: params})
        .then(d => {
            if (!d.task_id) throw new Error('模拟响应缺少 task_id');
            _simTaskId = d.task_id;
            addLog('📊 模拟任务已提交: ' + d.model);
            _renderSimStatus({running: true, status: 'running', progress: '已提交'});
            pollSimStatus();
        })
        .catch(e => {
            if (startBtn) startBtn.disabled = false;
            if (stopBtn) stopBtn.disabled = true;
            addLog('❌ 模拟启动失败: ' + e.message);
        });
}

function stopSim() {
    if (!_simTaskId) {
        apiFetch('/api/sim/stop', {method: 'POST'})
            .then(d => addLog('⏹ 正在停止模拟: ' + (d.message || d.status)))
            .catch(e => addLog('❌ 停止模拟失败: ' + e.message));
        return;
    }
    apiFetch(`/api/sim/stop?task_id=${encodeURIComponent(_simTaskId)}`, {method: 'POST'})
        .then(d => {
            addLog('⏹ 正在停止模拟: ' + d.status);
            _renderSimStatus({running: false, status: 'stopping', progress: '停止中'});
        })
        .catch(e => addLog('❌ 停止模拟失败: ' + e.message));
}

function pollSimStatus() {
    if (_simPollTimer) clearInterval(_simPollTimer);
    const url = _simTaskId
        ? `/api/sim/status?task_id=${encodeURIComponent(_simTaskId)}`
        : '/api/sim/status';
    _simPollTimer = setInterval(() => {
        apiFetch(url).then(d => {
            if (d.task_id) _simTaskId = d.task_id;
            _renderSimStatus(d);
            if (!d.running && (d.status === 'done' || d.status === 'finished')) {
                clearInterval(_simPollTimer); _simPollTimer = null;
                loadSimResult();
            }
        }).catch(e => addLog('❌ 读取模拟状态失败: ' + e.message));
    }, 2000);
}

function loadSimResult() {
    const url = _simTaskId
        ? `/api/sim/result?task_id=${encodeURIComponent(_simTaskId)}`
        : '/api/sim/result';
    apiFetch(url).then(d => {
        if (d.status !== 'done') return;
        const results = document.getElementById('simResults');
        if (results) {
            results.style.display = 'block';
            results.offsetHeight; // reflow
        }
        const m = d.metrics || {};
        if (d.ledger_id) {
            document.getElementById('simLedgerInfo').textContent = '📒 账本: ' + d.ledger_id;
            document.getElementById('simLedgerDate').textContent = new Date().toLocaleString('zh-CN');
        }
        const fmt = (v, suffix = '%') => (v > 0 ? '+' : '') + v.toFixed(2) + suffix;
        const setText = (id, text, color) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.textContent = text;
            if (color) el.style.color = color;
        };
        setText('simKpiReturn', fmt(m.total_return || 0), (m.total_return || 0) >= 0 ? '#f44336' : '#4caf50');
        setText('simKpiAnnReturn', fmt(m.annualized_return || 0), (m.annualized_return || 0) >= 0 ? '#f44336' : '#4caf50');
        setText('simKpiBench', fmt(m.benchmark_return || 0));
        setText('simKpiExcess', fmt(m.excess_return || 0), (m.excess_return || 0) >= 0 ? '#f44336' : '#4caf50');
        setText('simKpiSharpe', (m.sharpe_ratio || 0).toFixed(3));
        setText('simKpiCalmar', (m.calmar_ratio || 0).toFixed(3));
        setText('simKpiVol', (m.volatility || 0).toFixed(1) + '%');
        setText('simKpiDD', (m.max_drawdown || 0).toFixed(1) + '%');
        setText('simKpiWinRate', (m.win_rate || 0).toFixed(1) + '%');
        setText('simKpiStopLoss', m.stop_loss_count || 0);
        setText('simKpiTrades', m.total_trades || 0);
        renderSimNavChart(d.weekly_nav || d.nav_curve || []);
        renderSimWeeklyTable(d.weekly_nav || []);
        renderSimHoldings(d.holdings || []);
        renderSimTrades(d.trades || []);
        loadSimLedgers();
    }).catch(e => addLog('❌ 读取模拟结果失败: ' + e.message));
}

function loadSimLedgers() {
    apiFetch('/api/sim/ledgers').then(d => {
        const sel = document.getElementById('simLedgerSelect');
        if (!sel) return;
        sel.innerHTML = '<option value="">-- 已保存账本 --</option>' +
            (d.ledgers || []).map(l =>
                `<option value="${l.id}">${l.id} (${l.model || '-'} · ${l.start_date || ''}~${l.end_date || ''})</option>`
            ).join('');
    }).catch(e => addLog('❌ 加载账本列表失败: ' + e.message));
}

function renderSimNavChart(curve) {
    if (!curve.length) return;
    const el = document.getElementById('simNavChart');
    el.offsetHeight;
    if (el.clientWidth < 10) { setTimeout(() => renderSimNavChart(curve), 200); return; }
    const old = echarts.getInstanceByDom(el);
    if (old) old.dispose();
    const chart = echarts.init(el, 'dark');
    // 兼容 daily nav_curve 和 weekly_nav 两种格式
    const dates = curve.map(c => c.date || c.week_end);
    const navsRaw = curve.map(c => c.nav);
    const bench = curve.map(c => c.benchmark_nav);
    const navStart = navsRaw[0] || 1;
    const navs = navsRaw.map(n => n / navStart);
    const benchStart = bench[0] || 1;
    const benchNorm = bench.map(b => b / benchStart);

    chart.setOption({
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis', backgroundColor: '#1a2a3a', borderColor: '#2a3a4a',
            textStyle: { color: '#e0e0e0', fontSize: 12 },
            formatter: params => {
                let s = '<b>' + params[0].axisValue + '</b><br/>';
                params.forEach(p => {
                    s += p.marker + ' ' + p.seriesName + ': ' + Number(p.value).toFixed(2) + 'x<br/>';
                });
                return s;
            }
        },
        legend: { data: ['策略净值', '沪深300'], textStyle: { color: '#8899aa' }, top: 5 },
        grid: { left: 70, right: 20, top: 40, bottom: 70 },
        dataZoom: [
            { type: 'inside', start: 0, end: 100 },
            { type: 'slider', start: 0, end: 100, height: 20, bottom: 5,
              borderColor: '#2a3a4a', fillerColor: 'rgba(79,195,247,0.15)',
              handleStyle: { color: '#4fc3f7' }, textStyle: { color: '#8899aa', fontSize: 10 } }
        ],
        xAxis: { type: 'category', data: dates, axisLabel: { color: '#667788', fontSize: 10, rotate: 30 } },
        yAxis: { type: 'value', axisLabel: { color: '#667788', formatter: v => v.toFixed(1) + 'x' },
            splitLine: { lineStyle: { color: '#1a2a3a' } } },
        series: [
            { name: '策略净值', type: 'line', data: navs, smooth: true, lineStyle: { width: 2 },
                itemStyle: { color: '#f44336' }, areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(244,67,54,0.3)'},{offset:1,color:'rgba(244,67,54,0.02)'}]) },
                symbol: 'none' },
            { name: '沪深300', type: 'line', data: benchNorm, smooth: true, lineStyle: { width: 1.5, type: 'dashed' },
                itemStyle: { color: '#ff9800' }, symbol: 'none' },
        ]
    });
    window.addEventListener('resize', () => chart.resize());
}

function renderSimWeeklyTable(weekly) {
    const el = document.getElementById('simWeeklyTable');
    if (!el) return;
    if (!weekly.length) { el.innerHTML = '<span style="color:#667788">无周数据</span>'; return; }
    let html = '<table style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:11px">';
    html += '<colgroup><col style="width:13%"><col style="width:13%"><col style="width:13%"><col style="width:14%"><col style="width:13%"><col style="width:13%"><col style="width:13%"></colgroup>';
    html += '<tr style="color:#4fc3f7;border-bottom:1px solid #2a3a4a"><th style="text-align:left;padding:4px 6px">周起始</th><th style="text-align:left;padding:4px 6px">周结束</th><th style="text-align:right;padding:4px 6px">期末净值</th><th style="text-align:right;padding:4px 6px">基准净值</th><th style="text-align:right;padding:4px 6px">周收益</th><th style="text-align:right;padding:4px 6px">基准周收益</th><th style="text-align:right;padding:4px 6px">周内最高</th></tr>';
    weekly.forEach(w => {
        const retColor = w.week_return >= 0 ? '#f44336' : '#4caf50';
        const benchColor = w.bench_return >= 0 ? '#f44336' : '#4caf50';
        html += `<tr style="border-bottom:1px solid #0d1b2a">
            <td style="padding:4px 6px;color:#8899aa;overflow:hidden;text-overflow:ellipsis">${w.week_start}</td>
            <td style="padding:4px 6px;color:#8899aa;overflow:hidden;text-overflow:ellipsis">${w.week_end}</td>
            <td style="padding:4px 6px;text-align:right">${w.nav.toFixed(0)}</td>
            <td style="padding:4px 6px;text-align:right">${w.benchmark_nav.toFixed(4)}</td>
            <td style="padding:4px 6px;text-align:right;color:${retColor};font-weight:bold">${w.week_return >= 0 ? '+' : ''}${w.week_return.toFixed(2)}%</td>
            <td style="padding:4px 6px;text-align:right;color:${benchColor}">${w.bench_return >= 0 ? '+' : ''}${w.bench_return.toFixed(2)}%</td>
            <td style="padding:4px 6px;text-align:right;color:#8899aa">${w.max_nav.toFixed(0)}</td>
        </tr>`;
    });
    html += '</table>';
    el.innerHTML = html;
}

function renderSimHoldings(holdings) {
    const el = document.getElementById('simHoldings');
    if (!holdings.length) { el.innerHTML = '<span style="color:#667788">无持仓</span>'; return; }
    el.innerHTML = holdings.map((h, hi) => {
        const stocks = (h.stocks||[]);
        // 判断每只股票的状态: 调入/持有/调出
        const prevHoldings = hi > 0 ? holdings[hi-1].stocks : [];
        const prevCodes = new Set(prevHoldings.map(s => s.code));
        const curCodes = new Set(stocks.map(s => s.code));
        let tbl = `<div style="border-bottom:2px solid #1a2a3a;padding:8px 0;margin-bottom:4px">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="color:#4fc3f7;font-weight:bold">${h.date}</span>
                <span style="color:#8899aa">净值 ¥${Number(h.nav).toLocaleString()} | 现金 ¥${Number(h.cash).toLocaleString()}</span>
            </div>
            <table style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:11px">
            <colgroup><col style="width:16%"><col style="width:8%"><col style="width:10%"><col style="width:10%"><col style="width:14%"><col style="width:8%"><col style="width:10%"><col style="width:12%"><col style="width:12%"></colgroup>
            <tr style="color:#4fc3f7;border-bottom:1px solid #2a3a4a">
                <th style="text-align:left;padding:4px 6px">代码</th>
                <th style="text-align:right;padding:4px 6px">持股</th>
                <th style="text-align:right;padding:4px 6px">买入价</th>
                <th style="text-align:right;padding:4px 6px">现价</th>
                <th style="text-align:right;padding:4px 6px">市值</th>
                <th style="text-align:right;padding:4px 6px">权重</th>
                <th style="text-align:right;padding:4px 6px">盈亏%</th>
                <th style="text-align:center;padding:4px 6px">调入日期</th>
                <th style="text-align:center;padding:4px 6px">状态</th>
            </tr>`;
        stocks.forEach(s => {
            const pnlColor = s.pnl_pct >= 0 ? '#f44336' : '#4caf50';
            // 状态判断: 用 action 字段（后端已设置）
            let statusLabel, statusColor;
            if (s.action === 'stop_loss') {
                statusLabel = '🔴止损';
                statusColor = '#ff9800';
            } else if (s.action === 'take_profit') {
                statusLabel = '🟢获利了结';
                statusColor = '#26c6da';
            } else if (s.action === 'sell') {
                statusLabel = '调出';
                statusColor = '#f44336';
            } else if (s.entry_date === h.date) {
                statusLabel = '调入';
                statusColor = '#ff9800';
            } else {
                statusLabel = '持有';
                statusColor = '#4caf50';
            }
            tbl += `<tr style="border-bottom:1px solid #0d1b2a">
                <td style="padding:4px 6px;font-family:monospace;overflow:hidden;text-overflow:ellipsis">${s.code}</td>
                <td style="padding:4px 6px;text-align:right">${s.shares}</td>
                <td style="padding:4px 6px;text-align:right;color:#8899aa">${s.buy_price.toFixed(2)}</td>
                <td style="padding:4px 6px;text-align:right">${s.cur_price.toFixed(2)}</td>
                <td style="padding:4px 6px;text-align:right">¥${Number(s.market_value).toLocaleString()}</td>
                <td style="padding:4px 6px;text-align:right">${s.weight}%</td>
                <td style="padding:4px 6px;text-align:right;color:${pnlColor}">${s.pnl_pct !== 0 ? (s.pnl_pct > 0 ? '+' : '') + s.pnl_pct.toFixed(1) + '%' : (s.action === 'hold' ? '+0.0%' : '-')}</td>
                <td style="padding:4px 6px;text-align:center;color:#667788;font-size:10px">${s.entry_date || '-'}</td>
                <td style="padding:4px 6px;text-align:center;color:${statusColor};font-weight:bold;font-size:10px">${statusLabel}</td>
            </tr>`;
        });
        tbl += '</table></div>';
        return tbl;
    }).join('');
}

function renderSimTrades(trades) {
    const el = document.getElementById('simTrades');
    if (!trades.length) { el.innerHTML = '<span style="color:#667788">无交易</span>'; return; }
    const PAGE_SIZE = 100;
    let currentPage = Math.ceil(trades.length / PAGE_SIZE); // 默认最后一页
    function render() {
        const totalPages = Math.ceil(trades.length / PAGE_SIZE);
        const start = (currentPage - 1) * PAGE_SIZE;
        const page = trades.slice(start, start + PAGE_SIZE);
        let html = '<table style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:11px">';
        html += '<colgroup><col style="width:14%"><col style="width:14%"><col style="width:8%"><col style="width:11%"><col style="width:11%"><col style="width:10%"><col style="width:16%"><col style="width:10%"></colgroup>';
        html += '<tr style="color:#4fc3f7;border-bottom:1px solid #2a3a4a"><th style="text-align:left;padding:4px 6px">日期</th><th style="padding:4px 6px">代码</th><th style="padding:4px 6px">方向</th><th style="text-align:right;padding:4px 6px">买入价</th><th style="text-align:right;padding:4px 6px">卖出价</th><th style="text-align:right;padding:4px 6px">股数</th><th style="text-align:right;padding:4px 6px">金额</th><th style="text-align:right;padding:4px 6px">盈亏</th></tr>';
        page.forEach(t => {
            const sideColor = t.side === 'buy' ? '#4caf50' : t.side === 'stop_loss' ? '#ff9800' : t.side === 'take_profit' ? '#26c6da' : '#f44336';
            const sideLabel = t.side === 'buy' ? '买入' : t.side === 'stop_loss' ? '🔴止损' : t.side === 'take_profit' ? '🟢获利了结' : '卖出';
            const pnlColor = (t.pnl||0) >= 0 ? '#f44336' : '#4caf50';
            html += `<tr style="border-bottom:1px solid #0d1b2a">
                <td style="padding:4px 6px;color:#8899aa;overflow:hidden;text-overflow:ellipsis">${t.date}</td>
                <td style="padding:4px 6px;text-align:center;font-family:monospace;overflow:hidden;text-overflow:ellipsis">${t.stock_code}</td>
                <td style="padding:4px 6px;text-align:center;color:${sideColor};font-weight:bold">${sideLabel}</td>
                <td style="padding:4px 6px;text-align:right">${t.buy_price ? t.buy_price.toFixed(2) : '-'}</td>
                <td style="padding:4px 6px;text-align:right">${t.sell_price ? t.sell_price.toFixed(2) : '-'}</td>
                <td style="padding:4px 6px;text-align:right">${t.shares}</td>
                <td style="padding:4px 6px;text-align:right">¥${Number(t.amount).toLocaleString()}</td>
                <td style="padding:4px 6px;text-align:right;color:${pnlColor}">${t.pnl ? (t.pnl > 0 ? '+' : '') + t.pnl.toFixed(0) : '-'}</td>
            </tr>`;
        });
        html += '</table>';
        if (totalPages > 1) {
            html += `<div style="display:flex;justify-content:center;align-items:center;gap:8px;padding:6px;color:#667788;font-size:11px">`;
            html += `<button onclick="simTradesPageGo(${currentPage - 1})" ${currentPage <= 1 ? 'disabled' : ''} style="background:#1a2a3a;color:#8899aa;border:1px solid #2a3a4a;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:11px">◀</button>`;
            html += `<span>${currentPage} / ${totalPages}</span>`;
            html += `<button onclick="simTradesPageGo(${currentPage + 1})" ${currentPage >= totalPages ? 'disabled' : ''} style="background:#1a2a3a;color:#8899aa;border:1px solid #2a3a4a;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:11px">▶</button>`;
            html += `<span style="margin-left:8px">共 ${trades.length} 笔</span>`;
            html += `</div>`;
        }
        el.innerHTML = html;
    }
    window._simTradesData = trades;
    window._simTradesPage = currentPage;
    window._simTradesRender = render;
    render();
}
function simTradesPageGo(page) {
    const trades = window._simTradesData || [];
    const totalPages = Math.ceil(trades.length / 100);
    if (page < 1 || page > totalPages) return;
    window._simTradesPage = page;
    const PAGE_SIZE = 100;
    const start = (page - 1) * PAGE_SIZE;
    const pageData = trades.slice(start, start + PAGE_SIZE);
    // Re-render with new page
    const el = document.getElementById('simTrades');
    let html = '<table style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:11px">';
    html += '<colgroup><col style="width:14%"><col style="width:14%"><col style="width:8%"><col style="width:11%"><col style="width:11%"><col style="width:10%"><col style="width:16%"><col style="width:10%"></colgroup>';
    html += '<tr style="color:#4fc3f7;border-bottom:1px solid #2a3a4a"><th style="text-align:left;padding:4px 6px">日期</th><th style="padding:4px 6px">代码</th><th style="padding:4px 6px">方向</th><th style="text-align:right;padding:4px 6px">买入价</th><th style="text-align:right;padding:4px 6px">卖出价</th><th style="text-align:right;padding:4px 6px">股数</th><th style="text-align:right;padding:4px 6px">金额</th><th style="text-align:right;padding:4px 6px">盈亏</th></tr>';
    pageData.forEach(t => {
        const sideColor = t.side === 'buy' ? '#4caf50' : t.side === 'stop_loss' ? '#ff9800' : t.side === 'take_profit' ? '#26c6da' : '#f44336';
        const sideLabel = t.side === 'buy' ? '买入' : t.side === 'stop_loss' ? '🔴止损' : t.side === 'take_profit' ? '🟢获利了结' : '卖出';
        const pnlColor = (t.pnl||0) >= 0 ? '#f44336' : '#4caf50';
        html += `<tr style="border-bottom:1px solid #0d1b2a">
            <td style="padding:4px 6px;color:#8899aa;overflow:hidden;text-overflow:ellipsis">${t.date}</td>
            <td style="padding:4px 6px;text-align:center;font-family:monospace;overflow:hidden;text-overflow:ellipsis">${t.stock_code}</td>
            <td style="padding:4px 6px;text-align:center;color:${sideColor};font-weight:bold">${sideLabel}</td>
            <td style="padding:4px 6px;text-align:right">${t.buy_price ? t.buy_price.toFixed(2) : '-'}</td>
            <td style="padding:4px 6px;text-align:right">${t.sell_price ? t.sell_price.toFixed(2) : '-'}</td>
            <td style="padding:4px 6px;text-align:right">${t.shares}</td>
            <td style="padding:4px 6px;text-align:right">¥${Number(t.amount).toLocaleString()}</td>
            <td style="padding:4px 6px;text-align:right;color:${pnlColor}">${t.pnl ? (t.pnl > 0 ? '+' : '') + t.pnl.toFixed(0) : '-'}</td>
        </tr>`;
    });
    html += '</table>';
    html += `<div style="display:flex;justify-content:center;align-items:center;gap:8px;padding:6px;color:#667788;font-size:11px">`;
    html += `<button onclick="simTradesPageGo(${page - 1})" ${page <= 1 ? 'disabled' : ''} style="background:#1a2a3a;color:#8899aa;border:1px solid #2a3a4a;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:11px">◀</button>`;
    html += `<span>${page} / ${totalPages}</span>`;
    html += `<button onclick="simTradesPageGo(${page + 1})" ${page >= totalPages ? 'disabled' : ''} style="background:#1a2a3a;color:#8899aa;border:1px solid #2a3a4a;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:11px">▶</button>`;
    html += `<span style="margin-left:8px">共 ${trades.length} 笔</span>`;
    html += `</div>`;
    el.innerHTML = html;
}

// ─── v9 四段周期策略选项 ───
function onSimV9Change() {
    const enabled = document.getElementById('simV9Enabled')?.checked;
    const params = document.getElementById('simV9Params');
    if (params) {
        params.style.display = enabled ? 'block' : 'none';
    }
    if (enabled) {
        addLog('🎯 v9 四段周期策略已启用', 'simulation');
    }
}

// 获取 v9 策略参数
function getV9SimParams() {
    const enabled = document.getElementById('simV9Enabled')?.checked;
    if (!enabled) return {};
    return {
        v9_enabled: true,
        v9_a1_threshold: parseFloat(document.getElementById('simV9A1Threshold')?.value || 0.5),
        v9_a2_threshold: parseFloat(document.getElementById('simV9A2Threshold')?.value || 0.5),
        v9_peak_threshold: parseFloat(document.getElementById('simV9PeakThreshold')?.value || 0.5),
        v9_b1_threshold: parseFloat(document.getElementById('simV9B1Threshold')?.value || 0.5),
    };
}
