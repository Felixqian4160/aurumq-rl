/** tabs/data_mgmt.js — 数据管理 Tab（可复用模块）
 * 依赖全局: fetch / addLog / apiFetch(可选)
 * 由 index.html <script src="/static/tabs/data_mgmt.js"> 加载
 */
function dmSaveToken() {
    const token = document.getElementById('dmTokenInput').value.trim();
    if (!token) { document.getElementById('dmTokenResult').innerHTML = '<span style="color:#f44336">请输入 Token</span>'; return; }
    apiFetch('/api/data/token', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({token})})
    .then(d => {
        document.getElementById('dmTokenResult').innerHTML = '<span style="color:#4caf50">✅ Token 已保存</span>';
        try { addLog('✅ Token 已保存', 'data_mgmt'); } catch (_) {}
        dmCheckTokenStatus(true);
    }).catch(e => { document.getElementById('dmTokenResult').innerHTML = '<span style="color:#f44336">❌ '+e.message+'</span>'; });
}
function dmTestToken() {
    const token = document.getElementById('dmTokenInput').value.trim();
    if (!token) return;
    document.getElementById('dmTokenResult').innerHTML = '<span style="color:#667788">⏳ 测试中...</span>';
    apiFetch('/api/data/token/test', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({token})})
    .then(d => {
        const c = d.status==='ok' ? '#4caf50' : '#f44336';
        document.getElementById('dmTokenResult').innerHTML = `<span style="color:${c}">${d.status==='ok'?'✅':'❌'} ${d.message}</span>`;
        try { addLog((d.status === 'ok' ? '✅' : '❌') + ' Token 测试: ' + (d.message || d.status), 'data_mgmt'); } catch (_) {}
    }).catch(e => {
        document.getElementById('dmTokenResult').innerHTML = '<span style="color:#f44336">❌ '+e.message+'</span>';
        try { addLog('❌ Token 测试失败: ' + e.message, 'data_mgmt'); } catch (_) {}
    });
}
function dmCheckTokenStatus(force) {
    const badge = document.getElementById('dmTokenBadge');
    if (!badge) return;
    const setBadge = (text, cls) => {
        badge.textContent = text;
        badge.className = `status-badge ${cls}`;
        badge.title = '';
    };
    setBadge('⏳ 检测中...', 'status-loading');
    apiFetch('/api/data/token/status').then(d => {
        if (d && d.status === 'valid') {
            setBadge('✅ 有效', 'status-done');
            badge.title = d.message || 'Token 有效';
            if (force) { try { addLog('✅ Token 状态: 有效', 'data_mgmt'); } catch (_) {} }
        } else if (d && d.status === 'invalid') {
            setBadge('❌ 无效', 'status-failed');
            badge.title = d.message || 'Token 无效';
            if (force) { try { addLog('❌ Token 状态: 无效', 'data_mgmt'); } catch (_) {} }
        } else {
            setBadge('⚠️ 不可用', 'status-warning');
            badge.title = (d && (d.message || d.status)) ? String(d.message || d.status) : 'Token 服务返回未知状态';
        }
    }).catch(e => {
        setBadge('⚠️ 不可用', 'status-warning');
        badge.title = 'Token 状态检测失败: ' + (e && e.message ? e.message : String(e));
        try { addLog('⚠️ Token 状态检测失败: ' + (e && e.message ? e.message : String(e)), 'data_mgmt'); } catch (_) {}
    });
}
function dmSetDateAll() {
    document.getElementById('dmStartDate').value='20040102';
    const today = new Date();
    const todayStr = today.getFullYear().toString() + String(today.getMonth()+1).padStart(2,'0') + String(today.getDate()).padStart(2,'0');
    document.getElementById('dmEndDate').value=todayStr;
    const btn = event.target;
    const orig = btn.textContent;
    btn.textContent = '✅ 已设置';
    btn.style.background = '#2e7d32';
    setTimeout(() => { btn.textContent = orig; btn.style.background = ''; }, 1500);
}
const dmActiveDownloads = {};
function _dmStatusHtml(taskId, text) {
    const stop = taskId ? `<button class="btn btn-danger btn-sm" style="margin-left:6px;padding:2px 7px" onclick="dmStopDownload('${taskId}', this.parentElement.id)">⏹ 停止</button>` : '';
    return text + stop;
}
function _dmStopFile(taskId) {
    return '/api/data/download/stop?task_id=' + encodeURIComponent(taskId);
}
function dmStopDownload(taskId, statusId) {
    if (!taskId) return;
    apiFetch('/api/data/download/stop?task_id='+encodeURIComponent(taskId), {method:'POST'}).then(d => {
        const el = document.getElementById(statusId);
        if (el) el.textContent = '⏹ 正在停止...';
        try { addLog('⏹ 已请求停止下载: ' + taskId, 'data_mgmt'); } catch (_) {}
    }).catch(e => {
        const el = document.getElementById(statusId);
        if (el) el.textContent = '❌ 停止失败: ' + e.message;
    });
}
function dmDownload(pool, dataType) {
    const sd = document.getElementById('dmStartDate').value || '20040102';
    const ed = document.getElementById('dmEndDate').value || '';
    const statusId = dataType==='ohlcv'?'dmOhlcvStatus':dataType==='daily_basic'?'dmBasicStatus':'dmFinaStatus';
    const el = document.getElementById(statusId);
    if (el) el.textContent = '⏳ 下载中...';
    apiFetch('/api/data/download', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({pool, start_date:sd, end_date:ed, data_type:dataType})})
    .then(d => {
        if (!d.task_id || !Number.isFinite(Number(d.total))) throw new Error('下载响应缺少 task_id/total');
        const sEl = document.getElementById(statusId);
        if (sEl) sEl.innerHTML = _dmStatusHtml(d.task_id, `✅ 已启动 (${d.total}只) · <span style="color:#4fc3f7" id="dmDlPct_${dataType}">0%</span>`);
        dmActiveDownloads[statusId] = d.task_id;
        dmPollDownload(d.task_id, statusId);
    }).catch(e => { const sEl=document.getElementById(statusId); if(sEl) sEl.textContent='❌ '+e.message; });
}
function dmPollDownload(taskId, statusId) {
    let lastLog = '';
    const poll = () => {
        apiFetch('/api/data/download/progress?task_id='+encodeURIComponent(taskId)).then(d => {
            if (d.status==='running' || d.status==='starting' || d.status==='stopping') {
                const target = document.getElementById('dmDlPct_'+ (statusId==='dmOhlcvStatus'?'ohlcv':statusId==='dmBasicStatus'?'daily_basic':'fina_indicator')) || document.getElementById(statusId);
                if (target && target.id && target.id.startsWith('dmDlPct_')) target.textContent = (d.pct||0)+'% | '+(d.done||0)+'/'+(d.total||0);
                if (d.status==='stopping') {
                    const sEl=document.getElementById(statusId); if(sEl) sEl.innerHTML='⏹ 正在停止...';
                }
                if (d.log && d.log !== lastLog) { try{ addLog(d.log, 'data_mgmt'); }catch(_){} lastLog = d.log; }
                setTimeout(poll, 3000);
            } else if (d.status==='done') {
                const failedInfo = d.failure_categories ? ' | '+Object.entries(d.failure_categories).map(([k,v])=>k+'='+v).join(', ') : '';
                const msg = '✅ 下载完成 ('+(d.done||0)+'/'+(d.total||0)+') '+(d.failed?'❌'+d.failed:'')+failedInfo+' | 耗时'+(d.elapsed||0)+'s';
                const sEl=document.getElementById(statusId); if(sEl) sEl.textContent = msg;
                delete dmActiveDownloads[statusId];
                try{ addLog(msg, 'data_mgmt'); }catch(_){}
                dmRefreshCache();
            } else if (['failed','stopped','blocked','stale'].includes(d.status)) {
                const icon = d.status==='stopped' ? '⏹' : '❌';
                const reason = d.error || d.reason || (d.failure_samples && d.failure_samples[0] && d.failure_samples[0].error) || ('任务状态: '+d.status);
                const msg = icon+' 下载'+(d.status==='stopped'?'已停止':'失败')+'：'+reason;
                const sEl=document.getElementById(statusId); if(sEl) { sEl.textContent = msg; sEl.style.color = '#f44336'; }
                delete dmActiveDownloads[statusId];
                try{ addLog(msg, 'data_mgmt'); }catch(_){}
                const updateTask = statusId === 'dmUpdateStatus';
                if (updateTask) {
                    const b1=document.getElementById('dmUpdateBtn'); if(b1) b1.disabled=false;
                    const b2=document.getElementById('dmStopUpdateBtn'); if(b2) b2.disabled=true;
                }
            } else { setTimeout(poll, 5000); }
        }).catch(()=>{ setTimeout(poll, 5000); });
    };
    poll();
}
function dmUpdateAll() {
    const b1=document.getElementById('dmUpdateBtn'); if(b1) b1.disabled=true;
    const b2=document.getElementById('dmStopUpdateBtn'); if(b2) b2.disabled=false;
    const st=document.getElementById('dmUpdateStatus'); if(st) st.textContent='⏳ 更新中...';
    apiFetch('/api/data/update-all', {method:'POST'}).then(d => {
        if (!d.task_id || !Number.isFinite(Number(d.total))) throw new Error('更新响应缺少 task_id/total');
        const sEl=document.getElementById('dmUpdateStatus'); if(sEl) sEl.innerHTML=_dmStatusHtml(d.task_id, `任务已启动 (${d.total}只)`);
        dmActiveDownloads.dmUpdateStatus = d.task_id;
        dmPollDownload(d.task_id, 'dmUpdateStatus');
    }).catch(e => { const sEl=document.getElementById('dmUpdateStatus'); if(sEl) sEl.textContent='❌ '+e.message; });
}
function dmStopUpdate() {
    apiFetch('/api/data/update-all/stop', {method:'POST'}).then(d => {
        const sEl=document.getElementById('dmUpdateStatus'); if(sEl) sEl.textContent='⏹ 已停止';
        const b1=document.getElementById('dmUpdateBtn'); if(b1) b1.disabled=false;
        const b2=document.getElementById('dmStopUpdateBtn'); if(b2) b2.disabled=true;
    });
}
function dmRefreshCache() {
    apiFetch('/api/data/cached').then(d => {
        const el=document.getElementById('dmCacheOhlcv'); if(!el) return;
        el.textContent = d.stocks+'只'; el.style.color = d.stocks>0?'#4fc3f7':'#f44336';
        try { addLog(`📊 缓存刷新: OHLCV ${d.stocks}只 / adj ${d.adj_pct ?? 0}%`, 'data_mgmt'); } catch (_) {}
    }).catch(e => { try { addLog('❌ 缓存刷新失败: ' + e.message, 'data_mgmt'); } catch (_) {} });
    apiFetch('/api/data/adj-status').then(d => {
        const el=document.getElementById('dmCacheAdj'); if(!el) return;
        el.textContent = d.pct+'%'; el.style.color = d.pct>=99?'#ce93d8':d.pct>=90?'#ffb74d':'#f44336';
    }).catch(()=>{});
    apiFetch('/api/data/daily-basic-status').then(d => {
        const el=document.getElementById('dmCacheBasic'); if(!el) return;
        el.textContent = d.stocks+'只'; el.style.color = d.stocks>0?'#81c784':'#f44336';
    }).catch(()=>{});
    apiFetch('/api/data/fina-indicator-status').then(d => {
        const el=document.getElementById('dmCacheFina'); if(!el) return;
        el.textContent = d.stocks+'只'; el.style.color = d.stocks>0?'#a5d6a7':'#f44336';
    }).catch(()=>{});
}
function dmVerify() {
    const el = document.getElementById('dmVerifyResult');
    if(!el) return;
    el.style.display='block'; el.innerHTML='<span style="color:#667788">🔍 检查中（分四项校验新鲜度，稍等...）</span>';
    apiFetch('/api/data/verify').then(d => {
        const det = d.detail || {};
        const fmtDetail = (name, v) => {
            if (!v || v.total===0) return `<div style="display:flex;gap:8px;padding:2px 0;align-items:center"><span style="min-width:118px">${name}</span><span style="color:#f44336">0（未下载）</span></div>`;
            const c = v.fresh_pct>=90 ? '#4caf50' : v.fresh_pct>=70 ? '#ff9800' : '#f44336';
            const stale = v.stale_samples && v.stale_samples.length ? `<span style="color:#556677;font-size:10px;max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">示例 ${v.stale_samples[0]}</span>` : '';
            return `<div style="display:flex;gap:10px;padding:2px 0;align-items:center;flex-wrap:wrap"><span style="min-width:118px">${name}</span><span style="color:${c};font-weight:600">${v.fresh}/${v.total} ${v.fresh_pct}%</span><span style="color:#667788;font-size:10px">预期 ${v.expected}±${v.tolerance_days||5}天</span>${stale}</div>`;
        };
        const header = `<div style="color:${d.status==='ok'?'#4caf50':'#ff9800'};font-weight:600">${d.status==='ok'?'✅':'⚠'} ${d.healthy}/${d.total} OHLCV 新鲜</div>`;
        const issues = d.issues && d.issues.length ? `<div style="margin-top:4px;color:#ff9800;font-size:11px">${d.issues.join('；')}</div>` : '';
        const adjRow = det.adj ? `<div style="display:flex;gap:10px;padding:2px 0;align-items:center"><span style="min-width:118px">🔄 复权因子</span><span style="color:${(det.adj?.pct||0)>=90?'#4caf50':'#f44336'};font-weight:600">${det.adj?.with_adj||0}/${det.adj?.total||0} ${det.adj?.pct||0}%</span><span style="color:#667788;font-size:10px">${det.adj?.fresh||0} 新鲜</span></div>` : '';
        const detailHtml = det.ohlcv || det.adj ? `<div style="margin-top:6px;padding-top:6px;border-top:1px solid #1a2a3a;font-size:11px;line-height:1.35">${fmtDetail('📈 OHLCV',det.ohlcv)}${fmtDetail('📋 daily_basic',det.daily_basic)}${fmtDetail('📊 fina_indicator',det.fina_indicator)}${adjRow}</div>` : '';
        el.innerHTML = header + detailHtml + issues;
        try { addLog(`🔍 完整性检查: ${d.status}，OHLCV ${d.healthy || 0}/${d.total || 0} 新鲜`, 'data_mgmt'); } catch (_) {}
    }).catch(e => { el.innerHTML='<span style="color:#f44336">❌ '+e.message+'</span>'; try { addLog('❌ 完整性检查失败: ' + e.message, 'data_mgmt'); } catch (_) {} });
}
// 页面加载时初始化（由 index.html 触发，或自行兜底）
setTimeout(()=>{ try{ dmCheckTokenStatus(); dmRefreshCache(); }catch(_){} }, 1000);
