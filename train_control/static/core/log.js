/**
 * core/log.js — 统一日志面板
 * 负责 addLog / renderLogs / log SSE 单例 / recent 拉取。
 */
(function(window){
  const MAX_LOGS = 500;
  const logEl = () => document.getElementById('unifiedLogPanel') || document.getElementById('logPanel');
  let _es = null;
  let _dedupe = new Set();

  function escapeHtml(t){
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
  }

  function addLog(msg, source){
    source = source || 'app';
    const el = logEl();
    if (!el) return;
    const time = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.className = 'log-line';
    line.innerHTML = `<span class="log-time">${time}</span> <span class="log-source">[${escapeHtml(source)}]</span> ${escapeHtml(String(msg))}`;
    el.appendChild(line);
    while (el.children.length > MAX_LOGS) el.removeChild(el.firstChild);
    el.scrollTop = el.scrollHeight;
  }

  function appendLogEvent(x){
    if (!x || !x.msg) return;
    const key = (x.source||'') + '|' + x.msg;
    if (_dedupe.has(key)) return;
    _dedupe.add(key);
    if (_dedupe.size > 2000) { _dedupe.clear(); }
    addLog(x.msg, x.source || 'app');
  }

  function loadRecentStructuredLogs(){
    return fetch('/api/logs/recent?limit=100').then(r=>r.json()).then(d=>{
      const events = d.events || d.logs || [];
      events.forEach(appendLogEvent);
    }).catch(()=>{});
  }

  function connectSSE(){
    if (_es) return _es;
    _es = new EventSource('/api/logs');
    _es.onmessage = (ev)=>{
      try {
        const x = JSON.parse(ev.data);
        appendLogEvent(x);
      } catch(_){}
    };
    _es.onerror = ()=>{ /* auto-reconnect by browser */ };
    return _es;
  }

  window.coreApi = window.coreApi || {};
  window.coreLog = { addLog: addLog, appendLogEvent: appendLogEvent, loadRecentStructuredLogs: loadRecentStructuredLogs, connectSSE: connectSSE, escapeHtml: escapeHtml };
})(window);
