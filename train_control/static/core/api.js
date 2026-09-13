/**
 * core/api.js — 统一 fetch 封装
 * 所有前端请求走此封装，便于统一加 Cache-Control、错误抛 detail、日志过滤。
 */
function formatApiError(detail, fallback) {
  if (typeof detail === 'string' && detail) return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map(item => {
      if (typeof item === 'string') return item;
      if (item && typeof item === 'object') {
        const msg = item.msg || item.message || '';
        const loc = Array.isArray(item.loc) ? item.loc.join('.') : '';
        return loc && msg ? `${loc}: ${msg}` : (msg || JSON.stringify(item));
      }
      return String(item);
    }).filter(Boolean);
    if (messages.length) return messages.join('; ');
  }
  if (detail && typeof detail === 'object') {
    if (detail.message) return String(detail.message);
    try { return JSON.stringify(detail); } catch (_) {}
  }
  return fallback;
}

function apiFetch(path, opts) {
  opts = opts || {};
  const headers = Object.assign({ 'Cache-Control': 'no-cache' }, (opts.headers || {}));
  const request = Object.assign({}, opts, { headers });
  if (request.body !== undefined && request.body !== null &&
      typeof request.body === 'object' &&
      !(request.body instanceof FormData)) {
    if (!headers['Content-Type']) headers['Content-Type'] = 'application/json';
    request.body = JSON.stringify(request.body);
  }
  return fetch(path, request)
    .then(async r => {
      const payload = await r.json().catch(() => ({}));
      if (r.ok) return payload;
      throw new Error(formatApiError(payload.detail, 'HTTP ' + r.status));
    });
}

function apiFetchRaw(path, opts) {
  opts = opts || {};
  const headers = Object.assign({ 'Cache-Control': 'no-cache' }, (opts.headers || {}));
  return fetch(path, Object.assign({}, opts, { headers }));
}
