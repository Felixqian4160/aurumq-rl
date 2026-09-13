/** components/badge.js — 状态徽章 */
function statusBadge(status){
  const map = { running: ['运行中','badge-running'], done: ['完成','badge-done'], failed: ['失败','badge-failed'], idle: ['空闲','badge-idle'] };
  const v = map[status] || [status, 'badge-idle'];
  return '<span class="status-badge '+v[1]+'">'+v[0]+'</span>';
}
