/** components/chart.js — 图表复用（echarts 封装） */
function renderChart(containerId, option){
  const el = document.getElementById(containerId);
  if (!el || typeof echarts === 'undefined') return;
  const inst = echarts.getInstanceByDom(el) || echarts.init(el);
  inst.setOption(option, true);
}
