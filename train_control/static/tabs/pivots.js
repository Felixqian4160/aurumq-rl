// tabs/pivots.js — 📍 高低点标注校验 Tab
function _pivotFetch(url, opts) {
  return apiFetch(url, opts || {});
}
let _pivotChart = null;
let _pivotData = null;

async function initPivotsTab() {
  try {
    await loadPivotPanels();
    const panel = document.getElementById('pivotPanel').value;
    const s = await _pivotFetch('/api/pivots/stats?panel='+encodeURIComponent(panel));
    const v = (s.panel || '').includes('wavehunter_v10_1_') ? 'v10_1' :
      ((s.panel || '').includes('wavehunter_v10_') ? 'v10' : 'v9');
    const peak = s[v+'_zig_peak'];
    const valley = s[v+'_zig_valley'];
    const a1 = s[v+'_a1_point'];
    const a2 = s[v+'_a2_interval'];
    document.getElementById('pivotStats').innerHTML =
      `<span style="color:#4fc3f7">面板 ${s.panel}</span> | 总行 ${s.total_rows.toLocaleString()} |
       ${v}_zig_peak ${peak} (${s[v+'_zig_peak_pct']}%) |
       ${v}_zig_valley ${valley} (${s[v+'_zig_valley_pct']}%) |
       ${v}_a1_point ${a1} (${s[v+'_a1_point_pct']}%) |
       ${v}_a2_interval ${a2} (${s[v+'_a2_interval_pct']}%) |
       旧a1 ${s.a1_reversal_start||0} 旧a2 ${s.a2_start||0}`;
  } catch(e) { console.error(e); }
  loadPivotStocks();
}
async function loadPivotPanels() {
  const d = await _pivotFetch('/api/pivots/panels');
  const sel = document.getElementById('pivotPanel');
  const cur = sel.value;
  sel.innerHTML = d.panels.map(p=>`<option value="${p.name}">${p.name} (${p.size_mb}MB, ${p.cols}列${p.has_v9?' ·v9':''})</option>`).join('');
  if (cur && d.panels.some(p=>p.name===cur)) sel.value = cur;
  else if (d.default) sel.value = d.default;
}
async function loadPivotStocks() {
  const q = document.getElementById('pivotSearch').value.trim();
  const panel = document.getElementById('pivotPanel').value;
  const d = await _pivotFetch('/api/pivots/stocks?panel='+encodeURIComponent(panel)+'&limit=80&q='+encodeURIComponent(q));
  const sel = document.getElementById('pivotStock');
  sel.innerHTML = d.stocks.map(c=>`<option value="${c}">${c}</option>`).join('');
  if (d.stocks.length) loadPivotData();
}
async function loadPivotData() {
  const stock = document.getElementById('pivotStock').value;
  const panel = document.getElementById('pivotPanel').value;
  const limit = parseInt(document.getElementById('pivotLimit').value)||500;
  const offset = parseInt(document.getElementById('pivotOffset').value)||0;
  if (!stock) return;
  document.getElementById('pivotInfo').textContent = '加载中...';
  try {
    const d = await _pivotFetch(`/api/pivots/data?stock=${encodeURIComponent(stock)}&panel=${encodeURIComponent(panel)}&limit=${limit}&offset=${offset}`);
    _pivotData = d;
    document.getElementById('pivotInfo').innerHTML =
      `<span style="color:#4fc3f7">${d.stock}</span> ${d.panel} | 共 ${d.total} 条, 显示 ${d.returned} 条 (offset ${d.offset})
       | 峰 ${d.peaks.length} 谷 ${d.valleys.length} A1 ${d.a1.length} A2起点 ${d.a2_start.length}`;
    renderPivotChart(d);
    renderPivotTable(d);
  } catch(e) {
    document.getElementById('pivotInfo').textContent = '❌ '+e.message;
  }
}
function renderPivotChart(d) {
  const el = document.getElementById('pivotChart');
  if (!el) return;
  if (_pivotChart) { try{_pivotChart.dispose()}catch(e){} }
  _pivotChart = echarts.init(el);
  const showOld = document.getElementById('pivotShowOld').checked;
  const v = (d.panel || '').includes('wavehunter_v10_1_') ? 'v10_1' :
    ((d.panel || '').includes('wavehunter_v10_') ? 'v10' : 'v9');
  const label = name => `${v}_${name}`;
  // K 线 + 当前面板版本的峰谷散点
  const kData = d.ohlc; // [open, close, low, high]
  const dates = d.dates;
  const markPeaks = d.peaks.map(p=>({coord:[p.idx, kData[p.idx]?kData[p.idx][3]:0]}));
  const markValleys = d.valleys.map(p=>({coord:[p.idx, kData[p.idx]?kData[p.idx][2]:0]}));
  const markA1 = d.a1.map(p=>({coord:[p.idx, kData[p.idx]?kData[p.idx][1]:0]}));
  const markA2 = d.a2_start.map(p=>({coord:[p.idx, kData[p.idx]?kData[p.idx][1]:0]}));
  // A2 区间背景
  const a2Area = [];
  if (d.a2_interval && d.a2_interval.length) {
    let inA2=false, start=0;
    for(let i=0;i<d.a2_interval.length;i++){
      if(d.a2_interval[i]===1 && !inA2){ inA2=true; start=i; }
      if((d.a2_interval[i]!==1 || i===d.a2_interval.length-1) && inA2){
        inA2=false;
        a2Area.push([{xAxis:start},{xAxis:i}]);
      }
    }
  }
  const prefix = (d.panel || '').includes('wavehunter_v10_1_') ? 'v10_1' :
    ((d.panel || '').includes('wavehunter_v10_') ? 'v10' : 'v9');
  const prefixLabel = name => `${name}(${prefix})`;
  const option = {
    backgroundColor: 'transparent',
    textStyle:{color:'#8899aa'},
    tooltip:{trigger:'axis', axisPointer:{type:'cross'}},
    legend:{data:['K线',prefixLabel('峰'),prefixLabel('谷'),'A1','A2起点'].concat(showOld?['旧A1','旧A2']:[]), textStyle:{color:'#8899aa'}, top:0},
    grid:{left:'3%', right:'3%', top:48, bottom:'12%'},
    xAxis:{type:'category', data:dates, axisLabel:{color:'#667788', fontSize:10, interval: Math.floor(dates.length/12)}},
    yAxis:{scale:true, axisLabel:{color:'#667788'}},
    dataZoom:[{type:'inside'},{type:'slider', height:20, bottom:8}],
    series:[
      {name:'K线', type:'candlestick', data:kData, itemStyle:{color:'#ef5350', color0:'#26a69a', borderColor:'#ef5350', borderColor0:'#26a69a'},
        markArea:{silent:true, itemStyle:{color:'rgba(33,150,243,0.08)'}, data: a2Area},
        markPoint:{
          symbol:'pin', symbolSize:32,
          data: [].concat(
            markPeaks.map(m=>({...m, value:'峰', itemStyle:{color:'#ff9800'}})),
            markValleys.map(m=>({...m, value:'谷', itemStyle:{color:'#4fc3f7'}})),
            markA1.map(m=>({...m, value:'A1', itemStyle:{color:'#e91e63'}})),
            markA2.map(m=>({...m, value:'A2', itemStyle:{color:'#8bc34a'}}))
          )
        }
      }
    ]
  };
  if (showOld) {
    option.series.push(
      {name:'旧A1', type:'scatter', data: d.old_a1.map(p=>[p.idx, kData[p.idx]?kData[p.idx][1]:0]), symbol:'triangle', symbolSize:8, itemStyle:{color:'rgba(233,30,99,0.35)'}},
      {name:'旧A2', type:'scatter', data: d.old_a2.map(p=>[p.idx, kData[p.idx]?kData[p.idx][1]:0]), symbol:'triangle', symbolSize:8, itemStyle:{color:'rgba(139,195,74,0.35)'}}
    );
  }
  _pivotChart.setOption(option);
  window.addEventListener('resize', ()=>_pivotChart&&_pivotChart.resize(), {once:false});
}
function renderPivotTable(d){
  const tb=document.getElementById('pivotTableBody');
  if(!tb) return;
  const rows=[];
  // 合并所有标注按日期排序
  const events=[];
  const eventVersion = (d.panel||'').includes('wavehunter_v10_1_') ? 'v10.1' :
    ((d.panel||'').includes('wavehunter_v10_') ? 'v10' : 'v9');
  d.peaks.forEach(p=>events.push({date:p.date, type:`峰(${eventVersion})`, cls:'peak'}));
  d.valleys.forEach(p=>events.push({date:p.date, type:`谷(${eventVersion})`, cls:'valley'}));
  d.a1.forEach(p=>events.push({date:p.date, type:'A1', cls:'a1'}));
  d.a2_start.forEach(p=>events.push({date:p.date, type:'A2起点', cls:'a2'}));
  events.sort((a,b)=>a.date.localeCompare(b.date));
  // 最近 100 个事件
  const show=events.slice(-100);
  tb.innerHTML = show.map(e=>`<tr><td>${e.date}</td><td><span class="badge-${e.cls}">${e.type}</span></td><td>${e.date}</td></tr>`).join('') || '<tr><td colspan=3 style="color:#667788">无标注</td></tr>';
}

document.addEventListener('DOMContentLoaded', ()=>{
  const t=document.getElementById('tabbtn-pivots');
  if(t) t.addEventListener('click', ()=>setTimeout(initPivotsTab,100));
});
