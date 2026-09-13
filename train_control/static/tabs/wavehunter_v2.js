// tabs/wavehunter_v2.js — 🌊 主升浪猎手 v2 (训练 + 矩阵)
// 从 index.html inline JS 抽离；依赖全局 fetch/addLog/echarts
function whV2Val(id) { return document.getElementById(id)?.value; }
function loadWhV2Panels() {
    fetch('/api/wavehunter/panels').then(r=>r.json()).then(d=>{
        const selects=[document.getElementById('whv2Panel'),document.getElementById('whv2_panel')].filter(Boolean);
        const ps=(d.panels||[]).filter(x=>x.name.includes('wavehunter_'));
        selects.forEach(s=>{s.innerHTML=ps.map(x=>`<option value="${x.name}">${x.name} (${x.size_mb} MB)</option>`).join('')||'<option value="">无面板</option>';if(ps.some(x=>x.name===window._whv2PanelDefault))s.value=window._whv2PanelDefault;});
    }).catch(()=>{});
}
function whV2Params(){return {...whV2SchemaParams()};}
function whV2SchemaParams(){
    const params = {};
    (window._whv2Schema || []).forEach(p => {
        const el = document.getElementById('whv2_' + p.key);
        if (!el) return;
        params[p.key] = el.value;
    });
    // schema 尚未完成加载时，至少使用当前面板选择值。
    if (params.panel === undefined) {
        const panel = document.getElementById('whv2_panel') || document.getElementById('whv2Panel');
        if (panel) params.panel = panel.value;
    }
    if (params.n_steps === undefined) {
        params.n_steps = 32;
    }
    if (params.batch_size === undefined) {
        params.batch_size = 32;
    }
    if (params.lstm_hidden === undefined) {
        params.lstm_hidden = 64;
    }
    return params;
}
function whV2Write(p){document.querySelectorAll('[data-v2-key]').forEach(e=>{if(p[e.dataset.v2Key]!==undefined)e.value=p[e.dataset.v2Key];});}
function loadWhV2Configs(){fetch('/api/wavehunter/v2/configs').then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})).then(d=>{const s=document.getElementById('whv2SavedConfigs');if(s)s.innerHTML='<option value="">-- 已保存参数 --</option>'+(d.configs||[]).map(x=>`<option value="${x.name}">${x.name}</option>`).join('');}).catch(e=>addLog('❌ v2预设加载失败: '+e.message));}
function loadWhV2Config(n){if(!n)return;fetch('/api/wavehunter/v2/configs').then(r=>r.json()).then(d=>{const x=(d.configs||[]).find(y=>y.name===n);if(x)whV2Write(x.params);});}
function saveWhV2Config(){const n=prompt('v2参数预设名称:');if(!n)return;fetch('/api/wavehunter/v2/configs/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n,params:whV2Params()})}).then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})).then(()=>{loadWhV2Configs();addLog('💾 已保存v2预设: '+n);}).catch(e=>addLog('❌ v2预设保存失败: '+e.message));}
function deleteWhV2Config(){const n=document.getElementById('whv2SavedConfigs').value;if(!n||!confirm('删除v2预设 '+n+'？'))return;fetch('/api/wavehunter/v2/configs/'+encodeURIComponent(n),{method:'DELETE'}).then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})).then(()=>{loadWhV2Configs();addLog('🗑 已删除v2预设: '+n);}).catch(e=>addLog('❌ v2预设删除失败: '+e.message));}
function resetWhV2Config(){loadWhV2Schema();addLog('↺ v2参数已从后端schema恢复默认');}
function loadWhV2Models(){
    fetch('/api/joint/models').then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})).then(d=>{
        const b=document.getElementById('whv2ModelList'); if(!b) return;
        const models=(d.models||[]).filter(x=>x.name.startsWith('whv2_')||x.name.startsWith('whv3_')||x.name.startsWith('matrix_'));
        b.innerHTML=models.map(x=>`<div class="file-item" style="display:flex;justify-content:space-between;align-items:center"><span title="${x.name}">${x.name} ${x.has_onnx||x.has_ppo?'✅':'⏳'} ${x.total_timesteps||x.timesteps||''}</span><button class="btn btn-sm btn-danger" onclick="deleteWhV2Model('${x.name}')" style="padding:2px 6px">🗑</button></div>`).join('')||'<span style="color:#667788">暂无已训练联合模型</span>';
    }).catch(e=>{const b=document.getElementById('whv2ModelList');if(b)b.innerHTML='<span style="color:#f44336">加载失败</span>';addLog('❌ 联合模型列表加载失败: '+e.message);});
}
function deleteWhV2Model(n){if(!confirm('删除v2模型 '+n+'？'))return;fetch('/api/wavehunter/v2/models/'+encodeURIComponent(n),{method:'DELETE'}).then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})).then(()=>{loadWhV2Models();addLog('🗑 已删除v2模型: '+n);}).catch(e=>addLog('❌ v2模型删除失败: '+e.message));}
function whV2Body(){
    const dyn=whV2SchemaParams();
    const num=k=>{
        const v=dyn[k];
        if(v===undefined||v===null||v==='') return undefined;
        const n=Number(v);
        return Number.isFinite(n)?n:undefined;
    };
    // 从 schema 动态构建参数，确保新参数自动包含
    const body = {};
    (window._whv2Schema || []).forEach(p => {
        if (p.key === 'panel' || p.key === 'start_date' || p.key === 'end_date' ||
            p.key === 'universe_filter') {
            body[p.key] = dyn[p.key];
        } else if (p.type === 'bool') {
            body[p.key] = dyn[p.key] === 'true' || dyn[p.key] === true;
        } else {
            const n = num(p.key);
            if (n !== undefined) body[p.key] = n;
        }
    });
    // 安全默认值：兜底 v2 训练验证要求 (lstm_hidden<=64, n_steps<=32, batch_size<=32)
    if (body.lstm_hidden === undefined || body.lstm_hidden > 64) body.lstm_hidden = 64;
    if (body.n_steps === undefined || body.n_steps > 32) body.n_steps = 32;
    if (body.batch_size === undefined || body.batch_size > 32) body.batch_size = 32;
    return body;
}
function startWhV2Train(event){
    if (event) { event.preventDefault(); event.stopPropagation(); }
    const btn=document.getElementById('whv2StartBtn');
    const badge=document.getElementById('whv2Badge');
    const setState=(text, cls, disabled)=>{if(badge){badge.textContent=text;badge.className='status-badge '+cls;}if(btn)btn.disabled=disabled;};
    setState('提交中','status-running',true);
    let body;
    try { body=whV2Body(); }
    catch(e) { setState('参数错误','status-error',false); addLog('❌ v2参数生成失败: '+e.message); return; }
    fetch('/api/wavehunter/v2/train/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
      .then(async r=>{const d=await r.json();if(!r.ok)throw Error(d.detail||d.error||'启动失败');return d;})
      .then(d=>{setState('训练中','status-running',true);addLog('🚀 WaveHunter v2 已启动: '+d.out_name);pollWhV2();})
      .catch(e=>{setState('启动失败','status-error',false);addLog('❌ v2启动失败: '+e.message);});
}
function stopWhV2Train(){fetch('/api/wavehunter/v2/train/stop',{method:'POST'}).then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})).then(()=>{document.getElementById('whv2Badge').textContent='停止中';addLog('⏹ v2停止请求已发送');}).catch(e=>addLog('❌ v2停止失败: '+e.message));}
function pollWhV2(){fetch('/api/wavehunter/v2/train/status').then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})).then(d=>{const b=document.getElementById('whv2Badge');const p=document.getElementById('whv2Progress');if(p)p.textContent=d.running?`训练进度 ${d.current||0}/${d.total||'?'} (${d.percent??0}%) | Reward: ${d.reward||'absolute_return'}`:(d.status||'空闲');if(d.log_path)fetch('/api/wavehunter/v2/train/log').then(x=>x.json()).then(z=>{const lines=z.lines||[];lines.slice(-20).forEach(line=>addLog('[WaveHunter v2] '+line,'wavehunter_v2'));}).catch(()=>{});if(d.running){b.textContent='训练中';b.className='status-badge status-running';setTimeout(pollWhV2,3000);}else{b.textContent=d.stopping?'已停止':(d.status==='finished'?'完成':'空闲');b.className='status-badge '+(d.status==='finished'?'status-done':'status-idle');loadWhV2Models();}}).catch(e=>{addLog('❌ v2状态读取失败: '+e.message);setTimeout(pollWhV2,5000);});}
function initWaveHunterV2(){loadWhV2Configs();loadWhV2Models();loadWhV2Schema();pollWhV2();}

// ─── v3 双头训练 (与 v2 共用同一表单, 仅追加 v3 专属参数) ───
function startWhV3Train(event){
    if (event) { event.preventDefault(); event.stopPropagation(); }
    const btn=document.getElementById('whv3StartBtn');
    const badge=document.getElementById('whv2Badge');
    const setState=(text, cls, disabled)=>{if(badge){badge.textContent=text;badge.className='status-badge '+cls;}if(btn)btn.disabled=disabled;};
    setState('v3 提交中','status-running',true);
    let body;
    try { body = whV2Body(); }
    catch(e) { setState('参数错误','status-error',false); addLog('❌ v3参数生成失败: '+e.message); return; }
    // 追加 v3 专属字段
    body.wave_threshold = parseFloat(document.getElementById('v3WaveThreshold').value);
    body.min_pullback_days = parseInt(document.getElementById('v3MinPullbackDays').value);
    body.hit_rate_weight = parseFloat(document.getElementById('v3HitRateWeight').value);
    body.continuation_weight = parseFloat(document.getElementById('v3ContinuationWeight').value);
    body.drawdown_penalty = parseFloat(document.getElementById('v3DrawdownPenalty').value);
    fetch('/api/wavehunter/v3/train/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
      .then(async r=>{const d=await r.json();if(!r.ok)throw Error(d.detail||d.error||'启动失败');return d;})
      .then(d=>{setState('v3 训练中','status-running',true);addLog('🚀 WaveHunter v3 (双头) 已启动: '+d.out_name);pollWhV3();})
      .catch(e=>{setState('启动失败','status-error',false);addLog('❌ v3启动失败: '+e.message);});
}
function stopWhV3Train(){fetch('/api/wavehunter/v3/train/stop',{method:'POST'}).then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})).then(()=>{document.getElementById('whv2Badge').textContent='v3 停止中';addLog('⏹ v3停止请求已发送');}).catch(e=>addLog('❌ v3停止失败: '+e.message));}
function pollWhV3(){
    fetch('/api/wavehunter/v3/train/status').then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);}))
      .then(d=>{
        const b=document.getElementById('whv2Badge'), btn=document.getElementById('whv3StartBtn');
        const p=document.getElementById('whv2Progress');
        if(p) p.textContent=d.running?`v3 训练进度 ${d.current||0}/${d.total||'?'} (${d.percent??0}%)`:(d.status||'v3 空闲');
        if(d.log_path){
            fetch('/api/wavehunter/v3/train/log').then(x=>x.json()).then(z=>{
                const lines=z.lines||[];lines.slice(-20).forEach(line=>addLog('[WaveHunter v3] '+line,'wavehunter_v3'));
            }).catch(()=>{});
        }
        if(d.running){b.textContent='v3 训练中';b.className='status-badge status-running';btn.disabled=true;setTimeout(pollWhV3,3000);}
        else{b.textContent=d.status==='done'?'v3 完成':(d.status==='failed'?'v3 失败':(d.status==='stopped'?'v3 已停止':'空闲'));b.className='status-badge '+(d.status==='done'?'status-done':'status-idle');btn.disabled=false;loadWhV2Models();}
      }).catch(e=>{addLog('❌ v3状态读取失败: '+e.message);setTimeout(pollWhV3,5000);});
}

// v2 参数 schema 单一真源：后端定义范围/默认值，前端只负责渲染和读取。
function loadWhV2Schema(){
    fetch('/api/wavehunter/v2/schema').then(r=>r.json()).then(s=>{
        window._whv2Schema=s.parameters||[];
        const panel=s.parameters.find(p=>p.key==='panel'); if(panel) window._whv2PanelDefault=panel.default;
        const grid=document.getElementById('whv2DynamicParams'); if(!grid)return;
        grid.innerHTML=(s.parameters||[]).map(p=>{
            const id='whv2_'+p.key;
            let input='';
            if(p.type==='select') input=`<select id="${id}" data-v2-key="${p.key}">${(p.options||[]).map(o=>`<option value="${o}" ${o===p.default?'selected':''}>${o}</option>`).join('')}</select>`;
            else if(p.type==='panel') input=`<select id="${id}" data-v2-key="${p.key}"><option value="${p.default}">${p.default}</option></select>`;
            else input=`<input id="${id}" data-v2-key="${p.key}" type="${p.type==='date'?'date':'number'}" value="${p.default}" ${p.min!==undefined?'min="'+p.min+'"':''} ${p.max!==undefined?'max="'+p.max+'"':''} ${p.step!==undefined?'step="'+p.step+'"':''}>`;
            return `<div class="param-item"><div class="plabel">${p.label||p.key} <span class="pen">${p.key}</span></div><div class="pdesc">动态参数 · 默认 ${p.default??''} ${p.min!==undefined?'· 范围 '+p.min+'~'+p.max:''}</div><div class="pinput">${input}<span class="phint">${p.unit||''}</span></div></div>`;
        }).join('');
        loadWhV2Panels();
    }).catch(e=>addLog('❌ v2 schema加载失败: '+e.message));
}
function whV2mEsc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
let _whv2mSort = {key: 'sharpe', dir: 'desc'};
function whv2mFmtNum(v, digits=4){if(v===null||v===undefined||isNaN(v))return '';return Number(v).toFixed(digits);}
function whv2mSortKey(metrics, key, row){
    if(key==='run_id') {
        const r=row?.run_id||'';
        const m=r.match(/^([a-z]+)_(\d+m)_(\d+k)_/);
        if(m) return `${m[1]}_${parseInt(m[2])}m_${parseInt(m[3])}k`;
        return r;
    }
    if(key==='status') return String(row?.status||'');
    if(key==='finished_at') return String(row?.finished_at||'');
    if(!metrics) return -Infinity;
    const v = metrics[key];
    if(key==='total_return') return metrics.total_return;
    if(key==='excess_return') return metrics.excess_return;
    if(key==='sharpe') return metrics.sharpe_ratio ?? metrics.sharpe;
    if(key==='drawdown') return metrics.max_drawdown;
    if(key==='win_rate') return metrics.win_rate;
    return -Infinity;
}
function whv2mFmtRow(x){
    const m=x.metrics||{};
    const statusClass={done:'status-done',running:'status-running',failed:'status-error',blocked:'status-error',stopped:'status-idle',pending:'status-idle'}[x.status]||'status-idle';
    const statusLabel={done:'完成',running:'运行中',failed:'失败',blocked:'阻塞',stopped:'已停止',pending:'待执行'}[x.status]||x.status||'';
    return `<td class="whv2m-run-cell" title="${whV2mEsc(x.run_id)}">${whV2mEsc(x.run_id)}</td>`
        +`<td class="whv2m-status-cell"><span class="status-badge ${statusClass}">${whV2mEsc(statusLabel)}</span></td>`
        +`<td>${whv2mFmtNum(m.total_return)}</td>`
        +`<td>${whv2mFmtNum(m.excess_return)}</td>`
        +`<td>${whv2mFmtNum(m.sharpe_ratio ?? m.sharpe)}</td>`
        +`<td>${whv2mFmtNum(m.max_drawdown)}</td>`
        +`<td>${whv2mFmtNum(m.win_rate, 1)}</td>`
        +`<td class="whv2m-time-cell">${whV2mEsc(x.finished_at||'')}</td>`;
}
function whv2mTh(key, label){
    const arrow = _whv2mSort.key===key ? (_whv2mSort.dir==='asc'?'▲':'▼') : '';
    return `<th class="whv2m-th" onclick="whv2mSortBy('${key}')">${label}${arrow}</th>`;
}
function whv2mSortBy(key){
    if(_whv2mSort.key===key) _whv2mSort.dir = _whv2mSort.dir==='asc'?'desc':'asc';
    else _whv2mSort = {key, dir:'desc'};
    if(window._whv2mLastRuns) whv2mRenderRuns(window._whv2mLastRuns);
}
function whv2mRenderRuns(runs){
    const sorted = [...runs].sort((a,b)=>{
        const va = whv2mSortKey(a.metrics, _whv2mSort.key, a);
        const vb = whv2mSortKey(b.metrics, _whv2mSort.key, b);
        if(va===vb) return 0;
        if(typeof va==='string' || typeof vb==='string') {
            const cmp=String(va).localeCompare(String(vb));
            return _whv2mSort.dir==='asc' ? cmp : -cmp;
        }
        return _whv2mSort.dir==='asc' ? va - vb : vb - va;
    });
    let h='<table class="whv2m-table">'
        +'<colgroup><col style="width:29%"><col style="width:11%"><col style="width:9%"><col style="width:9%"><col style="width:9%"><col style="width:9%"><col style="width:8%"><col style="width:16%"></colgroup>'
        +'<thead><tr>'
        +whv2mTh('run_id','Run')+whv2mTh('status','状态')+whv2mTh('total_return','收益')
        +whv2mTh('excess_return','超额')+whv2mTh('sharpe','Sharpe')+whv2mTh('drawdown','回撤')
        +whv2mTh('win_rate','胜率')+whv2mTh('finished_at','完成时间')+'</tr></thead><tbody>';
    sorted.forEach(x=>{h+='<tr style="border-top:1px solid #1f3a52">'+whv2mFmtRow(x)+'</tr>';});
    h+='</table>';
    document.getElementById('whv2mRuns').innerHTML=h;
}
function loadWhV2Matrix(){
    Promise.all(['config','runs','status','events','results'].map(k=>fetch('/api/wavehunter/v2/matrix/'+k).then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})))).then(([c,r,s,e,res])=>{
        document.getElementById('whv2mSummary').textContent=(c.total||0)+' Run | 当前 pool: '+(c.current_pool||'hs300')+' | '+Object.entries(c.pools||{}).map(([k,v])=>k+(v.exists?'✅':'❌')).join(' ')+' | reward=absolute_return_v3 | phase='+(c.phase||'smoke')+' | smoke='+(c.smoke_steps||[]).map(x=>x/1000+'k').join('/')+' full='+(c.full_steps||[]).map(x=>x/1000+'k').join('/');
        const eta = s.eta_seconds && s.running ? ` | ⏳ 剩余 ~${Math.round(s.eta_seconds/60)} 分钟` : (s.eta_seconds ? ` | ⏳ 估算完成 ~${Math.round(s.eta_seconds/60)} 分钟` : '');
        const displayRun = s.running ? s.current_run : (s.last_run || s.current_run);
        const progressText = displayRun ? (s.running ? `当前 ${displayRun}` : `已停止 · 最后 ${displayRun}`) : '';
        document.getElementById('whv2mProgress').textContent=(s.status||'idle')+(progressText?` | ${progressText}`:'')+' | 已完成 '+(s.done||s.completed||0)+'/'+(s.total||c.total||0)+eta;
        document.getElementById('whv2mEta').textContent = s.running ? `当前阶段: pool=${c.current_pool||'hs300'} phase=${s.phase||'smoke'} | pending: ${s.pending||0} | failed: ${s.failed||0} | blocked: ${s.blocked||0}` : '';
        document.getElementById('whv2mPhaseTip').textContent = (c.phase_tips||{})[c.phase||'smoke'] || '';
        // 更新 pool radio 状态徽章 + 自动选中当前 pool
        for (const p of ['hs300','csi500','cs800']){
          const el = document.getElementById('whv2mPool'+p.charAt(0).toUpperCase()+p.slice(1));
          if (el) el.textContent = (c.pools?.[p]?.exists) ? '✅' : '❌';
        }
        if (c.current_pool){
          const r = document.querySelector(`input[name="whv2mPool"][value="${c.current_pool}"]`);
          if (r) r.checked = true;
        }
        document.getElementById('whv2mStart').disabled=!!s.running;
        document.getElementById('whv2mStop').disabled=!s.running;
        const resetBtn=document.getElementById('whv2mReset');
        if(resetBtn) resetBtn.disabled=!!s.running;
        const b=document.getElementById('whv2mBadge'); const labels={done:'已完成',stopped:'已暂停',failed:'失败',blocked:'阻塞',stale:'状态过期',idle:'空闲'}; const badgeStatus=s.running?'running':(s.status||'idle'); b.textContent=s.running?'运行中':(labels[badgeStatus]||badgeStatus); b.className='status-badge '+(s.running?'status-running':(badgeStatus==='done'?'status-done':((badgeStatus==='failed'||badgeStatus==='blocked'||badgeStatus==='stale')?'status-error':'status-idle')));
        // runs 是状态真源，results 只补充已完成 Run 的 metrics；不能用空数组触发错误短路。
        const resultMap = new Map((res.results||[]).map(x=>[x.run_id, x]));
        const rows = (r.runs||[]).map(x=>{
            const z = resultMap.get(x.run_id)||{};
            return {run_id:x.run_id, status:x.status, metrics:z.metrics||x.metrics||{}, finished_at:z.finished_at||x.finished_at, error:z.error||x.error||''};
        });
        window._whv2mLastRuns = rows;
        whv2mRenderRuns(rows);
        document.getElementById('whv2mEvents').innerHTML=(e.events||[]).slice(-100).map(x=>{
            const cfg=x.config||{}; const t=x.time||''; const err=x.error||cfg.error||'';
            const p=cfg.train||{}; const sim=cfg.simulation||{};
            const detail=[x.kind, x.run_id, err, x.index!==undefined?`Run ${x.index}/${x.total||0}`:'', p.n_steps!==undefined?`训练 ${p.n_steps}/${p.batch_size}/${p.lstm_hidden}`:'', sim.cost_bps!==undefined?`成本 ${sim.cost_bps}bps`:'', x.kind==='run_start'?'source=wavehunter_v2_matrix':''].filter(Boolean).join(' | ');
            const fullJson = JSON.stringify(x);
            const shortJson = fullJson.length > 220 ? fullJson.slice(0, 220) + '…' : fullJson;
                        return `<div title="${whV2mEsc(shortJson)}"><span style="color:#78909c">${whV2mEsc(t)}</span> ${whV2mEsc(detail)}</div>`;
        }).join('')||'暂无事件';
        if(s.running){clearTimeout(window._whv2mTimer);window._whv2mTimer=setTimeout(loadWhV2Matrix,5000);}
    }).catch(err=>{addLog('❌ v2矩阵刷新失败: '+err.message);document.getElementById('whv2mProgress').textContent='刷新失败: '+err.message;});
}
function whv2mPhaseChange(){const v=document.querySelector('input[name="whv2mPhase"]:checked')?.value||'smoke';addLog(`🔄 阶段已切换为 ${v}（下次启动生效）`);}
function whv2mPoolChange(){const v=document.querySelector('input[name="whv2mPool"]:checked')?.value||'hs300';addLog(`🔄 股票池已切换为 ${v}（下次启动生效）`);}
function startWhV2Matrix(){if(!document.getElementById('whv2mConfirm').checked){addLog('⚠ 请先确认v2面板和链路');return;}const phase=document.querySelector('input[name="whv2mPhase"]:checked')?.value||'smoke';const pool=document.querySelector('input[name="whv2mPool"]:checked')?.value||'hs300';fetch('/api/wavehunter/v2/matrix/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm:true,resume:true,phase,pool,version:'v3'})}).then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})).then(d=>{if(d.error||d.detail)throw Error(d.error||d.detail);addLog('🚀 v3矩阵已启动: pool='+(d.pool||'hs300')+'/'+(d.phase||'smoke')+' version='+(d.version||'v3')+' '+(d.total||9)+' Run');loadWhV2Matrix();}).catch(e=>{addLog('❌ v2矩阵启动失败: '+e.message);});}
function stopWhV2Matrix(){fetch('/api/wavehunter/v2/matrix/stop',{method:'POST'}).then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})).then(()=>{addLog('⏹ v2矩阵停止请求已发送');loadWhV2Matrix();}).catch(e=>addLog('❌ v2矩阵停止失败: '+e.message));}
function resetWhV2Matrix(){if(!confirm('⚠ 确认清空 v2 矩阵?\n将删除: registry/summary/events/results + matrix_*.log + matrix_*.json ledger + matrix_* 模型目录\n保留: 其他模型 + 矩阵面板 + 训练/模拟日志')){addLog('⚠ 已取消清空');return;}const keep=confirm('保留 matrix_* 模型目录?\n确定 = 仅清状态文件 (可重跑后处理)\n取消 = 同时删除模型目录 (彻底从头)');const btn=document.getElementById('whv2mReset');if(btn){btn.disabled=true;btn.textContent='🧹 清空中...';}fetch('/api/wavehunter/v2/matrix/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm:true,keep_models:keep})}).then(r=>r.ok?r.json():r.json().then(e=>{throw Error(e.detail||'HTTP '+r.status);})).then(d=>{const rm=d.removed||{};addLog('🧹 v2矩阵已清空: registry='+rm.registry+' summary='+rm.summary+' events='+rm.events+' results='+rm.results+' model_dirs='+rm.model_dirs+' logs='+rm.training_logs+' ledgers='+rm.ledgers+' (keep_models='+d.keep_models+')');loadWhV2Matrix();}).catch(e=>{addLog('❌ v2矩阵清空失败: '+e.message);}).finally(()=>{if(btn){btn.disabled=false;btn.textContent='🧹 清空矩阵';}});}

// 数据管理逻辑已抽至 /static/tabs/data_mgmt.js

// 供权重矩阵子账本复用：采集当前 v2 表单为训练请求体（缺则取默认值）
function whV2CollectParams(){
  const body={};
  try{
    const fp=(typeof getV2FormParams==='function'?getV2FormParams():null);
    if(fp && typeof fp==='object') Object.assign(body, fp);
  }catch(_){}
  // Fallback defaults aligned with v2 schema
  if(!body.panel) body.panel='wavehunter_v8_hs300_20040102_20260827.parquet';
  if(!body.start_date) body.start_date='2023-01-01';
  if(!body.end_date) body.end_date='2023-03-31';
  if(!body.total_timesteps) body.total_timesteps=50000;
  if(!body.window) body.window=20;
  if(!body.top_k) body.top_k=20;
  if(!body.lstm_hidden) body.lstm_hidden=64;
  if(!body.n_steps) body.n_steps=32;
  if(!body.batch_size) body.batch_size=32;
  if(!body.seed) body.seed=42;
  return body;
}
window.whV2CollectParams = whV2CollectParams;


// ─── v9 四段训练 (与 v2 共用同一表单, 自动路由到 v9 脚本) ───
function startWhV9Train(event){
    if (event) { event.preventDefault(); event.stopPropagation(); }
    const btn=document.getElementById('whv9StartBtn');
    const badge=document.getElementById('whv2Badge');
    const setState=(text, cls, disabled)=>{if(badge){badge.textContent=text;badge.className='status-badge '+cls;}if(btn)btn.disabled=disabled;};
    setState('v9 提交中','status-running',true);
    let body;
    try { body = whV2Body(); }
    catch(e) { setState('参数错误','status-error',false); addLog('❌ v9参数生成失败: '+e.message); return; }
    // 设置 v9 面板（自动路由到 v9 训练脚本）
    body.panel = 'wavehunter_v9_hs300_20040102_20260804.parquet';
    body.label_version = 'v9';
    fetch('/api/wavehunter/v2/train/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
      .then(async r=>{const d=await r.json();if(!r.ok)throw Error(d.detail||d.error||'启动失败');return d;})
      .then(d=>{setState('v9 训练中','status-running',true);addLog('🚀 WaveHunter v9 (四段) 已启动: '+d.out_name);pollWhV2();})
      .catch(e=>{setState('启动失败','status-error',false);addLog('❌ v9启动失败: '+e.message);});
}
