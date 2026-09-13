// tabs/chart.js — 📊 图表 Tab
// 从 index.html inline JS 抽离；依赖全局 fetch/toast/addLog/echarts/refreshChartModels
// 图表
// ══════════════════════════════════════════════════
function loadChartData(value) {
    if (!value || !value.includes(':')) {
        document.getElementById('chartContainer').style.display = 'none';
        return;
    }
    const idx = value.indexOf(':');
    if (idx < 0) {
        document.getElementById('chartContainer').style.display = 'none';
        return;
    }
    const source = value.substring(0, idx);
    const realName = value.substring(idx + 1);
    document.getElementById('chartContainer').style.display = 'block';
    const url = '/api/train/metrics-detail/' + encodeURIComponent(source) + '/' + encodeURIComponent(realName);
    fetch(url).then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.detail || 'HTTP ' + r.status); })).then(d => {
        if (!d.metrics || d.metrics.length === 0) {
            toast('❌ 该模型无训练指标数据');
            return;
        }
        // 解析数据
        const steps = d.metrics.map(m => m.timestep ?? m.num_timesteps ?? m.step ?? 0);
        const extra = m => m.extra || {};
        const rewards = d.metrics.map(m => {
            const ex = extra(m);
            return m.episode_reward_mean ?? m.reward ?? m.rewards ?? m['rollout/ep_rew_mean']
                ?? ex.episode_reward_mean ?? ex.reward ?? ex.rewards ?? ex['rollout/ep_rew_mean'] ?? null;
        });
        const policyLoss = d.metrics.map(m => m.policy_loss ?? extra(m)['train/policy_gradient_loss'] ?? null);
        const valueLoss = d.metrics.map(m => m.value_loss ?? extra(m)['train/value_loss'] ?? null);
        const entropy = d.metrics.map(m => m.entropy ?? extra(m)['train/entropy_loss'] ?? null);
        const approxKl = d.metrics.map(m => m.approx_kl ?? extra(m)['train/approx_kl'] ?? null);
        const fps = d.metrics.map(m => m.fps ?? extra(m)['time/fps'] ?? null);
        const lr = d.metrics.map(m => m.learning_rate ?? extra(m)['train/learning_rate'] ?? null);
        const extraSeries = (key) => d.metrics.map(m => m[key] ?? (m.extra && m.extra[key]) ?? null);

        // ECharts 主题: 暗色
        const theme = {
            backgroundColor: 'transparent',
            textStyle: { color: '#e0e0e0' },
            grid: { left: '8%', right: '6%', top: '12%', bottom: '14%' },
            xAxis: { type: 'category', axisLabel: { color: '#8899aa', fontSize: 10 }, axisLine: { lineStyle: { color: '#2a3a4a' } } },
            yAxis: { type: 'value', axisLabel: { color: '#8899aa', fontSize: 10 }, splitLine: { lineStyle: { color: '#1a2a3a' } } },
            tooltip: { trigger: 'axis', backgroundColor: '#1a2a3a', borderColor: '#4fc3f7', textStyle: { color: '#e0e0e0' } },
        };

        function makeChart(el, data, name, color) {
            const c = echarts.init(document.getElementById(el));
            const valid = data.filter(v => v !== null && v !== undefined && Number.isFinite(Number(v)));
            const displayName = name.startsWith('Reward') ? name : (name === '奖励均值' ? 'Reward（absolute_return，扣成本后组合收益）' : name);
            c.setOption({
                ...theme,
                xAxis: { ...theme.xAxis, data: steps },
                yAxis: { ...theme.yAxis },
                series: [{ type: 'line', data, name: displayName, smooth: true, symbol: 'none',
                    lineStyle: { width: 2, color }, itemStyle: { color } }],
            });
            c.resize();
        }

        const rewardSeries = rewards.some(v => v !== null && v !== undefined && Number.isFinite(Number(v))) ? rewards : extraSeries('rollout/ep_rew_mean');
        makeChart('chartReward', rewardSeries, 'Reward（absolute_return，扣成本后组合收益）', '#4fc3f7');
        // 胜率曲线
        const winRateSeries = extraSeries('reward/win_rate');
        makeChart('chartWinRate', winRateSeries, '胜率（最近20步正收益占比）', '#66bb6a');
        makeChart('chartPolicyLoss', policyLoss, '策略损失', '#ff7043');
        makeChart('chartValueLoss', valueLoss, '价值损失', '#ced4da');

        const meta = {...(d.metadata?.metadata||{}), ...(d.metadata?.training_summary||{})};
        const metaKeys = ['model_version','algorithm','env_type','reward_type','reward_definition','total_timesteps','window','top_k','max_position_pct','rebalance_days','cost_bps','learning_rate','lstm_hidden','lstm_layers','seed','n_factors','train_start_date','train_end_date','panel'];
        document.getElementById('chartTrainMetadata').textContent = metaKeys.filter(k=>meta[k]!==undefined).map(k=>`${k}: ${meta[k]}`).join('  |  ') || '该训练产物没有 metadata.json；仅显示训练指标';

        // 熵值 + KL 双轴
        const ec = echarts.init(document.getElementById('chartEntropy'));
        ec.setOption({
            ...theme,
            xAxis: { ...theme.xAxis, data: steps },
            yAxis: [
                { type: 'value', name: '熵值', nameTextStyle: { color: '#ce93d8' }, axisLabel: { color: '#ce93d8' }, splitLine: { show: false } },
                { type: 'value', name: 'KL', nameTextStyle: { color: '#ffcc02' }, axisLabel: { color: '#ffcc02' }, splitLine: { show: false } }
            ],
            series: [
                { type: 'line', data: entropy, name: '熵值', smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#ce93d8' }, yAxisIndex: 0 },
                { type: 'line', data: approxKl, name: 'KL散度', smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#ffcc02' }, yAxisIndex: 1 }
            ],
        });
        ec.resize();

        // FPS + 学习率 双轴
        const ef = echarts.init(document.getElementById('chartFps'));
        ef.setOption({
            ...theme,
            xAxis: { ...theme.xAxis, data: steps },
            yAxis: [
                { type: 'value', name: 'FPS', nameTextStyle: { color: '#4fc3f7' }, axisLabel: { color: '#4fc3f7' }, splitLine: { show: false } },
                { type: 'value', name: '学习率', nameTextStyle: { color: '#81c784' }, axisLabel: { color: '#81c784' }, splitLine: { show: false } }
            ],
            series: [
                { type: 'line', data: fps, name: 'FPS', smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#4fc3f7' }, yAxisIndex: 0, connectNulls: true },
                { type: 'line', data: lr, name: '学习率', smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#81c784' }, yAxisIndex: 1 }
            ],
        });
        ef.resize();

        addLog(`📊 已加载图表: ${realName} (${d.count} 个时间点)`);
    }).catch(e => { toast('❌ 加载图表失败: ' + e.message); addLog('❌ 图表加载失败: ' + e.message); });
}

refreshChartModels();

// ══════════════════════════════════════════════════
