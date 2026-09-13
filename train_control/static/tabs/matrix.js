/**
 * tabs/matrix.js — 矩阵（全局规律 + 账本）双视图
 * 顶置 subTab 切换，全局规律走 /api/wavehunter/matrix/*，账本走 /api/matrix-ledger/* + /api/sim/ledgers
 * index.html 内已注入 subtab 结构与懒加载，此文件仅兜底暴露 switchMatrixSubTab。
 */
(function(){
  function switchMatrixSubTab(name){
    // 切换按钮与面板
    document.querySelectorAll('#subtab-matrix, #subtab-ledger, #subtab-weight').forEach(el=>el.classList.remove('active'));
    const btn = document.getElementById('subtab-' + name);
    const pane = document.getElementById('subtab-content-' + name);
    if (btn) btn.classList.add('active');
    document.querySelectorAll('#subtab-content-matrix, #subtab-content-ledger, #subtab-content-weight').forEach(el=>{
      el.style.display = 'none';
      el.classList.remove('active');
    });
    if (pane) { pane.style.display = ''; pane.classList.add('active'); }
    // 懒加载：账本/矩阵各自刷新
    if (name === 'ledger' && typeof mlRefresh === 'function') try{ mlRefresh(); }catch(_){}
    if (name === 'matrix' && typeof loadWhMatrix === 'function') try{ loadWhMatrix(); }catch(_){}
    if (name === 'weight' && typeof loadWeightMatrix === 'function') try{ loadWeightMatrix(); }catch(_){}
  }
  window.switchMatrixSubTab = switchMatrixSubTab;
})();
