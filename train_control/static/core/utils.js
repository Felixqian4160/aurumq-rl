/**
 * core/utils.js — 通用工具
 */
function formatDate(d){
  if (!d) return '';
  return String(d).slice(0, 10);
}

function debounce(fn, ms){
  let t = null;
  return function(){
    const args = arguments;
    clearTimeout(t);
    t = setTimeout(()=>fn.apply(null, args), ms);
  };
}
