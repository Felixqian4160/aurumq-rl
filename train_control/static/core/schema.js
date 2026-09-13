/**
 * core/schema.js — 统一 module schema 加载
 */
function loadModuleSchema(module, containerId){
  return fetch('/api/'+module+'/schema').then(r=>r.json()).then(s=>{
    const el = document.getElementById(containerId);
    if (!el || !s || !s.parameters) return s;
    el.innerHTML = '';
    s.parameters.forEach(f=>{
      const row = document.createElement('div');
      row.className = 'form-row';
      row.innerHTML = '<label>'+ (f.label||f.key) + '</label><input id="'+containerId+'_'+f.key+'" value="'+(f.default||'')+'" />';
      el.appendChild(row);
    });
    return s;
  });
}

function moduleSchemaParams(module, containerId){
  const out = {};
  const el = document.getElementById(containerId);
  if (!el) return out;
  el.querySelectorAll('input, select').forEach(inp=>{
    const k = inp.id.replace(containerId+'_', '');
    out[k] = inp.value;
  });
  return out;
}
