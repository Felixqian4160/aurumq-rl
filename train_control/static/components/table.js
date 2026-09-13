/** components/table.js — 表格复用 */
function renderTable(containerId, rows, columns){
  const el = document.getElementById(containerId);
  if (!el) return;
  const head = '<tr>' + columns.map(c=>'<th>'+c.label+'</th>').join('') + '</tr>';
  const body = rows.map(r=>'<tr>' + columns.map(c=>'<td>'+(r[c.key]??'')+'</td>').join('') + '</tr>').join('');
  el.innerHTML = '<table class="data-table"><thead>'+head+'</thead><tbody>'+body+'</tbody></table>';
}
