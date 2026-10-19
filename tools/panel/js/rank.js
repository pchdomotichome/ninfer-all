import { $, esc, api, flash } from './core.js';

// The Ranking view.
//
// Two things are deliberately not done here. There is no single score: a model
// that is fast and wrong is not "second best", it is disqualified, and one number
// would hide that. And rows are never silently dropped: a model with speed data
// and no protocol data shows the gap, because that gap is the finding.
//
// The gates are pass/fail on evidence. They are ordered by how expensive they
// are to discover late, not by how interesting they are.
const GATES = [
  {id:'probe', label:'probe on record', kind:'data',
   why:'without an llmprobe report nothing else can be compared'},
  {id:'conformance', label:'chat protocol MUST', kind:'pct', min:80, field:'conformance',
   why:'the engine must implement what it advertises before model quality means anything'},
  {id:'capability', label:'capability ≥ 70%', kind:'pct', min:70, field:'capability',
   why:'below this the model is not usable for agent work'},
  {id:'correctness', label:'fidelity correct', kind:'text', ok: r => !/0\/\d+ items/.test(r.correctness||''),
   why:'zero correct items means the output is unusable no matter how fast it is'},
];

const AXES = [
  {id:'conformance', label:'protocol',   get:r=>r.probe?.conformance,  max:100, unit:'%',  better:'up'},
  {id:'capability',  label:'capability', get:r=>r.probe?.capability,   max:100, unit:'%',  better:'up'},
  {id:'decode',      label:'decode',     get:r=>r.probe?.decode_tps,   max:120, unit:'t/s',better:'up'},
  {id:'ttft',        label:'TTFT',       get:r=>r.probe?.ttft_ms,      max:400, unit:'ms', better:'down'},
  {id:'p95',         label:'p95',        get:r=>r.bench?.latency_p95,  max:8000,unit:'ms', better:'down'},
  {id:'throughput',  label:'out tok/s',  get:r=>r.bench?.out_tps_avg, max:300, unit:'t/s',better:'up'},
];

function gateState(row, g){
  const r = row.probe;
  if (!r) return g.kind === 'data' ? 'fail' : 'unknown';
  if (g.kind === 'pct') return (r[g.field] ?? -1) >= g.min ? 'pass' : 'fail';
  if (g.kind === 'text') return g.ok(r) ? 'pass' : 'fail';
  return 'pass';
}
const WHY = {pass:'', fail:'', unknown:'no probe report on record'};

export async function loadRank(){
  const d = await api('/api/rank').catch(()=>({rows:[]}));
  const order = $('#rank-order')?.value || 'agentic';
  const rows = d.rows || [];
  const hidden = d.hidden_bench || [];

  $('#rank-gates').innerHTML = `
    <div class="sechead"><h2>Gates</h2><span class="note">hard gates first; failing one is not a lower score, it is a disqualification</span></div>
    <div class="quick">${GATES.map(g=>{
      const n = rows.filter(r=>gateState(r,g)==='pass').length;
      const f = rows.filter(r=>gateState(r,g)==='fail').length;
      return `<span class="pill gate ${n>0?'ok':'fail'}" title="${esc(g.why)}">${esc(g.label)} <b>${n}/${rows.length}</b>${f?` <span class="dim">${f} fail</span>`:''}</span>`;
    }).join('')}</div>`;

  const score = r => order === 'decode' ? (r.probe?.decode_tps || 0) : (r.probe?.capability || 0);
  const sorted = [...rows].sort((a,b)=>{
    const ga = GATES.every(g=>gateState(a,g)==='pass'), gb = GATES.every(g=>gateState(b,g)==='pass');
    if (ga !== gb) return ga ? -1 : 1;
    return (score(b)||0)-(score(a)||0);
  });

  const head = AXES.map(ax=>`<th class="${ax.better==='down'?'lo':''}">${esc(ax.label)}<span class="dim">${ax.unit}</span></th>`).join('');
  const body = sorted.map(r=>{
    const cells = AXES.map(ax=>{
      const v = ax.get(r);
      if (v==null) return '<td class="num dim">—</td>';
      const w = Math.min(100, v/ax.max*100);
      const c = v>=ax.max*0.7 ? 'var(--teal)' : v>=ax.max*0.35 ? 'var(--blue)' : 'var(--amber)';
      return `<td class="num"><span class="gauge"><i style="width:${w.toFixed(0)}%;background:${c}"></i></span>${v.toFixed(ax.unit==='%'?1:0)}</td>`;
    }).join('');
    const failed = GATES.filter(g=>gateState(r,g)==='fail').map(g=>g.label);
    return `<tr class="${failed.length?'':'live'}">
      <td class="nm">${esc(r.model)}${failed.length?`<span class="badget" title="${esc(failed.join(' · '))}">${failed.length} gate${failed.length>1?'s':''}</span>`:''}</td>
      <td class="dim">${esc(r.state||'—')}</td>${cells}
      <td class="dim">${[r.probe?'':'no probe', r.bench?'':'no bench'].filter(Boolean).join(' · ')||'—'}</td>
    </tr>`;
  }).join('');
  $('#rank-rows').innerHTML = sorted.length ? `
    <table class="rank"><thead><tr><th>model</th><th>state</th>${head}<th>gaps</th></tr></thead>
    <tbody>${body}</tbody></table>` : '<div class="empty" style="padding:18px">Nothing to rank yet.</div>';
  // Named, not swallowed: these are AIPerf runs whose serving id the router no
  // longer has, so they cannot be compared against anything.
  if (hidden.length) $('#rank-rows').insertAdjacentHTML('beforeend',
    `<div style="padding:10px 18px;color:var(--dim);font:11.5px var(--mono)">Not ranked — AIPerf runs for <code>${esc(hidden.join(', '))}</code>: aliases the router no longer serves.</div>`);
}

export function wireRank(){
  const s = $('#rank-order');
  if (s) s.onchange = loadRank;
}