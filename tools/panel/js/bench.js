import { $, state, esc, num, flat, specKey, api, flash } from './core.js';
import { loadConvertFiles, syncDelCtl } from './convert.js';

const COLS = [
  // The ones needed to decide, always on screen.
  {k:'model',      t:'modelo',       l:true,  f:r => r.model},
  {k:'engine',     t:'motor',        l:true,  f:r => (r.server?.binary)||'—'},
  {k:'spec',       t:'spec',         l:true,  f:r => r.server?.spec_backend
      ? r.server.spec_backend + (r.server.draft_depth ? '/'+r.server.draft_depth : '') : 'none'},
  {k:'dataset',    t:'dataset',      l:true,  f:r => r.dataset},
  {k:'concurrency',t:'c',            d:0,     f:r => r.concurrency},
  {k:'out_tps_avg',t:'gen tok/s',    d:1,     f:r => r.out_tps_avg},
  {k:'prefill_tps',t:'prefill tok/s',d:0,     f:r => r.prefill_tps},
  {k:'latency_p95',t:'p95 ms',       d:0,     f:r => r.latency_p95},
  {k:'ttft_avg',   t:'ttft ms',      d:0,     f:r => r.ttft_avg},
  {k:'spec_decode_acceptance_length', t:'acc', d:2, f:r => r.spec_decode_acceptance_length},
  // The ones reached for when something does not add up, one click away.
  {k:'isl_avg',    t:'isl',    more:true, d:0, f:r => r.isl_avg},
  {k:'osl_avg',    t:'osl',    more:true, d:0, f:r => r.osl_avg},
  {k:'latency_p50',t:'p50 ms',more:true, d:0, f:r => r.latency_p50},
  {k:'latency_p99',t:'p99 ms',more:true, d:0, f:r => r.latency_p99},
  {k:'spec_decode_draft_acceptance_rate',t:'draft %',more:true, d:1,
      f:r => r.spec_decode_draft_acceptance_rate},
  {k:'spec_decode_accepted_per_verified',t:'apv',more:true, d:2,
      f:r => r.spec_decode_accepted_per_verified},
  {k:'effective_concurrency',t:'c_ef',more:true,d:2,f:r => r.effective_concurrency},
  {k:'req_per_s',  t:'req/s',  more:true, d:2, f:r => r.req_per_s},
  {k:'error_rate', t:'err %',  more:true, d:2, f:r => r.error_rate},
  {k:'duration_s', t:'seg',    more:true, d:1, f:r => r.duration_s},
  {k:'request_count_target',t:'n',more:true,d:0,f:r => r.request_count_target},
  {k:'output_tokens_mean',t:'osl obj',more:true,d:0,f:r => r.output_tokens_mean},
];
let sortK='ts', sortDir=1;
function updateCompareBtn(){
  const btn = $('#btn-compare');
  if (!btn) return;
  const count = state.selected.size;
  btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/></svg> Comparar (${count}/2)`;
  btn.disabled = count !== 2;
  btn.classList.toggle('primary', count === 2);
}

export function showCompareModal(){
  if (state.selected.size !== 2) return;
  const ids = Array.from(state.selected);
  const rA = state.runs.find(r => r.run_id === ids[0]);
  const rB = state.runs.find(r => r.run_id === ids[1]);
  if (!rA || !rB) return;

  const diffPct = (a, b) => {
    if (a == null || b == null || a === 0) return '';
    const pct = ((b - a) / a) * 100;
    const sign = pct > 0 ? '+' : '';
    const cls = pct > 0 ? 'diff-up' : pct < 0 ? 'diff-down' : 'diff-eq';
    return `<span class="${cls}">(${sign}${pct.toFixed(1)}%)</span>`;
  };

  const metrics = [
    { label: 'Modelo', a: rA.model, b: rB.model, type: 'str' },
    { label: 'Spec-Decode', a: rA.server?.spec_backend || 'off', b: rB.server?.spec_backend || 'off', type: 'str' },
    { label: 'Concurrencia', a: rA.concurrency, b: rB.concurrency, type: 'num', d: 0 },
    { label: 'Throughput (gen tok/s)', a: rA.out_tps_avg, b: rB.out_tps_avg, type: 'num', higher_is_better: true },
    { label: 'Prefill (tok/s)', a: rA.prefill_tps, b: rB.prefill_tps, type: 'num', higher_is_better: true },
    { label: 'Latencia p95 (ms)', a: rA.latency_p95, b: rB.latency_p95, type: 'num', higher_is_better: false },
    { label: 'TTFT (ms)', a: rA.ttft_avg, b: rB.ttft_avg, type: 'num', higher_is_better: false },
    { label: 'Acceptance Length', a: rA.spec_decode_acceptance_length, b: rB.spec_decode_acceptance_length, type: 'num', higher_is_better: true, d: 2 },
    { label: 'Draft Acceptance Rate', a: rA.spec_decode_draft_acceptance_rate != null ? rA.spec_decode_draft_acceptance_rate + '%' : null, b: rB.spec_decode_draft_acceptance_rate != null ? rB.spec_decode_draft_acceptance_rate + '%' : null, type: 'str' }
  ];

  const tableRows = metrics.map(m => {
    let diff = '';
    if (m.type === 'num' && m.a != null && m.b != null) {
      const pct = ((m.b - m.a) / (m.a || 1)) * 100;
      const isGood = m.higher_is_better ? pct > 0 : pct < 0;
      const cls = Math.abs(pct) < 0.1 ? 'diff-eq' : isGood ? 'diff-up' : 'diff-down';
      diff = `<span class="${cls}">${pct > 0 ? '+' : ''}${pct.toFixed(1)}%</span>`;
    }
    const valA = m.type === 'num' ? num(m.a, m.d ?? 1) : esc(m.a || '—');
    const valB = m.type === 'num' ? num(m.b, m.d ?? 1) : esc(m.b || '—');
    return `<tr>
      <td class="l" style="font-weight:600;color:var(--text-muted)">${esc(m.label)}</td>
      <td>${valA}</td>
      <td>${valB}</td>
      <td style="text-align:center">${diff || '—'}</td>
    </tr>`;
  }).join('');

  $('#modal-compare-body').innerHTML = `
    <div style="display:flex;justify-content:space-between;margin-bottom:14px;font:11px var(--mono)">
      <div style="background:var(--panel2);padding:8px 12px;border-radius:6px;border:1px solid var(--line-subtle);flex:1;margin-right:10px">
        <b style="color:var(--teal)">Baseline (A):</b> ${esc(rA.run_id)}
      </div>
      <div style="background:var(--panel2);padding:8px 12px;border-radius:6px;border:1px solid var(--line-subtle);flex:1">
        <b style="color:var(--green)">Target (B):</b> ${esc(rB.run_id)}
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th class="l">Metric</th>
          <th>Baseline (A)</th>
          <th>Target (B)</th>
          <th style="text-align:center">Delta (B vs A)</th>
        </tr>
      </thead>
      <tbody>${tableRows}</tbody>
    </table>
  `;
  $('#modal-compare').style.display = 'flex';
}

export function exportCsv(){
  if (!state.runs.length) { flash('No runs to export.', true); return; }
  const cols = COLS.filter(c => state.more || !c.more);
  const header = ['run_id', ...cols.map(c => c.k)].join(',');
  const rows = state.runs.map(r => {
    const f = flat(r);
    const values = cols.map(c => {
      let v = c.k === 'engine' ? f.engine : c.k === 'spec' ? f.spec : r[c.k];
      return `"${String(v ?? '').replace(/"/g, '""')}"`;
    });
    return [`"${r.run_id}"`, ...values].join(',');
  });
  const csvContent = 'data:text/csv;charset=utf-8,' + encodeURIComponent([header, ...rows].join('\n'));
  const link = document.createElement('a');
  link.setAttribute('href', csvContent);
  link.setAttribute('download', `ninfer-benchmarks-${new Date().toISOString().slice(0,10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ---------------------------------------------------------------- render
function renderStrip(){
  const runs = state.runs;
  const spec = runs.filter(r => r.spec_decode_acceptance_length!=null);
  const eng = [...new Set(runs.map(r=>r.engine).filter(Boolean))];
  const best = runs.filter(r=>r.out_tps_avg!=null).sort((a,b)=>b.out_tps_avg-a.out_tps_avg)[0];
  const a = state.cfg?.aiperf;
  $('#strip').innerHTML = `
    <span><span class="dot ${a?.ready?'ok':'bad'}"></span><b>AIPerf</b>
      <span class="k">${a?.ready ? esc(a.why) : esc(a?.why||'?')}</span></span>
    <span><b>${runs.length}</b> <span class="k">corridas</span></span>
    <span><b>${best ? best.out_tps_avg.toFixed(0)+' tok/s' : '—'}</b>
      <span class="k">mejor</span></span>
    <span><b>${spec.length}</b> <span class="k">with spec-decode</span></span>
    ${eng.length?`<span><span class="k">motores</span> <b>${esc(eng.join(' · '))}</b></span>`:''}
    <span><span class="k">actualizado</span> <b id="upd">ahora</b></span>`;
}

function renderHead(){
  const selectTh = `<th style="width:34px;text-align:center" title="Tick to compare">#</th>`;
  $('#thead').innerHTML = selectTh + COLS.filter(c=>state.more||!c.more).map(c=>{
    const arrow = sortK===c.k ? `<span class="arrow">${sortDir>0?'▼':'▲'}</span>` : '';
    return `<th data-k="${c.k}" class="${c.l?'l':''}">${c.t} ${arrow}</th>`;
  }).join('');
  $('#thead').querySelectorAll('th').forEach(th=>th.onclick=()=>{
    const k = th.dataset.k;
    if (sortK===k) sortDir = -sortDir;
    else { sortK = k; sortDir = 1; }
    state.page = 1;
    load();
  });
}

function renderRows(){
  const cols = COLS.filter(c=>state.more||!c.more);
  const tb = $('#tbody');
  const empty = $('#empty');
  if (!state.runs.length){
    tb.innerHTML = '';
    const sinFiltros = !['#q-profile','#q-engine','#q-spec','#q-dataset']
      .some(s => $(s).value);
    empty.hidden = false;
    empty.innerHTML = sinFiltros
      ? `<b>No runs yet</b>Benchmarks are archived here.<br>
         <span class="hint">Pick a profile and a dataset above and press
         <code>Correr</code>.</span>`
      : `<b>No runs match these filters</b>
         <span class="hint">Try dropping a filter, or
         <a href="#" id="limpiar" style="color:var(--teal)">limpiar todos</a>.</span>`;
    $('#limpiar')?.addEventListener('click', e => {
      e.preventDefault();
      ['#q-profile','#q-engine','#q-spec','#q-dataset'].forEach(s => { $(s).value=''; });
      state.page = 1; load();
    });
    renderStrip(); renderPager({runs:[], total:0, page:1, size:state.size, pages:1});
    return;
  }
  empty.hidden = true;
  tb.innerHTML = state.runs.map((raw,i)=>{
    const r = flat(raw);
    const eng = (r.engine||'').includes('llama') ? 'llama' : 'ninfer';
    const isSel = state.selected.has(raw.run_id);
    const checkTd = `<td style="text-align:center;padding:4px" onclick="event.stopPropagation()">
      <input type="checkbox" class="run-select" data-id="${esc(raw.run_id)}" ${isSel?'checked':''}>
    </td>`;
    const cells = checkTd + cols.map(c=>{
      let v = c.k==='engine' ? r.engine : c.k==='spec' ? r.spec : raw[c.k];
      let cell;
      if (c.k==='engine'){
        cell = !raw.server || !raw.server.binary
          ? '<span class="muted" title="run from before the server snapshot existed">sin dato</span>'
          : `<span class="pill ${eng}">${esc(v)}</span>`;
      }
      else if (c.k==='spec'){
        cell = !raw.server || !raw.server.binary
          ? '<span class="muted">—</span>'
          : v==='none' ? '<span class="muted">off</span>'
          : `<span class="pill mtp">${esc(v)}</span>`;
      }
      else if (c.k==='model') cell = esc(String(v||'').replace(/\.(ninfer|gguf)$/,'').slice(0,34));
      else cell = num(v, c.d===undefined?2:c.d);
      return `<td class="${c.l?'l':''}">${cell}</td>`;
    }).join('');
    const detail = state.open.has(raw.run_id) ? detailRow(raw) : '';
    return `<tr class="run" data-i="${i}">${cells}</tr>${detail}`;
  }).join('');
  tb.querySelectorAll('tr.run').forEach(tr=>tr.onclick=()=>{
    const r = state.runs[+tr.dataset.i];
    state.open.has(r.run_id) ? state.open.delete(r.run_id) : state.open.add(r.run_id);
    renderRows();
  });
  tb.querySelectorAll('input.run-select').forEach(cb => {
    cb.onchange = e => {
      e.stopPropagation();
      const id = cb.dataset.id;
      if (cb.checked) {
        if (state.selected.size >= 2) {
          // Once there are 2, the older one goes
          const first = state.selected.values().next().value;
          state.selected.delete(first);
        }
        state.selected.add(id);
      } else {
        state.selected.delete(id);
      }
      updateCompareBtn();
      renderRows();
    };
  });
  renderStrip();
}

function detailRow(r){
  const s = r.server || {};
  const hist = r.acceptance_histogram
    ? Object.entries(r.acceptance_histogram).sort((a,b)=>+a[0]-+b[0]) : null;
  const maxh = hist ? Math.max(...hist.map(h=>+h[1])) : 1;
  const WANT = /^--(spec|draft|lm-head|adaptive|ngram|kv-|max-|min-|host-|device-|prefill|use-alt|structured|expert)/;
  const argv = s.argv || [];
  const pairs = [];
  for (let i=0;i<argv.length;i++){
    if (!WANT.test(argv[i])) continue;
    const next = argv[i+1];
    pairs.push(next && !next.startsWith('--') ? `${argv[i]} ${next}` : argv[i]);
  }
  const flags = pairs.filter(p=>!p.startsWith('--model')).join('  ');
  return `<tr class="detail"><td colspan="99"><div class="detailbox">
    <div class="dgroup"><h4>Servidor</h4><dl class="kv">
      <dt>motor</dt><dd>${esc(s.binary||'—')}</dd>
      <dt>modelo</dt><dd>${esc(s.model||'—')}</dd>
      <dt>carriles</dt><dd>${s.slots??'—'}</dd>
      <dt>contexto</dt><dd>${s.max_context??'—'}</dd>
      <dt>spec</dt><dd>${!s.binary ? '<span class="muted">—</span>'
        : esc(specKey(s)||'off')}</dd>
      <dt>cuant</dt><dd>${esc(s.quant||'—')}</dd>
    </dl></div>
    <div class="dgroup"><h4>Flags de capacidad y spec</h4>
      <pre>${esc(flags || '(ninguna)')}</pre></div>
    <div class="dgroup"><h4>Spec-decode</h4><dl class="kv">
      <dt>acceptance length</dt><dd>${num(r.spec_decode_acceptance_length,3)}</dd>
      <dt>weighted by token</dt><dd>${num(r.spec_decode_token_weighted_acceptance_length,3)}</dd>
      <dt>draft acceptance</dt><dd>${num(r.spec_decode_draft_acceptance_rate,1)}%</dd>
      <dt>acc/verificado</dt><dd>${num(r.spec_decode_accepted_per_verified,3)}</dd>
      <dt>drafts</dt><dd>${num(r.spec_decode_draft_tokens,0)} → ${num(r.spec_decode_accepted_draft_tokens,0)}</dd>
      <dt>rondas</dt><dd>${num(r.spec_decode_steps,0)}</dd>
    </dl>
    ${hist?`<div class="hist">${hist.map(([k,v])=>
      `<div style="height:${(v/maxh*100).toFixed(1)}%" title="position ${k}: ${v}"><span>${k}</span></div>`).join('')}
      </div><div style="font:10px var(--mono);color:var(--dimmer);margin-top:20px">
      pooled histogram by position</div>`:''}
    </div>
    <div class="dgroup"><h4>Carga</h4><dl class="kv">
      <dt>dataset</dt><dd>${esc(r.dataset)}</dd>
      <dt>concurrencia</dt><dd>${r.concurrency}</dd>
      <dt>requests</dt><dd>${num(r.request_count_target,0)}</dd>
      <dt>osl objetivo</dt><dd>${num(r.output_tokens_mean,0)}</dd>
      <dt>isl / osl real</dt><dd>${num(r.isl_avg,1)} / ${num(r.osl_avg,1)}</dd>
      <dt>errores</dt><dd>${num(r.error_rate,2)}%</dd>
      <dt>tokenizer</dt><dd>${esc(r.tokenizer||'—')}</dd>
      <dt>aiperf</dt><dd>${esc(r.aiperf_version||'—')}</dd>
      <dt>ninfer git</dt><dd>${esc(r.ninfer_git||'—')}</dd>
    </dl></div>
    <div class="dgroup" style="grid-column:1/-1"><h4>Comando AIPerf</h4>
      <pre>${esc(r.aiperf_cmd||'(no guardado)')}</pre></div>
  </div></td></tr>`;
}

function renderPager(d){
  const el = $('#pager');
  if (!d || (d.total <= d.runs.length && d.pages <= 1)){
    el.innerHTML = d && d.total > d.runs.length
      ? `<span style="color:var(--dimmer)">${d.total} runs in total — raise the page size to see more</span>` : '';
    return;
  }
  const btn = (label, page, dis) => `<button data-p="${page}" ${dis?'disabled':''}>${label}</button>`;
  el.innerHTML = `
    <span>page <b style="color:var(--text)">${d.page}</b> of ${d.pages}</span>
    ${btn('‹ anterior', d.page-1, d.page<=1)}
    ${btn('siguiente ›', d.page+1, d.page>=d.pages)}
    <span class="spacer" style="flex:1"></span>
    <span>per page</span>
    <select id="pg-size">
      ${[10,25,50,100,0].map(n=>
        `<option value="${n}" ${n===d.size?'selected':''}>${n||'todas'}</option>`).join('')}
    </select>`;
  el.querySelectorAll('button[data-p]').forEach(b=>b.onclick=()=>{
    state.page = +b.dataset.p; load();
  });
  $('#pg-size').onchange = e => { state.size = +e.target.value; state.page = 1; load(); };
}

// ---------------------------------------------------------------- datos (registro)
export async function load(){
  const q = new URLSearchParams({sort:sortK, dir: sortDir>0 ? '1' : '0',
    page:String(state.page), size:String(state.size)});
  const f = {profile:'q-profile', engine:'q-engine', spec:'q-spec', dataset:'q-dataset'};
  for (const [k,id] of Object.entries(f)) { const v=$('#'+id).value; if(v) q.set(k,v); }
  const d = await api('/api/bench/runs?'+q);
  state.runs = d.runs; state.total = d.total;
  fillFilter('#q-profile','profile', r=>r.profile);
  fillFilter('#q-engine','engine', r=>r.engine);
  fillFilter('#q-spec','spec', r=>r.spec);
  fillFilter('#q-dataset','dataset', r=>r.dataset);
  $('#resnote').textContent = d.total===d.runs.length
    ? `${d.total} corridas` : `${d.runs.length} de ${d.total}`;
  renderHead(); renderRows(); renderPager(d);
}
function fillFilter(sel, key, get){
  const el=$(sel), cur=el.value;
  const vals=[...new Set(state.runs.map(get).filter(Boolean))].sort();
  el.innerHTML = '<option value="">'+({profile:'todos',engine:'cualquier motor',
    spec:'cualquier spec',dataset:'cualquier dataset'}[key])+'</option>'+
    vals.map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('');
  el.value = cur;
}

// ---------------------------------------------------------------- bench
export async function runBench(){
  const body={ profile:$('#f-profile').value, dataset:$('#f-dataset').value,
    model:$('#f-model').value, concurrency:+$('#f-conc').value,
    requests:+$('#f-req').value, output_tokens:+$('#f-osl').value||128,
    tokenizer:$('#f-tok').value.trim(),
    url:state.cfg.defaults.url };
  $('#btn-run').disabled=true;
  try{
    const r = await api('/api/bench/run',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    state.job = {run_id:r.run_id, total:body.requests, progress:{records:0}};
    if (!state.live){ state.live=true; connect(); }
  }catch(e){ flash(e.message, true); }
  finally{ $('#btn-run').disabled=false; }
}

function renderLive(job){
  if (!job){ $('#livejobs').innerHTML=''; $('#btn-stop').disabled=true; return; }
  $('#btn-stop').disabled = false;
  const p = job.progress||{}, done = p.records||0, want = job.total||1;
  const PHASE = {starting:'starting', preparing:'preparing · downloading dataset',
    measuring:'measuring', finished:'finished', failed:'failed'};
  $('#livejobs').innerHTML = `<div class="live" style="margin:0 15px 15px;border-radius:8px;padding:12px 14px">
    <div style="display:flex;gap:16px;align-items:baseline;font:12px var(--mono);flex-wrap:wrap">
      <b style="color:${p.phase==='failed'?'var(--red)':'var(--green)'}">● ${esc(PHASE[p.phase]||'live')}</b>
      <span>${esc(job.run_id)}</span>
      <span>${done} / ${want} requests</span>
      ${done?`<span>lat ${num(p.latency_avg,0)} ms</span>
              <span>${num(p.out_tps_avg,1)} tok/s</span>`:''}
    </div>
    <div class="bar"><i style="width:${Math.min(100, done/want*100).toFixed(0)}%"></i></div>
    ${!done && p.log_tail ? `<div style="font:11px var(--mono);color:var(--dim);margin-top:6px;
       overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(p.log_tail)}</div>` : ''}
  </div>`;
}

export function connect(){
  const es = new EventSource('/api/bench/stream');
  es.onmessage = async ev => {

    const m = JSON.parse(ev.data);
    if (m.type==='progress' && state.job && m.run_id===state.job.run_id){
      state.job.progress = m.progress; renderLive(state.job);
    } else if (m.type==='finished'){
      $('#btn-stop').disabled=true;
      if (state.job?.run_id===m.run_id){
        state.job=null; renderLive(null);
        if (m.ok){
          flash(`Corrida <code>${m.run_id}</code> registrada — ${num(m.run.out_tps_avg,1)} tok/s,
            ${num(m.run.spec_decode_acceptance_length,2)} acceptance.`);
        } else {
          $('#banner').innerHTML = `<div class="err"><b>La corrida
            <code>${esc(m.run_id)}</code> registered no results</b>
            (exit ${m.exit}).<pre style="margin-top:9px">${esc((m.log_tail||[]).join('\n'))}</pre></div>`;
        }
      }
      load();
    } else if (m.type==='convert' && state.convert?.run_id===m.run_id){
      const pre = $('#convlogpre');
      if (pre){ pre.textContent = m.log_tail.join('\n'); pre.scrollTop = pre.scrollHeight; }
    } else if (m.type==='convert_done'){
      $('#c-run').disabled = false;
      if (state.convert?.run_id===m.run_id){
        state.convert = null;
        const pre = $('#convlogpre');
        if (pre) pre.insertAdjacentText('beforeend',
          m.ok ? '\n\nCONVERSION OK' : '\n\nFAILED (see the log above)');
        if (m.ok && m.deleted) flash('Conversion OK · downloaded source deleted from the inbox.');
        loadConvertFiles(); syncDelCtl();
      }
    } else if (m.type==='download_done' && state.download?.run_id===m.run_id){
      // redundant: download progress is followed by polling (pollDownload).
    }
  };
  es.onerror = ()=>{ /* el EventSource reconecta solo */ };
}

// The model is not picked blind: it comes from the catalog the server declares.
export async function loadEngine(){
  const sel = $('#f-model'), bar = $('#engbar');
  try{
    const e = await api('/api/engine');
    if (!e.online){
      sel.disabled = $('#btn-run').disabled = true;
      return;
    }
    sel.innerHTML = e.catalog.map(m=>{
      const loaded = e.loaded.includes(m.id);
      return `<option value="${esc(m.id)}">${esc(m.id)} · ${loaded?'loaded':(m.status||'unloaded')}</option>`;
    }).join('');
    if (e.loaded.length) sel.value = e.loaded[0];
    sel.disabled = false;
    // The tokenizer follows the model: with another one AIPerf produces no
    // results and the run never reaches the registry.
    const syncTok = () => {
      const map = state.cfg?.aiperf?.tokenizers || {};
      $('#f-tok').value = map[sel.value] || state.cfg?.aiperf?.tokenizer || '';
    };
    sel.onchange = syncTok; syncTok();
  }catch(err){ /* server down: the live tab says so */ }
}

