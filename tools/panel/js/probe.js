import { $, esc, num, api, state, flash } from './core.js';

// The LLM Probe view: what the engine implements, and what the model can do.
// A report is one (model, config) pair, so the list keeps every run rather than
// collapsing per model: two runs of the same model are two different questions.
const pct = v => v==null ? '—' : v.toFixed(1)+'%';
const col = v => v==null ? 'var(--dim)' : v>=90 ? 'var(--green)' : v>=70 ? 'var(--amber)' : 'var(--red)';

export function probeOptions(catalog){
  const sel = $('#probe-model');
  const cur = sel.value;
  sel.innerHTML = catalog.map(m=>`<option value="${esc(m.id)}">${esc(m.id)}</option>`).join('');
  if (catalog.some(m=>m.id===cur)) sel.value = cur;
}

export async function loadProbe(){
  const [d, cat] = await Promise.all([
    api('/api/probe/runs').catch(()=>({runs:[]})), api('/api/engine').catch(()=>({catalog:[]}))]);
  probeOptions(cat.catalog || []);
  $('#probenote').textContent = d.version ? `llmprobe ${d.version} · ${d.runs.length} reports` : 'llmprobe not installed';

  // What the engine is missing is the part we can fix in code, so it goes first.
  const gaps = new Map();
  for (const r of d.runs) for (const m of (r.missing_surface||[])) gaps.set(m, (gaps.get(m)||0)+1);
  $('#probe-gate').innerHTML = gaps.size ? `
    <div class="sechead"><h2>Surface gaps</h2><span class="note">missing in the latest report per surface; an engine defect, not a model one</span></div>
    <div class="quick">${[...gaps].sort((a,b)=>b[1]-a[1])
      .map(([m,n])=>`<span class="pill gap">${esc(m)} <span class="dim">${n}</span></span>`).join('')}</div>`
    : '<div class="empty">No surface gaps recorded.</div>';

  $('#probe-runs').innerHTML = d.runs.length ? d.runs.map(r=>`
    <div class="model-row rich">
      <span class="nm" title="${esc(r.file)}">${esc(r.model||r.file)}</span>
      <span class="mt">conf <b style="color:${col(r.conformance)}">${pct(r.conformance)}</b>
        <span class="dim">${esc(r.conformance_items)}</span> · cap <b style="color:${col(r.capability)}">${pct(r.capability)}</b>
        · fid ${pct(r.fidelity)} · ${r.decode_tps?r.decode_tps+' tok/s':'no bench'}${r.ttft_ms?' · '+r.ttft_ms+' ms TTFT':''}
        · ${r.duration_s}s · ${esc((r.started_at||'').replace('T',' ').slice(0,19))}</span>
      <span class="acts"><span class="dim">${esc(r.version)}</span></span>
    </div>`).join('') : '<div class="empty" style="padding:18px">No reports yet. Run one.</div>';

  const st = await api('/api/probe/state').catch(()=>null);
  if (st?.running){
    $('#probe-run').disabled = true;
    $('#probe-run').textContent = `probing ${st.model}…`;
    $('#probe-cards').innerHTML = `<div class="empty" style="padding:14px">running: ${esc(st.lines.slice(-1)[0]||'')}</div>`;
    setTimeout(loadProbe, 2000);
  } else {
    $('#probe-run').disabled = false;
    $('#probe-run').textContent = 'Run probe';
  }
}

export function wireProbe(){
  $('#probe-run').onclick = async ()=>{
    const model = $('#probe-model').value;
    if (!model) return;
    $('#probe-run').disabled = true;
    try{
      await api('/api/probe/run', {method:'POST', headers:{'Content-Type':'application/json'},
                                    body: JSON.stringify({model})});
      loadProbe();
    }catch(e){ flash(e.message, true); $('#probe-run').disabled = false; }
  };
}