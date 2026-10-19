import { $, state, esc, num, api, flash } from './core.js';

// ---------------------------------------------------------------- convert
export const fmtSize = n => n==null ? '—' : n>2**30 ? (n/2**30).toFixed(1)+' GiB'
  : n>2**20 ? (n/2**20).toFixed(1)+' MiB' : n>2**10 ? (n/1024).toFixed(0)+' KiB' : n+' B';

export async function loadConvertFiles(){
  let d, recs;
  try{
    [d, recs] = await Promise.all([
      api('/api/convert/files'),
      api('/api/convert/recipes').catch(()=>({ok:false})),
    ]);
  }catch(e){ $('#convnote').textContent = 'Could not scan: '+e.message; return; }
  const sa = $('#c-artifacts'), ss = $('#c-sources');
  sa.innerHTML = '<option value="">— pick an artifact —</option>' +
    d.artifacts.map(a=>`<option value="${esc(a.path)}">
      ${esc(a.path.split('/').pop())} (${fmtSize(a.size)})</option>`).join('');
  const srcOpts = [];
  for (const s of d.sources) srcOpts.push(`<option value="${esc(s.path)}">
    ${esc(s.path.split('/').pop())} (${fmtSize(s.size)})</option>`);
  for (const c of d.checkpoints) srcOpts.push(`<option value="${esc(c.path)}">
    ${esc(c.path.split('/').pop())}</option>`);
  ss.innerHTML = '<option value="">— pick a source —</option>' + srcOpts.join('');
  ss.onchange = syncDelCtl; syncDelCtl();
  // Real recipes from tools/convert, not hardcoded: MoE/35B/Flash-Next included.
  if (recs && recs.ok){
    $('#c-recipe').innerHTML = Object.keys(recs.recipes).sort().map(r=>
      `<option value="${esc(r)}">${esc(r)}</option>`).join('');
    $('#convnote').textContent =
      `${d.artifacts.length} artifacts · ${d.sources.length} fuentes · ${d.checkpoints.length} checkpoints · ${Object.keys(recs.recipes).length} recipes`;
  } else {
    $('#convnote').textContent = `could not read the recipes: ${recs?.error||'?'}`;
  }
  // Local GGUF basenames -> real path, to mark "✓ you already have it" in the
  // HF search and to pick the local file for one of them.
  hfLocal = new Map(d.sources.map(s=>[s.path.split('/').pop(), s.path]));
}

// ---------------------------------------------------------------- buscar en HF
let hfTimer = null, hfLocal = new Map();

export function hfOpen(){
  $('#modal-hf').style.display = 'flex';
  $('#hf-results').innerHTML = '';
  $('#hf-status').textContent = '';
  const inp = $('#hf-q'); inp.value = ''; inp.focus();
}
export function hfClose(){ $('#modal-hf').style.display = 'none'; }

export async function hfSearch(q){
  const box = $('#hf-results');
  $('#hf-status').textContent = 'buscando…';
  let r;
  try{ r = await api('/api/convert/search?q=' + encodeURIComponent(q)); }
  catch(e){ $('#hf-status').textContent = e.message; return; }
  const hits = r.hits || [];
  $('#hf-status').textContent = r.error || (hits.length ? hits.length + ' results' : 'no results');
  if (!hits.length){ box.innerHTML = `<div class="empty" style="padding:14px">nothing for "${esc(q)}"</div>`; return; }
  box.innerHTML = hits.map(h=>`
    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;padding:8px 12px;
      border:1px solid var(--line-subtle);border-radius:var(--radius-sm);margin-bottom:6px;cursor:pointer"
      onclick="hfPick('${encodeURIComponent(h.id)}')">
      <span style="font:12.5px var(--mono);color:${h.gguf!==null?'var(--teal)':'var(--text)'}">${esc(h.id)}</span>
      <span style="display:flex;gap:8px;align-items:center">
        ${h.gguf!==null?'<span style="font:10px var(--mono);color:var(--green);border:1px solid rgba(52,211,153,.4);padding:1px 6px;border-radius:8px">GGUF</span>':''}
        <span style="font:11px var(--mono);color:var(--dim)">♥ ${h.likes}</span>
      </span>
    </div>`).join('');
}

async function hfPick(encModel){
  const model = decodeURIComponent(encModel);
  $('#hf-status').textContent = 'loading GGUFs from ' + model + '…';
  let g;
  try{ g = await api('/api/convert/ggufs?model=' + encodeURIComponent(model)); }
  catch(e){ $('#hf-status').textContent = e.message; return; }
  if (!g || !g.length){ $('#hf-results').innerHTML = '<div class="empty" style="padding:14px">no .gguf in this repo</div>'; return; }
  // Group by quant: the shards of one quantisation download together.
  const groups = new Map();
  for (const x of g){
    if (!groups.has(x.quant)) groups.set(x.quant, []);
    groups.get(x.quant).push(x);
  }
  const quantLabel = q => q.startsWith('__') ? q.slice(2).toUpperCase() : q;
  $('#hf-results').innerHTML = `
    <div style="margin-bottom:10px;font:12px var(--mono);color:var(--dim)">
      <b style="color:var(--teal)">${esc(model)}</b> — pick a quantisation (✓ = already local, ⬇ = download):
      <span style="float:right;cursor:pointer;color:var(--amber)" onclick="hfSearch(document.getElementById('hf-q').value)">← back</span>
    </div>
    ${[...groups.entries()].map(([q, fs]) => {
      const total = fs.reduce((a,x)=>a+(x.size||0),0);
      const allLocal = fs.every(x=>hfLocal.has(x.path.split('/').pop()));
      return `<div style="display:flex;justify-content:space-between;align-items:center;gap:10px;padding:7px 12px;margin-bottom:5px;
        border:1px solid ${allLocal?'rgba(52,211,153,.35)':'var(--line-subtle)'};border-radius:var(--radius-sm);cursor:pointer"
        data-model="${btoa(model)}"
        data-quant="${btoa(q)}"
        data-files="${btoa(JSON.stringify(fs.map(x=>x.path)))}"
        data-first="${btoa(allLocal ? hfLocal.get(fs[0].path.split('/').pop())
                             : 'hf://' + model + ':' + fs[0].path)}"
        data-total="${total}" data-local="${allLocal?'1':'0'}" onclick="hfPickRow(this)">
        <span style="font:12px var(--mono)">${esc(quantLabel(q))}
          <span style="font:10.5px var(--mono);color:var(--dim)">${fs.length>1?fs.length+' archivos · ':''}${fmtSize(total)}</span></span>
        <span style="display:flex;gap:8px;align-items:center;font:11px var(--mono)">
          <span class="hfbadge" style="color:${allLocal?'var(--green)':'var(--dim)'}">${allLocal?'✓ local':'🔗 HF'}</span>
          ${allLocal?'':'<button data-act="dl" onclick="event.stopPropagation(); hfDl(this)" style="padding:3px 8px" title="Download to inbox">⬇</button>'}
        </span>
      </div>`;}).join('')}`;
}

function hfPickRow(el){
  if (el.dataset.act === 'dl') return;      // the click was on the download button
  const model = atob(el.dataset.model);
  const val = atob(el.dataset.first);
  const label = val.split('/').pop();
  const sel = $('#c-sources');
  if (![...sel.options].some(o=>o.value===val)){
    const opt = document.createElement('option');
    opt.value = val;
    opt.textContent = (val.startsWith('hf://') ? '🔗 ' : '✓ ') + label;
    sel.insertBefore(opt, sel.firstChild);
  }
  sel.value = val;
  hfClose();
  syncDelCtl();
  flash(val.startsWith('hf://')
    ? 'Remote source marked — use ⬇ to download it into the inbox.'
    : 'Fuente local seleccionada.');
}

// Deleting the source only applies to what the panel downloaded (inbox/).
export function syncDelCtl(){
  const w = $('#c-del-wrap');
  if (!w) return;
  const src = ($('#c-sources').value || '');
  const inbox = src.includes('/ninfer/inbox/');
  $('#c-del').disabled = !inbox;
  $('#c-del').checked = false;
  w.title = inbox ? 'Opt-in: borra ninfer/inbox/ completo de ese repo al convertir OK.' 
                  : 'Only valid for sources the panel downloaded (inbox/).';
  probeSource(src);
}

// Auto-recipe + sources from the source: read it and assemble recipe/base/gguf/
// dflash2/vision/components/template. state.probe alimenta runConvert.
let probeCache = {};
async function probeSource(src){
  const box = $('#c-flagsum');
  if (!src || src.startsWith('hf://')){ box.innerHTML = ''; state.probe = null; return; }
  state.probe = null;
  box.innerHTML = '<span style="color:var(--dim)">analyzing source…</span>';
  let pr;
  if (probeCache[src]){ pr = probeCache[src]; }
  else{
    try{ pr = await api('/api/convert/probe?path=' + encodeURIComponent(src)); }
    catch(e){ box.innerHTML = `<span class="err">probe failed: ${esc(e.message)}</span>`; return; }
    if (pr && !pr.error) probeCache[src] = pr;
  }
  state.probe = pr;
  const rec = $('#c-recipe');
  if (pr && pr.known && pr.recipe){
    const opt = [...rec.options].find(o=>o.value===pr.recipe);
    if (opt) rec.value = pr.recipe;
    const parts = [];
    if (pr.base) parts.push('base=' + pr.base.split('/').pop());
    if (pr.sources.gguf) parts.push('gguf=' + pr.sources.gguf.split('/').pop());
    ['dflash2','vision','imatrix'].forEach(k=>{
      if (pr.sources[k]) parts.push(k + '=' + pr.sources[k].split('/').pop());
    });
    box.innerHTML = `<span style="color:var(--green)">${esc(pr.recipe)}</span> · ` +
      `<span style="color:var(--dim)">${esc(parts.join(' · '))}</span>` +
      (pr.components.length>1?` · <span style="color:var(--cyan)">comp: ${esc(pr.components.join(','))}</span>`:'') +
      (pr.template?` · tmpl:${esc(pr.template.split('/').slice(-2)[0])}`:'');
  } else {
    box.innerHTML = `<span class="err">⚠ unknown recipe for this source — pick the recipe manually.</span>`;
  }
}

async function hfDl(el){
  const row = el.closest('[data-model]');
  const model = atob(row.dataset.model);
  const files = JSON.parse(atob(row.dataset.files));
  const total = +row.dataset.total;
  try{
    const r = await api('/api/convert/download', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model, files, total})});
    if (r.error){ flash(r.error, true); return; }
    state.download = {run_id:r.run_id, model, quant: atob(row.dataset.quant)};
    const badge = row.querySelector('.hfbadge');
    badge.innerHTML = '⬇ 0%';
    flash(`Downloading ${files.length} file(s) (${fmtSize(total)})…`);
    pollDownload();
  }catch(e){ flash('Could not start the download: ' + e.message, true); }
}

async function pollDownload(){
  const dl = state.download;
  if (!dl) return;
  let st;
  try{ st = await api('/api/convert/download?run_id=' + encodeURIComponent(dl.run_id)); }
  catch(e){ setTimeout(pollDownload, 1000); return; }
  if (st.error){ state.download = null; flash(st.error, true); return; }
  const row = [...document.querySelectorAll('#hf-results > div[data-quant]')]
              .find(r=>atob(r.dataset.quant)===dl.quant);
  const badge = row?.querySelector('.hfbadge');
  if (badge && st.status==='running') badge.innerHTML = `⬇ ${st.pct}%`;
  if (st.status==='running'){
    $('#hf-status').textContent =
      `downloading… ${st.pct}% (${fmtSize(st.done)} / ${fmtSize(st.total)})`;
    setTimeout(pollDownload, 1000);
    return;
  }
  state.download = null;
  $('#hf-status').textContent = '';
  if (st.status==='failed'){
    flash('Download failed: ' + (st.error || 'see log').slice(0,200), true);
    return;
  }
  // done
  for (const p of st.files){ hfLocal.set(p.split('/').pop(), p); }
  flash(`Descarga terminada — ${st.files.length} archivo(s) en ${st.dest.split('/').slice(-2).join('/')}.`);
  if (dl.model){
    await hfPick(dl.model);
    const rows = [...document.querySelectorAll('#hf-results > div[data-quant]')];
    const target = rows.find(r=>atob(r.dataset.quant)===dl.quant)
                || rows.find(r=>r.dataset.local==='1');
    if (target) hfPickRow(target);
  }
}

export async function inspectArtifact(){
  const path = $('#c-artifacts').value.trim();
  const box = $('#convresult');
  if (!path) return;
  box.innerHTML = '<div class="empty" style="padding:14px">Reading artifact…</div>';
  try{
    const r = await api('/api/convert/inspect?path=' + encodeURIComponent(path));
    if (r.error){ box.innerHTML = `<div class="err">${esc(r.error)}</div>`; return; }
    const comps = Object.entries(r.components||{}).map(([n,c])=>{
      const cfg=c.config||{};
      return `<div class="dgroup"><h4>${esc(n)}</h4><dl class="kv">
        <span>architecture</span><span>${esc((cfg.architectures||[]).join(', ')||'—')}</span>
        <span>layers</span><span>${cfg.num_hidden_layers??'—'}</span>
        <span>hidden</span><span>${cfg.hidden_size??'—'}</span>
        <span>vocab</span><span>${cfg.vocab_size??'—'}</span>
        <span>native ctx</span><span>${cfg.max_position_embeddings??'—'}</span>
      </dl></div>`;
    }).join('');
    box.innerHTML = `<div class="detailbox" style="width:100%;max-width:none">
      <div class="dgroup"><h4>Artifact</h4><dl class="kv">
        <span>name</span><span>${esc(r.name||'—')}</span>
        <span>version</span><span>${r.version}</span>
        <span>size</span><span>${fmtSize(r.file_bytes)}</span>
        <span>tensors</span><span>${r.tensors}</span>
        <span>formats</span><span>${esc(Object.entries(r.formats||{}).map(([k,v])=>k+': '+v).join(', '))}</span>
      </dl></div>
      ${comps}
    </div>`;
  }catch(e){ box.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}

export async function runConvert(){
  const body = { model:$('#c-sources').value.trim(), recipe:$('#c-recipe').value };
  if (!body.model){ flash('Pick a source first.'); return; }
  if (body.model.startsWith('hf://')){
    flash('That source is remote (step 2: download). For now pick a local one or tick the download.', true);
    return;
  }
  body.delete_source = !$('#c-del').disabled && $('#c-del').checked;
  // Auto-sources from the probe: base/gguf/dflash2/vision/components/template.
  const pr = state.probe;
  if (pr && pr.known && pr.base){
    body.base = pr.base;
    body.sources = pr.sources || {};
    body.components = pr.components;
    body.template = pr.template;
  }
  $('#convlog').innerHTML = '<div class="empty" style="padding:14px">Starting conversion…</div>';
  $('#c-run').disabled = true;
  try{
    const r = await api('/api/convert/run', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    if (r.error){ $('#convlog').innerHTML = `<div class="err">${esc(r.error)}</div>`;
      $('#c-run').disabled = false; return; }
    state.convert = {run_id:r.run_id, log:[]};
    $('#convlog').innerHTML = `<div class="live" style="border-radius:8px;padding:12px 14px">
      <b style="color:var(--green)">● converting</b>
      <span style="font:12px var(--mono);color:var(--dim)">${esc(r.run_id)}</span>
      <pre id="convlogpre" style="margin-top:8px;max-height:280px;overflow-y:auto"></pre>
    </div>`;
  }catch(e){
    $('#convlog').innerHTML = `<div class="err">${esc(e.message)}</div>`;
    $('#c-run').disabled = false;
  }
}


// The SSE stream is opened once at startup (see boot.js), so there is nothing
// left for ensureStream() here: with no job running the server stays quiet.

export function wireHfSearch(){
  $('#hf-q').addEventListener('input', e => {
    clearTimeout(hfTimer);
    const q = e.target.value.trim();
    if (q.length >= 3) hfTimer = setTimeout(()=>hfSearch(q), 350);
  });
}
