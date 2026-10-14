import { $, state, esc, num, api } from './core.js';

// ---------------------------------------------------------------- requests
const REQ_COLS = [
  {k:'ts', t:'hora', l:true, d:0, f:r=>r.ts_ms},
  {k:'model', t:'modelo', l:true, f:r=>r.model},
  {k:'spec_backend', t:'spec', l:true, f:r=>r.spec_backend},
  {k:'ttft_ms', t:'ttft ms', d:0},
  {k:'decode_s', t:'decode s', d:2},
  {k:'prefill_s', t:'prefill s', d:3},
  {k:'total_s', t:'total s', d:2},
  {k:'completion_tokens', t:'out tok', d:0},
  {k:'cache_hit', t:'cache', d:0},
  {k:'acceptance_avg', t:'acc len', d:2},
  {k:'draft_rate', t:'draft %', d:1},
  // bajo demanda
  {k:'protocol', t:'protocolo', more:true, l:true},
  {k:'prefill_tokens', t:'prefill tok', more:true, d:0},
  {k:'finish', t:'finish', more:true, l:true},
  {k:'draft_window', t:'draft win', more:true, d:0},
  {k:'accepted_tokens', t:'aceptados', more:true, d:0},
  {k:'drafted_tokens', t:'drafteados', more:true, d:0},
];
let reqMore = false;

export async function loadRequests(){
  let d;
  try{ d = await api('/api/requests?n=60'); }
  catch(e){ return; }
  if (!d.ok){ return; }
  // No re-renderizar si nada cambió: el polling de 3s no debe destruir un
  // detalle abierto ni resetear el scroll cada vez.
  const prev = state.reqs || [];
  const same = d.requests.length === prev.length &&
    d.requests[0]?.ts_ms === prev[0]?.ts_ms &&
    d.requests[d.requests.length-1]?.ts_ms === prev[prev.length-1]?.ts_ms;
  state.reqs = d.requests;
  state.reqOpen = state.reqOpen ?? new Set();
  renderSpecPos();
  if (same && $('#tbody-req').children.length) {
    $('#reqs-note').innerHTML = reqLogNote(d);
    return;
  }
  renderRequests();
  $('#reqs-note').innerHTML = reqLogNote(d);
}

// Aceptación por posición: la TASA real de cada posición del draft, no un
// conteo crudo — el denominador son las rondas de verificación, que es lo que
// hace comparables posiciones distintas y requests de distinta longitud.
//
// Se agrega sólo entre requests del mismo backend y ventana: mezclar ventanas
// distintas haría la posición 4 existir en unos y no en otros, y la media
// mentiría.
function renderSpecPos(){
  const box = $('#spec-pos'), note = $('#sp-note');
  const rs = (state.reqs||[]).filter(r => r.spec_rounds>0 && (r.histogram||[]).length);
  if (!rs.length){
    box.innerHTML = `<div class="sp-empty">sin spec-decode en los últimos requests
      <span style="color:var(--dimmer)">aparece cuando el modelo corre con <code>--spec</code></span></div>`;
    note.textContent = '';
    return;
  }
  const groups = new Map();
  for (const r of rs){
    const k = (r.spec_backend||'?')+' · ventana '+(r.draft_window??'?');
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r);
  }
  const [key, g] = [...groups.entries()].sort((a,b)=>b[1].length-a[1].length)[0];
  const use = g.slice(-50);
  // `accepted_per_position` siempre viene con 15 ranuras (la del array de
  // métricas), pero sólo --draft-tokens posiciones pueden tener drafts. Dibujar
  // las 15 pondría 12 barras siempre en cero; se recorta a la ventana real.
  const len = Math.max(...use.map(r=>r.histogram.length));
  let W = use[0].draft_window;
  if (!(W>0) || W>len) W = len - 1;
  while (W>1 && use.every(r => !(r.histogram[W-1]>0))) W--;
  const acc = new Array(W).fill(0);
  let rounds = 0, accepted = 0, drafted = 0;
  for (const r of use){
    for (let i=0;i<W;i++) acc[i] += r.histogram[i] || 0;
    rounds += r.spec_rounds;
    accepted += r.accepted_tokens || 0;
    drafted += r.drafted_tokens || 0;
  }
  const rates = acc.map(a => rounds ? a/rounds : 0);
  box.innerHTML = `<div class="specpos">` + rates.map((v,i)=>{
    // Escalar contra el 100% no contra el máximo del grupo: si nadie pasa el
    // 20%, unas barras llenarían la caja y harían creer que el spec va bien.
    const h = Math.min(100, v*100);
    return `<div class="sp${v<=0.001?' zero':''}">
      <div class="spv">${(v*100).toFixed(0)}%</div>
      <div class="spbwrap"><div class="spb" style="height:${h.toFixed(1)}%"></div></div>
      <div class="spi">pos ${i}</div></div>`;
  }).join('') + `</div>`;
  const perStep = rounds ? accepted/rounds : 0;
  note.textContent = `${key} · ${use.length} req · ${perStep.toFixed(2)} tok/ronda`
    + (drafted ? ` · ${(accepted/drafted*100).toFixed(0)}% de los drafts` : '');
}

// Un jsonl congelado se lee como vivo: las cifras "último gen" son fijas y
// parecen fresco. Si el motor no está escribiendo, el panel lo dice.
function reqLogNote(d){
  const newest = d.requests?.[0]?.ts_ms;
  const base = `${d.requests.length} de ${d.lines} líneas del jsonl`;
  if (!newest) return base;
  const mins = Math.floor((Date.now() - newest) / 60000);
  if (mins < 10) return base;
  const when = mins < 60 ? `${mins} min`
    : mins < 1440 ? `${Math.floor(mins/60)} h` : `${Math.floor(mins/1440)} d`;
  return `${base} · <span style="color:var(--amber)">sin escribir desde hace ${when}</span>`;
}

function renderRequests(){
  const d = state.reqs;
  if (!d) return;
  const cols = REQ_COLS.filter(c=>reqMore||!c.more);
  $('#thead-req').innerHTML = cols.map(c=>
    `<th class="${c.l?'l':''}">${c.t}</th>`).join('');
  const tb = $('#tbody-req');
  if (!d.length){
    tb.innerHTML=''; $('#empty-req').hidden=false;
    return;
  }
  $('#empty-req').hidden = true;
  tb.innerHTML = d.map((r,i)=>{
    const open = state.reqOpen.has(r.ts_ms);
    const cells = cols.map(c=>{
      let v = r[c.k];
      if (c.k==='ts') v = new Date(r.ts_ms).toLocaleTimeString('es-AR',{hour12:false});
      if (c.k==='spec_backend') v = v ? v+(r.draft_window!=null?'/'+r.draft_window:'') : 'off';
      if (c.k==='cache_hit') v = r.cache_hit!=null ? r.cache_hit+' tok' : v;
      if (c.k==='model') v = String(v||'').slice(0,30);
      if (c.k==='spec_backend') return `<td class="l">${v==='off'
        ?'<span class="muted">off</span>':`<span class="pill mtp">${esc(v)}</span>`}</td>`;
      return `<td class="${c.l?'l':''}">${v==null||v===''?'<span class="muted">—</span>':(typeof v==='number'?v.toFixed(c.d??1):esc(v))}</td>`;
    }).join('');
    const detail = open ? detailReq(r) : '';
    return `<tr class="run" data-ts="${r.ts_ms}">${cells}</tr>${detail}`;
  }).join('');
  tb.querySelectorAll('tr.run').forEach(tr=>tr.onclick=()=>{
    const ts = +tr.dataset.ts;
    state.reqOpen.has(ts)?state.reqOpen.delete(ts):state.reqOpen.add(ts);
    renderRequests();      // solo redibuja: el toggle no necesita fetch
  });
}

function detailReq(r){
  const hist = (r.histogram||[]);
  const maxh = hist.length?Math.max(...hist):1;
  return `<tr class="detail"><td colspan="99">
    <div class="detailbox">
      <div class="dgroup"><h4>Request</h4><dl class="kv">
        <dt>hora</dt><dd>${esc(new Date(r.ts_ms).toLocaleTimeString('es-AR',{hour12:false}))}</dd>
        <dt>modelo</dt><dd>${esc(r.model||'—')}</dd>
        <dt>protocolo</dt><dd>${esc(r.protocol||'—')}</dd>
        <dt>finish</dt><dd>${esc(r.finish||'—')}</dd>
      </dl></div>
      <div class="dgroup"><h4>Timings</h4><dl class="kv">
        <dt>ttft</dt><dd>${num(r.ttft_ms,1)} ms</dd>
        <dt>prefill</dt><dd>${num(r.prefill_s,3)} s</dd>
        <dt>decode</dt><dd>${num(r.decode_s,2)} s</dd>
        <dt>total</dt><dd>${num(r.total_s,2)} s</dd>
      </dl></div>
      <div class="dgroup"><h4>Spec-decode</h4><dl class="kv">
        <dt>backend</dt><dd>${esc(r.spec_backend||'—')}</dd>
        <dt>draft window</dt><dd>${r.draft_window??'—'}</dd>
        <dt>aceptados</dt><dd>${num(r.accepted_tokens,0)} de ${num(r.drafted_tokens,0)}</dd>
        <dt>acc promedio</dt><dd>${num(r.acceptance_avg,2)}</dd>
        <dt>draft rate</dt><dd>${num(r.draft_rate,1)}%</dd>
      </dl>
      ${hist.length?`<div class="hist">${hist.map((v,pos)=>
        `<div style="height:${(v/maxh*100).toFixed(1)}%" title="pos ${pos}: ${v}"><span>${pos}</span></div>`).join('')}
        </div><div style="font:10px var(--mono);color:var(--dimmer);margin-top:20px">
        aceptados por posición</div>`:''}
      </div>
    </div></td></tr>`;
}


export function toggleReqCols(){
  reqMore = !reqMore;
  $('#q-reqmore').textContent = reqMore ? 'Menos columnas' : 'Más columnas';
  loadRequests();
}
