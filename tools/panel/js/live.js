import { $, state, esc, num, api, flash, refresh } from './core.js';
import { modelArgs, modelRows, renderQuickLoad, wireModelRows, SVG_GEAR } from './models.js';

const LIVE_TILES = [
  {k:'running', label:'corriendo', max:'max_concurrency'},
  {k:'admitted', label:'admitidos', max:'max_admitted_requests'},
  {k:'waiting', label:'en cola', max:null},
  {k:'device_main_kv_tokens', label:'KV device', max:'kv_capacity_tokens', unit:'tok'},
];

export function initCharts(){
  const def = {responsive:true, maintainAspectRatio:false,
    scales:{x:{display:true,ticks:{maxTicksLimit:8,color:'#5d6b81'},grid:{color:'#1c2534'}},
             y:{ticks:{color:'#5d6b81',maxTicksLimit:5},grid:{color:'#1c2534'}}},
    // Sin legend: con una sola serie, el cuadradito de la leyenda solo hacía
    // ruido. El nombre de la serie vive en el tooltip ahora.
    plugins:{legend:{display:false}},
    animation:false};
  const mk = (id, color, name) => new Chart(document.getElementById(id), {
    type:'line', data:{labels:[],datasets:[{label:name,data:[],borderColor:color,
      backgroundColor:color+'22',fill:true,borderWidth:1.7,pointRadius:0,tension:.25}]},
    options:def});
  state.charts = {
    tp: mk('c-tp','#38bdf8','decode tok/s'), decode: mk('c-decode','#60a5fa','prefill tok/s'),
  };
  for (const [k,c] of Object.entries(state.charts)){
    const wins = document.getElementById('w-'+k);
    ['1m','5m','15m','retained'].forEach(w=>{
      const b=document.createElement('button');
      b.textContent=w;
      b.className = state.win===w ? 'on' : '';
      b.onclick=()=>{
        state.win=w;
        // Cambiar ventana = recortar, no solo limitar los pushes futuros.
        // Se limpian las series y se re-backfillea desde la historia.
        for (const c2 of Object.values(state.charts)){
          c2.data.labels=[]; c2.data.datasets[0].data=[]; c2.update('none');
        }
        state.histLoaded=false;
        wins.querySelectorAll('button').forEach(x=>x.classList.toggle('on', x===b));
        loadLive();
      };
      wins.appendChild(b);
    });
  }
  state.series = {counters:{}, up:{}};
}

function chartPush(ch, label, v, t){
  const WT = {'1m':60, '5m':300, '15m':900};
  ch.data.labels.push(label);
  ch.data.datasets[0].data.push(v);
  const max = state.win==='retained' ? 720 : WT[state.win];
  if(ch.data.labels.length > max){
    ch.data.labels.shift(); ch.data.datasets[0].data.shift();
  }
  ch.update('none');
}

// El router en reposo responde 404 model_not_found a /v1/load, /stats y /slots:
// no es un error, es "ningún modelo cargado". Sin modelo no hay ocupación ni KV.

function sparkSvg(vals){
  const v = (vals||[]).map(x=>x==null?0:x);
  if (v.length < 2) return '<svg class="spark" viewBox="0 0 100 26" preserveAspectRatio="none"></svg>';
  const top = Math.max(...v, 1e-9);
  const pts = v.map((x,i)=>[(i/(v.length-1))*100, 24-(x/top)*20]);
  const line = pts.map((p,i)=>`${i?'L':'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join('');
  return `<svg class="spark" viewBox="0 0 100 26" preserveAspectRatio="none" aria-hidden="true">
    <path class="area" d="${line}L100,26L0,26Z"/><path class="line" d="${line}"/></svg>`;
}

export function liveModels(d){
  return ((d.models?.data)||[]).filter(m=>
    ['loaded','sleeping','suspended','loading'].includes(m.status?.value));
}

// Tasas por modelo, de los contadores acumulados de SU /v1/load.
// El delta se guarda por modelo: con dos Engines en memoria, usar el delta
// global mezclaría las velocidades de uno con los del otro.
function engineRates(engines){
  const now = Date.now()/1000;
  const out = {};
  state.engPrev = state.engPrev || {};
  for (const [mid, e] of Object.entries(engines||{})){
    const c = (e.load && e.load.counters) || {};
    const prev = state.engPrev[mid];
    let tg = null, pg = null;
    if (prev){
      const dt = now - prev.t;
      if (dt > 0){
        tg = Math.max(0, (c.committed_decode_tokens||0) - prev.tok) / dt;
        pg = Math.max(0, (c.computed_prefill_tokens||0) - prev.pt) / dt;
      }
    }
    state.engPrev[mid] = {t: now, tok: c.committed_decode_tokens||0, pt: c.computed_prefill_tokens||0};
    const busy = (e.load && e.load.requests && e.load.requests.running) || 0;
    // En reposo los contadores no avanzan y la tasa es 0 por definición: se
    // muestra el último valor real del request log de ese modelo.
    const last = (state.reqs||[]).find(r=>r.model===mid && r.decode_s>0 && r.completion_tokens);
    const prevReq = (state.reqs||[]).find(r=>r.model===mid && r.prefill_s>0 && r.prefill_tokens);
    out[mid] = {tg, pg, busy,
                lastTps: last ? last.completion_tokens/last.decode_s : null,
                lastPrefill: prevReq ? prevReq.prefill_tokens/prevReq.prefill_s : null};
  }
  return out;
}

function fmtRate(v, last, unit){
  if (v != null && v > 0.5) return v.toFixed(0)+' '+unit;
  if (last != null) return last.toFixed(0)+' '+unit;
  return '—';
}

// Una línea por Engine en memoria. Mismos campos en todas, que es lo que las
// hace comparables de un vistazo; lo que cambia de un estado a otro se rotula.
function renderRailModels(d, rates){
  const host = $('#rail-models');
  const engines = d.engines || {};
  const ids = Object.keys(engines);
  if (!ids.length){ host.innerHTML = ''; return; }
  host.innerHTML = ids.map(mid=>{
    const e = engines[mid];
    const l = e.load || {};
    const res = e.residency || {};
    const on = e.state === 'loaded';
    const occ = l.occupancy||{}, cap = l.capacity||{};
    const r = rates[mid] || {};
    const hostMiB = (res.host_snapshot_bytes||0)/2**20;
    // "tamaño" para uno dormido, "VRAM" para uno cargado: el mismo número no
    // significa lo mismo, y llamarle VRAM a los dos sería mentir.
    const bytes = res.releasable_device_bytes || 0;
    const memCol = on ? (bytes/2**30 >= 18 ? 'var(--amber)' : 'var(--teal)') : 'var(--dim)';
    const ctxPct = cap.max_context ? Math.min(100, (occ.device_main_kv_tokens||0)/cap.max_context*100) : 0;
    const stateLbl = {loaded:'activo', sleeping:'dormido', suspended:'suspendido', loading:'cargando'}[e.state] || e.state;
    return `<div class="rail-model" data-model="${esc(mid)}">
      <span class="dot ${on?'ok':e.state==='suspended'?'blue':e.state==='loading'?'warn':'off'}"></span>
      <span class="nm">${esc(mid)}</span>
      <span class="dim2">${stateLbl}</span>
      <span class="sep">|</span>
      <span style="color:${memCol}">${bytes?(bytes/2**30).toFixed(1)+' GiB':'—'}</span>
      <span class="dim2">${on?'VRAM':'tamaño'}</span>
      ${!on && hostMiB>1 ? `<span class="dim2">· ${hostMiB.toFixed(0)} MiB en host</span>` : ''}
      <span class="sep">|</span>
      <span style="color:${ctxPct>85?'var(--amber)':'var(--text-muted)'}">ctx ${(occ.device_main_kv_tokens||0).toLocaleString()}<span class="dim2">/${(cap.max_context||0).toLocaleString()}</span></span>
      <span class="sep">|</span>
      <span title="tok/s de generación${r.tg>0.5?' (en vivo)':''}">TG ${fmtRate(r.tg, r.lastTps, 'tok/s')}</span>
      <span class="sep">|</span>
      <span title="tok/s de prefill${r.pg>0.5?' (en vivo)':''}">PG ${fmtRate(r.pg, r.lastPrefill, 'tok/s')}</span>
      <span class="sep">|</span>
      <span style="color:${r.busy?'var(--teal)':'var(--dimmer)'}">${r.busy||0} en vuelo</span>
      ${on ? `<button data-act="sleep" data-model="${esc(mid)}">Dormir</button>`
           : `<button data-act="load" data-model="${esc(mid)}">Despertar</button>`}
    </div>`;
  }).join('');
  wireModelRows($('#rail-models'), d);
}

function renderRail(d, gpu, loadOk, rates){
  const live = liveModels(d);
  const load = loadOk ? (d.load||{}) : {};
  const cap = load.capacity||{}, occ = load.occupancy||{}, req = load.requests||{};
  // El modelo corre en un device puntual (--device N del preset): el gauge debe
  // mostrar ESA GPU, no la primera del host.
  const loadedModel = live.find(m=>m.status?.value==='loaded') || live[0];
  const dev = loadedModel ? Number(modelArgs(loadedModel).device) : NaN;
  const g = gpu && gpu.online && gpu.gpus.length
    ? (gpu.gpus.find(x=>x.index===dev) || gpu.gpus[0]) : null;
  const vramPct = g ? g.mem_used_mib/g.mem_total_mib*100 : null;
  const vramCls = vramPct==null ? '' : vramPct>=95 ? 'bad' : vramPct>=85 ? 'warn' : '';
  const vramCol = vramPct==null ? 'var(--dim)' : vramPct>=95?'var(--red)':vramPct>=85?'var(--amber)':'var(--teal)';
  const kvPct = (occ.device_main_kv_tokens!=null && cap.kv_capacity_tokens)
    ? occ.device_main_kv_tokens/cap.kv_capacity_tokens*100 : null;
  const kvCol = kvPct==null ? 'var(--dim)' : kvPct>=95?'var(--red)':kvPct>=85?'var(--amber)':'var(--teal)';
  const tempCol = g ? (g.temp_c>=85?'var(--red)':g.temp_c>=75?'var(--amber)':'var(--green)') : 'var(--dim)';
  const uCol = g ? (g.util>=90?'var(--red)':g.util>=60?'var(--amber)':'var(--blue)') : 'var(--dim)';
  $('#rail-global').innerHTML = `
    <span class="item"><span class="dot ${live.length?'ok':'off'}"></span>
      <span class="val">${live.length?live.length+' en memoria':'sin modelo'}</span></span>
    <span class="item"><span class="lbl">VRAM</span>
      <span class="gauge ${vramCls}"><i style="width:${(vramPct??0).toFixed(0)}%"></i></span>
      <span class="val" style="color:${vramCol}">${g?(g.mem_used_mib/1024).toFixed(1):'—'}<span class="dim">/${g?(g.mem_total_mib/1024).toFixed(0):'—'}G</span></span></span>
    <span class="item"><span class="lbl">KV</span>
      <span class="gauge ${kvPct!=null&&kvPct>90?'warn':''}"><i style="width:${(kvPct??0).toFixed(0)}%"></i></span>
      <span class="val" style="color:${kvCol}">${occ.device_main_kv_tokens!=null?occ.device_main_kv_tokens.toLocaleString():'—'}<span class="dim">/${cap.kv_capacity_tokens!=null?cap.kv_capacity_tokens.toLocaleString():'—'}</span></span></span>
    ${g?`<span class="item"><span class="lbl">GPU ${g.index}</span>
      <span class="val" style="color:${uCol}">${g.util.toFixed(0)}%</span>
      <span class="val" style="color:${tempCol}">${g.temp_c.toFixed(0)}°C</span>
      <span class="val" style="color:var(--purple)">${g.power_w.toFixed(0)}W</span></span>`:''}
    <span class="item"><span class="lbl">en vuelo</span>
      <span class="val" style="color:${(req.running||0)>0?'var(--teal)':'var(--dim)'}">${req.running??0}</span></span>
    <span class="push"></span>
    ${live.length?`<button id="rail-unload" style="padding:3px 10px;font-size:11px">Descargar todo</button>`:''}`;
  renderRailModels(d, rates);
  const ub = $('#rail-unload');
  if (ub) ub.onclick = async () => {
    ub.disabled = true; ub.textContent = 'descargando…';
    for (const m of live){
      try{ await api('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({action:'unload',model:m.id})}); }catch(e){}
    }
    loadLive();
  };
}

// El veredicto: una frase que conteste "¿está bien?". El nivel tiñe sólo la
// línea, y el detalle va al lado en vez de competir con ella.
const VERDICT_DOT = {good:'ok pulse', warn:'warn', bad:'bad', idle:'off'};
const ALERT_RANK = {critical:3, serious:2, warning:1};
const ALERT_ICON = {
  critical:'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
  serious:'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  warning:'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>',
};

function renderVerdict(lvl, txt, detail){
  const head = $('#now').querySelector('.sechead');
  head.className = 'sechead verdict ' + lvl;
  $('#vd-dot').className = 'dot ' + VERDICT_DOT[lvl];
  $('#verdict').textContent = txt;
  $('#vd-detail').textContent = detail || '';
}

// La alerta manda sobre el estado: si algo está mal, el veredicto lo dice
// primero, no "en reposo".
function renderAlerts(d){
  const box = $('#alerts');
  const list = (d.alerts||[]).slice().sort(
    (a,b) => (ALERT_RANK[b.level]||0) - (ALERT_RANK[a.level]||0));
  if (!list.length){ box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  box.innerHTML = `<div class="alerts-row">` + list.map(a=>`
    <div class="alert ${esc(a.level)}" role="alert">
      <span class="ico">${ALERT_ICON[a.level] || ALERT_ICON.warning}</span>
      <span class="txt"><span class="t">${esc(a.title)}</span>
        <span class="d">${esc(a.detail||'')}</span></span>
      <span class="lvl">${esc(a.level)}</span>
    </div>`).join('') + `</div>`;
}

function renderNow(d, load, cap, occ, req, loadOk, rates){
  // El estado se lee de `engines`, que ya resolvió la contradicción entre
  // /models (que sigue diciendo "loaded" a un modelo suspendido) y residency.
  // Antes se leía de /models y por eso el veredicto decía "dormido" con un
  // modelo activo en memoria: promediaba dos Engines distintos.
  const eng = d.engines || {};
  const st = id => (eng[id]||{}).state;
  const ids = Object.keys(eng);
  const running = occ.running ?? req.running ?? 0;
  const detail = ids.map(id => {
    const s = st(id);
    return s === 'loaded' ? id : `${id} ${({sleeping:'dormido', suspended:'suspendido', loading:'cargando'})[s]||s}`;
  }).join(' · ');
  const alerts = (state.alerts && state.alerts.alerts) || [];
  const worst = alerts.reduce((w,a)=> (ALERT_RANK[a.level]||0) > (ALERT_RANK[w]||0) ? a.level : w, null);
  if (worst){
    const n = alerts.length;
    renderVerdict(worst==='critical'?'bad':'warn',
      n===1 ? (alerts[0].title) : `${n} alertas`,
      n===1 ? '' : alerts.map(a=>a.title).join(' · '));
  }
  else if (ids.some(id=>st(id)==='loading')) renderVerdict('warn', 'cargando modelo', detail);
  else if (running>0)                          renderVerdict('good', 'decodificando', `${running} en vuelo · ${detail}`);
  else if (ids.some(id=>st(id)==='loaded'))    renderVerdict('good', 'en reposo', detail);
  else if (ids.some(id=>st(id)==='suspended')) renderVerdict('idle', 'suspendido · vive en el host', detail);
  else if (ids.some(id=>st(id)==='sleeping'))  renderVerdict('idle', 'dormido · despierta al próximo request', detail);
  else                                         renderVerdict('idle', 'sin modelo cargado', 'VRAM libre · carga uno para ver ocupación y KV');

  $('#now-metrics').innerHTML = renderEngineCards(d, rates);
  wireModelRows($('#now-metrics'), d);
}

// Un recuadro por Engine en memoria. Con dos modelos cargados una sola fila de
// números era ambigua: el veredicto mezclaba "dormido" con "activo" y cada cifra
// era de un solo modelo. Acá cada caja es de un modelo y rotula su estado.
function renderEngineCards(d, rates){
  const ids = Object.keys(d.engines || {});
  if (!ids.length){
    return `<div class="now-metrics">
      <div class="now-m"><span class="k">carriles</span><b>—</b></div>
      <div class="now-m"><span class="k">en vuelo</span><b>0</b></div>
      <div class="now-m"><span class="k">ctx usado</span><b>—</b></div></div>`;
  }
  return `<div class="engine-cards">` + ids.map(mid=>{
    const e = d.engines[mid], l = e.load || {}, res = e.residency || {};
    const on = e.state === 'loaded';
    const occ = l.occupancy||{}, cap = l.capacity||{}, req = l.requests||{};
    const r = rates[mid] || {};
    const bl = ((state.alerts && state.alerts.models) || {})[mid];
    const bytes = res.releasable_device_bytes || 0;
    const hostMiB = (res.host_snapshot_bytes||0)/2**20;
    const ctxCap = cap.max_context || cap.kv_capacity_tokens || 0;
    const ctxPct = ctxCap ? Math.min(100, (occ.device_main_kv_tokens||0)/ctxCap*100) : 0;
    return `<div class="engine-card${on?'':' off'}">
      <div class="ec-head">
        <span class="dot ${on?'ok pulse':e.state==='suspended'?'blue':e.state==='loading'?'warn':'off'}"></span>
        <b>${esc(mid)}</b>
        <span class="ec-state">${esc({loaded:'activo',sleeping:'dormido',suspended:'suspendido en host',loading:'cargando'}[e.state]||e.state)}</span>
        <span class="spacer"></span>
        ${on ? `<button data-act="sleep" data-model="${esc(mid)}">Dormir</button>`
             : `<button data-act="load" data-model="${esc(mid)}">Despertar</button>`}
        <button class="iconbtn" data-cfg="${esc(mid)}" title="Config del modelo">${SVG_GEAR}</button>
      </div>
      <div class="ec-metrics">
        <div class="now-m"><span class="k">${on?'VRAM':'tamaño'}</span>
          <b style="color:${on?'#fff':'var(--dim)'}">${bytes?(bytes/2**30).toFixed(1):'—'} <small>GiB</small></b>
          ${!on&&hostMiB>1?`<small class="sub">${hostMiB.toFixed(0)} MiB en host</small>`:''}</div>
        <div class="now-m"><span class="k">ctx usado</span>
          <b style="color:${ctxPct>85?'var(--amber)':'#fff'}">${(occ.device_main_kv_tokens||0).toLocaleString()} <small>/ ${ctxCap.toLocaleString()}</small></b>
          ${ctxCap?`<div class="progress"><i style="width:${ctxPct.toFixed(0)}%"></i></div>`:''}</div>
        <div class="now-m"><span class="k">carriles</span><b>${req.running??0}/${cap.max_concurrency??'—'}</b>
          ${cap.max_concurrency?`<div class="progress"><i style="width:${Math.min(100,(req.running||0)/cap.max_concurrency*100).toFixed(0)}%"></i></div>`:''}</div>
        <div class="now-m"><span class="k">en cola</span><b style="color:${(req.waiting||0)>0?'var(--amber)':'#fff'}">${req.waiting??0}</b></div>
        <div class="now-m"><span class="k">${r.tg>0.5?'gen ahora':'último gen'}</span>
          <b style="color:${r.tg>0.5?'var(--teal)':'#fff'}">${fmtRate(r.tg, r.lastTps, '').trim()} <small>tok/s</small></b></div>
        <div class="now-m"><span class="k">${r.pg>0.5?'prefill ahora':'último prefill'}</span>
          <b style="color:${r.pg>0.5?'var(--teal)':'#fff'}">${fmtRate(r.pg, r.lastPrefill, '').trim()} <small>tok/s</small></b></div>
        <div class="now-m" title="mediana aprendida de la velocidad del servidor para este modelo"><span class="k">referencia</span>
          <b style="color:var(--muted)">${bl? (bl.samples>=12?bl.median_tok_s.toFixed(0):`aprendiendo ${bl.samples}/12`) : '—'} <small>${bl&&bl.samples>=12?'tok/s':''}</small></b></div>
      </div>
    </div>`;
  }).join('') + `</div>`;
}

function slotsHtml(d, occ){
  const slots = d.slots || [];
  if (!slots.length) return '';
  return `<div class="slots-strip">
      <div class="slots-strip-head">
        <b><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg> Paged KV Cache — Slots de Ejecución (${slots.length} carriles)</b>
        <span style="color:var(--dim)">${occ.device_main_kv_tokens != null ? occ.device_main_kv_tokens.toLocaleString() + ' tok activos' : ''}</span>
      </div>
      <div class="slots-grid">
        ${slots.map(sl => {
          const isBusy = sl.is_processing, isRetained = sl.retained;
          const stateCls = isBusy ? 'busy' : isRetained ? 'retained' : '';
          const statusTxt = isBusy ? 'decodificando' : isRetained ? 'en caché' : 'libre';
          const statusCls = isBusy ? 'active' : isRetained ? 'cached' : 'free';
          const tok = sl.n_prompt_tokens || sl.n_prompt_tokens_cache || 0;
          return `<div class="slot-cell ${stateCls}">
            <div class="slot-num">Slot ${sl.id}</div>
            <div class="slot-status ${statusCls}">● ${statusTxt}</div>
            <div class="slot-meta" title="${tok} tok prompt">${tok ? tok + ' tok' : '—'}</div>
          </div>`;
        }).join('')}
      </div>
    </div>`;
}

export async function loadAlerts(){
  let d;
  try{ d = await api('/api/alerts'); }
  catch(e){ return; }
  if (!d || !d.ok) return;
  // Las alertas y la referencia cambian mucho más lento que el estado: no
  // hay que re-renderizar si el contenido es el mismo.
  const sig = JSON.stringify([d.alerts, d.models]);
  if (sig === state.alertsSig) return;
  state.alertsSig = sig;
  state.alerts = d;
  renderAlerts(d);
}


refresh.live = loadLive;

export async function loadLive(){
  const errhost = $('#livebody');
  let d;
  try{ d = await api('/api/live'); }
  catch(e){ errhost.innerHTML = `<div class="err" style="margin:15px">${esc(e.message)}</div>`; return; }
  if (!d.online){
    errhost.innerHTML = `<div class="err" style="margin:15px">El servidor <code>${esc(d.url)}</code> no responde.</div>`;
    renderVerdict('bad', 'el servidor no responde', d.url || '');
    $('#now-metrics').innerHTML = '';
    return;
  }
  errhost.innerHTML = '';

  // Backfill: las gráficas arrancan con la historia del jsonl, no vacías.
  if (!state.histLoaded){
    state.histLoaded = true;
    try{
      const h = await api('/api/history');
      if (h.ok){
        for (const p of h.points.slice(-300)){
          const lbl = new Date(p.ts).toLocaleTimeString('es-AR',{hour12:false});
          chartPush(state.charts.tp, lbl, p.decode ?? 0, p.ts/1000);
          chartPush(state.charts.decode, lbl, p.prefill ?? 0, p.ts/1000);
        }
      }
    }catch(e){}
  }
  const loadOk = !!d.load && !d.load.error;
  const load = loadOk ? d.load : {};
  const cap = load.capacity||{}, occ = load.occupancy||{}, req = load.requests||{};
  const t = Date.now()/1000;
  const lbl = new Date().toLocaleTimeString('es-AR',{hour12:false});

  // series desde /v1/load counters. `lastDT !== undefined` es el "hay muestra
  // previa": el objeto se reemplaza entero, así que no existe una clave `last`.
  if (loadOk){
    const c = load.counters||{};
    if(state.series.counters.lastDT !== undefined){
      const dt = t - state.series.counters.lastT || 1;
      const dtok = (c.committed_decode_tokens||0) - state.series.counters.lastDT;
      chartPush(state.charts.tp, lbl, dtok<=0?0:dtok/dt, t);
      const ptok = (c.computed_prefill_tokens||0) - state.series.counters.lastPT;
      chartPush(state.charts.decode, lbl, ptok<=0?0:ptok/dt, t);
    }
    state.series.counters = {lastDT: c.committed_decode_tokens||0,
                             lastPT: c.computed_prefill_tokens||0, lastT: t};
  }
  let gpu=null;
  try{ gpu = await api('/api/gpu'); }catch(e){}

  const rates = engineRates(d.engines);
  renderRail(d, gpu, loadOk, rates);
  loadAlerts();
  renderNow(d, load, cap, occ, req, loadOk, rates);

  // Tiles: ratio acotado -> barra; tasa -> sparkline. El tono dice si hay que
  // mirarlo (normal, ámbar >75%, rojo >90%).
  const tiles = LIVE_TILES.map(ti => {
    const v = occ[ti.k] ?? req[ti.k];
    const max = cap[ti.max] ?? occ[ti.max];
    const pct = (v!=null && max) ? Math.min(100, v/max*100) : null;
    const tone = pct==null ? '' : pct>90 ? 'bad' : pct>75 ? 'warn' : '';
    return `<div class="tile ${tone}">
      <div class="k">${ti.label}</div>
      <div class="v">${v??'—'}${ti.unit?` <small>${ti.unit}</small>`:''}</div>
      <div class="s">${max!=null ? `de ${max}` : ''}</div>
      ${pct!=null?`<div class="bar"><i style="width:${pct.toFixed(0)}%"></i></div>`:''}
    </div>`;
  }).join('');

  const gpuTiles = gpu && gpu.online && gpu.gpus.length ? gpu.gpus.map(g=>{
    const vramPct = g.mem_used_mib/g.mem_total_mib*100;
    const vramCol = vramPct>=95?'var(--red)':vramPct>=85?'var(--amber)':'var(--teal)';
    const tCol = g.temp_c>=85?'var(--red)':g.temp_c>=75?'var(--amber)':'var(--green)';
    const uCol = g.util>=90?'var(--red)':g.util>=60?'var(--amber)':'var(--blue)';
    return `<div class="tile">
      <div class="k">GPU ${g.index}</div>
      <div class="v" style="color:${uCol}">${g.util.toFixed(0)}<small>% carga</small></div>
      <div class="s"><span class="c-temp" style="font-weight:600">${g.temp_c.toFixed(0)}°C</span> ·
        <span class="c-pow" style="font-weight:600">${g.power_w.toFixed(0)}W</span></div>
      <div class="bar-row"><span class="bl">VRAM</span>
        <span class="bv" style="color:${vramCol}">${(g.mem_used_mib/1024).toFixed(1)}/${(g.mem_total_mib/1024).toFixed(0)} GiB</span></div>
      <div class="bar"><i style="width:${vramPct.toFixed(0)}%;background:${vramCol}"></i></div></div>`;
  }).join('') : '';

  const tpData = (state.charts.tp?.data.datasets[0].data)||[];
  const sparkTile = `<div class="tile info"><div class="k">Throughput</div>
    <div class="v">${tpData.length?(tpData[tpData.length-1]||0).toFixed(0):'—'} <small>tok/s</small></div>
    <div class="s">${state.win==='retained'?'retenido':state.win}</div>
    ${sparkSvg(tpData.slice(-40))}</div>`;

  // Sin modelo no hay ocupación ni KV que mostrar: esos tiles serían cuatro
  // guiones. Quedan sólo los de GPU, que siempre tienen dato.
  $('#tiles').innerHTML = sparkTile + (loadOk ? tiles : '') + gpuTiles;
  $('#livenote').textContent = loadOk ? '' : 'sin modelo cargado — cargá uno para ver ocupación y KV';
  $('#slots-host').innerHTML = slotsHtml(d, occ);
  $('#modelrows').innerHTML = modelRows(d, req, false);
  $('#modelnote').textContent = `${(d.models?.data||[]).length} en el catálogo`;
  renderQuickLoad(d);
  $('#liveupd').textContent = 'actualizado ' + new Date().toLocaleTimeString('es-AR',{hour12:false});
  wireModelRows($('#modelrows'), d);
  wireModelRows($('#quickload'), d);
}
