import { $, state, esc, num, api, flash, refresh } from './core.js';
import { modelArgs, modelRows, renderQuickLoad, renderCtlRows, wireModelRows, SVG_GEAR } from './models.js';

const LIVE_TILES = [
  {k:'running', label:'running', max:'max_concurrency'},
  {k:'admitted', label:'admitted', max:'max_admitted_requests'},
  {k:'waiting', label:'queued', max:null},
  {k:'device_main_kv_tokens', label:'KV device', max:'kv_capacity_tokens', unit:'tok'},
];

export function initCharts(){
  const def = {responsive:true, maintainAspectRatio:false,
    scales:{x:{display:true,ticks:{maxTicksLimit:8,color:'#5d6b81'},grid:{color:'#1c2534'}},
             y:{ticks:{color:'#5d6b81',maxTicksLimit:5},grid:{color:'#1c2534'}}},
    // No legend: with a single series the legend box was only noise. The series
    // name lives in the tooltip now.
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
        // Changing the window trims; it does not only cap future pushes. The
        // series are cleared and backfilled again from history.
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

// An idle router answers 404 model_not_found to /v1/load, /stats and /slots:
// that is not an error, it means "no model loaded". With no model there is no
// occupancy and no KV to report.

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

// Per-model rates, from the cumulative counters of ITS OWN /v1/load. The
// previous reading is kept per model: with two Engines in memory a global
// delta would mix one model's speed with the other's.
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
    // At rest the counters do not advance and the rate is zero by definition, so
    // the last real value from that model's request log is shown instead.
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

// One line per Engine in memory. The same fields on every one, which is what
// makes them comparable at a glance; what changes with state is labelled.
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
    // "size" for a sleeping model, "VRAM" for a loaded one: the same number
    // does not mean the same, and calling both VRAM would be lying.
    const bytes = res.releasable_device_bytes || 0;
    const memCol = on ? (bytes/2**30 >= 18 ? 'var(--amber)' : 'var(--teal)') : 'var(--dim)';
    const ctxPct = cap.max_context ? Math.min(100, (occ.device_main_kv_tokens||0)/cap.max_context*100) : 0;
    const stateLbl = {loaded:'loaded', sleeping:'sleeping', suspended:'suspended', loading:'loading'}[e.state] || e.state;
    return `<div class="rail-model" data-model="${esc(mid)}">
      <span class="dot ${on?'ok':e.state==='suspended'?'blue':e.state==='loading'?'warn':'off'}"></span>
      <span class="nm">${esc(mid)}</span>
      <span class="dim2">${stateLbl}</span>
      <span class="sep">|</span>
      <span style="color:${memCol}">${bytes?(bytes/2**30).toFixed(1)+' GiB':'—'}</span>
      <span class="dim2">${on?'VRAM':'size'}</span>
      ${!on && hostMiB>1 ? `<span class="dim2">· ${hostMiB.toFixed(0)} MiB in host</span>` : ''}
      <span class="sep">|</span>
      <span style="color:${ctxPct>85?'var(--amber)':'var(--text-muted)'}">ctx ${(occ.device_main_kv_tokens||0).toLocaleString()}<span class="dim2">/${(cap.max_context||0).toLocaleString()}</span></span>
      <span class="sep">|</span>
      <span title="generation tok/s${r.tg>0.5?' (live)':''}">TG ${fmtRate(r.tg, r.lastTps, 'tok/s')}</span>
      <span class="sep">|</span>
      <span title="tok/s de prefill${r.pg>0.5?' (en vivo)':''}">PG ${fmtRate(r.pg, r.lastPrefill, 'tok/s')}</span>
      <span class="sep">|</span>
      <span style="color:${r.busy?'var(--teal)':'var(--dimmer)'}">${r.busy||0} in flight</span>
      ${on ? `<button data-act="sleep" data-model="${esc(mid)}">Sleep</button>`
           : `<button data-act="load" data-model="${esc(mid)}">Wake</button>`}
    </div>`;
  }).join('');
  wireModelRows($('#rail-models'), d);
}

// The header strip: the machine, not the engine. One chip per GPU with the card
// name that used to sit frozen in the version badge, plus RAM and CPU load. The
// rail below answers the other question (what the engine is doing), so VRAM and
// GPU live here and only here.
function renderHostChips(gpu){
  const host = gpu?.host || {};
  const gib = b => (b/1073741824).toFixed(1);
  const chip = (k, pct, col, val, dim, extra='') =>
    `<span class="hchip${pct>=90?' hot':''}"><span class="k">${esc(k)}</span>${extra}
      <span class="gauge"><i style="width:${Math.min(100,pct).toFixed(0)}%;background:${col}"></i></span>
      <span class="v">${val}</span>${dim?`<span class="d">${dim}</span>`:''}</span>`;
  const bar = (p, warn, bad) => p>=bad ? 'var(--red)' : p>=warn ? 'var(--amber)' : 'var(--teal)';
  let h = '';
  for (const g of (gpu?.gpus||[])){
    const vm = g.mem_used_mib/g.mem_total_mib*100;
    // "NVIDIA GeForce RTX 3090" -> "RTX 3090": the vendor prefix is noise here.
    const nm = (g.name||'').replace(/^NVIDIA\s*/i,'').replace(/^GeForce\s*/i,'').replace(/\s+/g,'');
    const uc = g.util>=90?'var(--red)':g.util>=60?'var(--amber)':'var(--blue)';
    const tc = g.temp_c>=85?'var(--red)':g.temp_c>=75?'var(--amber)':'var(--dim)';
    const vr = (g.mem_used_mib/1024).toFixed(1)+'/'+(g.mem_total_mib/1024).toFixed(0);
    h += `<span class="hchip${vm>=95?' hot':''}" title="${esc(g.name)} · ${vr} · ${g.util.toFixed(0)}% load · ${g.temp_c.toFixed(0)}°C · ${g.power_w.toFixed(0)}W">
      <span class="k">GPU ${g.index}</span><span class="nm">${esc(nm)}</span>
      <span class="gauge"><i style="width:${Math.min(100,vm).toFixed(0)}%;background:${bar(vm,85,95)}"></i></span>
      <span class="v">${vr}</span>
      <span class="d"><span style="color:${uc}">${g.util.toFixed(0)}%</span> <span style="color:${tc}">${g.temp_c.toFixed(0)}°</span></span></span>`;
  }
  const r = host.ram;
  if (r?.total){
    const p = r.used/r.total*100;
    h += chip('RAM', p, bar(p,80,90), `${gib(r.used)}/${gib(r.total)}`,
              `${gib(r.available)} free`);
  }
  const c = host.cpu;
  if (c?.cores){
    const p = c.load5/c.cores*100;
    h += `<span class="hchip" title="load average 1m ${c.load1.toFixed(2)} · 5m ${c.load5.toFixed(2)} · 15m ${c.load15.toFixed(2)} on ${c.cores} threads">
      <span class="k">CPU</span>
      <span class="gauge"><i style="width:${Math.min(100,p).toFixed(0)}%;background:${p>=90?'var(--red)':p>=75?'var(--amber)':'var(--blue)'}"></i></span>
      <span class="v">${c.load1.toFixed(2)}</span>
      <span class="d">/${c.cores}</span></span>`;
  }
  $('#hostchips').innerHTML = h;
}

function renderRail(d, loadOk, rates){
  const live = liveModels(d);
  const load = loadOk ? (d.load||{}) : {};
  const cap = load.capacity||{}, occ = load.occupancy||{}, req = load.requests||{};
  const kvPct = (occ.device_main_kv_tokens!=null && cap.kv_capacity_tokens)
    ? occ.device_main_kv_tokens/cap.kv_capacity_tokens*100 : null;
  const kvCol = kvPct==null ? 'var(--dim)' : kvPct>=95?'var(--red)':kvPct>=85?'var(--amber)':'var(--teal)';
  $('#rail-global').innerHTML = `
    <span class="item"><span class="dot ${live.length?'ok':'off'}"></span>
      <span class="val">${live.length?live.length+' in memory':'no model'}</span></span>
    <span class="item"><span class="lbl">KV</span>
      <span class="gauge ${kvPct!=null&&kvPct>90?'warn':''}"><i style="width:${(kvPct??0).toFixed(0)}%"></i></span>
      <span class="val" style="color:${kvCol}">${occ.device_main_kv_tokens!=null?occ.device_main_kv_tokens.toLocaleString():'—'}<span class="dim">/${cap.kv_capacity_tokens!=null?cap.kv_capacity_tokens.toLocaleString():'—'}</span></span></span>
    <span class="item"><span class="lbl">in flight</span>
      <span class="val" style="color:${(req.running||0)>0?'var(--teal)':'var(--dim)'}">${req.running??0}</span></span>
    <span class="push"></span>
    ${live.length?`<button id="rail-unload" style="padding:3px 10px;font-size:11px">Unload all</button>`:''}`;
  renderRailModels(d, rates);
  const ub = $('#rail-unload');
  if (ub) ub.onclick = async () => {
    ub.disabled = true; ub.textContent = 'downloading…';
    for (const m of live){
      try{ await api('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({action:'unload',model:m.id})}); }catch(e){}
    }
    loadLive();
  };
}

// The verdict: one sentence answering "is it fine?". The level tints only that
// line; the detail sits beside it rather than competing with it.
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

// An alert outranks the state: when something is wrong the verdict says so
// first, instead of "idle".
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
  // State is read from `engines`, which already resolves the contradiction
  // between /models (which still reports a suspended model as loaded) and
  // residency. Reading /models is what made the verdict say "asleep" while a
  // model was running: it was averaging two different Engines.
  const eng = d.engines || {};
  const st = id => (eng[id]||{}).state;
  const ids = Object.keys(eng);
  const running = occ.running ?? req.running ?? 0;
  const detail = ids.map(id => {
    const s = st(id);
    return s === 'loaded' ? id : `${id} ${({sleeping:'sleeping', suspended:'suspended', loading:'loading'})[s]||s}`;
  }).join(' · ');
  const alerts = (state.alerts && state.alerts.alerts) || [];
  const worst = alerts.reduce((w,a)=> (ALERT_RANK[a.level]||0) > (ALERT_RANK[w]||0) ? a.level : w, null);
  if (worst){
    const n = alerts.length;
    renderVerdict(worst==='critical'?'bad':'warn',
      n===1 ? (alerts[0].title) : `${n} alertas`,
      n===1 ? '' : alerts.map(a=>a.title).join(' · '));
  }
  else if (ids.some(id=>st(id)==='loading')) renderVerdict('warn', 'loading model', detail);
  else if (running>0)                          renderVerdict('good', 'decoding', `${running} in flight · ${detail}`);
  else if (ids.some(id=>st(id)==='loaded'))    renderVerdict('good', 'idle', detail);
  else if (ids.some(id=>st(id)==='suspended')) renderVerdict('idle', 'suspendido · vive en el host', detail);
  else if (ids.some(id=>st(id)==='sleeping'))  renderVerdict('idle', 'sleeping · wakes on the next request', detail);
  else                                         renderVerdict('idle', 'no model loaded', 'free VRAM · load one to see occupancy and KV');

  $('#now-metrics').innerHTML = renderEngineCards(d, rates);
  wireModelRows($('#now-metrics'), d);
}

// One box per Engine in memory. With two models loaded a single row of numbers
// was ambiguous: the verdict mixed "asleep" with "running", and every figure
// belonged to one model. Each box is one model and labels its state.
function renderEngineCards(d, rates){
  const ids = Object.keys(d.engines || {});
  if (!ids.length){
    return `<div class="now-metrics">
      <div class="now-m"><span class="k">lanes</span><b>—</b></div>
      <div class="now-m"><span class="k">in flight</span><b>0</b></div>
      <div class="now-m"><span class="k">ctx used</span><b>—</b></div></div>`;
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
        <span class="ec-state">${esc({loaded:'loaded',sleeping:'sleeping',suspended:'suspended on host',loading:'loading'}[e.state]||e.state)}</span>
        <span class="spacer"></span>
        ${on ? `<button data-act="sleep" data-model="${esc(mid)}">Sleep</button>`
             : `<button data-act="load" data-model="${esc(mid)}">Wake</button>`}
        <button class="iconbtn" data-cfg="${esc(mid)}" title="Model config">${SVG_GEAR}</button>
      </div>
      <div class="ec-metrics">
        <div class="now-m"><span class="k">${on?'VRAM':'size'}</span>
          <b style="color:${on?'#fff':'var(--dim)'}">${bytes?(bytes/2**30).toFixed(1):'—'} <small>GiB</small></b>
          ${!on&&hostMiB>1?`<small class="sub">${hostMiB.toFixed(0)} MiB in host</small>`:''}</div>
        <div class="now-m"><span class="k">ctx used</span>
          <b style="color:${ctxPct>85?'var(--amber)':'#fff'}">${(occ.device_main_kv_tokens||0).toLocaleString()} <small>/ ${ctxCap.toLocaleString()}</small></b>
          ${ctxCap?`<div class="progress"><i style="width:${ctxPct.toFixed(0)}%"></i></div>`:''}</div>
        <div class="now-m"><span class="k">lanes</span><b>${req.running??0}/${cap.max_concurrency??'—'}</b>
          ${cap.max_concurrency?`<div class="progress"><i style="width:${Math.min(100,(req.running||0)/cap.max_concurrency*100).toFixed(0)}%"></i></div>`:''}</div>
        <div class="now-m"><span class="k">queued</span><b style="color:${(req.waiting||0)>0?'var(--amber)':'#fff'}">${req.waiting??0}</b></div>
        <div class="now-m"><span class="k">${r.tg>0.5?'generating':'last gen'}</span>
          <b style="color:${r.tg>0.5?'var(--teal)':'#fff'}">${fmtRate(r.tg, r.lastTps, '').trim()} <small>tok/s</small></b></div>
        <div class="now-m"><span class="k">${r.pg>0.5?'prefilling':'last prefill'}</span>
          <b style="color:${r.pg>0.5?'var(--teal)':'#fff'}">${fmtRate(r.pg, r.lastPrefill, '').trim()} <small>tok/s</small></b></div>
        <div class="now-m" title="learned median of the server speed for this model"><span class="k">baseline</span>
          <b style="color:var(--muted)">${bl? (bl.samples>=12?bl.median_tok_s.toFixed(0):`learning ${bl.samples}/12`) : '—'} <small>${bl&&bl.samples>=12?'tok/s':''}</small></b></div>
      </div>
    </div>`;
  }).join('') + `</div>`;
}

function slotsHtml(d, occ){
  const slots = d.slots || [];
  if (!slots.length) return '';
  return `<div class="slots-strip">
      <div class="slots-strip-head">
        <b><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg> Paged KV Cache — Execution Slots (${slots.length} lanes)</b>
        <span style="color:var(--dim)">${occ.device_main_kv_tokens != null ? occ.device_main_kv_tokens.toLocaleString() + ' tokens active' : ''}</span>
      </div>
      <div class="slots-grid">
        ${slots.map(sl => {
          const isBusy = sl.is_processing, isRetained = sl.retained;
          const stateCls = isBusy ? 'busy' : isRetained ? 'retained' : '';
          const statusTxt = isBusy ? 'decoding' : isRetained ? 'cached' : 'free';
          const statusCls = isBusy ? 'active' : isRetained ? 'cached' : 'free';
          const tok = sl.n_prompt_tokens || sl.n_prompt_tokens_cache || 0;
          return `<div class="slot-cell ${stateCls}">
            <div class="slot-num">Slot ${sl.id}</div>
            <div class="slot-status ${statusCls}">● ${statusTxt}</div>
            <div class="slot-meta" title="${tok} tok prompt of ${sl.n_ctx} ctx">
              ${tok ? tok.toLocaleString() + ' tok' : '—'}
              ${tok && sl.n_ctx ? ` · ${(tok/sl.n_ctx*100).toFixed(1)}%` : ''}</div>
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
  // Alerts and the learned baseline move far more slowly than the state, so
  // nothing is re-rendered while the content is unchanged.
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
    errhost.innerHTML = `<div class="err" style="margin:15px">The server <code>${esc(d.url)}</code> is not responding.</div>`;
    renderVerdict('bad', 'the server is not responding', d.url || '');
    $('#now-metrics').innerHTML = '';
    return;
  }
  errhost.innerHTML = '';

  // Backfill: the charts start with the jsonl history instead of empty.
  if (!state.histLoaded){
    state.histLoaded = true;
    try{
      const h = await api('/api/history');
      if (h.ok){
        for (const p of h.points.slice(-300)){
          const lbl = new Date(p.ts).toLocaleTimeString('en-US',{hour12:false});
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
  const lbl = new Date().toLocaleTimeString('en-US',{hour12:false});

  // series from the /v1/load counters. `lastDT !== undefined` means "there is a
  // previous sample": the object is replaced whole, so no `last` key exists.
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
  renderHostChips(gpu);
  renderRail(d, loadOk, rates);
  loadAlerts();
  renderNow(d, load, cap, occ, req, loadOk, rates);

  // Tiles: a bounded ratio gets a bar, a rate gets a sparkline. The tone says
  // whether it is worth looking at (blue normal, amber >75%, red >90%).
  const tiles = LIVE_TILES.map(ti => {
    const v = occ[ti.k] ?? req[ti.k];
    const max = cap[ti.max] ?? occ[ti.max];
    const pct = (v!=null && max) ? Math.min(100, v/max*100) : null;
    const tone = pct==null ? '' : pct>90 ? 'bad' : pct>75 ? 'warn' : '';
    return `<div class="tile ${tone}">
      <div class="k">${ti.label}</div>
      <div class="v">${v??'—'}${ti.unit?` <small>${ti.unit}</small>`:''}</div>
      <div class="s">${max!=null ? `de ${max}` : ''}</div>
      ${pct!=null && pct>0 ? `<div class="s pct">${pct.toFixed(pct<10?1:0)}%</div>` : ''}
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
      <div class="v" style="color:${uCol}">${g.util.toFixed(0)}<small>% load</small></div>
      <div class="s"><span class="c-temp" style="font-weight:600">${g.temp_c.toFixed(0)}°C</span> · <span class="c-pow" style="font-weight:600">${g.power_w.toFixed(0)}W</span></div>
      <div class="bar-row"><span class="bl">VRAM</span>
        <span class="bv" style="color:${vramCol}">${(g.mem_used_mib/1024).toFixed(1)}/${(g.mem_total_mib/1024).toFixed(0)} GiB</span></div>
      <div class="bar"><i style="width:${vramPct.toFixed(0)}%;background:${vramCol}"></i></div></div>`;
  }).join('') : '';

  // The host, not just the card: a sleeping model costs host RAM, and a model
  // that will not fit is usually a disk or a reservation question, not a GPU one.
  const h = gpu?.host || {};
  const gib = b => (b/1073741824).toFixed(1);
  const hostTiles = [];
  if (h.ram?.total){
    const p = h.ram.used/h.ram.total*100;
    const col = p>=90?'var(--red)':p>=80?'var(--amber)':'var(--teal)';
    hostTiles.push(`<div class="tile">
      <div class="k">Host RAM</div>
      <div class="v">${gib(h.ram.used)}<small> GiB used</small></div>
      <div class="s">${gib(h.ram.available)} GiB available of ${gib(h.ram.total)}</div>
      <div class="bar"><i style="width:${p.toFixed(0)}%;background:${col}"></i></div></div>`);
  }
  if (h.cpu?.cores){
    const p = Math.min(100, h.cpu.load5/h.cpu.cores*100);
    const col = p>=90?'var(--red)':p>=75?'var(--amber)':'var(--blue)';
    hostTiles.push(`<div class="tile">
      <div class="k">CPU load</div>
      <div class="v">${h.cpu.load1.toFixed(2)}<small> / ${h.cpu.cores}</small></div>
      <div class="s">5 min ${h.cpu.load5.toFixed(2)} · 15 min ${h.cpu.load15.toFixed(2)}</div>
      <div class="bar"><i style="width:${p.toFixed(0)}%;background:${col}"></i></div></div>`);
  }
  for (const d of (h.disks||[])){
    const p = d.used/d.total*100;
    const col = p>=95?'var(--red)':p>=85?'var(--amber)':'var(--teal)';
    const name = d.path==='/' ? 'Root' : d.path.split('/').pop();
    hostTiles.push(`<div class="tile">
      <div class="k">${esc(name)}</div>
      <div class="v">${gib(d.free)}<small> GiB free</small></div>
      <div class="s">${esc(d.path)} · ${p.toFixed(0)}% used</div>
      <div class="bar"><i style="width:${p.toFixed(0)}%;background:${col}"></i></div></div>`);
  }

  const tpData = (state.charts.tp?.data.datasets[0].data)||[];
  const sparkTile = `<div class="tile info"><div class="k">Throughput</div>
    <div class="v">${tpData.length?(tpData[tpData.length-1]||0).toFixed(0):'—'} <small>tok/s</small></div>
    <div class="s">${state.win==='retained'?'retained':state.win}</div>
    ${sparkSvg(tpData.slice(-40))}</div>`;

  // With no model there is no occupancy or KV to show: those tiles would be
  // four dashes. Only the GPU ones stay, which always have data.
  $('#tiles').innerHTML = sparkTile + (loadOk ? tiles : '') + gpuTiles + hostTiles.join('');
  $('#livenote').textContent = loadOk ? '' : 'no model loaded — load one to see occupancy and KV';
  $('#slots-host').innerHTML = slotsHtml(d, occ);
  $('#modelrows').innerHTML = modelRows(d, req, false);
  $('#modelnote').textContent = `${(d.models?.data||[]).length} in the catalog`;
  renderQuickLoad(d);
  $('#liveupd').textContent = 'updated ' + new Date().toLocaleTimeString('en-US',{hour12:false});
  wireModelRows($('#modelrows'), d);
  wireModelRows($('#quickload'), d);

  // The Control page's table is drawn by loadControl() at boot and after a save
  // only, so it froze on the state of that moment: rows claiming a model was
  // loaded when it wasn't, and a catalog count from before the last restart.
  // Same payload as the rail, so refresh it here rather than polling twice.
  state.ctlLive = d;
  renderCtlRows(d);
}
