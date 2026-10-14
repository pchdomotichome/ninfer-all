import { $, state, esc, api, flash, refresh } from './core.js';

export function modelArgs(m){
  const a = (m.status?.args)||[];
  const get = k => { const i=a.indexOf(k); return i>=0 ? a[i+1] : null; };
  const has = k => a.includes(k);
  const spec = get('--spec'), draft = get('--draft-tokens');
  return { ctx: get('--max-context'), device: get('--device'),
           spec: spec ? spec + (draft?'/'+draft:'') : (has('--no-spec')?'off':null),
           vision: has('--vision') };
}

// ── Favoritos → quick-load del Dashboard ───────────────────────────────────
// Sin backend: localStorage, por navegador. Cuando el panel tenga config
// persistente esto se muda ahí y pasa a ser compartido entre navegadores.
const FAV_KEY = 'ninfer.favs';
export let favs = new Set();
try{ favs = new Set(JSON.parse(localStorage.getItem(FAV_KEY) || '[]')); }catch(e){}
function favSave(){ try{ localStorage.setItem(FAV_KEY, JSON.stringify([...favs])); }catch(e){} }

// Estrella de favorito: hueca en gris, rellena y ámbar cuando está marcada.
const SVG_STAR = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>';// Engranaje de verdad (el mismo trazo que la cabecera del modal), no un "sol".
export const SVG_GEAR = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.08a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h.08a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.08a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>';

// Una fila por modelo, no una card. El cargado primero; el resto es catálogo.
// `rich` agrega la estrella de favorito y el engranaje de config: es la vista de
// la pestaña Modelos. La del Dashboard va sin ellas, sólo acciones de carga.
function buildModelRow(m, d, rich){
  const st = m.status?.value || 'unloaded';
  const ar = modelArgs(m);
  const dotCls = st==='loaded'?'ok':st==='loading'?'warn':st==='sleeping'?'warn'
    : st==='suspended'?'blue':'off';
  const running = ((d.load && !d.load.error) ? d.load.requests : {}).running || 0;
  const busy = running > 0 && st === 'loaded';
  const act = st==='loaded'
    ? `<button data-act="sleep" data-model="${esc(m.id)}" ${busy?'disabled':''} title="${busy?'Hay requests en vuelo':''}">Dormir</button>
       <button data-act="unload" data-model="${esc(m.id)}" class="danger">Descargar</button>`
    : st==='sleeping' ? `<button data-act="load" data-model="${esc(m.id)}">Despertar</button>`
    : st==='suspended' ? `<button data-act="resume" data-model="${esc(m.id)}">Reanudar</button>`
    : st==='loading' ? `<button disabled>cargando…</button>`
    : `<button data-act="load" data-model="${esc(m.id)}">Cargar</button>`;
  const star = rich ? `<button class="iconbtn starbtn${favs.has(m.id)?' on':''}" data-fav="${esc(m.id)}"
      title="${favs.has(m.id)?'Quitar del quick-load del Dashboard':'Agregar al quick-load del Dashboard'}">${SVG_STAR}</button>` : '';
  const gear = rich ? `<button class="iconbtn" data-cfg="${esc(m.id)}"
      title="Config del modelo: flags de lanzamiento">${SVG_GEAR}</button>` : '';
  const meta = `ctx ${ar.ctx??'—'}${ar.spec?` · ${ar.spec}`:''}${ar.vision?' · vision':''}`;
  return `<div class="model-row${rich?' rich':''} ${st!=='unloaded'?'live':''}">
    ${star}
    <span class="dot ${dotCls}"></span>
    <span class="nm" title="${esc(m.id)}">${esc(m.id)}</span>
    <span class="mt">${meta}</span>
    <span class="acts">${act}${gear}</span>
  </div>`;
}

export function modelRows(d, req, rich){
  const models = (d.models?.data)||[];
  if (!models.length) return '<div class="empty" style="padding:18px">El servidor no declara modelos.</div>';
  const order = {loaded:0, loading:1, sleeping:2, suspended:3, unloaded:4};
  const rows = [...models].sort((a,b)=>
    (order[a.status?.value]??9)-(order[b.status?.value]??9) || a.id.localeCompare(b.id));
  return rows.map(m=>buildModelRow(m, d, rich)).join('');
}

// Quick-load del Dashboard: los favoritos marcados en la pestaña Modelos.
export function renderQuickLoad(d){
  const box = $('#quickload');
  const fav = ((d.models?.data)||[]).filter(m=>favs.has(m.id));
  if (!fav.length){ box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  box.innerHTML = `<div class="quick-label">Quick load</div><div class="chips">` +
    fav.map(m=>{
      const st = m.status?.value || 'unloaded';
      const on = st==='loaded' || st==='sleeping' || st==='suspended';
      const dot = st==='loaded' ? 'ok' : st==='sleeping' ? 'warn'
        : st==='suspended' ? 'blue' : 'off';
      return `<button class="chip${on?' on':''}" data-act="${on?'unload':'load'}"
          data-model="${esc(m.id)}" title="${on?'Descargar':'Cargar'} ${esc(m.id)}">
          <span class="dot ${dot}"></span>${esc(m.id)}${st==='loaded'?' · activo':''}</button>`;
    }).join('') + '</div>';
}

// Un solo wiring para las tres tablas de modelos (dashboard, quick-load, control):
// `data-*` en el botón, no la posición en el DOM.
export function wireModelRows(host, d){
  host.querySelectorAll('button[data-act]').forEach(b=>{
    b.onclick = async () => {
      b.disabled = true;
      try{
        const r = await api('/api/control', {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({action:b.dataset.act, model:b.dataset.model})});
        if (r.error) flash(String(r.error).slice(0,160), true);
      }catch(e){ flash(e.message, true); }
      if (host.id === 'ctl-rows'){
        // Re-pedir /api/live: si se re-renderiza con el payload viejo la fila
        // seguiría diciendo "Dormir" sobre un modelo que ya duerme. No se llama
        // loadControl() completo porque resetearía el formulario de recipes.
        const live = await api('/api/live').catch(()=>null);
        if (live){ state.ctlLive = live; renderCtlRows(live); }
      } else refresh.live?.();
    };
  });
  host.querySelectorAll('button[data-fav]').forEach(b=>{
    b.onclick = () => {
      const id = b.dataset.fav;
      if (favs.has(id)) favs.delete(id); else favs.add(id);
      favSave();
      renderCtlRows(state.ctlLive);   // barato: no re-pide /api/live
    };
  });
  host.querySelectorAll('button[data-cfg]').forEach(b=>{
    b.onclick = () => {
      const m = ((d.models?.data)||[]).find(x=>x.id===b.dataset.cfg);
      if (m) openModelCfg(m);
    };
  });
}

// Config de un modelo: los flags reales con los que el router lo lanzaría.
// Es de sóla lectura a propósito — el panel lee router-preset.ini, no lo escribe —
// y el aviso de abajo lo dice, en vez de fingir un botón "Guardar".
export function openModelCfg(m){
  const ar = modelArgs(m);
  const args = (m.status?.args)||[];
  const res = m.residency || {};
  const stDot = m.status?.value==='loaded' ? 'ok' : m.status?.value==='sleeping' ? 'warn'
    : m.status?.value==='suspended' ? 'blue' : 'off';
  // Un Map por flag: --device viene dos veces (default del preset + CLI) y
  // duplicarlo sería ruido. host/port son del proceso, no del modelo.
  const map = new Map();
  for (let i=0;i<args.length;i++){
    const a = args[i];
    if (!a.startsWith('--') || a==='--host' || a==='--port') continue;
    const next = args[i+1];
    const hasVal = next!=null && !next.startsWith('--');
    const prev = map.get(a);
    map.set(a, (prev ? prev + ', ' : '') + (hasVal ? next : 'sí'));
    if (hasVal) i++;
  }
  $('#modal-cfg-title').textContent = 'Config · ' + m.id;
  $('#modal-cfg-body').innerHTML = `
    <dl class="kv">
      <dt>estado</dt><dd><span class="dot ${stDot}" style="display:inline-block;margin-right:6px"></span>${esc(m.status?.value || '—')}</dd>
      <dt>contexto</dt><dd>${ar.ctx ?? '—'}</dd>
      <dt>spec decode</dt><dd>${ar.spec ?? 'off'}</dd>
      <dt>residencia</dt><dd>${esc(res.state || '—')}</dd>
      <dt>GPU</dt><dd>${esc(modelArgs(m).device ?? '—')}</dd>
      ${[...map.entries()].map(([k,v])=>
        `<dt style="color:var(--teal)">${esc(k)}</dt><dd style="color:var(--text)">${esc(v)}</dd>`).join('')}
    </dl>
    <div class="quick-label" style="margin-top:16px">Comando de lanzamiento</div>
    <div class="cfg-args">${esc(args.join(' '))}</div>
    <div class="filenote">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--amber)" stroke-width="2" stroke-linecap="round" style="flex:none;margin-top:1px"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      <div>Estos flags vienen de <code>router-preset.ini</code> y del catálogo de
      <code>--models-dir</code>. El panel los lee, no los escribe: para cambiarlos hay
      que editar ese archivo y <b>reiniciar el router</b> — no hay hot-reload del catálogo.</div>
    </div>`;
  $('#modal-cfg').style.display = 'flex';
}

// Alertas y referencias aprendidas. El backend corre su propio muestreo cada
// 2 s aunque nadie mire la página; esto sólo lee el resultado.
// Re-render barato: sólo las filas, sin volver a pedir /api/live ni a recalcular
// el formulario de recipes (que perdería lo que el usuario estuviera escribiendo).
export function renderCtlRows(d){
  const host = $('#ctl-rows');
  if (!host || !d || !d.online) return;
  host.innerHTML = modelRows(d, null, true);
  const nFav = ((d.models?.data)||[]).filter(m=>favs.has(m.id)).length;
  $('#ctl-note').textContent = `${(d.models?.data||[]).length} modelos` +
    (nFav ? ` · ${nFav} favoritos` : '');
  wireModelRows(host, d);
}

