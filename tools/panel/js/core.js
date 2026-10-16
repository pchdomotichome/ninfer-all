

export const $ = s => document.querySelector(s);
export const state = { cfg:null, runs:[], open:new Set(), selected:new Set(), more:false, live:false, job:null,
                 page:1, size:10, total:0, convert:null, download:null, probe:null, win:'5m', hist:[],
                 histLoaded:false, series:{counters:{}, up:{}}, charts:{}, reqs:[], ctlLive:null,
                 alerts:null, alertsSig:'', recipeModel:null, recipeArtifact:null, recipeFlags0:null};
export const esc = s => String(s??'').replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
export function num(v, d=1){
  if (v===null||v===undefined) return '<span class="muted">—</span>';
  return typeof v==='number' ? v.toFixed(d) : esc(v);
}
export function flat(r){
  const s = r.server||{};
  return {...r, engine:s.binary||'', spec: s.spec_backend
    ? s.spec_backend + (s.draft_depth?'/'+s.draft_depth:'') : 'none'};
}
export function specKey(s){
  if (!s || !s.spec_backend) return '';
  return s.spec_backend + (s.draft_depth ? '/'+s.draft_depth : '');
}

// One hue per component name, so `dflash2` reads the same colour in the
// benchmark table, the request list and an artifact's components. The hue is
// taken from the name before the slash, since the tables carry `mtp/3` while
// the artifact inspector carries `mtp`.
const COMP_HUE = {text:'text', vision:'vision', mtp:'mtp', dflash:'dflash', dflash2:'dflash2'};
export function compPill(spec){
  const n = String(spec ?? '').trim();
  if (!n || n === 'none' || n === 'off') return '<span class="muted">off</span>';
  const hue = COMP_HUE[n.split('/')[0]] || '';
  return `<span class="pill ${hue}">${esc(n)}</span>`;
}


// live.js publishes here and models.js asks, so the two never import each other
// in the way a direct call would force.
export const refresh = { live: null };

export async function api(path, opts){
  const res = await fetch(path, opts);
  const body = await res.json().catch(()=>({error:res.statusText}));
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}
export function flash(msg, bad){
  $('#banner').innerHTML = `<div class="${bad?'err':'ok-msg'}">${esc(msg)}</div>`;
  setTimeout(()=>$('#banner').innerHTML='', 5000);
}

