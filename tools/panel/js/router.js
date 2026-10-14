// Router tab: browse the catalog, rename aliases, restart the router.
// The catalog is the source of truth (models_dir + router-preset.ini); the panel
// never invents an entry. Alias lives in the preset's [model] section.

import { $, esc, api, flash } from './core.js';

let dirty = false;

export async function loadRouter() {
  const res = await fetch('/api/router/catalog').catch(() => null);
  const d = res ? await res.json().catch(() => ({})) : {};
  const rows = (d.models || []);
  const body = $('#router-status');
  if (body) {
    body.innerHTML = d.preset ? `<span class="note">preset: ${esc(d.preset)} · models-dir: ${esc(d.models_dir)}</span>` : '';
  }
  const cont = $('#router-rows');
  if (!cont) return;
  if (!rows.length) { cont.innerHTML = '<div class="empty" style="padding:18px">The router declares no models.</div>'; return; }
  cont.innerHTML = rows.map(m => {
    const al = m.alias || '';
    return `<div class="model-row" style="align-items:center">
      <span class="nm" title="${esc(m.id)}">${esc(m.id)}</span>
      <span class="mt">${esc(m.artifact || '')}${m.preset ? '' : ' · (disk only)'}</span>
      <span style="flex:1"></span>
      <input data-alias="${esc(m.id)}" value="${esc(al)}" placeholder="alias" style="width:180px;margin-right:8px">
      <button data-rename="${esc(m.id)}">Rename</button>
      <span class="note" id="rn-${esc(m.id)}"></span>
    </div>`;
  }).join('');
  // alias rename wiring
  cont.querySelectorAll('button[data-rename]').forEach(b => {
    b.addEventListener('click', async () => {
      const mid = b.dataset.rename;
      const inp = cont.querySelector(`input[data-alias="${CSS.escape(mid)}"]`);
      const r = await api('POST', '/api/router/alias', { model: mid, alias: inp?.value || '' });
      if (r.ok) { flash(`Alias updated for ${mid} (restart required).`, false); setDirty(true); }
      else flash((r.error || 'alias update failed'), true);
    });
  });
  $('#router-note').textContent = `${rows.length} models in catalog`;
  refreshRestartNote();
}

function setDirty(v) {
  dirty = dirty || v;
  refreshRestartNote();
}

function refreshRestartNote() {
  const el = $('#router-restart-note');
  if (el) el.textContent = dirty ? 'A router restart is required to apply these changes.' : '';
}

// Restart button
function wireRestart() {
  const btn = $('#router-restart');
  if (!btn || btn.dataset.bound) return;
  btn.dataset.bound = '1';
  btn.addEventListener('click', async () => {
    if (!confirm('Restart ninfer-router? Brief downtime, models unload.')) return;
    const r = await api('POST', '/api/router/restart', {});
    if (r.ok) { flash('Router restarted.', false); dirty = false; refreshRestartNote(); await loadRouter(); }
    else flash('Restart failed: ' + (r.error || r.stderr || 'unknown'), true);
  });
}
wireRestart();