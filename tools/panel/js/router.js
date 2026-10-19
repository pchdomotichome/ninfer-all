// Router tab: the catalog of models the router serves.
//
// The source of truth is the router's own input: the .ninfer artifacts under
// --models-dir plus router-preset.ini. The panel edits that file (the [model]
// sections only, preserving the rest) and never invents catalog entries.
//
// A model id is immutable: the router keys /v1 models on it. The "alias" is the
// human name the router also accepts, and it is what the panel shows.

import { $, esc, api, flash } from './core.js';

let dirty = false;
let flagHints = [];

// api(path, opts) is fetch's signature, not (method, path, body). Every write
// here went through the wrong one, so it fetched the URL "POST".
const post = (path, body) =>
  api(path, {method:'POST', headers:{'Content-Type':'application/json'},
              body:JSON.stringify(body || {})});

export async function loadRouter() {
  const [cat, helps] = await Promise.all([
    fetch('/api/router/catalog').then(r => r.json()).catch(() => ({ models: [] })),
    flagHints.length ? null : fetch('/api/help').then(r => r.json()).catch(() => null),
  ]);
  if (helps && helps.flags) flagHints = helps.flags.map(f => f.flag);
  const info = $('#router-status');
  if (info) {
    info.innerHTML = `<span class="note">preset: ${esc(cat.preset || '')} · models-dir: ${esc(cat.models_dir || '')}</span>`;
  }
  const models = cat.models || [];
  $('#router-note').textContent = `${models.length} models in catalog`;
  const host = $('#router-rows');
  if (!host) return;
  host.innerHTML = `<div class="router-head">
      <span>model id (immutable)</span><span>alias (what you see)</span>
      <span>artifact</span><span></span></div>` +
    models.map(renderRow).join('') + renderAdd(cat);
  wire(host);
  refreshRestartNote();
}

function flagChips(flags) {
  const keys = Object.keys(flags || {});
  if (!keys.length) return '<span class="note">no flags</span>';
  return keys.map(k => `<span class="router-flag">${esc(k)}=${esc(flags[k])}</span>`).join(' ');

}

function renderRow(m) {
  const real = m.artifact_real
    ? `<div class="real" title="${esc(m.artifact_real)}">→ ${esc(m.artifact_real)}</div>` : '';
  return `<div class="router-row" data-row="${esc(m.id)}" data-flags="${esc(JSON.stringify(m.flags || {}))}">
    <span class="router-id" title="${esc(m.id)}">${esc(m.id)}</span>
    <span><input data-alias="${esc(m.id)}" value="${esc(m.alias || '')}"
      placeholder="(no alias)" style="width:100%"></span>
    <span class="router-art" title="${esc(m.artifact || '')}">${esc(m.artifact || '')}${real}
      <div style="margin-top:3px">${flagChips(m.flags)}</div></span>
    <span class="router-acts">
      <button data-edit="${esc(m.id)}">Edit</button>
      <button data-del="${esc(m.id)}" class="danger">Delete</button>
    </span>
  </div>
  <div class="router-edit" id="edit-${esc(m.id)}" hidden></div>`;
}

function renderAdd(cat) {
  return `<div class="router-row" style="border-top:1px solid var(--line);margin-top:4px">
    <span><input id="router-new-id" placeholder="new-model-id" style="width:100%"></span>
    <span><input id="router-new-alias" placeholder="alias" style="width:100%"></span>
    <span><input id="router-new-artifact" placeholder="/path/to/model.ninfer" list="router-artifacts" style="width:100%">
      <datalist id="router-artifacts"></datalist></span>
    <span class="router-acts"><button id="router-add">Add</button></span>
  </div>`;
}

function flagLines(flags) {
  return Object.keys(flags || {}).map(k => `${k} = ${flags[k]}`).join('\n');
}

function parseFlags(text) {
  const out = {};
  for (const raw of (text || '').split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq < 0) continue;
    out[line.slice(0, eq).trim().replace(/^--/, '')] = line.slice(eq + 1).trim();
  }
  return out;
}

async function writeModel(id, body) {
  const r = await post('/api/router/save', { model: id, ...body });
  if (r.ok) { dirty = true; flash(`${id} saved (restart required).`, false); await loadRouter(); }
  else flash(r.error || 'save failed', true);
}

function wire(host) {
  host.querySelectorAll('input[data-alias]').forEach(inp => {
    inp.addEventListener('change', async () => {
      const r = await post('/api/router/alias',
        { model: inp.dataset.alias, alias: inp.value });
      if (r.ok) { dirty = true; refreshRestartNote(); flash('Alias saved (restart required).', false); }
      else flash(r.error || 'alias save failed', true);
    });
  });

  host.querySelectorAll('button[data-edit]').forEach(b => {
    b.addEventListener('click', () => openEditor(host, b.dataset.edit));
  });

  host.querySelectorAll('button[data-del]').forEach(b => {
    b.addEventListener('click', async () => {
      const id = b.dataset.del;
      if (!confirm(`Delete ${id} from the catalog?`)) return;
      const delLink = confirm('Also remove its symlink in --models-dir?\n' +
        'The artifact it points at is never touched.');
      const r = await post('/api/router/delete', { model: id, delete_link: delLink });
      if (r.ok) { dirty = true; flash(`${id} deleted (restart required).`, false); await loadRouter(); }
      else flash(r.error || 'delete failed', true);
    });
  });

  const add = $('#router-add');
  if (add) add.addEventListener('click', async () => {
    const id = $('#router-new-id').value.trim();
    const alias = $('#router-new-alias').value.trim();
    const artifact = $('#router-new-artifact').value.trim();
    if (!id || !artifact) { flash('An id and an artifact are required.', true); return; }
    await writeModel(id, { alias, artifact });
  });

  // Artifact suggestions (datalist) are loaded lazily, once.
  const list = $('#router-artifacts');
  if (list && !list.childElementCount) {
    fetch('/api/convert/files').then(r => r.json()).then(d => {
      list.innerHTML = (d.artifacts || [])
        .map(a => `<option value="${esc(a.path)}">`).join('');
    }).catch(() => {});
  }
}

function openEditor(host, id) {
  const box = host.querySelector(`#edit-${CSS.escape(id)}`);
  if (!box || !box.hidden) { if (box) box.hidden = true; return; }
  const row = host.querySelector(`div[data-row="${CSS.escape(id)}"]`);
  const alias = row?.querySelector('input[data-alias]')?.value || '';
  // The row carries the flags as JSON so the editor round-trips them exactly.
  let flags = {};
  try { flags = JSON.parse(row?.dataset.flags || '{}'); } catch { flags = {}; }
  const artifact = row?.querySelector('.router-art')?.getAttribute('title') || '';
  box.hidden = false;
  box.innerHTML = `<div class="dgroup" style="margin:10px 18px">
    <h4>${esc(id)}</h4>
    <label class="f" style="margin-bottom:6px">artifact<input id="e-art" value="${esc(artifact)}" style="width:100%"></label>
    <label class="f" style="margin-bottom:6px">flags (one <code>key = value</code> per line)
      <textarea id="e-flags" rows="6" style="width:100%;font:12px var(--mono)">${esc(flagLines(flags))}</textarea></label>
    <div class="router-acts">
      <button class="primary" data-save="${esc(id)}">Save</button>
      <button data-cancel="${esc(id)}">Cancel</button>
    </div>
  </div>`;
  box.querySelector('[data-cancel]').onclick = () => { box.hidden = true; };
  box.querySelector('[data-save]').onclick = async () => {
    const flags = parseFlags(box.querySelector('#e-flags').value);
    await writeModel(id, { alias, artifact: box.querySelector('#e-art').value.trim(), flags });
  };
}

function refreshRestartNote() {
  const el = $('#router-restart-note');
  if (el) el.textContent = dirty
    ? 'A router restart is required to apply these changes.' : '';
}

function wireRestart() {
  const btn = $('#router-restart');
  if (!btn || btn.dataset.bound) return;
  btn.dataset.bound = '1';
  btn.addEventListener('click', async () => {
    if (!confirm('Restart ninfer-router? Brief downtime, loaded models unload.')) return;
    const r = await post('/api/router/restart');
    if (r.ok) { dirty = false; refreshRestartNote(); flash('Router restarted.', false); await loadRouter(); }
    else flash('Restart failed: ' + (r.error || r.stderr || 'unknown'), true);
  });
}
wireRestart();