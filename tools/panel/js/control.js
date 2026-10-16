import { $, state, esc, num, api, flash } from './core.js';
import { modelRows, wireModelRows, renderCtlRows, favs } from './models.js';
import { fmtSize } from './convert.js';
import { liveModels } from './live.js';

export async function loadControl(){
  const body = $('#ctl-body');
  try{
    // /api/live rather than /api/engine: status.args carries the model's flags
    // (ctx, spec, draft), and the same source feeds the Dashboard, so there is
    // one row renderer instead of two.
    const [h, live] = await Promise.all([
      api('/api/help').catch(()=>({flags:[]})),
      api('/api/live').catch(()=>({online:false, url:'', models:{data:[]}})),
    ]);
    state.ctlLive = live;
    body.innerHTML = '<div class="model-rows" id="ctl-rows"></div>';
    const host = $('#ctl-rows');
    const n = (live.models?.data||[]).length;
    if (!live.online){
      host.className = 'empty';
      host.style.maxHeight = 'none';
      host.textContent = `The server ${live.url || ''} is not responding.`;
      $('#ctl-note').textContent = 'no connection';
    } else {
      host.innerHTML = modelRows(live, null, true);
      const nFav = ((live.models?.data)||[]).filter(m=>favs.has(m.id)).length;
      $('#ctl-note').textContent = `${n} models` + (nFav ? ` · ${nFav} favorites` : '');
      $('#ctl-upd').textContent = 'updated ' + new Date().toLocaleTimeString('en-US',{hour12:false});
      wireModelRows(host, live);
    }
    // recipe form: artifact dropdown + flags
    if (h.flags && h.flags.length){
      $('#rec-note').textContent = `${h.flags.length} flags read from ${h.binary}`;
      let files = {artifacts:[]};
      try{ files = await api('/api/convert/files'); }catch(err){}
      $('#r-artifact').innerHTML = '<option value="">— pick an artifact —</option>' +
        files.artifacts.map(a=>`<option value="${esc(a.path)}">
          ${esc(a.path.split('/').pop())} (${fmtSize(a.size)})</option>`).join('');
      // Picking an artifact is the event that drives the whole form: it decides
      // which components exist, hence which flags, hence the preset preview.
      $('#r-artifact').onchange = e => loadArtifactRecipe(e.target.value);
      $('#r-save').onclick = () => savePreset(false);
      $('#r-launch').onclick = () => savePreset(true);
      renderRecipeFlags(h.flags);
    } else {
      $('#r-flags').innerHTML = '<div class="err">Could not read the binary\'s --help.</div>';
    }
    // endpoint for agents: base url + the loaded model from the catalog
    const base = state.cfg?.endpoint?.base_url || '';
    if (base){
      const loaded = liveModels(live).map(m=>m.id);
      const model = loaded.length ? loaded.join(', ') : ((live.models?.data||[])[0]?.id || '');
      const key = state.cfg.endpoint.api_key || '(no api key)';
      $('#ep-kv').innerHTML = `
        <dt>base_url</dt><dd id="ep-base">${esc(base)}</dd>
        <dt>model</dt><dd id="ep-model">${esc(model)}</dd>
        <dt>api_key</dt><dd>${esc(key)}</dd>
        <dt>models</dt><dd>${(live.models?.data||[]).length} in catalog</dd>`;
      const activeModel = model.split(',')[0].trim() || 'qwen35-9b';
      const epSnippets = {
        curl: `# OpenAI-compatible cURL\n` +
              `curl ${base}/chat/completions \\\n` +
              `  -H 'Content-Type: application/json' \\\n` +
              (state.cfg.endpoint.api_key ? `  -H 'Authorization: Bearer ${state.cfg.endpoint.api_key}' \\\n` : '') +
              `  -d '{"model":"${activeModel}","messages":[{"role":"user","content":"hello"}],"max_tokens":128}'`,
        python: `# Python (openai SDK)\nfrom openai import OpenAI\n\n` +
                `client = OpenAI(base_url="${base}", api_key="${state.cfg.endpoint.api_key || 'dummy'}")\n\n` +
                `resp = client.chat.completions.create(\n` +
                `    model="${activeModel}",\n` +
                `    messages=[{"role": "user", "content": "Hello ninfer!"}],\n` +
                `    max_tokens=128\n)\nprint(resp.choices[0].message.content)`,
        js: `// TypeScript / Node (openai SDK)\nimport OpenAI from "openai";\n\n` +
            `const client = new OpenAI({ baseURL: "${base}", apiKey: "${state.cfg.endpoint.api_key || 'dummy'}" });\n\n` +
            `const res = await client.chat.completions.create({\n` +
            `  model: "${activeModel}",\n` +
            `  messages: [{ role: "user", content: "Hello ninfer!" }],\n` +
            `  max_tokens: 128\n});\nconsole.log(res.choices[0].message.content);`
      };

      let activeLang = 'curl';
      const renderSnippet = () => {
        $('#ep-example').textContent = epSnippets[activeLang] || epSnippets.curl;
        document.querySelectorAll('.code-tab').forEach(t => {
          t.classList.toggle('active', t.dataset.lang === activeLang);
        });
      };
      renderSnippet();

      document.querySelectorAll('.code-tab').forEach(tab => {
        tab.onclick = () => {
          activeLang = tab.dataset.lang;
          renderSnippet();
        };
      });

      $('#ep-copy').onclick = () => {
        navigator.clipboard.writeText($('#ep-example').textContent)
          .then(()=>flash('Code copied to the clipboard', false))
          .catch(()=>flash('No se pudo copiar', true));
      };
    }
  }catch(e){
    body.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}

// ── Recipe form: artifact -> recipe -> flags -> preset preview -> save ───────
// The artifact is the input: it knows which components it carries, and those
// decide which flags the launch needs. Nothing here is hardcoded per model.

const RECIPE_DEFAULTS = { '--kv-capacity':'32768', '--max-context':'32768',
                          '--max-concurrency':'2' };

/**
 * Flags an artifact's components require: vision and/or a spec backend.
 * The draft depth is not in the artifact, so these are the lab's defaults for
 * each backend; the form is there to change them.
 */
function flagsFromComponents(comps){
  const flags = {};
  if (comps.vision) flags['--vision'] = 'true';
  if (comps.mtp) {
    flags['--spec'] = 'mtp';
    flags['--draft-tokens'] = '3';
  } else if (comps.dflash2) {
    flags['--spec'] = 'dflash2';
    flags['--draft-tokens'] = '4';
  }
  return flags;
}

function renderRecipeFlags(flags){
  // 154 flags at once is noise: the relevant ones first, the rest under "more".
  // They are grouped by category from the flag's prefix (model, context,
  // spec-decode, etc.) so the form stays navigable.
  const GROUPS = [
    {name:'model', re:/^--(model-id|models|chat-template|mmproj|vision|expert)/},
    {name:'context and kv', re:/^--(max-context|kv-|host-|device-|prefill|context-)/},
    {name:'spec-decode', re:/^--(spec|draft|lm-head|adaptive|ngram|verbose|mtp)/},
    {name:'execution', re:/^--(max-concurrency|max-pending|sleep-idle|unload-idle|model-suspend|no-models)/},
    {name:'output and api', re:/^--(api-key|host|port|cors|no-webui|response|revision)/},
  ];
  const REST = 'sampling and rest';
  // Keep whatever is already in the form: loadControl() re-renders after a
  // launch, and wiping the inputs there would silently discard the recipe.
  const keep = new Map([...$('#r-flags').querySelectorAll('[data-flag]')]
    .map(i => [i.dataset.flag, i.type === 'checkbox' ? (i.checked ? 'true' : '')
                                                  : i.value]));

  // One row per flag, label left and control right: a boolean asking for a
  // value invites "yes"/"1"/"true" and three different meanings in one preset.
  // The full --help description rides in the title, since the placeholder has
  // room for about six words.
  const mk = f => {
    const name = esc(f.flag.slice(2));
    const title = esc(f.desc || f.flag);
    const cur = keep.has(f.flag) ? keep.get(f.flag) : (RECIPE_DEFAULTS[f.flag] ?? '');
    let ctl;
    if (!f.takes_value){
      ctl = `<label class="sw" title="${title}">
        <input type="checkbox" data-flag="${esc(f.flag)}" ${cur === 'true' ? 'checked' : ''}>
        <i></i><span class="sw-s">${cur === 'true' ? 'true' : 'off'}</span></label>`;
    } else if (f.enum && f.enum.length){
      const opts = f.enum.map(o =>
        `<option value="${esc(o)}"${o === cur ? ' selected' : ''}>${esc(o)}</option>`).join('');
      ctl = `<select data-flag="${esc(f.flag)}" title="${title}">
        <option value=""${cur ? '' : ' selected'}>—</option>${opts}</select>`;
    } else {
      ctl = `<input data-flag="${esc(f.flag)}" value="${esc(cur)}"
        placeholder="${esc((f.desc || '').slice(0, 48))}" title="${title}">`;
    }
    return `<div class="flag-row"><label class="flag-name" title="${title}">${name}</label>
      ${ctl}</div>`;
  };
  let html = '';
  // The last group is whatever the categories above did not name, so a flag that
  // exists is never unreachable: hiding 93 of 162 is worse than a long list.
  const named = new Set(GROUPS.flatMap(g => flags.filter(f=>g.re.test(f.flag))
                                                     .map(f=>f.flag)));
  const buckets = GROUPS.map(g => [g.name, flags.filter(f=>g.re.test(f.flag))]);
  buckets.push([REST, flags.filter(f=>!named.has(f.flag))]);
  for(const [name, inGroup] of buckets){
    if (!inGroup.length) continue;
    html += `<details ${name==='model'?'open':''}>
      <summary style="cursor:pointer;color:var(--teal);font:600 11px system-ui;
        text-transform:uppercase;letter-spacing:.6px;margin:8px 0 5px">${name}
        (${inGroup.length})</summary>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 18px;margin-bottom:6px">
      ${inGroup.map(mk).join('')}
      </div></details>`;
  }
  $('#r-flags').innerHTML = html;
  $('#r-flags').oninput = renderPresetPreview;
  $('#r-flags').onchange = renderPresetPreview;
  renderPresetPreview();
}

/**
 * Flag a flag, a value that is neither empty nor the built-in default is one
 * someone chose: tint the row so the configured recipe reads at a glance without
 * checking every field. Also refreshes the switch's own text, which otherwise
 * keeps saying "off" when the preset filled it in without a click.
 * Runs after the artifact fills the values and again on every change.
 */
function markCustom(){
  $('#r-flags').querySelectorAll('.flag-row').forEach(row=>{
    const ctl = row.querySelector('[data-flag]');
    if (!ctl) return;
    const val = ctl.type === 'checkbox' ? (ctl.checked ? 'true' : '')
                                        : ctl.value.trim();
    row.classList.toggle('custom',
      val !== '' && val !== (RECIPE_DEFAULTS[ctl.dataset.flag] ?? ''));
    const text = row.querySelector('.sw-s');
    if (text) text.textContent = ctl.checked ? 'true' : 'off';
  });
}

/**
 * The values of the form, as preset entries: "spec = mtp", not "--spec mtp".
 * An unchecked switch is absent from the preset, which is how a flag is turned
 * off without leaving `key = false` behind.
 */
function recipeFlags(){
  const out = {};
  $('#r-flags').querySelectorAll('[data-flag]').forEach(i=>{
    const key = i.dataset.flag.replace(/^--/, '');
    if (i.type === 'checkbox'){
      if (i.checked) out[key] = 'true';
      return;
    }
    const v = i.value.trim();
    if (v) out[key] = v;
  });
  return out;
}

/**
 * The preview is the literal preset section that Save would write, so what is
 * on screen is what lands in router-preset.ini, not a paraphrase of it.
 */
function renderPresetPreview(){
  const pre = $('#r-preview');
  if (!pre) return;
  const artifact = state.recipeArtifact || $('#r-artifact')?.value || '';
  const flags = recipeFlags();
  if (!artifact){
    pre.textContent = '(pick an artifact to see its recipe)';
    markCustom();
    return;
  }
  const id = state.recipeModel || artifact.split('/').pop().replace(/\.ninfer$/, '');
  const lines = [`[${id}]`, `artifact = ${artifact}`];
  for(const [k,v] of Object.entries(flags)) lines.push(`${k} = ${v}`);
  pre.textContent = lines.join('\n');
  markCustom();
  return id;
}

/** Load an artifact, read its components, and fill the form from them. */
async function loadArtifactRecipe(path){
  state.recipeModel = null;
  state.recipeArtifact = path;
  if (!path){ renderPresetPreview(); return; }
  let info;
  try{
    info = await api(`/api/convert/inspect?path=${encodeURIComponent(path)}`);
  }catch(e){
    $('#r-recipe').textContent = `could not read the artifact: ${e.message}`;
    renderPresetPreview();
    return;
  }
  const comps = info.components || {};
  const names = Object.keys(comps);
  $('#r-recipe').textContent =
    `${info.name || path.split('/').pop()} · components: ${names.join(', ') || 'none'}`;
  // If the router already serves this artifact, that entry is the truth: the
  // form mirrors it exactly, so Save without edits is a no-op. Only an artifact
  // outside the catalog gets the values derived from its components.
  let hit = null;
  try{
    const cat = await api('/api/router/catalog');
    hit = (cat.models||[]).find(m =>
      m.artifact === path || m.artifact_real === path ||
      (m.artifact||'').split('/').pop() === path.split('/').pop()) || null;
  }catch(e){ /* the catalog is optional context, not a blocker */ }

  const wanted = hit ? hit.flags || {}
                    : {...RECIPE_DEFAULTS, ...flagsFromComponents(comps)};
  $('#r-flags').querySelectorAll('[data-flag]').forEach(i=>{
    const v = wanted[i.dataset.flag.replace(/^--/, '')];
    if (i.type === 'checkbox') i.checked = v === 'true' || v === 'on';
    else i.value = v == null ? '' : v;
  });
  if (hit){
    state.recipeModel = hit.id;
    state.recipeFlags0 = { ...(hit.flags || {}) };
    // Keep the catalog's artifact path (the symlink under --models-dir): that is
    // what the existing sections use, and rewriting it to the real path would be
    // a silent change of convention.
    state.recipeArtifact = hit.artifact || path;
    $('#rec-note').textContent = `editing ${hit.id} (alias: ${hit.alias || 'none'})`;
  } else {
    state.recipeFlags0 = {};
    $('#rec-note').textContent = 'artifact not in the router catalog; Save will add it';
  }
  renderPresetPreview();
}

/** Poll the catalog until the router answers again after a restart. */
async function waitForRouter(timeoutMs){
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline){
    try{
      // /api/live answers 200 even when the engine is down; its own `online`
      // flag is what says the router is serving.
      const r = await fetch('/api/live');
      const body = r.ok ? await r.json() : null;
      if (body?.online) return true;
    }catch(e){ /* not up yet */ }
    await new Promise(res => setTimeout(res, 500));
  }
  return false;
}

async function savePreset(thenLaunch){
  const artifact = state.recipeArtifact || $('#r-artifact').value;
  if (!artifact){ flash('Pick an artifact first.', true); return; }
  const body = { artifact, flags: recipeFlags() };
  const id = state.recipeModel || artifact.split('/').pop().replace(/\.ninfer$/, '');
  body.model = id;
  // A restart unloads every model, so only do it when the flags really differ
  // from what the preset already has.
  const before = (state.recipeFlags0 || {});
  const changed = JSON.stringify(Object.entries(body.flags).sort())
               !== JSON.stringify(Object.entries(before).sort());
  let r;
  try{
    r = await api('/api/router/save', { method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
  }catch(e){
    flash('Preset not saved: ' + e.message, true);
    return;
  }
  flash(`Preset saved for ${id}. ` +
    (changed ? 'Restart the router to apply it.' : 'No change.'), false);
  if (!thenLaunch) return;
  // The router read the preset at startup, so launching before a restart would
  // load the model with the OLD flags: restart first, then load.
  if (changed){
    try{
      const rs = await api('/api/router/restart', { method:'POST',
        headers:{'Content-Type':'application/json'}, body:JSON.stringify({}) });
      if (!rs.ok){ flash('Saved, but the restart failed: ' + (rs.error||rs.stderr||'?'), true); return; }
      // systemctl returns as soon as the unit is up; the socket is not serving
      // yet, and a load sent now gets a connection refused (502 through here).
      await waitForRouter(30000);
    }catch(e){ flash('Saved, but the restart failed: ' + e.message, true); return; }
  }
  try{
    await api('/api/control', { method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ action:'load', model:id }) });
    flash(`Saved and launched ${id}${changed ? ' with the new flags' : ''}.`, false);
    loadControl();
  }catch(e){
    flash('Saved, but the launch failed: ' + e.message, true);
  }
}

