import { $, state, esc, num, api, flash } from './core.js';
import { modelRows, wireModelRows, renderCtlRows, favs } from './models.js';
import { fmtSize } from './convert.js';
import { liveModels } from './live.js';

export async function loadControl(){
  const body = $('#ctl-body');
  try{
    // /api/live en vez de /api/engine: status.args trae los flags del modelo
    // (ctx, spec, draft) y la misma fuente alimenta el Dashboard, así que hay
    // una sola renderer de filas en vez de dos.
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
      host.textContent = `El servidor ${live.url || ''} no responde.`;
      $('#ctl-note').textContent = 'sin conexión';
    } else {
      host.innerHTML = modelRows(live, null, true);
      const nFav = ((live.models?.data)||[]).filter(m=>favs.has(m.id)).length;
      $('#ctl-note').textContent = `${n} modelos` + (nFav ? ` · ${nFav} favoritos` : '');
      $('#ctl-upd').textContent = 'actualizado ' + new Date().toLocaleTimeString('es-AR',{hour12:false});
      wireModelRows(host, live);
    }
    // recipe form: dropdown de artifacts + flags
    if (h.flags && h.flags.length){
      $('#rec-note').textContent = `${h.flags.length} flags leídas de ${h.binary}`;
      let files = {artifacts:[]};
      try{ files = await api('/api/convert/files'); }catch(err){}
      $('#r-artifact').innerHTML = '<option value="">— elegir artifact —</option>' +
        files.artifacts.map(a=>`<option value="${esc(a.path)}">
          ${esc(a.path.split('/').pop())} (${fmtSize(a.size)})</option>`).join('');
      renderRecipeFlags(h.flags);
    } else {
      $('#r-flags').innerHTML = '<div class="err">No se pudo leer --help del binario.</div>';
    }
    // endpoint para agentes: base url + modelo activo del catálogo
    const base = state.cfg?.endpoint?.base_url || '';
    if (base){
      const loaded = liveModels(live).map(m=>m.id);
      const model = loaded.length ? loaded.join(', ') : ((live.models?.data||[])[0]?.id || '');
      const key = state.cfg.endpoint.api_key || '(sin api key)';
      $('#ep-kv').innerHTML = `
        <dt>base_url</dt><dd id="ep-base">${esc(base)}</dd>
        <dt>model</dt><dd id="ep-model">${esc(model)}</dd>
        <dt>api_key</dt><dd>${esc(key)}</dd>
        <dt>modelos</dt><dd>${(live.models?.data||[]).length} en catálogo</dd>`;
      const activeModel = model.split(',')[0].trim() || 'qwen35-9b';
      const epSnippets = {
        curl: `# OpenAI-compatible cURL\n` +
              `curl ${base}/chat/completions \\\n` +
              `  -H 'Content-Type: application/json' \\\n` +
              (state.cfg.endpoint.api_key ? `  -H 'Authorization: Bearer ${state.cfg.endpoint.api_key}' \\\n` : '') +
              `  -d '{"model":"${activeModel}","messages":[{"role":"user","content":"hola"}],"max_tokens":128}'`,
        python: `# Python (openai SDK)\nfrom openai import OpenAI\n\n` +
                `client = OpenAI(base_url="${base}", api_key="${state.cfg.endpoint.api_key || 'dummy'}")\n\n` +
                `resp = client.chat.completions.create(\n` +
                `    model="${activeModel}",\n` +
                `    messages=[{"role": "user", "content": "Hola ninfer!"}],\n` +
                `    max_tokens=128\n)\nprint(resp.choices[0].message.content)`,
        js: `// TypeScript / Node (openai SDK)\nimport OpenAI from "openai";\n\n` +
            `const client = new OpenAI({ baseURL: "${base}", apiKey: "${state.cfg.endpoint.api_key || 'dummy'}" });\n\n` +
            `const res = await client.chat.completions.create({\n` +
            `  model: "${activeModel}",\n` +
            `  messages: [{ role: "user", content: "Hola ninfer!" }],\n` +
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
          .then(()=>flash('Código copiado al portapapeles', false))
          .catch(()=>flash('No se pudo copiar', true));
      };
    }
  }catch(e){
    body.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}

function renderRecipeFlags(flags){
  // 154 flags a la vez es ruido: los relevantes primero, el resto bajo "más".
  // Se agrupan por categoría según el prefijo del flag (modelo, contexto,
  // spec-decode, etc.) para que el formulario sea navegable.
  const GROUPS = [
    {name:'modelo', re:/^--(model-id|models|chat-template|mmproj|vision|expert)/},
    {name:'contexto y kv', re:/^--(max-context|kv-|host-|device-|prefill|context-)/},
    {name:'spec-decode', re:/^--(spec|draft|lm-head|adaptive|ngram|verbose|mtp)/},
    {name:'ejecución', re:/^--(max-concurrency|max-pending|sleep-idle|unload-idle|model-suspend|no-models)/},
    {name:'salida y api', re:/^--(api-key|host|port|cors|no-webui|response|revision)/},
  ];
  const byFlag = Object.fromEntries(flags.map(f=>[f.flag, f]));
  const vals = { '--spec':'mtp','--draft-tokens':'3','--kv-capacity':'32768',
    '--max-context':'32768','--max-concurrency':'2' };
  const mk = f => `<label class="f" style="margin-bottom:6px">${esc(f.flag.slice(2))}
    <input data-flag="${esc(f.flag)}" value="${esc(vals[f.flag] ?? '')}"
      placeholder="${esc((f.desc||'').slice(0,60))}"></label>`;
  let html = '';
  for(const g of GROUPS){
    const inGroup = flags.filter(f=>g.re.test(f.flag));
    if (!inGroup.length) continue;
    html += `<details ${g.name==='modelo'?'open':''}>
      <summary style="cursor:pointer;color:var(--teal);font:600 11px system-ui;
        text-transform:uppercase;letter-spacing:.6px;margin:8px 0 5px">${g.name}
        (${inGroup.length})</summary>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;margin-bottom:6px">
      ${inGroup.map(mk).join('')}
      </div></details>`;
  }
  $('#r-flags').innerHTML = html;
  const preview = () => {
    const lines = [];
    const seen = new Set();
    $('#r-flags').querySelectorAll('input[data-flag]').forEach(i=>{
      const v = i.value.trim();
      if (!v || seen.has(i.dataset.flag)) return;  // un flag puede existir dos veces en --help
      seen.add(i.dataset.flag);
      lines.push(`${i.dataset.flag} = ${v}`);
    });
    $('#r-preview').textContent = lines.join('\n') || '(sin flags)';
  };
  $('#r-flags').addEventListener('input', preview); preview();
}

