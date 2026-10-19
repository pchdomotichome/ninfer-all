import { $, state, esc, api } from './core.js';

// The Fit view answers one question: does this configuration fit on this card?
//
// Every constant comes from fit-data.json, copied from the measured DATA block of
// docs/config-calculator.html (its `source` field says so). Nothing here measures anything:
// the arithmetic only combines those constants, so if the two ever disagree, regenerate
// fit-data.json from that page rather than editing it.
//
// The one thing worth saying twice: KV cost per token comes from the model's ARCHITECTURE, not
// from its quantisation. Weights scale with the quant, the cache does not. That is why the
// profile is the control that has to be right, and why a wrongly matched profile gives a wrong
// verdict that still looks plausible.

const GIB = 1024 ** 3, MIB = 1024 ** 2;

export function pageRoundUp(tokens, page) {
  return Math.ceil(tokens / page) * page;
}

// Same shape as the calculator: a byte count, or [limit, bytes] steps read as "up to this
// context", with null standing for an open-ended last step. Only three contexts were measured
// for the stepped cases, so a value between two measured points takes the higher one.
export function graphBytes(spec, ctx) {
  const g = spec.graphBytes;
  if (typeof g === 'number') return g;
  for (const [limit, bytes] of g) if (limit === null || ctx <= limit) return bytes;
  return g[g.length - 1][1];
}

/* Cache bytes per token: the format's own cost plus whatever the speculation backend adds to
   it. MTP buffers its extra draft tokens *in the KV format*, so its term is a ratio of the
   format's bytes; DFlash instead pays a fixed 4,096 B/token, which is why modelling that as a
   ratio is what made it look format-dependent. */
export function kvPerToken(profile, kv, specKey) {
  const per = profile.kv[kv];
  if (per === null || per === undefined) return null;
  const spec = profile.spec[specKey];
  return per * spec.kvRatio + spec.kvExtraPerToken;
}

/* Everything resident that does not scale with the requested context, evaluated at that
   context: weights plus what the speculative heads add, the workspace and CUDA graphs, and the
   fixed part of the sequence state. Taking the artifact's real size rather than the profile's
   measured reference quant is the whole point of the prober. */
export function fixedBytes(profile, specKey, ctx, weightsBytes) {
  const spec = profile.spec[specKey];
  return weightsBytes + (spec.weightsGiB - profile.spec.none.weightsGiB) * GIB +
         profile.workspaceBytes + graphBytes(spec, ctx) + spec.seqFixedBytes;
}

/* The whole budget for one configuration. `usable` is the VRAM the model may spend; the caller
   decides what that means (free VRAM minus whatever reserve it was told to keep). */
export function budget(data, profile, specKey, kv, ctx, weightsBytes, usable) {
  const spec = profile.spec[specKey];
  const per = kvPerToken(profile, kv, specKey);
  const haveKv = per !== null;
  /* KV is reserved by the physical page the requested context rounds up to, not by the exact
     token count (paged_kv_cache.h, layouts_impl.h), so charging ctx * bytes understates the
     reservation by up to one page. */
  const kvBytes = haveKv ? per * pageRoundUp(ctx, data.pageTokens) : 0;
  const seqBytes = haveKv ? spec.seqFixedBytes + spec.seqPerToken * ctx : 0;
  const weights = weightsBytes + (spec.weightsGiB - profile.spec.none.weightsGiB) * GIB;
  const workspaceBytes = profile.workspaceBytes + graphBytes(spec, ctx);
  const total = weights + kvBytes + seqBytes + workspaceBytes;
  return {spec, haveKv, kvBytes, seqBytes, weightsBytes: weights, workspaceBytes,
          total, free: usable - total};
}

/* The largest context that fits, at a page boundary -- and therefore always achievable: the
   engine rounds a page-partial request *up*, past what would have been checked, so the only
   safe answer is the largest page that fits in full. Within one page the reservation is fixed,
   so the hardest token count in it is its last one.
     The fixed part was measured at the displayed context, but the graph allowance steps with
   context, so a value solved that way could cross a step, have been computed against the
   smaller allowance, and come back too large. Re-solve against the candidate and repeat: a
   bigger allowance only ever shrinks the answer, so this converges in as many passes as there
   are steps. */
export function maxContext(data, profile, specKey, kv, weightsBytes, usable) {
  const per = kvPerToken(profile, kv, specKey);
  if (per === null || per <= 0) return null;
  const page = data.pageTokens;
  let solveFixed = fixedBytes(profile, specKey, profile.nativeMaxContext, weightsBytes);
  let best = 0;
  for (let i = 0; i < 8; i++) {
    const room = usable - solveFixed;
    const candidate = Math.max(0, page * Math.floor(room / (page * per)));
    const candidateFixed = fixedBytes(profile, specKey, candidate, weightsBytes);
    if (candidate === best && candidateFixed === solveFixed) { best = candidate; break; }
    best = candidate;
    solveFixed = candidateFixed;
  }
  return Math.min(best, profile.nativeMaxContext);
}

// ── Formatting ────────────────────────────────────────────────────────────────
const giB = b => (b / GIB).toFixed(2) + ' GiB';
const miB = b => Math.round(b / MIB).toLocaleString('en-US') + ' MiB';
const toks = n => Math.round(n).toLocaleString('en-US');
const signed = (v, d = 4) => (v > 0 ? '+' : '') + v.toFixed(d);

// ── The view ──────────────────────────────────────────────────────────────────

// Weights live off the GPU in exactly these two modes, which is the whole subject of the host
// RAM warning below.
const HOST_BACKED = {
  host: 'pinned host memory, with a device cache of the most used banks',
  disk: "the artifact's files, read into a device cache through the OS page cache",
};
const SEGS = [
  ['--blue', 'Weights', b => b.weightsBytes],
  ['--teal', 'KV cache', b => b.kvBytes],
  ['--cyan', 'Sequence state', b => b.seqBytes],
  ['--purple', 'Workspace + graphs', b => b.workspaceBytes],
];

/* The profile is the one control where a wrong value is invisible in the output, so a model
   whose id or artifact does not name exactly one measured profile is called out instead of
   guessed at: "27b" in a name is a hint, not a promise about the architecture. */
export function profileFor(data, model) {
  if (!model) return null;
  const hay = `${model.id} ${model.artifact_real || model.artifact || ''}`.toLowerCase();
  const hits = Object.keys(data.models).filter(k => hay.includes(k));
  return hits.length === 1 ? hits[0] : null;
}

/* The measured spec rows are named after the flags that produce them -- `mtp3` is `--spec mtp
   --draft-tokens 3`, `dflash2-7` is `--spec dflash2 --draft-tokens 7`, and `+head` is
   `--lm-head-draft` on top -- so a model's own flags select its row. A combination nobody
   measured falls back to none, which is the baseline, rather than to the nearest guess. */
export function specFor(profile, flags) {
  flags = flags || {};
  const n = Number(flags['draft-tokens']);
  const base = !flags.spec ? 'none'
    : !n ? flags.spec
    : flags.spec === 'mtp' ? `mtp${n}` : `${flags.spec}-${n}`;
  const key = base + (String(flags['lm-head-draft']) === 'true' ? '+head' : '');
  return profile.spec[key] ? key : 'none';
}

export async function loadFit() {
  if (!state.fit) state.fit = {open: null, hf: null, wired: false, profile: null};
  const ui = state.fit;
  /* Wired before anything is fetched, and the fetches kept out of the wiring's scope by
     declaring their results here: a rejected handler leaves three controls that look fine and
     do nothing -- an empty list, an empty box and a dead button -- with nothing on screen to
     say why. That is exactly how the undefined `data` below reached the user. */
  if (!ui.wired) { bindFit(); ui.wired = true; }
  let data, cat, gpu;
  try {
    [data, cat, gpu] = await Promise.all([
      api('/api/fit/data'), api('/api/router/catalog'), api('/api/gpu'),
    ]);
    ui.data = data;
    ui.catalog = cat.models || [];
    ui.gpus = gpu.gpus || [];
    ui.host = gpu.host || {};
    // Artifact sizes, so a router model's weights are the file that will actually be mapped
    // rather than its measured reference quant.
    try {
      const files = await api('/api/convert/files');
      ui.sizes = Object.fromEntries((files.artifacts || []).map(a => [a.path, a.size]));
    } catch (e) { ui.sizes = {}; }
  } catch (err) {
    // Without constants there is no arithmetic to show, and a rejected handler would leave the
    // section silently empty.
    $('#fit-out').innerHTML = `<div class="err">The Fit constants could not be read:
      ${esc(err.message)}</div>`;
    return;
  }
  if (ui.open === null) ui.open = ui.catalog.length ? ui.catalog[0].id : '';
  // First visit only: after that the controls are the user's, not ours.
  if (!ui.profile) suggestFit(ui, data);
  if (ui.gpu === undefined && ui.gpus.length) {
    // The card with the most room, since that is the one worth asking the question about.
    ui.gpu = ui.gpus.reduce((a, b) =>
      (b.mem_total_mib - b.mem_used_mib) > (a.mem_total_mib - a.mem_used_mib) ? b : a).index;
  }
  fillControls(data);
  renderFit();
}

/* The model's own flags pick the profile and the spec row: a model whose flags say
   `--spec mtp --draft-tokens 3` is sized as mtp3, not as the baseline. */
function suggestFit(ui, data) {
  const cur = ui.catalog.find(m => m.id === ui.open);
  ui.profile = profileFor(data, cur) || ui.profile || Object.keys(data.models)[0];
  ui.spec = specFor(data.models[ui.profile], cur && cur.flags);
}

function bindFit() {
  $('#fit-src').onchange = e => {
    const ui = state.fit;
    ui.open = e.target.value; ui.hf = null;
    $('#fit-size').value = '';
    suggestFit(ui, ui.data);
    fillControls(ui.data);
    renderFit();
  };
  $('#fit-profile').onchange = e => {
    const ui = state.fit;
    ui.profile = e.target.value;
    // The spec rows are per profile -- dflash2-7 on one, dflash-3 on another -- so the row that
    // was selected may not exist here.
    suggestFit(ui, ui.data);
    fillControls(ui.data);
    renderFit();
  };
  for (const id of ['#fit-spec', '#fit-kv', '#fit-gpu']) {
    $(id).onchange = e => { state.fit[id.slice(5)] = e.target.value; renderFit(); };
  }
  $('#fit-ctx').oninput = e => { state.fit.ctx = Number(e.target.value); renderFit(); };
  $('#fit-reserve').oninput = e => { state.fit.reserve = Number(e.target.value); renderFit(); };
  $('#fit-size').oninput = () => renderFit();
  $('#fit-hf-go').onclick = hfProbe;
  $('#fit-hf-repo').onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); hfProbe(); } };
}

function fillControls(data) {
  const ui = state.fit;
  $('#fit-src').innerHTML = (ui.catalog.length
      ? `<option value="">— pick a model —</option>`
      : `<option value="">the router catalog is empty — is the router running?</option>`) +
    ui.catalog.map(m => {
      const size = ui.sizes[m.artifact_real || m.artifact];
      const p = profileFor(data, m);
      return `<option value="${esc(m.id)}">${esc(m.id)}${size ? ' · ' + giB(size) : ''}${
        p ? ' · ' + p : ''}</option>`;
    }).join('');
  $('#fit-src').value = ui.open;

  $('#fit-profile').innerHTML = Object.entries(data.models)
    .map(([k, p]) => `<option value="${esc(k)}">${esc(k)} — ${esc(p.label || k)}</option>`).join('');
  $('#fit-profile').value = ui.profile;

  const profile = data.models[ui.profile];
  $('#fit-spec').innerHTML = Object.entries(profile.spec)
    .map(([k, s]) => `<option value="${esc(k)}">${esc(s.label || k)}</option>`).join('');
  if (!profile.spec[ui.spec]) ui.spec = Object.keys(profile.spec)[0];
  if (ui.spec) $('#fit-spec').value = ui.spec;
  ui.spec = $('#fit-spec').value;

  const formats = data.kvOrder.filter(k => profile.kv[k] !== null && profile.kv[k] !== undefined);
  $('#fit-kv').innerHTML = formats.map(k => `<option value="${esc(k)}">${esc(k)}</option>`).join('');
  if (!formats.includes(ui.kv)) ui.kv = 'int8';
  $('#fit-kv').value = ui.kv;
  ui.kv = $('#fit-kv').value;

  $('#fit-gpu').innerHTML = ui.gpus.length
    ? ui.gpus.map(g => `<option value="${g.index}">GPU ${g.index} · ${esc(g.name)} · ${
        giB((g.mem_total_mib - g.mem_used_mib) * MIB)} free</option>`).join('')
    : '<option value="">no GPU</option>';
  if (ui.gpu !== undefined && ui.gpus.some(g => String(g.index) === String(ui.gpu))) {
    $('#fit-gpu').value = ui.gpu;
  }
  ui.gpu = $('#fit-gpu').value;

  if (!ui.ctx) ui.ctx = 32768;
  if (!ui.reserve) ui.reserve = 1024;
  $('#fit-ctx').value = ui.ctx;
  $('#fit-reserve').value = ui.reserve;
}

// The weights size for the verdict: whatever is typed wins, so the prober and the verdict are
// the same control rather than two code paths that can disagree.
function weightsBytes(ui) {
  const typed = Number($('#fit-size').value);
  if (typed > 0) return typed * GIB;
  const cur = (ui.catalog || []).find(m => m.id === ui.open);
  const onDisk = cur && ui.sizes && ui.sizes[cur.artifact_real || cur.artifact];
  return onDisk || null;
}

async function hfProbe() {
  const ui = state.fit;
  const repo = $('#fit-hf-repo').value.trim();
  if (!repo) return;
  if (!ui.data) {
    $('#fit-hf-out').innerHTML = `<div class="err">The Fit constants are not loaded, so there is
      nothing to size these files against.</div>`;
    return;
  }
  $('#fit-hf-out').innerHTML = '<div class="note" style="padding:10px 0 4px">reading the Hub…</div>';
  try {
    // Only the weights size comes from the repo. The cache cost never does: it is the profile's.
    ui.hf = await api(`/api/convert/ggufs?model=${encodeURIComponent(repo)}`);
    ui.hfRepo = repo;
  } catch (e) {
    ui.hf = {error: e.message};
  }
  // Whatever went wrong, it has to end up on screen: a render that throws here used to leave
  // "reading the Hub…" as the only thing on screen, which reads exactly like doing nothing.
  try {
    renderFit();
  } catch (err) {
    $('#fit-hf-out').innerHTML = `<div class="err">${esc(err.message)}</div>`;
  }
}

function renderFit() {
  const ui = state.fit, data = ui.data;
  const profile = data.models[ui.profile];
  const kv = ui.kv, specKey = ui.spec || Object.keys(profile.spec)[0];
  const ctx = Math.max(1, Number(ui.ctx) || 1);
  const reserve = Math.max(0, Number(ui.reserve) || 0) * MIB;
  const gpu = (ui.gpus || []).find(g => String(g.index) === String(ui.gpu));
  const cur = (ui.catalog || []).find(m => m.id === ui.open);
  const wb = weightsBytes(ui);
  const asked = cur && Number((cur.flags || {})['max-context'] || 0);
  const out = $('#fit-out');
  renderHfOut();

  const warn = [];
  if (cur && !profileFor(data, cur)) {
    warn.push(`<b>${esc(cur.id)}</b> names no measured profile, so it is being sized on
      <b>${esc(profile.label || ui.profile)}</b>. The cache cost below — the largest term, and the
      one that decides the verdict — is this profile's architecture: check it is the same before
      trusting the answer.`);
  }
  if (!gpu) {
    out.innerHTML = `<div class="err">No GPU. The panel reads nvidia-smi; with no card there is
      no VRAM figure to fit against.</div>`;
    return;
  }
  // The memory available is worth showing before a model is picked, so nothing here returns
  // early on a missing size: the weights simply count as zero and the verdict says so.
  const haveWeights = wb !== null;

  // Free VRAM, as the user asked: the whole card is not available when something else already
  // holds part of it. The reserve is the additional headroom refused on top of that.
  const freeVram = (gpu.mem_total_mib - gpu.mem_used_mib) * MIB;
  const usable = Math.max(0, freeVram - reserve);
  const b = budget(data, profile, specKey, kv, ctx, haveWeights ? wb : 0, usable);
  const fits = haveWeights && b.total <= usable;
  const maxCtx = haveWeights ? maxContext(data, profile, specKey, kv, wb, usable) : null;
  const ram = (ui.host || {}).ram;

  /* Host RAM is the real bound in the two modes where weights live off the GPU, and getting this
     wrong is not abstract: it is what had the OOM killer take flash-next with its experts
     mapped, while VRAM still had room. */
  const residency = cur && cur.flags && cur.flags['expert-residency'];
  if (haveWeights && residency && HOST_BACKED[residency]) {
    const fitsHost = ram ? wb <= ram.available : null;
    warn.push(`<b>Host RAM is the bound here, not VRAM.</b> This model keeps its experts in
      ${HOST_BACKED[residency]}, so the artifact has to fit in host memory too.
      ${ram ? `The host has <b>${giB(ram.available)}</b> free of ${giB(ram.total)} and this
      artifact is <b>${giB(wb)}</b>${fitsHost ? ' — it fits for now'
        : ' — <b>more than is free</b>'}.` : ''} Disk residency reads those pages through the OS
      page cache, which competes with everything else on the box: this, not the card, is what
      killed flash-next.`);
  }
  if (cur && cur.flags && cur.flags['ngram-residency'] === 'ram') {
    warn.push(`The n-gram table is loaded into host RAM (<code>--ngram-residency ram</code>) and
      is not part of the weights figure above.`);
  }

  /* The bar is drawn against the budget while the budget is the larger of the two, and against
     the total once the total has passed it: scaling every segment by the same span keeps their
     proportions honest in both states, and the red line is the budget itself. */
  const span = Math.max(usable, b.total);
  const over = b.total > usable;
  const bar = SEGS.map(([v, name, get]) => {
    const w = Math.max(0, Math.min(100, 100 * get(b) / span));
    return `<i style="width:${w}%;background:var(${v})" title="${esc(name)}: ${giB(get(b))}"></i>`;
  }).join('') + `<i style="width:${Math.max(0, Math.min(100, 100 * reserve / span))}%;
    background:var(--amber);opacity:.45" title="Reserve: ${miB(reserve)}"></i>`;
  const budgetLine = over
    ? `<b class="stack-line" style="left:${(100 * usable / span).toFixed(2)}%"
        title="the budget: ${giB(usable)}"></b>` : '';

  const ppl = data.perplexity, base = profile.kv['int8'];
  const per = kvPerToken(profile, kv, specKey);
  const atPage = pageRoundUp(ctx, data.pageTokens);

  out.innerHTML = `
    ${warn.map(w => `<div class="warn-msg">${w}</div>`).join('')}
    <div class="tiles">
      <div class="tile"><div class="k">Verdict</div>
        <div class="v" style="color:${fits ? 'var(--green)' : haveWeights ? 'var(--red)' : 'var(--dim)'}">${
          haveWeights ? (fits ? 'Fits' : 'No fit') : '—'}</div>
        <div class="s">${!haveWeights ? 'no weights size yet'
          : fits ? giB(b.free) + ' to spare' : 'over by ' + giB(-b.free)}</div>
        <div class="bar"><i style="width:${Math.min(100, 100 * b.total / usable)}%;
          background:${fits ? 'var(--green)' : haveWeights ? 'var(--red)' : 'var(--dim)'}"></i></div></div>
      <div class="tile"><div class="k">Max context at this size</div>
        <div class="v">${maxCtx ? toks(maxCtx) : '—'}</div>
        <div class="s">${!haveWeights ? 'needs a weights size'
          : per === null ? `this format has no cache to size`
          : !maxCtx ? 'nothing fits on this card'
          : maxCtx === profile.nativeMaxContext ? "the model's native limit"
          : 'largest context that fits'}${asked && maxCtx ? ` · the router asks ${toks(asked)}`
          : ''}</div></div>
      <div class="tile"><div class="k">Cache per token</div>
        <div class="v">${per === null ? '—' : toks(per) + ' B'}</div>
        <div class="s">${esc(kv)} · ${giB(b.kvBytes)} at ${toks(atPage)} tok</div></div>
      <div class="tile"><div class="k">Free on GPU ${esc(gpu.index)}</div>
        <div class="v">${giB(freeVram)}</div>
        <div class="s">of ${giB(gpu.mem_total_mib * MIB)} · keeping ${miB(reserve)} back</div>
        <div class="bar"><i style="width:${Math.min(100, 100 * freeVram / (gpu.mem_total_mib * MIB))}%;
          background:var(--teal)"></i></div></div>
    </div>

    <div class="stack">${budgetLine}${bar}</div>
    <div class="stack-legend">
      ${SEGS.map(([v, name, get]) => `<span><b style="background:var(${v})"></b>${esc(name)}
        <i>${giB(get(b))}</i></span>`).join('')}
      <span><b style="background:var(--amber);opacity:.45"></b>Reserve <i>${miB(reserve)}</i></span>
      <span><b style="background:rgba(255,255,255,.07)"></b>Free
        <i>${giB(Math.max(0, b.free))}</i></span>
    </div>
    ${haveWeights ? '' : `<div class="note" style="padding:14px 2px 0;white-space:normal">
      <b>No weights yet.</b> The card's room and the cache cost below are already the real ones;
      pick a model above, or type a size in <b>Weights, GiB</b>, and the weights join the
      budget.</div>`}

    <table style="margin-top:18px"><tbody>
      <tr><td class="l">Weights${b.weightsBytes !== wb ? ' + speculation heads' : ''}</td>
        <td>${haveWeights ? giB(b.weightsBytes) : '—'}</td>
        <td class="l muted">${!haveWeights ? 'not counted yet, no size chosen'
          : `${miB(wb)} for this quant${
            b.weightsBytes !== wb ? ', plus ' + miB(b.weightsBytes - wb) + ' of draft heads' : ''}`}</td></tr>
      <tr><td class="l">KV cache (${esc(kv)})</td><td>${giB(b.kvBytes)}</td>
        <td class="l muted">${per === null ? 'this format has no cache to size'
          : toks(per) + ' B/token × ' + toks(atPage) + ' tok (reserved by the page)'}</td></tr>
      <tr><td class="l">Sequence state, fixed part</td><td>${giB(b.seqBytes)}</td>
        <td class="l muted">${miB(b.spec.seqFixedBytes)} + ${b.spec.seqPerToken} B/token</td></tr>
      <tr><td class="l">Workspace + CUDA graphs</td><td>${giB(b.workspaceBytes)}</td>
        <td class="l muted">${miB(profile.workspaceBytes)} + ${miB(graphBytes(b.spec, ctx))}
          at this context</td></tr>
      <tr><td class="l">Reserved for the OS</td><td>${miB(reserve)}</td>
        <td class="l muted">kept back, never charged to the model</td></tr>
      <tr><td class="l"><b>Total</b></td><td><b>${giB(b.total)}</b></td>
        <td class="l muted">against ${giB(usable)} of usable room, from ${giB(freeVram)} free</td></tr>
    </tbody></table>

    <div class="sechead"><h2>Speculation, measured</h2>
      <span class="note">decode and draft acceptance for each backend on
        ${esc(profile.label || ui.profile)} — read the two columns together</span></div>
    <table><thead><tr><th class="l">Backend</th><th>Decode</th><th>Accept</th>
      <th>Room at ${toks(ctx)}</th></tr></thead><tbody>
      ${Object.entries(profile.spec).map(([k, s]) => {
        const mb = budget(data, profile, k, kv, ctx, wb, usable);
        const ok = mb.total <= usable;
        return `<tr><td class="l">${k === specKey
          ? `<span class="pill on">${esc(s.label || k)}</span>` : esc(s.label || k)}</td>
          <td>${s.decode} t/s</td>
          <td>${s.accept === null || s.accept === undefined ? '—' : s.accept.toFixed(3)}</td>
          <td style="color:${ok ? 'var(--green)' : 'var(--red)'}">${
            ok ? giB(mb.free) : 'over by ' + giB(-mb.free)}</td></tr>`;
      }).join('')}
    </tbody></table>
    <div class="note" style="padding:10px 2px 0">Accept is the share of drafted tokens that
      survive verification, so a backend with the lower accept can still win — and a backend that
      does not fit at all cannot, however fast it decodes.</div>

    <div class="sechead"><h2>KV format, measured</h2>
      <span class="note">the format is the one choice here that stays open after the model is
        loaded: it trades cache size against quality</span></div>
    <table><thead><tr><th class="l">Format</th><th>Cache/token</th><th>At ${toks(atPage)} tok</th>
      <th>Perplexity</th><th>vs int8</th><th>Cache vs int8</th><th>Room</th>
      </tr></thead><tbody>
      ${data.kvOrder.filter(k => profile.kv[k] !== null && profile.kv[k] !== undefined).map(k => {
        const mb = budget(data, profile, specKey, k, ctx, wb, usable);
        const pt = kvPerToken(profile, k, specKey);
        const ok = mb.total <= usable;
        return `<tr><td class="l">${k === kv ? `<span class="pill on">${esc(k)}</span>` : esc(k)}</td>
          <td>${toks(pt)} B</td><td>${giB(mb.kvBytes)}</td>
          <td>${ppl[k] ? ppl[k].toFixed(4) : '—'}</td>
          <td>${ppl[k] && ppl['int8'] ? signed(ppl[k] - ppl['int8']) : '—'}</td>
          <td>${pt === base ? 'the baseline' :
            (pt < base ? '' : '+') + ((pt / base - 1) * 100).toFixed(1) + '%'}</td>
          <td style="color:${ok ? 'var(--green)' : 'var(--red)'}">${
            ok ? giB(mb.free) : 'over by ' + giB(-mb.free)}</td></tr>`;
      }).join('')}
    </tbody></table>
    <div class="note" style="padding:10px 2px 0;white-space:normal"><b>${
      esc(data.kvLabels[kv] || kv)}</b> — ${esc(data.kvNotes[kv] || '')}</div>`;
}

// One row per GGUF in the repo: the question the prober answers is which quant of it reaches
// which context, so the sizes are read straight from the Hub and only the weights size is used.
function renderHfOut() {
  const ui = state.fit, hf = ui.hf;
  if (!hf) return;
  $('#fit-hf-out').innerHTML = hf.error
    ? `<div class="err">${esc(hf.error)}</div>`
    : `<div class="note" style="padding:10px 0 4px">${esc(ui.hfRepo)} · ${
        hf.length} GGUF files, sizes read from the Hub — only the weights size is taken from a
        repo, the cache cost is still ${esc((ui.data.models[ui.profile] || {}).label || ui.profile)}</div>
      <table><thead><tr><th class="l">Quant</th><th class="l">File</th><th>Weights</th>
        <th>Max context</th><th>Margin</th></tr></thead><tbody>
      ${hf.slice(0, 60).map(f => hfRow(ui, f)).join('')}</tbody></table>`;
}

function hfRow(ui, f) {
  const data = ui.data, profile = data.models[ui.profile];
  const gpu = (ui.gpus || []).find(g => String(g.index) === String(ui.gpu));
  if (!gpu || !f.size) return '';
  const reserve = Math.max(0, Number(ui.reserve) || 0) * MIB;
  const usable = Math.max(0, (gpu.mem_total_mib - gpu.mem_used_mib) * MIB - reserve);
  const max = maxContext(data, profile, ui.spec, ui.kv, f.size, usable);
  const set = ui.ctx || 32768;
  const b = budget(data, profile, ui.spec, ui.kv, set, f.size, usable);
  return `<tr><td class="l">${esc(f.quant)}</td>
    <td class="l muted">${esc(f.path.split('/').pop().slice(0, 46))}</td>
    <td>${giB(f.size)}</td>
    <td>${max === null ? '—' : toks(max)}</td>
    <td style="color:${b.total <= usable ? 'var(--green)' : 'var(--red)'}">${
      b.total <= usable ? giB(b.free) : 'over by ' + giB(-b.free)}</td></tr>`;
}
