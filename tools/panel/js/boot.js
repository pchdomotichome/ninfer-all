import { $, state, esc, num, api, flash } from './core.js';
import { loadConvertFiles, hfOpen, hfClose, hfSearch, inspectArtifact, runConvert, wireHfSearch } from './convert.js';
import { initCharts, loadLive } from './live.js';
import { loadRequests, toggleReqCols } from './requests.js';
import { load, runBench, connect, loadEngine, showCompareModal, exportCsv } from './bench.js';
import { loadControl } from './control.js';
import { loadRouter } from './router.js';
import { loadProbe, wireProbe } from './probe.js';
import { loadRank, wireRank } from './rank.js';

// Bench is one page with three views, not three pages: the subnav switches which
// container is visible inside #pg-bench. Clicking the group reveals it and, when
// it was closed, lands on the view the user last had.
export function initNav(){
  const nav = $('#nav');
  const benchSub = $('#subnav-bench');
  const show = (pg, sub) => {
    document.querySelectorAll('.pg').forEach(p=>p.classList.toggle('active', p.id==='pg-'+pg));
    if (pg === 'bench'){
      document.querySelectorAll('#pg-bench .subpg')
        .forEach(d=>d.classList.toggle('active', d.dataset.subpg===sub));
      benchSub.querySelectorAll('button')
        .forEach(x=>x.classList.toggle('active', x.dataset.sub===sub));
    }
    nav.querySelectorAll(':scope > button')
      .forEach(x=>x.classList.toggle('active', x.dataset.pg===pg));
  };
  state.benchSub = state.benchSub || 'aiperf';
  nav.querySelectorAll('button[data-pg]').forEach(b=>b.onclick=()=>{
    if (b.dataset.sub){ state.benchSub = b.dataset.sub; show('bench', b.dataset.sub);
      if (b.dataset.sub==='probe') loadProbe();
      if (b.dataset.sub==='rank') loadRank();
      return; }
    show(b.dataset.pg);
    if (b.dataset.pg === 'router'){ loadRouter(); }
    if (b.dataset.sub === 'probe'){ loadProbe(); }
    if (b.dataset.sub === 'rank'){ loadRank(); }
    if (b.dataset.pg === 'bench'){
      const open = benchSub.hidden;
      benchSub.hidden = !open;
      b.classList.toggle('grouped', open);
      if (open) show('bench', state.benchSub);
    }
  });
  return show;
}


// ---------------------------------------------------------------- boot
async function boot(){
  initNav();
  try{
    state.cfg = await api('/api/config');
  }catch(e){
    $('#banner').innerHTML = `<div class="err">Could not read the configuration: ${esc(e.message)}</div>`;
    return;
  }
  const a = state.cfg.aiperf;
  $('#hdrsub').textContent = a.ready ? ('aiperf ' + a.why) : 'AIPerf no disponible';
  const pf=$('#f-profile');
  pf.innerHTML = state.cfg.benchmarks.map(b=>
    `<option value="${b.id}">${esc(b.label)}</option>`).join('');
  const note = state.cfg.benchmarks.find(b=>b.id===pf.value)?.note;
  $('#benchnote').textContent = note||'';
  const syncDatasets=()=>{
    const b = state.cfg.benchmarks.find(x=>x.id===pf.value);
    $('#f-dataset').innerHTML = (b?.datasets||[]).map(d=>`<option>${esc(d)}</option>`).join('');
    if (b?.concurrency?.length) $('#f-conc').value = b.concurrency[0];
  };
  pf.onchange = syncDatasets; syncDatasets();

  initCharts();
  connect();
  loadEngine();
  loadLive();
  loadControl();
  loadConvertFiles();
  loadRouter();
  load();
  loadRequests();

  $('#q-reqmore').onclick = toggleReqCols;

  $('#btn-compare').onclick = showCompareModal;
  $('#btn-export-csv').onclick = exportCsv;
  $('#modal-compare-close').onclick = () => { $('#modal-compare').style.display = 'none'; };
  $('#modal-compare').onclick = (e) => { if (e.target.id === 'modal-compare') $('#modal-compare').style.display = 'none'; };
  // Buscador HF
  $('#c-hf').onclick = hfOpen;
  $('#modal-hf-close').onclick = hfClose;
  $('#modal-hf').onclick = (e) => { if (e.target.id === 'modal-hf') hfClose(); };
  // Model config (read-only; see openModelCfg)
  $('#modal-cfg-close').onclick = () => { $('#modal-cfg').style.display = 'none'; };
  $('#modal-cfg').onclick = (e) => { if (e.target.id === 'modal-cfg') $('#modal-cfg').style.display = 'none'; };
  wireHfSearch();
  wireProbe();
  wireRank();
  $('#q-more').onclick = e=>{ state.more=!state.more;
    e.target.textContent = state.more?'Fewer columns':'More columns'; load(); };
  ['#q-profile','#q-engine','#q-spec','#q-dataset'].forEach(s=>$(s).onchange=()=>{
    state.page = 1; load();
  });
  $('#btn-run').onclick = runBench;
  $('#btn-stop').onclick = async()=>{
    if(!state.job) return;
    await api('/api/bench/stop',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({run_id:state.job.run_id})});
  };
  $('#c-inspect').onclick = inspectArtifact;
  $('#c-run').onclick = runConvert;
}

boot();
setInterval(()=>{ if(!state.live) load(); }, 5000);
setInterval(()=>{ if(document.visibilityState==='visible') loadLive(); }, 2000);
setInterval(()=>{ if(document.visibilityState==='visible') loadRequests(); }, 3000);
