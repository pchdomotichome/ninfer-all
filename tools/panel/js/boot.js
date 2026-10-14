import { $, state, esc, num, api, flash } from './core.js';
import { loadConvertFiles, hfOpen, hfClose, hfSearch, inspectArtifact, runConvert, wireHfSearch } from './convert.js';
import { initCharts, loadLive } from './live.js';
import { loadRequests, toggleReqCols } from './requests.js';
import { load, runBench, connect, loadEngine, showCompareModal, exportCsv } from './bench.js';
import { loadControl } from './control.js';
import { loadRouter } from './router.js';

export function initNav(){
  $('#nav').querySelectorAll('button').forEach(b=>b.onclick=()=>{
    $('#nav').querySelectorAll('button').forEach(x=>x.classList.toggle('active', x===b));
    document.querySelectorAll('.pg').forEach(p=>p.classList.toggle('active',
      p.id==='pg-'+b.dataset.pg));
    if (b.dataset.pg === 'router') { loadRouter(); }
  });
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
