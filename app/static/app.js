'use strict';
const $=id=>document.getElementById(id);
const state={report:null,threshold:0,upload:null,requestId:0,explainId:0,timer:null};
const pct=x=>x==null?'—':(x*100).toFixed(1)+'%';
const number=x=>Number(x).toLocaleString();
function text(id,value){$(id).textContent=value}
function make(tag,content,cls){const node=document.createElement(tag);if(content!=null)node.textContent=String(content);if(cls)node.className=cls;return node}
async function read(response){const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'The request could not be completed.');return data}
function setThreshold(t){state.threshold=t;$('threshold').value=String(t);text('thresholdValue',t.toFixed(6));}
function renderRows(data,uploaded){
  const tbody=$('queueRows');tbody.replaceChildren();
  for(const r of data.rows){
    const tr=document.createElement('tr');
    const id=make('td',r.transaction_id);if(r.timestamp)id.append(make('span',r.timestamp.slice(0,10),'sub'));tr.append(id);
    tr.append(make('td',Number(r.amount).toFixed(2)));
    const context=make('td',r.merchant_category);context.append(make('span',`${r.new_device?'New device':'Known device'} · ${r.country_mismatch?'Country differs':'Country matches'}`,'sub'));tr.append(context);
    tr.append(make('td',Number(r.score).toFixed(6)));
    const status=make('td');status.append(make('span',r.flagged?'Review':'Below cutoff','pill'+(r.flagged?' review':'')));tr.append(status);
    const label=make('td');label.append(make('span',uploaded?'Not supplied':(r.is_fraud?'Positive':'Negative'),'pill'+(!uploaded&&r.is_fraud?' positive':'')));tr.append(label);const actions=make('td');const button=make('button','Inspect','explain-btn');button.type='button';button.setAttribute('aria-label',`Inspect score for ${r.transaction_id}`);button.addEventListener('click',()=>explain(r));actions.append(button);tr.append(actions);tbody.append(tr);
  }
  const total=uploaded?data.total:data.metrics.total;
  text('rowCount',`${number(data.rows.length)} / ${number(total)} shown`);
  text('queueTitle',uploaded?'Your CSV · highest scores':'Highest-scored transactions');
  text('queueSubtitle',uploaded?'Fictional uploaded data · ordered by score · no outcome labels':'Holdout data · ordered by score · labels are simulated');
  text('queueFoot',uploaded?`${number(data.flagged)} of ${number(total)} uploaded rows meet the cutoff. Showing up to ${data.shown_limit}; scores are rounded for display. Scores are rounded for display. Scroll the table horizontally if needed.`:`Showing up to ${data.shown_limit} of ${number(total)} test transactions, including rows below the cutoff when applicable. Scores are rounded for display. Scroll the table horizontally if needed.`);
}
async function explain(row){
  const ticket=++state.explainId;
  const panel=$('explanation');panel.hidden=false;panel.replaceChildren(make('p','Calculating score sensitivity…'));
  try{
    const data=await read(await fetch('/api/explain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(row)}));
    if(ticket!==state.explainId)return;
    panel.replaceChildren();panel.append(make('h3',`Score sensitivity · ${row.transaction_id}`));
    panel.append(make('p',`Score ${data.score.toFixed(6)}. Each change below replaces one value with a training-set reference while holding the other inputs fixed.`));
    const list=make('ul');
    for(const item of data.effects.slice(0,3)){
      const delta=item.score_difference;
      list.append(make('li',`${item.feature.replaceAll('_',' ')}: ${item.value} vs reference ${Number.isInteger(item.reference)?item.reference:(typeof item.reference==='number'?item.reference.toFixed(2):item.reference)} → score difference ${delta>=0?'+':''}${delta.toFixed(4)}`));
    }
    panel.append(list,make('p','A positive difference means the current value yields a higher score than its reference replacement. These are model sensitivities, not causal reasons, proof of fraud, or SHAP values. Replacements may be unrealistic.'));panel.scrollIntoView({block:'nearest',behavior:'auto'});
  }catch(e){if(ticket===state.explainId)panel.replaceChildren(make('p',`Could not inspect this score: ${e.message}`))}
}

async function refresh(){
  const ticket=++state.requestId, cutoff=state.threshold;
  try{
    const holdout=await read(await fetch(`/api/holdout?threshold=${cutoff}&limit=25`));
    if(ticket!==state.requestId)return;
    const m=holdout.metrics;
    text('workload',pct(m.review_rate));text('workloadDetail',`${number(m.selected)} of ${number(m.total)} transactions`);
    text('precision',pct(m.precision));text('recall',pct(m.recall));text('missed',number(m.fn));
    text('counts',`${number(m.tp)} true positives · ${number(m.fp)} false positives · ${number(m.fn)} false negatives · ${number(m.tn)} true negatives`);
    let uploaded=null;
    if(state.upload){const fd=new FormData();fd.append('file',state.upload);uploaded=await read(await fetch(`/api/score?threshold=${cutoff}`,{method:'POST',body:fd}));}
    if(ticket!==state.requestId)return;
    renderRows(uploaded||holdout,!!uploaded);state.explainId++;$('explanation').hidden=true;
    text('uploadStatus',uploaded?`${state.upload.name}: scored ${number(uploaded.total)} rows locally.`:'');
    text('status','');
  }catch(err){if(ticket!==state.requestId)return;text('status',err.message);if(state.upload){text('uploadStatus',err.message);$('queueRows').replaceChildren();text('queueTitle','CSV could not be scored');text('queueSubtitle','Choose a valid fictional CSV or show the synthetic holdout.');text('rowCount','0 shown');text('queueFoot','Previous results were cleared.');state.explainId++;$('explanation').hidden=true;}}
}
function describe(report){
  text('modelName',report.model);text('splitCounts',`${number(report.split.train.rows)} / ${number(report.split.validation.rows)} / ${number(report.split.test.rows)}`);
  text('validationThreshold',report.threshold.toFixed(6));text('baseRate',pct(report.test_metrics.positive/report.test_metrics.total));
  const boxes=$('comparison');boxes.replaceChildren();
  for(const [name,result] of Object.entries(report.comparison)){
    const box=make('div',null,'model-box');box.append(make('strong',`${name}${name===report.model?' · selected':''}`),make('span',`Validation AP ${result.validation_ap.toFixed(3)} · Brier ${result.validation_brier.toFixed(3)}`));boxes.append(box);
  }
  const container=$('evalDetails');container.replaceChildren();
  const lines=[`Chronological split: train ${report.split.train.start.slice(0,10)}–${report.split.train.end.slice(0,10)}, validation ${report.split.validation.start.slice(0,10)}–${report.split.validation.end.slice(0,10)}, test ${report.split.test.start.slice(0,10)}–${report.split.test.end.slice(0,10)}.`,
    `Initial validation cutoff: ${report.threshold.toFixed(6)}. Validation selected ${report.validation_metrics.selected} of ${report.validation_metrics.total} records (${pct(report.validation_metrics.review_rate)}); tied scores can change the target.`,
    `The selected model's untouched test average precision is ${report.test_metrics.average_precision.toFixed(3)} and its Brier score is ${report.test_metrics.brier.toFixed(3)}. These are results on this fixed synthetic simulation, not evidence of real-world performance.`,
    'Scores are simulated-model estimates and are not independently calibrated. No automatic blocking or investigation decisions are made.'];
  for(const line of lines)container.append(make('p',line));
}
$('threshold').addEventListener('input',event=>{state.threshold=Number(event.target.value);text('thresholdValue',state.threshold.toFixed(6));clearTimeout(state.timer);state.requestId++;state.timer=setTimeout(refresh,130)});
$('reset').addEventListener('click',()=>{setThreshold(state.report.threshold);refresh()});
$('csvFile').addEventListener('change',event=>{state.upload=event.target.files[0]||null;text('uploadStatus','Scoring file…');refresh()});
$('showHoldout').addEventListener('click',()=>{state.upload=null;$('csvFile').value='';refresh()});
(async()=>{try{state.report=await read(await fetch('/api/report'));describe(state.report);setThreshold(state.report.threshold);$('workspace').hidden=false;await refresh()}catch(e){text('status',`Could not load the experiment: ${e.message}`)}})();
