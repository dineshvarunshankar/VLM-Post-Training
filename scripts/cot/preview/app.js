const $ = id => document.getElementById(id);
const initial = new URLSearchParams(location.search);
let clips = [], activeTask = initial.get('task') || 'native_general_scene_description', sequence = 0, focused = false;
const params = extra => new URLSearchParams({task: activeTask, ...extra});
const pretty = value => JSON.stringify(value, null, 2);
async function get(url) {
  const response = await fetch(url, {cache:'no-store'});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Unable to load request.');
  return data;
}
function status(text, error=false) { $('status').textContent=text; $('status').classList.toggle('error',error); }
function tab(name) {
  document.querySelectorAll('[data-tab]').forEach(button => {
    const selected=button.dataset.tab===name;
    button.setAttribute('aria-selected',selected);
    button.tabIndex=selected?0:-1;
    $('panel-'+button.dataset.tab).hidden=!selected;
  });
}
document.querySelectorAll('[data-tab]').forEach(button => {
  button.addEventListener('click',()=>tab(button.dataset.tab));
  button.addEventListener('keydown',event=>{
    const buttons=[...document.querySelectorAll('[data-tab]')].filter(b=>!b.hidden); let i=buttons.indexOf(button);
    if(event.key==='ArrowRight') i=(i+1)%buttons.length;
    else if(event.key==='ArrowLeft') i=(i+buttons.length-1)%buttons.length;
    else return;
    event.preventDefault();buttons[i].focus();tab(buttons[i].dataset.tab);
  });
});
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{
  try { await navigator.clipboard.writeText($(button.dataset.copy).textContent);button.textContent='Copied'; }
  catch { button.textContent='Select text to copy'; }
  setTimeout(()=>button.textContent='Copy',1800);
}));
function resetEvidence() {
  $('tab-evidence').hidden=true;
  $('evidence-points').replaceChildren();$('evidence-spans').replaceChildren();
  $('evidence-frame-wrap').hidden=true;$('evidence-frame').removeAttribute('src');
  $('evidence-frame-label').hidden=true;$('evidence-crosshair').hidden=true;
  $('toggle-evidence-point').hidden=true;
}
function renderEvidence(data) {
  const evidence=data.human_evidence || {};
  $('tab-evidence').hidden=!data.task.includes('medical');
  $('evidence-json').textContent=pretty(evidence);
  $('evidence-note').textContent='Select a point to show its exact annotated frame. The magenta ring is a review overlay; it is not part of the video. Display enlargement adds no detail.';
  const rationale=data.result?.response?.reasoning_process || '';
  const citations=[...rationale.matchAll(/<evidence>[\s\S]*?<\/evidence>/g)].map(match=>new DOMParser().parseFromString(match[0],'application/xml').documentElement);
  const text=(node,key)=>node.querySelector(key)?.textContent || '';
  const pointMatches=(node,cue,point)=>{
    try {return text(node,'cue')===cue && text(node,'time')===point.time && text(node,'slot')===(point.slot || '') && JSON.stringify(JSON.parse(text(node,'point')))===JSON.stringify(point.point);}
    catch{return false;}
  };
  for(const [cue,values] of Object.entries(evidence)) {
    for(const point of values.points) {
      const row=document.createElement('div'),button=document.createElement('button'),details=document.createElement('div'),name=document.createElement('strong'),note=document.createElement('small');
      row.className='evidence-row';button.className='time-button';button.textContent=point.time;name.textContent=cue;
      note.textContent=`[${point.point.join(', ')}]${point.slot?' · '+point.slot:''} · `+(citations.some(node=>pointMatches(node,cue,point))?'Cited by model; matches human GT':'Human GT; not cited');
      button.addEventListener('click',()=>{
        tab('evidence');$('video').pause();
        const [minutes,seconds]=point.time.split(':').map(Number);$('video').currentTime=minutes*60+seconds;
        $('evidence-frame-wrap').hidden=true;$('evidence-crosshair').hidden=true;
        $('toggle-evidence-point').hidden=true;$('evidence-frame-label').hidden=false;
        $('evidence-frame-label').textContent=`${cue} · ${point.time} · [${point.point.join(', ')}] · Loading exact frame…`;
        const img=$('evidence-frame');
        img.onload=()=>{
          $('evidence-frame-wrap').hidden=false;$('evidence-crosshair').style.left=(point.point[0]/10)+'%';$('evidence-crosshair').style.top=(point.point[1]/10)+'%';
          $('evidence-crosshair').hidden=false;$('toggle-evidence-point').hidden=false;$('toggle-evidence-point').textContent='Hide point';
          $('evidence-frame-label').textContent=`${cue} · ${point.time} · [${point.point.join(', ')}] · Exact frame, normalized coordinates`;
        };
        img.onerror=()=>{$('evidence-frame-label').textContent='Could not load the evidence frame.';};
        img.src='/evidence-frame?'+new URLSearchParams({task:data.task,id:data.clip,category:data.category,frame:point.frame_index});
      });
      details.append(name,note);row.append(button,details);$('evidence-points').append(row);
    }
    for(const span of values.spans) {
      const p=document.createElement('p');p.textContent=`${cue}: ${span.start}–${span.end}`;$('evidence-spans').append(p);
    }
  }
}
$('toggle-evidence-point').addEventListener('click',()=>{
  $('evidence-crosshair').hidden=!$('evidence-crosshair').hidden;
  $('toggle-evidence-point').textContent=$('evidence-crosshair').hidden?'Show point':'Hide point';
});
function clearExample() {
  resetEvidence();
  $('video').pause();$('video').removeAttribute('src');$('video').load();
  $('video-wrap').hidden=true;$('video-empty').hidden=false;$('video-id').textContent='';
  $('result-status').textContent='No result loaded.';$('result-reasoning').textContent='';$('result-answer').textContent='';$('result-details').textContent='';
  $('source').textContent='No request loaded.';$('request').textContent='No request loaded.';
}
function renderSettings(settings) {
  $('settings').replaceChildren();
  for(const [key,value] of Object.entries(settings)) {
    const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');
    dt.textContent=key.replaceAll('_',' ');dd.textContent=value;row.append(dt,dd);$('settings').append(row);
  }
}
function filterClips() {
  const selected=$('clip').value,query=$('search').value.trim().toLowerCase();
  const visible=clips.filter(c=>c.id.toLowerCase().includes(query));
  $('clip').replaceChildren(new Option('Select a clip…',''));
  for(const c of visible) $('clip').append(new Option(`${c.id} · ${Number(c.duration_s).toFixed(1)}s`,c.id));
  if(visible.some(c=>c.id===selected)) $('clip').value=selected;
  $('clip-count').textContent=focused ? 'One paired pilot sample · four tasks' : `${visible.length} of ${clips.length} completed RGB clips`;
  $('load').disabled=!$('clip').value;
}
async function loadExample() {
  const id=$('clip').value;if(!id)return;
  const n=++sequence;clearExample();$('load').disabled=true;status('Preparing the request using generate.py…');
  $('user').textContent='Loading…';
  try {
    const data=await get('/api/clip?'+params({id,category:$('category').value}));if(n!==sequence)return;
    renderEvidence(data);
    $('category-field').hidden=!data.categories.length;
    $('category').replaceChildren(...data.categories.map(c=>new Option(c,c)));$('category').value=data.category;
    $('system').textContent=data.system_prompt;$('user').textContent=data.user_prompt;$('schema').textContent=pretty(data.schema);
    $('source').textContent=pretty({...data.inputs,context_exclusions:data.context_exclusions});$('request').textContent=pretty(data.request);
    $('example-note').textContent=data.note;$('input-note').textContent='Text from generate.py request_payload(), without rewriting.';
    $('video').src=data.video_url;$('video-wrap').hidden=false;$('video-empty').hidden=true;$('video-id').textContent=id;
    if(data.result) {
      $('result-status').textContent=(data.result.current?'Matches current inputs. ':data.result.request_unchanged?'Model inputs unchanged; preview code updated. ':'Saved result uses earlier inputs or settings. ')+(data.result.created_at || '');
      $('result-reasoning').textContent=data.result.response?.reasoning_process || 'Scene descriptions contain an answer only.';
      const answer=data.result.response?.answer;
      $('result-answer').textContent=typeof answer==='string'?answer:pretty(answer);
      const checks=data.result.checks || {};
      $('result-status').textContent+=` · Schema: ${checks.response_schema || 'not checked'}`;
      if(checks.human_answer_and_grounding) $('result-status').textContent+=` · GT answer and grounding: ${checks.human_answer_and_grounding}`;
      $('result-status').textContent+=' · Visual support and cross-task consistency are not automatically checked.';
      $('result-details').textContent=pretty({model:data.result.model,checks,usage:data.result.usage,review:data.result.review});tab('result');
    } else $('result-status').textContent='No saved result yet. Click Load example again after generation finishes.';
    renderSettings(data.settings);status('Input and result loaded. This page makes no model calls.');
  } catch(error) {
    if(n!==sequence)return;status(error.message,true);$('user').textContent='Request could not be prepared: '+error.message;
  } finally {if(n===sequence)$('load').disabled=false;}
}
async function selectTask(task) {
  $('category').replaceChildren();$('category-field').hidden=true;
  const n=++sequence;activeTask=task;clearExample();clips=[];filterClips();
  $('task').disabled=true;$('load').disabled=true;$('search').disabled=true;$('clip').disabled=true;
  $('system').textContent='Loading…';$('user').textContent='Loading…';status('Loading YAML…');
  try {
    const data=await get('/api/template?'+params({}));if(n!==sequence)return;
    focused=Boolean(data.focused);
    $('task').replaceChildren(...data.tasks.map(t=>new Option(t.title,t.id)));$('task').value=task;$('task').disabled=false;
    $('task-title').textContent=data.title;$('version').textContent='v'+data.prompt.version;
    $('variant').textContent=task.startsWith('stabilized')?'STABILIZED':'NATIVE';
    const note=data.production_request?'Request preview uses the existing generate.py runner.':'Template only: no generation runner is implemented for this task. No example input is assembled.';
    $('task-note').textContent=note;$('example-note').textContent=note;
    $('system').textContent=data.prompt.system_prompt;$('user').textContent=data.prompt.user_prompt;$('schema').textContent=pretty(data.prompt.response_schema);
    $('schema-note').textContent='Response schema from the YAML.';$('output-title').textContent='Response schema';
    $('input-note').textContent='Unfilled user template from the YAML.';
    $('request').textContent=data.production_request?'Load a clip to inspect the request.':'Unavailable: this task has no generation runner.';
    renderSettings(data.settings);
    if(!data.production_request) {status('Template only. Generation is not implemented.');return;}
    const index=await get('/api/clips?'+params({}));if(n!==sequence)return;
    clips=index.clips;filterClips();$('search').value='';filterClips();$('search').disabled=focused;$('clip').disabled=focused;
    if(clips.length) {const requested=initial.get('clip');$('clip').value=clips.some(c=>c.id===requested)?requested:clips[0].id;const category=initial.get('category');if(category && task.includes('medical'))$('category').append(new Option(category,category,true,true));await loadExample();}
    else status('No prepared RGB clips are available.',true);
  } catch(error) {if(n===sequence)status(error.message,true);$('task').disabled=false;}
}
$('task').addEventListener('change',()=>selectTask($('task').value));
$('clip').addEventListener('change',()=>{++sequence;clearExample();$('user').textContent='Load the selected clip to inspect its request.';$('load').disabled=!$('clip').value;});
$('search').addEventListener('input',()=>{filterClips();if(!$('clip').value){++sequence;clearExample();$('user').textContent='Select a clip to inspect its request.';}});
$('category').addEventListener('change',loadExample);
$('load').addEventListener('click',loadExample);
selectTask(activeTask);
