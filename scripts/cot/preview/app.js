const $ = id => document.getElementById(id);
let clips = [], requestSequence = 0, activeTask = '', livePreview = false;
async function get(url) {
  const response = await fetch(url, {cache: 'no-store'});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load the preview.');
  return data;
}
function status(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
function settings(values) {
  $('settings').replaceChildren();
  const names = {model: 'Model', video_fps: 'Requested video FPS', media_resolution: 'Media resolution', reasoning_effort: 'Reasoning effort', max_tokens: 'Output token limit'};
  for (const [key, label] of Object.entries(names)) {
    const row = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = label; dd.textContent = values[key] ?? 'Not set'; row.append(dt, dd); $('settings').append(row);
  }
}
function filterClips() {
  const selected = $('clip').value, query = $('search').value.trim().toLowerCase();
  const filtered = clips.filter(clip => clip.id.toLowerCase().includes(query));
  $('clip').replaceChildren(new Option('Select a clip…', ''));
  for (const clip of filtered) $('clip').append(new Option(clip.id, clip.id));
  if (filtered.some(clip => clip.id === selected)) $('clip').value = selected;
  $('load').disabled = !$('clip').value;
  $('clip-count').textContent = `${filtered.length} of ${clips.length} completed RGB clips · inputs checked when opened`;
}
$('search').addEventListener('input', filterClips);
$('clip').addEventListener('change', () => { $('load').disabled = !$('clip').value; });
$('load').addEventListener('click', async () => {
  const name = $('clip').value;
  if (!name || !livePreview) return;
  const sequence = ++requestSequence;
  $('load').disabled = true;
  status('Checking the video and annotation fingerprints, then rendering the complete input…');
  $('video').pause(); $('video').removeAttribute('src'); $('video').load(); $('video').hidden = true;
  $('video-empty').hidden = false; $('video-id').textContent = '';
  $('user').textContent = 'Loading the selected clip…'; $('request').textContent = 'Loading…';
  $('input-kind').textContent = 'Loading';
  try {
    const data = await get('/api/clip?task=' + encodeURIComponent(activeTask) + '&id=' + encodeURIComponent(name));
    if (sequence !== requestSequence) return;
    const request = data.request;
    $('system').textContent = request.messages[0].content;
    $('user').textContent = request.messages[1].content.find(part => part.type === 'text').text;
    $('schema').textContent = JSON.stringify(request.response_format.json_schema.schema, null, 2);
    $('request').textContent = JSON.stringify(request, null, 2);
    $('input-kind').textContent = 'Actual inputs';
    $('input-note').textContent = 'These are the complete clip-specific inputs. No medical evidence coordinates or additional images are attached.';
    settings(data.settings);
    $('video').src = '/video?id=' + encodeURIComponent(name); $('video').hidden = false; $('video-empty').hidden = true;
    $('video-id').textContent = name;
    status('Inputs verified. Ready to review — nothing has been uploaded or generated.');
  } catch (error) {
    if (sequence !== requestSequence) return;
    status(error.message, true); $('user').textContent = 'This clip could not be prepared. See the error beside the clip selector.';
    $('request').textContent = 'No request was assembled.'; $('input-kind').textContent = 'Unavailable';
  } finally { if (sequence === requestSequence) $('load').disabled = !$('clip').value; }
});
async function selectTask(task = '') {
  const sequence = ++requestSequence;
  livePreview = false;
  $('video').pause(); $('video').removeAttribute('src'); $('video').load();
  $('video').hidden = true; $('video-empty').hidden = false; $('video-id').textContent = '';
  for (const id of ['clip-card', 'media-card', 'settings-card', 'request-card']) $(id).hidden = true;
  for (const id of ['system', 'user', 'schema', 'input-contract']) $(id).textContent = 'Loading…';
  $('request').textContent = 'Select a clip to see the assembled request.';
  $('input-kind').textContent = 'Template';
  status('Loading the selected YAML…');
  try {
    const data = await get('/api/template?task=' + encodeURIComponent(task || 'native_general_scene_description'));
    if (sequence !== requestSequence) return;
    activeTask = data.task; livePreview = data.live_preview;
    $('task').replaceChildren(...data.tasks.map(item => new Option(item.title, item.id)));
    $('task').value = activeTask; $('task').disabled = false;
    $('task-title').textContent = data.title;
    $('task-mode').textContent = livePreview ? 'Template + actual clip inputs' : 'Template review';
    $('task-note').textContent = livePreview
      ? 'Review the YAML, then select a prepared clip to inspect the exact request.'
      : 'This YAML is available for review. Its clip-input assembly and generation runner are not implemented yet.';
    for (const id of ['clip-card', 'media-card', 'settings-card', 'request-card']) $(id).hidden = !livePreview;
    $('system').textContent = data.prompt.system_prompt;
    $('user').textContent = data.prompt.user_prompt;
    $('schema').textContent = JSON.stringify(data.prompt.response_schema, null, 2);
    const medical = activeTask.includes('medical');
    $('reasoning-field').hidden = !medical;
    $('output-title').textContent = medical ? 'Reasoning + answer' : 'Scene description';
    $('answer-description').textContent = medical
      ? 'The supplied category answer with its label keys and integer 0/1 values.'
      : 'A description of the designated subject and visible scene.';
    $('schema-note').textContent = medical
      ? 'Base schema from the YAML. The runner must specialize answer to the exact category label keys before generation; that runner is not implemented yet.'
      : 'The response schema from the YAML. No answer has been generated.';
    $('input-note').textContent = livePreview
      ? 'Choose a clip to replace the placeholders with its actual metadata, viewpoint, counts, and human transcription.'
      : 'These are the exact input placeholders in the YAML, not populated annotations or an assembled API request.';
    $('inputs-title').textContent = livePreview ? 'Input contract' : 'Planned input contract';
    const prompt = data.prompt.system_prompt;
    const inputStart = prompt.indexOf('\nINPUT\n');
    const nextSection = inputStart < 0 ? -1 : prompt.slice(inputStart + 7).search(/\n[A-Z][A-Z /_-]+\n/);
    $('input-contract').textContent = inputStart < 0
      ? data.prompt.user_prompt
      : prompt.slice(inputStart + 7, nextSection < 0 ? undefined : inputStart + 7 + nextSection).trim();
    settings(data.settings);
    status(livePreview ? 'Prompt loaded. Select a clip to see exactly what will be sent.' : 'Template loaded. No clip-specific request has been assembled.');
    filterClips();
  } catch (error) {
    if (sequence === requestSequence) status(error.message, true);
  }
}
$('task').addEventListener('change', () => selectTask($('task').value));
selectTask();
get('/api/clips').then(data => { clips = data.clips; filterClips(); }).catch(error => {
  $('clip').replaceChildren(new Option('Clips could not be loaded', '')); status(error.message, true);
});
