// The renderer is served by the sidecar, so the API is same-origin ('/api/...').
const $ = (id) => document.getElementById(id);
const els = {
  q: $('q'), go: $('go'), objects: $('objects'),
  byimage: $('byimage'), index: $('index'), stats: $('stats'),
  results: $('results'), empty: $('empty'),
  progress: $('progress'), barfill: $('barfill'), ptext: $('ptext'),
  status: $('status'),
};

// ---------- helpers ----------
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function setStatus(msg) {
  if (!msg) { els.status.classList.add('hidden'); return; }
  els.status.textContent = msg;
  els.status.classList.remove('hidden');
}

async function refreshStats() {
  try {
    const { count } = await getJSON('/api/stats');
    els.stats.textContent = `${count.toLocaleString()} indexed`;
  } catch { els.stats.textContent = ''; }
}

// ---------- rendering ----------
function card(item) {
  const el = document.createElement('div');
  el.className = 'card';
  el.title = item.path;

  const wrap = document.createElement('div');
  wrap.className = 'thumbwrap';
  const img = document.createElement('img');
  img.loading = 'lazy';
  img.src = `/api/thumb?path=${encodeURIComponent(item.path)}`;
  img.onerror = () => { wrap.textContent = '⚠︎'; };
  wrap.appendChild(img);
  if (typeof item.score === 'number') {
    const s = document.createElement('div');
    s.className = 'score';
    s.textContent = item.score.toFixed(3);
    wrap.appendChild(s);
  }

  const meta = document.createElement('div');
  meta.className = 'meta';
  const name = document.createElement('div');
  name.className = 'name';
  name.textContent = item.name;
  meta.appendChild(name);

  const actions = document.createElement('div');
  actions.className = 'actions';
  actions.appendChild(actionBtn('Open', (e) => { e.stopPropagation(); window.lumen.openPath(item.path); }));
  actions.appendChild(actionBtn('Reveal', (e) => { e.stopPropagation(); window.lumen.revealPath(item.path); }));
  actions.appendChild(actionBtn('Similar', (e) => { e.stopPropagation(); searchByImage(item.path); }));

  // click anywhere on the card opens the file
  el.addEventListener('click', () => window.lumen.openPath(item.path));

  el.append(wrap, meta, actions);
  return el;
}

function actionBtn(label, onClick) {
  const b = document.createElement('button');
  b.textContent = label;
  b.addEventListener('click', onClick);
  return b;
}

function render(results) {
  els.results.innerHTML = '';
  els.empty.classList.toggle('hidden', results.length > 0);
  if (!results.length) { setStatus('No matches.'); return; }
  const frag = document.createDocumentFragment();
  for (const item of results) frag.appendChild(card(item));
  els.results.appendChild(frag);
}

// ---------- searches ----------
async function searchText() {
  const q = els.q.value.trim();
  if (!q) return;
  setStatus('Searching…');
  try {
    const objects = els.objects.checked;
    const { results } = await getJSON(
      `/api/search?q=${encodeURIComponent(q)}&objects=${objects}&limit=80`
    );
    render(results);
    setStatus(`${results.length} result${results.length === 1 ? '' : 's'} for “${q}”`);
  } catch (e) { setStatus(`Search failed: ${e.message}`); }
}

async function searchByImage(path) {
  setStatus('Finding similar images…');
  try {
    const { results } = await getJSON(`/api/similar?path=${encodeURIComponent(path)}&limit=80`);
    render(results);
    els.q.value = '';
    setStatus(`Images similar to ${path.split('/').pop()}`);
  } catch (e) { setStatus(`Similar search failed: ${e.message}`); }
}

// ---------- indexing ----------
let pollTimer = null;

async function indexFolder() {
  const folder = await window.lumen.pickFolder();
  if (!folder) return;
  try {
    await getJSON(`/api/index?folder=${encodeURIComponent(folder)}`, { method: 'POST' });
    els.progress.classList.remove('hidden');
    els.barfill.style.width = '0%';
    els.ptext.textContent = 'Starting…';
    pollIndex(folder);
  } catch (e) { setStatus(`Could not start indexing: ${e.message}`); }
}

function pollIndex(folder) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    let st;
    try { st = await getJSON('/api/index/status'); } catch { return; }
    // Indeterminate count (we don't know the total up front), so show a pulse.
    els.ptext.textContent = st.running
      ? `Indexing ${folder.split('/').pop()} — ${st.done.toLocaleString()} images…`
      : 'Finishing…';
    els.barfill.style.width = st.running ? `${30 + (st.done % 70)}%` : '100%';

    if (!st.running) {
      clearInterval(pollTimer);
      setTimeout(() => els.progress.classList.add('hidden'), 800);
      const bits = [`added ${st.added}`];
      if (st.skipped) bits.push(`skipped ${st.skipped}`);
      if (st.failed) bits.push(`failed ${st.failed}`);
      if (st.error) bits.push(`error: ${st.error}`);
      setStatus(`Indexing done — ${bits.join(', ')}.`);
      refreshStats();
    }
  }, 400);
}

// ---------- wiring ----------
els.go.addEventListener('click', searchText);
els.q.addEventListener('keydown', (e) => { if (e.key === 'Enter') searchText(); });
els.index.addEventListener('click', indexFolder);
els.byimage.addEventListener('click', async () => {
  const p = await window.lumen.pickImage();
  if (p) searchByImage(p);
});

// drag-and-drop an image anywhere to search by it
window.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('dragging'); });
window.addEventListener('dragleave', (e) => { if (e.relatedTarget === null) document.body.classList.remove('dragging'); });
window.addEventListener('drop', (e) => {
  e.preventDefault();
  document.body.classList.remove('dragging');
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (!file) return;
  const path = window.lumen.getFilePath(file);
  if (path) searchByImage(path);
});

refreshStats();
