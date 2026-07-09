const showError = (errorEl, statusEl, message) => {
  statusEl.textContent = '';
  errorEl.textContent = 'Error: ' + message;
  errorEl.style.display = 'block';
};

document.querySelectorAll('#sidebar button').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#sidebar button').forEach((item) => item.classList.remove('active'));
    document.querySelectorAll('.page').forEach((page) => page.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('page-' + btn.dataset.page).classList.add('active');
  });
});

document.getElementById('yolo-input').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const status = document.getElementById('yolo-status');
  const errorEl = document.getElementById('yolo-error');
  const resultEl = document.getElementById('yolo-result');
  document.getElementById('yolo-file-name').textContent = file.name;

  status.textContent = 'Detecting...';
  errorEl.style.display = 'none';
  resultEl.classList.remove('visible');

  try {
    const formData = new FormData();
    formData.append('image', file);

    const res = await fetch('/detect-image', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);

    status.textContent = '';
    document.getElementById('yolo-count').textContent = `Detections (${data.detections.length})`;

    const tbody = document.getElementById('yolo-tbody');
    tbody.innerHTML = data.detections.length === 0
      ? '<tr><td colspan="3" style="text-align:center;color:#6b7280">No objects detected</td></tr>'
      : data.detections.map((d) => `
          <tr>
            <td>${d.class}</td>
            <td>${d.confidence.toFixed(4)}</td>
            <td class="mono">[${d.bbox.map((v) => v.toFixed(4)).join(', ')}]</td>
          </tr>
        `).join('');

    resultEl.classList.add('visible');

    const canvas = document.getElementById('yolo-canvas');
    const ctx = canvas.getContext('2d');
    const imgEl = new Image();
    imgEl.src = URL.createObjectURL(file);
    imgEl.onload = () => {
      canvas.width = imgEl.naturalWidth;
      canvas.height = imgEl.naturalHeight;
      ctx.drawImage(imgEl, 0, 0);

      data.detections.forEach((d) => {
        const [xc, yc, bw, bh] = d.bbox;
        const x = (xc - bw / 2) * canvas.width;
        const y = (yc - bh / 2) * canvas.height;
        const pw = bw * canvas.width;
        const ph = bh * canvas.height;

        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = Math.max(2, canvas.width / 400);
        ctx.strokeRect(x, y, pw, ph);

        const label = `${d.class} ${(d.confidence * 100).toFixed(1)}%`;
        const fontSize = Math.max(14, canvas.width / 60);
        ctx.font = `bold ${fontSize}px sans-serif`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = '#ef4444';
        ctx.fillRect(x, y - fontSize - 6, tw + 10, fontSize + 6);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x + 5, y - 4);
      });
    };
  } catch (err) {
    showError(errorEl, status, err.message);
  }

  event.target.value = '';
});

document.getElementById('nima-input').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const status = document.getElementById('nima-status');
  const errorEl = document.getElementById('nima-error');
  const resultEl = document.getElementById('nima-result');
  document.getElementById('nima-file-name').textContent = file.name;

  status.textContent = 'Running NIMA... first run may download model weights.';
  errorEl.style.display = 'none';
  resultEl.classList.remove('visible');
  document.getElementById('nima-img').src = URL.createObjectURL(file);

  try {
    const formData = new FormData();
    formData.append('image', file);

    const res = await fetch('/nima-score', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);

    status.textContent = '';
    document.getElementById('nima-score').textContent = data.score.toFixed(2);
    document.getElementById('nima-device').textContent = `Device: ${data.device}`;
    resultEl.classList.add('visible');
  } catch (err) {
    showError(errorEl, status, err.message);
  }

  event.target.value = '';
});

// ── Kakao Map (Leaflet + OpenStreetMap) ───────────────────────────────
let _leafletMap = null;
let _leafletMarkers = [];

function initLeafletMap() {
  if (_leafletMap) return;
  const lat = parseFloat(document.getElementById('kakao-lat').value) || 36.076;
  const lng = parseFloat(document.getElementById('kakao-lng').value) || 129.5665;
  _leafletMap = L.map('kakao-map-container').setView([lat, lng], 15);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(_leafletMap);
}

function clearLeafletMarkers() {
  _leafletMarkers.forEach((m) => m.remove());
  _leafletMarkers = [];
}

function renderMarkers(landmarks) {
  clearLeafletMarkers();
  if (landmarks.length === 0) return;
  const group = [];
  landmarks.forEach((place, i) => {
    const marker = L.marker([place.lat, place.lng])
      .addTo(_leafletMap)
      .bindPopup(`<b>${i + 1}. ${place.name}</b><br>${place.distance}m<br><span style="color:#6b7280;font-size:12px;">${place.address}</span>`);
    _leafletMarkers.push(marker);
    group.push([place.lat, place.lng]);
  });
  _leafletMap.fitBounds(group, { padding: [40, 40] });
}

function renderList(landmarks) {
  const wrap = document.getElementById('kakao-list-wrap');
  const tbody = document.getElementById('kakao-list-tbody');
  const title = document.getElementById('kakao-list-title');
  if (landmarks.length === 0) { wrap.style.display = 'none'; return; }
  title.textContent = `주변 관광명소 ${landmarks.length}곳`;
  tbody.innerHTML = landmarks.map((p, i) => `
    <tr>
      <td style="padding:7px 12px;border:1px solid #e5e7eb;color:#6b7280;">${i + 1}</td>
      <td style="padding:7px 12px;border:1px solid #e5e7eb;font-weight:600;">${p.name}</td>
      <td style="padding:7px 12px;border:1px solid #e5e7eb;color:#374151;">${p.distance}m</td>
      <td style="padding:7px 12px;border:1px solid #e5e7eb;color:#6b7280;font-size:12px;">${p.address}</td>
    </tr>
  `).join('');
  wrap.style.display = 'block';
}

document.getElementById('kakao-geolocate').addEventListener('click', () => {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition((pos) => {
    document.getElementById('kakao-lat').value = pos.coords.latitude.toFixed(6);
    document.getElementById('kakao-lng').value = pos.coords.longitude.toFixed(6);
  });
});

document.getElementById('kakao-search').addEventListener('click', async () => {
  const status = document.getElementById('kakao-status');
  const errorEl = document.getElementById('kakao-error');
  const lat = parseFloat(document.getElementById('kakao-lat').value);
  const lng = parseFloat(document.getElementById('kakao-lng').value);
  const radius = parseInt(document.getElementById('kakao-radius').value, 10) || 1000;

  if (isNaN(lat) || isNaN(lng)) {
    errorEl.textContent = '위도/경도를 입력하세요.';
    errorEl.style.display = 'block';
    return;
  }
  errorEl.style.display = 'none';
  status.textContent = '검색 중...';

  try {
    initLeafletMap();
    const res = await fetch(`/nearby-landmarks?lat=${lat}&lng=${lng}&radius_m=${radius}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);

    _leafletMap.setView([lat, lng], 15);
    renderMarkers(data.landmarks);
    renderList(data.landmarks);
    status.textContent = data.landmarks.length === 0 ? '주변 관광명소가 없습니다.' : '';
  } catch (err) {
    showError(errorEl, status, err.message);
  }
});

document.querySelector('[data-page="kakao-map"]').addEventListener('click', () => {
  setTimeout(() => {
    initLeafletMap();
    if (_leafletMap) _leafletMap.invalidateSize();
  }, 150);
});

// ── Capture Viewer (세션 로그) ──────────────────────────────────────────
const CAP_MENUS = [
  { key: '1_original', label: '1_original', sub: '장면탐색' },
  { key: '2_scan', label: '2_scan', sub: '구도' },
  { key: '3_nima-move', label: '3_nima-move', sub: '세부이동' },
];
let capSession = null;
let capMenu = '1_original';

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

async function loadSessionList() {
  const status = document.getElementById('captures-status');
  const errorEl = document.getElementById('captures-error');
  const select = document.getElementById('captures-select');
  errorEl.style.display = 'none';
  status.textContent = '세션 목록 불러오는 중...';
  try {
    const res = await fetch('/sessions');
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);
    const current = select.value;
    select.innerHTML = '<option value="">— 선택 —</option>' +
      data.sessions.map((n) => `<option value="${n}">${n}</option>`).join('');
    if (data.sessions.includes(current)) select.value = current;
    status.textContent = `${data.sessions.length}개 세션`;
  } catch (err) { showError(errorEl, status, err.message); }
}

async function loadSession(sid) {
  const status = document.getElementById('captures-status');
  const errorEl = document.getElementById('captures-error');
  const detail = document.getElementById('captures-detail');
  if (!sid) { detail.style.display = 'none'; return; }
  errorEl.style.display = 'none';
  status.textContent = '불러오는 중...';
  try {
    const res = await fetch(`/session/${encodeURIComponent(sid)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);
    capSession = data;
    capMenu = '1_original'; // 첫 진입 자동선택
    renderNimaScore(data.nima_score);
    renderMenuTabs();
    renderMenuContent();
    detail.style.display = 'block';
    status.textContent = '';
  } catch (err) { detail.style.display = 'none'; showError(errorEl, status, err.message); }
}

function renderNimaScore(list) {
  const el = document.getElementById('captures-nima');
  if (!list || !list.length) {
    el.innerHTML = '<div class="metric-title" style="font-size:13px;">NIMA 진행 점수</div><p style="color:#6b7280;margin:4px 0 0;">nima-score.json 없음</p>';
    return;
  }
  const vals = list.map((e) => e.nima);
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const rows = list.map((e) => {
    const w = mx > mn ? ((e.nima - mn) / (mx - mn)) * 100 : 50;
    return `<tr>
      <td class="mono" style="white-space:nowrap;padding-right:12px;">${escapeHtml(e.stage)}</td>
      <td style="width:220px;"><div style="background:#e5e7eb;border-radius:4px;overflow:hidden;"><div style="height:12px;width:${w}%;background:#2563eb;"></div></div></td>
      <td style="padding-left:10px;">${e.nima}</td></tr>`;
  }).join('');
  el.innerHTML = `<div class="metric-title" style="font-size:13px;margin-bottom:6px;">NIMA 진행 점수 (단계가 진행될수록 오르는지)</div><table><tbody>${rows}</tbody></table>`;
}

function renderMenuTabs() {
  const el = document.getElementById('captures-menu');
  el.innerHTML = CAP_MENUS.map((m) => {
    const n = (capSession.menus[m.key] || []).length;
    const active = m.key === capMenu;
    return `<button data-menu="${m.key}" style="padding:9px 16px;border-radius:8px;border:1px solid ${active ? '#2563eb' : '#d1d5db'};background:${active ? '#2563eb' : '#fff'};color:${active ? '#fff' : '#374151'};cursor:pointer;font-size:13px;">${m.label} <span style="opacity:.85;">(${m.sub})</span> <span style="opacity:.7;">· ${n}</span></button>`;
  }).join('');
  el.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    capMenu = b.dataset.menu; renderMenuTabs(); renderMenuContent();
  }));
}

function renderMenuContent() {
  const el = document.getElementById('captures-content');
  const groups = capSession.menus[capMenu] || [];
  if (!groups.length) { el.innerHTML = '<p style="color:#6b7280;">이 단계 데이터 없음</p>'; return; }
  el.innerHTML = groups.map(renderGroup).join('');
}

function renderGroup(g) {
  const imgs = g.images.length ? g.images.map((im) => `
    <div class="panel" style="width:220px;">
      <div class="image-frame"><img src="${im.url}" alt="${im.name}" style="width:100%;height:auto;display:block;" /></div>
      <div class="score-meta" style="margin-top:6px;">${im.name}</div>
    </div>`).join('') : '<p style="color:#6b7280;">이미지 없음</p>';
  const jsons = Object.entries(g.jsons).map(([name, obj]) => `
    <div class="panel" style="margin-top:10px;">
      <div class="metric-title" style="font-size:13px;">${name}</div>
      <pre style="margin:6px 0 0;max-height:360px;overflow:auto;font-size:12px;white-space:pre-wrap;word-break:break-all;">${obj === null ? '(파싱 실패)' : escapeHtml(JSON.stringify(obj, null, 2))}</pre>
    </div>`).join('');
  return `<div style="margin-bottom:28px;">
      <h3 style="margin:8px 0;">${escapeHtml(g.label)}</h3>
      <div style="display:flex;flex-wrap:wrap;gap:16px;">${imgs}</div>
      ${jsons}
    </div>`;
}

document.getElementById('captures-refresh').addEventListener('click', loadSessionList);
document.getElementById('captures-select').addEventListener('change', (e) => loadSession(e.target.value));
document.querySelector('[data-page="captures"]').addEventListener('click', loadSessionList);
