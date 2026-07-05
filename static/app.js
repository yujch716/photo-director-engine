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

// ── Capture Viewer ─────────────────────────────────────────────────────
async function loadCaptureList() {
  const status = document.getElementById('captures-status');
  const errorEl = document.getElementById('captures-error');
  const select = document.getElementById('captures-select');
  errorEl.style.display = 'none';
  status.textContent = '목록 불러오는 중...';
  try {
    const res = await fetch('/captures');
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);

    const current = select.value;
    select.innerHTML = '<option value="">— 선택 —</option>' +
      data.captures.map((n) => `<option value="${n}">${n}</option>`).join('');
    if (data.captures.includes(current)) select.value = current;
    status.textContent = `${data.captures.length}개 폴더`;
  } catch (err) {
    showError(errorEl, status, err.message);
  }
}

function renderCaptureImages(name, original, original2x, crops, result) {
  const wrap = document.getElementById('captures-images');
  const card = (title, file, extra = '') => `
    <div class="panel" style="width:280px;">
      <h3 style="font-size:13px;">${title}</h3>
      <div class="image-frame">
        <img src="/drone-data/${name}/${file}" alt="${file}" />
      </div>
      <div class="score-meta" style="margin-top:6px;">${file}</div>
      ${extra}
    </div>
  `;
  let html = '';
  if (original) html += card('Original (1x)', original);
  if (original2x) {
    const box = (result && result.original_2x) || null;
    const coords = box
      ? `<div class="score-meta mono" style="margin-top:4px;">crop 좌표: [${box.left}, ${box.top}, ${box.right}, ${box.bottom}]</div>`
      : '';
    html += card('Original (2x)', original2x, coords);
  }
  crops.forEach((f, i) => { html += card(`Crop ${i}`, f); });
  wrap.innerHTML = html || '<p style="color:#6b7280;">이미지 없음</p>';
}

function renderCaptureMeta(result) {
  const el = document.getElementById('captures-meta');
  if (!result) { el.innerHTML = '<p style="color:#6b7280;">result.json 없음</p>'; return; }
  const loc = result.location || {};
  const nearby = result.nearby_landmarks || [];
  const box2x = result.original_2x;
  const box2xText = box2x
    ? `[${box2x.left}, ${box2x.top}, ${box2x.right}, ${box2x.bottom}] (left, top, right, bottom)`
    : '없음';
  el.innerHTML = `
    <table>
      <tbody>
        <tr><td class="mono">위도 (lat)</td><td>${loc.lat ?? '없음'}</td></tr>
        <tr><td class="mono">경도 (lng)</td><td>${loc.lng ?? '없음'}</td></tr>
        <tr><td class="mono">주변 관광명소</td><td>${nearby.length ? nearby.map((p) => p.name).join(', ') : '(없음)'}</td></tr>
        <tr><td class="mono">타깃 수</td><td>${(result.targets || []).length}개</td></tr>
        <tr><td class="mono">2배율 crop 좌표</td><td class="mono">${box2xText}</td></tr>
      </tbody>
    </table>
  `;
}

function renderCaptureResults(result) {
  const el = document.getElementById('captures-results');
  const results = (result && result.results) || [];
  if (results.length === 0) { el.innerHTML = '<p style="color:#6b7280;">타깃 없음</p>'; return; }

  el.innerHTML = results.map((r, i) => {
    const scores = r.scores || {};
    const scoreRows = Object.entries(scores)
      .sort((a, b) => b[1] - a[1])
      .map(([label, s]) => `
        <tr>
          <td class="mono">${label}</td>
          <td>${(s * 100).toFixed(2)}%</td>
        </tr>
      `).join('');
    return `
      <div class="panel" style="margin-bottom:16px;">
        <h3 style="font-size:14px;">Crop ${i} — <span style="color:#2563eb;">${r.class}</span></h3>
        <table style="margin-bottom:10px;">
          <tbody>
            <tr><td class="mono">landmark</td><td>${r.landmark ?? '<span style="color:#b91c1c;">null (미확정)</span>'}</td></tr>
            <tr><td class="mono">confidence</td><td>${r.confidence != null ? (r.confidence * 100).toFixed(2) + '%' : '-'}</td></tr>
            <tr><td class="mono">bbox</td><td class="mono">[${(r.bbox || []).map((v) => v.toFixed(4)).join(', ')}]</td></tr>
          </tbody>
        </table>
        <div class="metric-title" style="font-size:13px;">Scores</div>
        <table><tbody>${scoreRows || '<tr><td colspan="2" style="color:#6b7280;">없음</td></tr>'}</tbody></table>
      </div>
    `;
  }).join('');
}

async function loadCaptureDetail(name) {
  const status = document.getElementById('captures-status');
  const errorEl = document.getElementById('captures-error');
  const detail = document.getElementById('captures-detail');
  if (!name) { detail.style.display = 'none'; return; }

  errorEl.style.display = 'none';
  status.textContent = '불러오는 중...';
  try {
    const res = await fetch(`/captures/${name}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);

    renderCaptureImages(data.name, data.original, data.original_2x, data.crops, data.result);
    renderCaptureMeta(data.result);
    renderCaptureResults(data.result);
    detail.style.display = 'block';
    status.textContent = '';
  } catch (err) {
    detail.style.display = 'none';
    showError(errorEl, status, err.message);
  }
}

document.getElementById('captures-refresh').addEventListener('click', loadCaptureList);
document.getElementById('captures-select').addEventListener('change', (e) => loadCaptureDetail(e.target.value));
document.querySelector('[data-page="captures"]').addEventListener('click', loadCaptureList);
