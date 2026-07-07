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

function renderCaptureImages(name, original, original2x, crops, best, result) {
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
  if (best) {
    const bb = (result && result.best && result.best.source_box_in_1x) || null;
    const coords = bb
      ? `<div class="score-meta mono" style="margin-top:4px;">구도 좌표: [${bb.join(', ')}]</div>`
      : '';
    html += card('★ 최종 구도 (best)', best, coords);
  }
  crops.forEach((f, i) => { html += card(`Crop ${i}`, f); });
  wrap.innerHTML = html || '<p style="color:#6b7280;">이미지 없음</p>';
}

function renderCaptureOverlay(name, original, result) {
  const canvas = document.getElementById('captures-overlay');
  const legend = document.getElementById('captures-overlay-legend');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  legend.innerHTML = '';

  if (!original) { legend.textContent = '원본 이미지 없음'; return; }

  const box2x = result && result.original_2x;         // {left, top, right, bottom}
  const best = result && result.best;                  // {source_box_in_1x: [l,t,r,b], ...}
  const bestBox = best && best.source_box_in_1x;       // [l, t, r, b]
  const offset = result && result.drone_offset;

  const img = new Image();
  img.onload = () => {
    const W = img.naturalWidth, H = img.naturalHeight;
    canvas.width = W;
    canvas.height = H;                                 // 원본 해상도로 그리고 CSS가 100% 폭으로 축소
    ctx.drawImage(img, 0, 0, W, H);

    const lw = Math.max(2, Math.round(W / 320));
    const fs = Math.max(14, Math.round(W / 42));

    const drawBox = (coords, color, label) => {
      if (!coords) return;
      const [l, t, r, b] = coords;
      ctx.lineWidth = lw;
      ctx.strokeStyle = color;
      ctx.strokeRect(l, t, r - l, b - t);
      ctx.font = `bold ${fs}px sans-serif`;
      const tw = ctx.measureText(label).width;
      const th = fs + 8;
      const ty = t >= th ? t - th : t;                 // 위 공간 없으면 박스 안쪽
      ctx.fillStyle = color;
      ctx.fillRect(l, ty, tw + 12, th);
      ctx.fillStyle = '#fff';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, l + 6, ty + th / 2);
    };

    drawBox(box2x ? [box2x.left, box2x.top, box2x.right, box2x.bottom] : null, '#3b82f6', '2배율(중앙)');
    drawBox(bestBox, '#22c55e', '최종 구도');

    // 이미지 중심 → best 중심 이동 화살표 + 중심점
    if (bestBox) {
      const [l, t, r, b] = bestBox;
      const bcx = (l + r) / 2, bcy = (t + b) / 2, icx = W / 2, icy = H / 2;
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.moveTo(icx, icy);
      ctx.lineTo(bcx, bcy);
      ctx.stroke();
      const dot = (x, y, c) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, y, lw * 2.2, 0, Math.PI * 2); ctx.fill(); };
      dot(icx, icy, '#3b82f6');
      dot(bcx, bcy, '#22c55e');
    }

    const items = ['<span style="color:#3b82f6;">■</span> 2배율(중앙) 박스'];
    if (bestBox) items.push('<span style="color:#22c55e;">■</span> 최종 구도 박스');
    if (offset) {
      const num = (v, d = 1) => (typeof v === 'number' ? v.toFixed(d) : v);
      // 극좌표(대각선 단일 이동)를 주로, dx/dy는 보조로 표기. 구버전 캡처는 dr/theta 없음.
      const polar = (offset.dr != null && offset.theta_deg != null)
        ? `dr=${num(offset.dr)}px, θ=${num(offset.theta_deg)}° (화면좌표: 0°=오른쪽, +90°=아래) · `
        : '';
      items.push(`<span style="color:#f59e0b;">→</span> 드론 이동: ${polar}dx=${offset.dx}, dy=${offset.dy}`);
    }
    if (!bestBox) items.push('<span style="color:#b91c1c;">최종 구도 없음</span> (임베딩 DB 없음/매칭 실패)');
    legend.innerHTML = items.join(' &nbsp;&nbsp; ');
  };
  img.onerror = () => { legend.textContent = '원본 이미지를 불러오지 못했습니다.'; };
  img.src = `/drone-data/${name}/${original}`;
}

function tagsTable(tags) {
  if (!tags) return '<p style="color:#6b7280;">태그 없음</p>';
  const row = (k, v) => `<tr><td class="mono">${k}</td><td>${v == null ? '<span style="color:#b91c1c;">null</span>' : v}</td></tr>`;
  const flags = Array.isArray(tags.flags) ? tags.flags.join(', ') : (tags.flags ?? '');
  return `<table><tbody>
    ${row('landmark', tags.landmark)}
    ${row('location', tags.location)}
    ${row('time', tags.time)}
    ${row('weather', tags.weather)}
    ${row('facing', tags.facing)}
    ${row('person_count', tags.person_count)}
    ${row('flags', flags)}
  </tbody></table>`;
}

function renderCaptureBest(result) {
  const el = document.getElementById('captures-best');
  const best = result && result.best;
  if (!best) {
    el.innerHTML = '<p style="color:#6b7280;">최종 구도 없음 (임베딩 DB 없음/매칭 실패 시)</p>';
    return;
  }
  const sim = typeof best.similarity === 'number' ? best.similarity.toFixed(4) : best.similarity;
  const refPath = best.reference || '(없음)';
  // reference 실제 사진: /refimages 로 시도, 없으면 숨기고 안내
  const refImg = best.reference
    ? `<img src="/refimages/${best.reference}" alt="reference"
           style="max-width:260px;border:1px solid #dde3ed;border-radius:6px;display:none;"
           onload="this.style.display='block';this.nextElementSibling.style.display='none';"
           onerror="this.style.display='none';" />
       <div class="score-meta" style="color:#6b7280;">reference 원본 이미지 없음 (임베딩·메타만 보유). ver_1/ 사진 폴더를 /refimages로 붙이면 표시됩니다.</div>`
    : '';

  el.innerHTML = `
    <div style="display:flex;flex-wrap:wrap;gap:24px;">
      <div style="min-width:240px;">
        <div class="metric-title" style="font-size:13px;">① 최종 구도(선택된 크롭)의 태그</div>
        ${tagsTable(best.candidate_tags)}
      </div>
      <div style="min-width:240px;">
        <div class="metric-title" style="font-size:13px;">② 유사도로 매칭된 reference (.npy)</div>
        <table><tbody>
          <tr><td class="mono">reference</td><td class="mono">${refPath}</td></tr>
          <tr><td class="mono">similarity</td><td>${sim}</td></tr>
          <tr><td class="mono">candidate_index</td><td>${best.candidate_index ?? '-'}</td></tr>
        </tbody></table>
        <div class="metric-title" style="font-size:13px;margin-top:8px;">reference 태그</div>
        ${tagsTable(best.reference_tags)}
      </div>
      <div style="min-width:240px;">
        <div class="metric-title" style="font-size:13px;">reference 원본 이미지</div>
        ${refImg}
      </div>
    </div>
  `;
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

    renderCaptureOverlay(data.name, data.original, data.result);
    renderCaptureImages(data.name, data.original, data.original_2x, data.crops, data.best, data.result);
    renderCaptureBest(data.result);
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
