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

// ── SAMP Score (구도 점수) ────────────────────────────────────────────
document.getElementById('samp-input').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const status = document.getElementById('samp-status');
  const errorEl = document.getElementById('samp-error');
  const resultEl = document.getElementById('samp-result');
  document.getElementById('samp-file-name').textContent = file.name;

  status.textContent = 'Running SAMP... first run may load model weights.';
  errorEl.style.display = 'none';
  resultEl.classList.remove('visible');
  document.getElementById('samp-img').src = URL.createObjectURL(file);

  try {
    const formData = new FormData();
    formData.append('image', file);

    const res = await fetch('/samp-score', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);

    status.textContent = '';
    document.getElementById('samp-score').textContent = data.score.toFixed(2);
    document.getElementById('samp-device').textContent =
      `Device: ${data.device} · 구도 점수 (1~5)`;
    document.getElementById('samp-dominant').textContent =
      data.dominant_pattern != null ? `주요 구도 패턴: #${data.dominant_pattern}` : '';

    const detail = document.getElementById('samp-detail');
    let html = '';
    if (Array.isArray(data.distribution)) {
      const mx = Math.max(...data.distribution);
      html += '<div class="metric-title" style="font-size:13px;margin-bottom:6px;">점수 분포 (1~5)</div>';
      html += data.distribution.map((p, i) => {
        const w = mx > 0 ? (p / mx) * 100 : 0;
        return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;font-size:12px;">
          <span style="width:12px;color:#6b7280;">${i + 1}</span>
          <div style="flex:1;background:#e5e7eb;border-radius:4px;overflow:hidden;"><div style="height:10px;width:${w}%;background:#2563eb;"></div></div>
          <span style="width:44px;text-align:right;">${p.toFixed(3)}</span>
        </div>`;
      }).join('');
    }
    if (data.attributes) {
      html += '<div class="metric-title" style="font-size:13px;margin:12px 0 6px;">구도 속성</div>';
      html += '<table><tbody>' + Object.entries(data.attributes).map(([k, v]) =>
        `<tr><td class="mono" style="padding-right:12px;">${escapeHtml(k)}</td><td>${v.toFixed(4)}</td></tr>`
      ).join('') + '</tbody></table>';
    }
    detail.innerHTML = html;

    resultEl.classList.add('visible');
  } catch (err) {
    showError(errorEl, status, err.message);
  }

  event.target.value = '';
});

// ── TOPIQ Score (무참조 품질) ─────────────────────────────────────────
document.getElementById('topiq-input').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const status = document.getElementById('topiq-status');
  const errorEl = document.getElementById('topiq-error');
  const resultEl = document.getElementById('topiq-result');
  document.getElementById('topiq-file-name').textContent = file.name;

  status.textContent = 'Running TOPIQ... first run may download model weights (~170MB).';
  errorEl.style.display = 'none';
  resultEl.classList.remove('visible');
  document.getElementById('topiq-img').src = URL.createObjectURL(file);

  try {
    const formData = new FormData();
    formData.append('image', file);

    const res = await fetch('/topiq-score', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);

    status.textContent = '';
    document.getElementById('topiq-score').textContent = data.score.toFixed(4);
    const dir = data.lower_better ? '낮을수록 좋음' : '높을수록 좋음';
    document.getElementById('topiq-meta').textContent =
      `metric: ${data.metric} · ${dir} · Device: ${data.device}`;
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
  { key: '3_detail-refine', label: '3_detail-refine', sub: '세부조정' },
  { key: '4_tilt', label: '4_tilt', sub: '틸트' },
  { key: '5_final', label: '5_final', sub: '최종' },
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

  if (capMenu === '5_final') { renderFinalCompare(el); return; }

  let html = '';
  // 1_original: 원본 위에 1.4배율 박스 + best 박스 + 이동 방향 오버레이
  if (capMenu === '1_original' && groups.length) {
    html += `<div class="panel" style="margin-bottom:20px;">
      <div class="metric-title" style="font-size:13px;">구도 이동 시각화 (1.4배율 박스 → best 박스, 각 중심 잇기)</div>
      <div class="image-frame" style="background:#0b1020;"><canvas id="orig-overlay" style="display:block;width:100%;height:auto;"></canvas></div>
      <div id="orig-overlay-legend" class="score-meta" style="margin-top:8px;line-height:1.7;"></div>
    </div>`;
  }
  // 3_detail-refine: 48분할 후보를 사진 대신 좌표로만 저장 → 원본 위에 박스로 그려 보여줌.
  const refineGroup = capMenu === '3_detail-refine'
    ? groups.find((g) => g.jsons && g.jsons['scores.json']) : null;
  if (refineGroup) {
    html += `<div class="panel" style="margin-bottom:20px;">
      <div class="metric-title" style="font-size:13px;">정밀 구도 시각화 (48분할 후보 · top3 · best, 좌표만 저장)</div>
      <div class="image-frame" style="background:#0b1020;"><canvas id="refine-overlay" style="display:block;width:100%;height:auto;"></canvas></div>
      <div id="refine-overlay-legend" class="score-meta" style="margin-top:8px;line-height:1.7;"></div>
    </div>`;
  }
  html += groups.length ? groups.map(renderGroup).join('') : '<p style="color:#6b7280;">이 단계 데이터 없음</p>';
  el.innerHTML = html;

  if (capMenu === '1_original' && groups.length) drawOriginalOverlay(groups[0]);
  if (refineGroup) drawRefineOverlay(refineGroup);
}

function drawRefineOverlay(group) {
  const canvas = document.getElementById('refine-overlay');
  const legend = document.getElementById('refine-overlay-legend');
  if (!canvas) return;
  const origImg = (group.images.find((im) => /original_frame/i.test(im.name)) || {}).url;
  const sc = group.jsons['scores.json'] || {};
  const cands = sc.candidates || [];                 // [{source_box, passed_filter, score}]
  const top3 = (sc.top3 || []).map((t) => t.source_box);
  const bestBox = sc.best_crop && sc.best_crop.source_box;
  const offset = sc.offset;
  if (!origImg) { legend.textContent = 'original_frame 이미지 없음'; return; }

  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.onload = () => {
    const W = img.naturalWidth, H = img.naturalHeight;
    canvas.width = W; canvas.height = H;
    ctx.drawImage(img, 0, 0, W, H);
    const lw = Math.max(2, Math.round(W / 320));
    const box = (coords, color, width, label) => {
      const [l, t, r, b] = coords;
      ctx.lineWidth = width; ctx.strokeStyle = color; ctx.strokeRect(l, t, r - l, b - t);
      if (label) {
        const fs = Math.max(12, Math.round(W / 48));
        ctx.font = `bold ${fs}px sans-serif`;
        const tw = ctx.measureText(label).width, th = fs + 6, ty = t >= th ? t - th : t;
        ctx.fillStyle = color; ctx.fillRect(l, ty, tw + 10, th);
        ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle'; ctx.fillText(label, l + 5, ty + th / 2);
      }
    };
    // 48분할 후보: 필터 통과=하늘색, 잘림=반투명 회색
    cands.forEach((c) => box(c.source_box, c.passed_filter ? 'rgba(56,189,248,0.9)' : 'rgba(148,163,184,0.35)', Math.max(1, lw - 1)));
    top3.forEach((t, i) => box(t, '#f59e0b', lw, `top${i + 1}`));   // top3 = 주황
    if (bestBox) {
      box(bestBox, '#22c55e', lw + 1, 'best');                      // best = 초록
      const [l, t, r, b] = bestBox;
      const bcx = (l + r) / 2, bcy = (t + b) / 2, icx = W / 2, icy = H / 2;
      ctx.strokeStyle = '#22c55e'; ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(icx, icy); ctx.lineTo(bcx, bcy); ctx.stroke();
      const dot = (x, y, c) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, y, lw * 2.2, 0, Math.PI * 2); ctx.fill(); };
      dot(icx, icy, '#3b82f6'); dot(bcx, bcy, '#22c55e');
    }
    const items = [
      `<span style="color:#38bdf8;">■</span> 48분할 후보(통과 ${cands.filter((c) => c.passed_filter).length}/${cands.length})`,
      '<span style="color:#f59e0b;">■</span> GAIC top3',
      '<span style="color:#22c55e;">■</span> 최종 best',
    ];
    if (offset) {
      const num = (v) => (typeof v === 'number' ? v.toFixed(1) : v);
      items.push(`<span style="color:#22c55e;">→</span> 이동: dr=${num(offset.dr)}px, θ=${num(offset.theta_deg)}° (dx=${num(offset.dx)}, dy=${num(offset.dy)})`);
    }
    if (sc.score_source) items.push(`점수: ${sc.score_source}`);
    legend.innerHTML = items.join(' &nbsp; ');
  };
  img.onerror = () => { legend.textContent = '이미지 로드 실패'; };
  img.src = origImg;
}

function drawOriginalOverlay(group) {
  const canvas = document.getElementById('orig-overlay');
  const legend = document.getElementById('orig-overlay-legend');
  if (!canvas) return;
  const origImg = (group.images.find((im) => /original_1x/i.test(im.name)) || {}).url;
  const report = group.jsons['report.json'] || group.jsons['result.json'] || {};
  const b2 = report.original_2x;                 // {left,top,right,bottom}
  const bestBox = report.best && report.best.source_box_in_1x; // [l,t,r,b]
  const offset = report.drone_offset;
  if (!origImg) { legend.textContent = 'original_1x 이미지 없음'; return; }

  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.onload = () => {
    const W = img.naturalWidth, H = img.naturalHeight;
    canvas.width = W; canvas.height = H;
    ctx.drawImage(img, 0, 0, W, H);
    const lw = Math.max(2, Math.round(W / 320));
    const fs = Math.max(14, Math.round(W / 42));
    const drawBox = (coords, color, label) => {
      if (!coords) return;
      const [l, t, r, b] = coords;
      ctx.lineWidth = lw; ctx.strokeStyle = color; ctx.strokeRect(l, t, r - l, b - t);
      ctx.font = `bold ${fs}px sans-serif`;
      const tw = ctx.measureText(label).width, th = fs + 8, ty = t >= th ? t - th : t;
      ctx.fillStyle = color; ctx.fillRect(l, ty, tw + 12, th);
      ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle'; ctx.fillText(label, l + 6, ty + th / 2);
    };
    drawBox(b2 ? [b2.left, b2.top, b2.right, b2.bottom] : null, '#3b82f6', '1.4배율(중앙)');
    drawBox(bestBox, '#22c55e', 'best');
    if (bestBox) {
      const [l, t, r, b] = bestBox;
      const bcx = (l + r) / 2, bcy = (t + b) / 2, icx = W / 2, icy = H / 2;
      ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(icx, icy); ctx.lineTo(bcx, bcy); ctx.stroke();
      const dot = (x, y, c) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, y, lw * 2.2, 0, Math.PI * 2); ctx.fill(); };
      dot(icx, icy, '#3b82f6'); dot(bcx, bcy, '#22c55e');
    }
    const items = ['<span style="color:#3b82f6;">■</span> 1.4배율(중앙) 박스'];
    if (bestBox) items.push('<span style="color:#22c55e;">■</span> best 박스');
    if (offset) {
      const num = (v) => (typeof v === 'number' ? v.toFixed(1) : v);
      const polar = (offset.dr != null && offset.theta_deg != null) ? `dr=${num(offset.dr)}px, θ=${num(offset.theta_deg)}° · ` : '';
      items.push(`<span style="color:#f59e0b;">→</span> 이동: ${polar}dx=${offset.dx}, dy=${offset.dy}`);
    }
    if (!bestBox) items.push('<span style="color:#b91c1c;">best 없음</span>');
    legend.innerHTML = items.join(' &nbsp; ');
  };
  img.onerror = () => { legend.textContent = '이미지 로드 실패'; };
  img.src = origImg;
}

function renderFinalCompare(el) {
  const fc = capSession.final_compare;
  if (!fc) { el.innerHTML = '<p style="color:#6b7280;">5_final 데이터 없음 (final.jpg 없음)</p>'; return; }
  const card = (title, side) => side ? `
    <div class="panel" style="width:340px;">
      <div class="metric-title" style="font-size:13px;">${title}</div>
      <div class="image-frame"><img src="${side.url}" style="width:100%;height:auto;display:block;" /></div>
      <div style="margin-top:8px;font-size:14px;">NIMA <b>${side.nima ?? '-'}</b></div>
    </div>` : `<div class="panel" style="width:340px;"><div class="metric-title" style="font-size:13px;">${title}</div><p style="color:#6b7280;">없음</p></div>`;
  let delta = '';
  if (fc.original && fc.final && fc.original.nima != null && fc.final.nima != null) {
    const d = fc.final.nima - fc.original.nima;
    delta = `<div style="margin:6px 0 16px;font-size:14px;">최종 − 원본 NIMA: <b style="color:${d >= 0 ? '#16a34a' : '#b91c1c'};">${d >= 0 ? '+' : ''}${d.toFixed(4)}</b></div>`;
  }
  const initialBlock = fc.initial ? `
    <div class="metric-title" style="font-size:13px;margin-bottom:6px;">최초 촬영본 (initial)</div>
    <div style="display:flex;flex-wrap:wrap;gap:16px;margin-bottom:20px;">${card('최초 (initial.jpg)', fc.initial)}</div>` : '';
  el.innerHTML = `${initialBlock}
    <div class="metric-title" style="font-size:13px;margin-bottom:6px;">원본 vs 최종 비교</div>
    ${delta}
    <div style="display:flex;flex-wrap:wrap;gap:16px;">${card('원본 (original_1x)', fc.original)}${card('최종 (final.jpg)', fc.final)}</div>
    <div class="metric-title" style="font-size:13px;margin:20px 0 6px;">1.4배율 비교 (원본 1.4배 vs 최종 중앙 1.4배 크롭)</div>
    <div style="display:flex;flex-wrap:wrap;gap:16px;">${card('원본 1.4배 (original_2x)', fc.original_2x)}${card('최종 1.4배 (final.jpg 중앙 1.4배 크롭)', fc.final_2x)}</div>`;
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
