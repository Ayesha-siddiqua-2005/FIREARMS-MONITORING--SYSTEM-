/* ================================================================
   SENTINEL AI — Multi-Page Shared Script
   Auto-detects current page and initializes relevant modules.
   ================================================================ */

document.addEventListener('DOMContentLoaded', () => {
  const page = document.body.dataset.page;

  /* ─── Global Modules ─── */
  ThemeManager.init();
  VoiceToggle.init();
  NavManager.init(page);

  /* ─── Page-Specific Init ─── */
  switch (page) {
    case 'home':       HomePage.init(); break;
    case 'detection':  DetectionEngine.init(); break;
    case 'monitoring': LiveMonitoring.init(); break;
    case 'crimemap':   CrimeMap.init(); break;
    case 'alerts':     AlertsTable.init(); break;
    case 'analytics':  AnalyticsDashboard.init(); break;
    // guide, about — static pages, no JS init needed
  }
});

/* ================================================================
   THEME MANAGER
   ================================================================ */
const ThemeManager = (() => {
  let current = 'dark';

  function init() {
    current = localStorage.getItem('sentinel-theme') || 'dark';
    apply(current);

    const btn = document.getElementById('themeToggle');
    if (btn) btn.addEventListener('click', toggle);
  }

  function toggle() {
    apply(current === 'dark' ? 'light' : 'dark');
  }

  function apply(theme) {
    current = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('sentinel-theme', theme);

    const icon = document.getElementById('themeIcon');
    if (icon) icon.textContent = theme === 'dark' ? '🌙' : '☀️';
  }

  return { init, toggle, get current() { return current; } };
})();

/* ================================================================
   VOICE TOGGLE — Global header icon for muting/unmuting voice alerts
   ================================================================ */
const VoiceToggle = (() => {
  function init() {
    const btn  = document.getElementById('voiceToggle');
    const icon = document.getElementById('voiceIcon');
    if (!btn || !icon) return;

    const isOn = localStorage.getItem('sentinel-voice') === 'true';
    apply(isOn, btn, icon);

    btn.addEventListener('click', () => {
      const next = localStorage.getItem('sentinel-voice') !== 'true';
      localStorage.setItem('sentinel-voice', next);
      apply(next, btn, icon);
      if (!next) stopAllAudio();
    });
  }

  function apply(on, btn, icon) {
    icon.textContent = on ? '\uD83D\uDD0A' : '\uD83D\uDD07';
    btn.classList.toggle('muted', !on);
    btn.title = on ? 'Voice Alerts: ON' : 'Voice Alerts: OFF';
  }

  return { init };
})();

function isAudioAllowed() {
  return localStorage.getItem('sentinel-voice') === 'true';
}

const WEAPON_LABELS = new Set(['PISTOL', 'KNIFE']);

function normalizeThreatLabel(label) {
  return String(label || '').trim().toUpperCase();
}

function isWeaponThreat(threatType) {
  return WEAPON_LABELS.has(normalizeThreatLabel(threatType));
}

function stopAllAudio() {
  if ('speechSynthesis' in window) {
    speechSynthesis.cancel();
  }
  document.querySelectorAll('audio').forEach((audioEl) => {
    audioEl.pause();
    audioEl.currentTime = 0;
  });
}

function speakThreatAlert({ soundToggleEl = null } = {}) {
  const localToggleEnabled = soundToggleEl ? !!soundToggleEl.checked : true;
  if (!isAudioAllowed() || !localToggleEnabled || !('speechSynthesis' in window)) return;

  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance('Warning. Firearm identified in Sector 1. Engaging surveillance lock.');
  utterance.rate = 1.0;
  utterance.pitch = 0.8;
  speechSynthesis.speak(utterance);
}

async function parseApiJsonResponse(response, endpointLabel) {
  const rawText = await response.text();
  let data;

  try {
    data = JSON.parse(rawText);
  } catch (_error) {
    const snippet = String(rawText || '').trim().slice(0, 160).replace(/\s+/g, ' ');
    throw new Error(`${endpointLabel} returned non-JSON (${response.status}). ${snippet || 'Empty response'}`);
  }

  if (!response.ok) {
    throw new Error(data.message || `${endpointLabel} failed (${response.status})`);
  }

  return data;
}

/* ================================================================
   NAVIGATION MANAGER — Highlight active link + hamburger
   ================================================================ */
const NavManager = (() => {
  function init(page) {
    // Highlight active nav link
    document.querySelectorAll('.nav-link').forEach(link => {
      if (link.dataset.page === page) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

    // Hamburger toggle
    const hamburger = document.getElementById('navHamburger');
    const links = document.getElementById('navLinks');
    if (hamburger && links) {
      hamburger.addEventListener('click', () => {
        links.classList.toggle('open');
      });
      // Close on link click (mobile)
      links.querySelectorAll('.nav-link').forEach(l => {
        l.addEventListener('click', () => links.classList.remove('open'));
      });
    }
  }

  return { init };
})();

/* ================================================================
   HOME PAGE — Tactical CCTV Monitor
   ================================================================ */
const HomePage = (() => {
  function init() {
    initCCTVTimestamp();
  }

  function initCCTVTimestamp() {
    const el = document.getElementById('cctvTimestamp');
    if (!el) return;

    function update() {
      const now = new Date();
      const hh = String(now.getHours()).padStart(2, '0');
      const mm = String(now.getMinutes()).padStart(2, '0');
      const ss = String(now.getSeconds()).padStart(2, '0');
      const ms = String(Math.floor(now.getMilliseconds() / 10)).padStart(2, '0');
      el.textContent = `${hh}:${mm}:${ss}:${ms}`;
    }
    update();
    setInterval(update, 40);            // ~25 fps for realistic frame counter
  }

  return { init };
})();

/* ================================================================
   DETECTION ENGINE — File upload + backend inference
   ================================================================ */
const DetectionEngine = (() => {
  let selectedFile = null;

  function init() {
    if (window.__disableDetectionEngine) {
      console.log('DetectionEngine skipped: disabled by page-level override');
      return;
    }

    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnClear = document.getElementById('btnClear');

    if (!uploadZone || !fileInput || !btnAnalyze) {
      console.error('DetectionEngine init failed', {
        uploadZone: !!uploadZone,
        fileInput: !!fileInput,
        btnAnalyze: !!btnAnalyze,
      });
      return;
    }

    console.log('DetectionEngine ready');

    uploadZone.addEventListener('click', () => {
      console.log('Upload zone clicked');
      fileInput.click();
    });

    uploadZone.addEventListener('dragover', (event) => {
      event.preventDefault();
      uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
      uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (event) => {
      event.preventDefault();
      uploadZone.classList.remove('drag-over');
      const droppedFile = event.dataTransfer?.files?.[0];
      console.log('File dropped:', droppedFile);
      if (!droppedFile) return;

      const transfer = new DataTransfer();
      transfer.items.add(droppedFile);
      fileInput.files = transfer.files;
      onFileSelected(droppedFile);
    });

    fileInput.addEventListener('change', () => {
      const file = fileInput.files?.[0] || null;
      console.log('File selected:', file);
      if (!file) return;
      onFileSelected(file);
    });

    btnAnalyze.addEventListener('click', (event) => {
      event.preventDefault();
      console.log('1) btnAnalyze clicked');
      console.log('2) default behavior prevented');
      analyzeSelectedFile();
    });

    if (btnClear) {
      btnClear.addEventListener('click', (event) => {
        event.preventDefault();
        clearResult();
      });
    }
  }

  function onFileSelected(file) {
    selectedFile = file;

    const previewImage = document.getElementById('previewImage');
    const previewVideo = document.getElementById('previewVideo');
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnClear = document.getElementById('btnClear');
    const resultText = document.getElementById('resultText');
    const resultStatus = document.getElementById('resultStatus');

    const extension = (file.name || '').split('.').pop()?.toLowerCase() || '';
    const isMp4OrMov = ['mp4', 'mov'].includes(extension);
    const isVideoMime = file.type === 'video/mp4' || file.type === 'video/quicktime' || file.type?.startsWith('video/');
    const isVideo = isMp4OrMov || isVideoMime;

    if (isVideo) {
      if (previewImage) {
        previewImage.src = '';
        previewImage.style.display = 'none';
      }
      if (previewVideo) {
        previewVideo.src = URL.createObjectURL(file);
        previewVideo.style.display = 'block';
        previewVideo.load();
      }
      if (previewPlaceholder) previewPlaceholder.style.display = 'none';
      console.log('Local video preview shown');
    } else {
      const localUrl = URL.createObjectURL(file);
      if (previewImage) {
        previewImage.src = localUrl;
        previewImage.style.display = 'block';
      }
      if (previewVideo) {
        previewVideo.src = '';
        previewVideo.style.display = 'none';
      }
      if (previewPlaceholder) previewPlaceholder.style.display = 'none';
      console.log('Local image preview shown');
    }

    if (resultText) resultText.textContent = 'Ready to analyze';
    if (resultStatus) resultStatus.className = 'result-status safe';

    if (btnAnalyze) {
      btnAnalyze.disabled = false;
      btnAnalyze.removeAttribute('disabled');
      console.log('Analyze button enabled');
    }

    if (btnClear) btnClear.style.display = 'inline-flex';
  }

  function analyzeSelectedFile() {
    const fileInput = document.getElementById('fileInput');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const previewImage = document.getElementById('previewImage');
    const resultText = document.getElementById('resultText');
    const resultStatus = document.getElementById('resultStatus');

    console.log('3) analyzeSelectedFile() started');
    const file = fileInput?.files?.[0];
    console.log('4) fileInput.files[0] =', file);
    if (!file) {
      console.error('5) STOP: fileInput.files[0] does not exist');
      if (resultText) resultText.textContent = 'No file selected';
      if (resultStatus) resultStatus.className = 'result-status danger';
      return;
    }

    const originalText = btnAnalyze ? btnAnalyze.textContent : 'ANALYZE';
    if (btnAnalyze) {
      btnAnalyze.textContent = 'ANALYZING...';
      btnAnalyze.disabled = true;
      console.log('6) Analyze button set to ANALYZING... and disabled');
    }

    const formData = new FormData();
    formData.append('file', file);
    console.log('7) FormData created and file appended under key "file"');
    console.log('SENDING DATA...');
    console.log('8) Sending fetch POST request to /upload');

    fetch('/upload', {
      method: 'POST',
      body: formData,
    })
      .then(async (response) => {
        console.log('9) Raw fetch response received. status =', response.status);
        const data = await parseApiJsonResponse(response, '/upload');
        return { data };
      })
      .then(({ data }) => {
        console.log('RESPONSE RECEIVED', data);
        console.log('10) Parsed response payload:', { ok: true, data });
        if (data.status !== 'success') {
          console.error('11) Backend returned non-success payload');
          throw new Error(data.message || 'Upload failed');
        }

        const previewVideo = document.getElementById('previewVideo');
        const previewPlaceholder = document.getElementById('previewPlaceholder');
        const processedUrl = data.processed_url;
        const fileType = (data.file_type || '').toLowerCase();

        if (fileType === 'video') {
          console.log('12) Updating previewVideo.src with processed_url');
          if (previewImage) {
            previewImage.src = '';
            previewImage.style.display = 'none';
          }
          if (previewVideo && processedUrl) {
            previewVideo.src = processedUrl + (processedUrl.includes('?') ? '&' : '?') + 't=' + Date.now();
            previewVideo.style.display = 'block';
            previewVideo.load();
            previewVideo.play().catch(() => {});
          }
        } else {
          console.log('12) Updating previewImage.src with processed_url');
          if (previewVideo) {
            previewVideo.src = '';
            previewVideo.style.display = 'none';
          }
          if (previewImage && processedUrl) {
            previewImage.src = processedUrl + (processedUrl.includes('?') ? '&' : '?') + 't=' + Date.now();
            previewImage.style.display = 'block';
          }
        }
        if (previewPlaceholder) previewPlaceholder.style.display = 'none';

        const threatDetected = !!data.threat;
        const objectOnly = !threatDetected && !!data.object_detected;

        if (threatDetected) {
          if (resultText) {
            resultText.textContent = `THREAT DETECTED (${data.confidence_score || 0}%)`;
          }
          if (resultStatus) resultStatus.className = 'result-status danger critical';
          speakThreatAlert();
        } else if (objectOnly) {
          if (resultText) resultText.textContent = 'Object detected (weapon without person overlap)';
          if (resultStatus) resultStatus.className = 'result-status warning';
        } else {
          if (resultText) resultText.textContent = 'Safe';
          if (resultStatus) resultStatus.className = 'result-status safe';
        }
      })
      .catch((error) => {
        console.error('13) Fetch failed:', error);
        if (resultText) resultText.textContent = `Error: ${error.message}`;
        if (resultStatus) resultStatus.className = 'result-status danger';
      })
      .finally(() => {
        console.log('14) Request lifecycle finished. Re-enabling Analyze button');
        if (btnAnalyze) {
          btnAnalyze.disabled = false;
          btnAnalyze.textContent = originalText;
        }
      });
  }

  function clearResult() {
    const previewImage = document.getElementById('previewImage');
    const previewVideo = document.getElementById('previewVideo');
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    const resultText = document.getElementById('resultText');
    const resultStatus = document.getElementById('resultStatus');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnClear = document.getElementById('btnClear');
    const fileInput = document.getElementById('fileInput');

    if (previewImage) {
      previewImage.src = '';
      previewImage.style.display = 'none';
    }
    if (previewVideo) {
      previewVideo.src = '';
      previewVideo.style.display = 'none';
    }
    if (previewPlaceholder) previewPlaceholder.style.display = 'block';
    if (resultText) resultText.textContent = 'Ready — upload an image to begin analysis';
    if (resultStatus) resultStatus.className = 'result-status safe';
    if (btnAnalyze) {
      btnAnalyze.disabled = true;
      btnAnalyze.textContent = '🔍 ANALYZE';
    }
    if (btnClear) btnClear.style.display = 'none';
    if (fileInput) fileInput.value = '';
    selectedFile = null;
    stopAllAudio();
    console.log('Detection view cleared');
  }

  return { init };
})();

/* ================================================================
   LIVE MONITORING — Backend stream + threat polling
   ================================================================ */
const LiveMonitoring = (() => {
  let running = false;
  let startTime = 0;
  let uptimeInterval = null;
  let threatPollInterval = null;
  let lastSpokenThreatState = false;

  function init() {
    if (typeof window.lastSpokenState !== 'boolean') {
      window.lastSpokenState = false;
    }
    const btnStart = document.getElementById('btnStartMonitor');
    const btnStop = document.getElementById('btnStopMonitor');
    if (btnStart) btnStart.addEventListener('click', start);
    if (btnStop) btnStop.addEventListener('click', stop);
  }

  function start() {
    if (running) return;
    const streamImage = document.getElementById('webcamVideo');
    if (streamImage) {
      streamImage.src = `/video_feed?t=${Date.now()}`;
      streamImage.style.display = 'block';
    }

    running = true;
    window.lastSpokenState = false;
    startTime = Date.now();
    document.getElementById('btnStartMonitor').disabled = true;
    document.getElementById('btnStopMonitor').disabled = false;
    document.getElementById('feedLabel').textContent = '● LIVE';

    uptimeInterval = setInterval(updateUptime, 1000);
    threatPollInterval = setInterval(pollThreatStatus, 1000);
  }

  function stop() {
    if (!running) return;
    running = false;

    if (uptimeInterval) {
      clearInterval(uptimeInterval);
      uptimeInterval = null;
    }
    if (threatPollInterval) {
      clearInterval(threatPollInterval);
      threatPollInterval = null;
    }

    const streamImage = document.getElementById('webcamVideo');
    if (streamImage) {
      streamImage.src = '';
    }

    document.getElementById('btnStartMonitor').disabled = false;
    document.getElementById('btnStopMonitor').disabled = true;
    document.getElementById('feedLabel').textContent = '● OFFLINE';
    document.getElementById('feedFps').textContent = '—';
    document.getElementById('detailFps').textContent = '—';
    document.getElementById('detailUptime').textContent = '00:00:00';
    setThreat({ isThreat: false });
    window.lastSpokenState = false;
    lastSpokenThreatState = false;
  }

  function updateUptime() {
    if (!running) return;

    const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
    const hours = String(Math.floor(elapsedSeconds / 3600)).padStart(2, '0');
    const minutes = String(Math.floor((elapsedSeconds % 3600) / 60)).padStart(2, '0');
    const seconds = String(elapsedSeconds % 60).padStart(2, '0');
    document.getElementById('detailUptime').textContent = `${hours}:${minutes}:${seconds}`;
    document.getElementById('feedFps').textContent = 'STREAM';
    document.getElementById('detailFps').textContent = 'STREAM';
  }

  async function pollThreatStatus() {
    if (!running) return;
    try {
      const response = await fetch('/threat_status');
      const data = await parseApiJsonResponse(response, '/threat_status');
      const threat = !!data.threat;
      const objectOnly = !threat && !!data.object;

      setThreat({
        isThreat: threat,
        isObjectOnly: objectOnly,
        threatCount: Number(data.threat_count || 0),
        objectCount: Number(data.object_count || 0),
        threatType: data.threat_type || 'Threat',
      });

      const soundToggle = document.getElementById('soundAlertsToggle');
      const isSoundEnabled = soundToggle ? soundToggle.checked : true;

      if (threat && isSoundEnabled && isAudioAllowed() && !window.lastSpokenState) {
        console.log('🔊 Triggering Tactical Voice Alert');
        window.speechSynthesis.cancel();
        const msg = new SpeechSynthesisUtterance('Warning. Threat detected. Sector 1 clear for intervention.');
        msg.rate = 1.1;
        msg.pitch = 0.8;
        window.speechSynthesis.speak(msg);
        window.lastSpokenState = true;
      } else if (!threat) {
        window.lastSpokenState = false;
      } else if (!isSoundEnabled || !isAudioAllowed()) {
        stopAllAudio();
      }

      lastSpokenThreatState = threat;
    } catch (error) {
      console.error('Threat polling failed:', error);
    }
  }

  function setThreat({ isThreat, isObjectOnly = false, threatCount = 0, objectCount = 0, threatType = 'Threat' } = {}) {
    const ring = document.getElementById('threatRing');
    const text = document.getElementById('threatText');
    const status = document.getElementById('detailStatus');
    const detections = document.getElementById('detailDetections');

    if (isThreat) {
      if (ring) ring.className = 'threat-ring critical';
      if (text) text.textContent = 'THREAT';
      if (status) {
        status.textContent = 'THREAT DETECTED';
        status.className = 'danger-text';
      }
      if (detections) detections.textContent = `${Math.max(threatCount, 1)}`;
    } else if (isObjectOnly) {
      if (ring) ring.className = 'threat-ring warning';
      if (text) text.textContent = 'OBJECT';
      if (status) {
        status.textContent = 'Weapon detected without person overlap';
        status.className = 'warning-text';
      }
      if (detections) detections.textContent = `${Math.max(objectCount, 1)}`;
    } else {
      if (ring) ring.className = 'threat-ring safe';
      if (text) text.textContent = 'SAFE';
      if (status) {
        status.textContent = 'All Clear';
        status.className = 'safe-text';
      }
      if (detections) detections.textContent = '0';
    }
  }

  return { init };
})();

/* ================================================================
   CRIME MAP — Leaflet.js
   ================================================================ */
const CrimeMap = (() => {
  function init() {
    const el = document.getElementById('crimeMap');
    if (!el || typeof L === 'undefined') return;

    const map = L.map('crimeMap', { zoomControl: true }).setView([20, 0], 2);

    // Tile layer — CartoDB Dark Matter (tactical dark mode)
    const darkTile = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
    const lightTile = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
    const theme = document.documentElement.getAttribute('data-theme');
    L.tileLayer(theme === 'light' ? lightTile : darkTile, {
      attribution: '&copy; CartoDB Dark Matter',
      maxZoom: 19,
    }).addTo(map);

    // High-risk smart city threat clusters — major urban areas only
    const incidents = [
      // North America
      { lat: 40.758, lng: -73.985, type: 'Firearm Discharged', date: 'Oct 12, 2025', city: 'New York, USA',          incidents: 187, size: 28 },
      { lat: 41.878, lng: -87.630, type: 'Multiple Shootings',  date: 'Mar 18, 2025', city: 'Chicago, USA',           incidents: 214, size: 30 },
      { lat: 34.052, lng: -118.244, type: 'Weapon Detected',    date: 'Jan 5, 2026',  city: 'Los Angeles, USA',       incidents: 142, size: 24 },
      { lat: 29.760, lng: -95.370, type: 'Firearm Discharged',  date: 'Jul 22, 2024', city: 'Houston, USA',           incidents: 98,  size: 20 },
      { lat: 25.761, lng: -80.192, type: 'Weapon Detected',     date: 'Sep 9, 2025',  city: 'Miami, USA',             incidents: 76,  size: 18 },
      { lat: 33.749, lng: -84.388, type: 'Firearm Discharged',  date: 'Jun 3, 2025',  city: 'Atlanta, USA',           incidents: 85,  size: 19 },
      { lat: 19.433, lng: -99.133, type: 'Firearm Discharged',  date: 'Dec 1, 2025',  city: 'Mexico City, Mexico',    incidents: 165, size: 26 },

      // South America
      { lat: -22.906, lng: -43.173, type: 'Active Threat Zone', date: 'Feb 20, 2026', city: 'Rio de Janeiro, Brazil', incidents: 231, size: 32 },
      { lat: -23.550, lng: -46.634, type: 'Firearm Discharged', date: 'Aug 9, 2024',  city: 'São Paulo, Brazil',      incidents: 178, size: 26 },
      { lat: 4.711,  lng: -74.072,  type: 'Weapon Detected',    date: 'Nov 14, 2025', city: 'Bogotá, Colombia',       incidents: 110, size: 22 },
      { lat: -34.604, lng: -58.382, type: 'Weapon Detected',    date: 'Jul 4, 2024',  city: 'Buenos Aires, Argentina',incidents: 62,  size: 16 },

      // Europe
      { lat: 51.507, lng: -0.128,  type: 'Weapon Detected',     date: 'Sep 3, 2025',  city: 'London, UK',             incidents: 89,  size: 20 },
      { lat: 48.857, lng: 2.352,   type: 'Firearm Discharged',  date: 'Nov 28, 2024', city: 'Paris, France',          incidents: 72,  size: 18 },
      { lat: 52.520, lng: 13.405,  type: 'Weapon Detected',     date: 'Feb 14, 2026', city: 'Berlin, Germany',        incidents: 45,  size: 14 },
      { lat: 41.390, lng: 2.154,   type: 'Suspicious Activity', date: 'Apr 22, 2025', city: 'Barcelona, Spain',       incidents: 38,  size: 13 },

      // Africa
      { lat: -33.925, lng: 18.424, type: 'Active Threat Zone',  date: 'Jan 30, 2026', city: 'Cape Town, South Africa',incidents: 196, size: 28 },
      { lat: -1.286, lng: 36.817,  type: 'Firearm Discharged',  date: 'Jan 19, 2026', city: 'Nairobi, Kenya',         incidents: 120, size: 22 },
      { lat: 6.524,  lng: 3.379,   type: 'Multiple Shootings',  date: 'Oct 7, 2025',  city: 'Lagos, Nigeria',         incidents: 148, size: 24 },
      { lat: 30.044, lng: 31.236,  type: 'Suspicious Activity', date: 'Mar 8, 2025',  city: 'Cairo, Egypt',           incidents: 55,  size: 16 },
      { lat: -26.204, lng: 28.046, type: 'Firearm Discharged',  date: 'May 18, 2025', city: 'Johannesburg, S. Africa',incidents: 172, size: 26 },

      // Asia
      { lat: 28.614, lng: 77.209,  type: 'Weapon Detected',     date: 'Apr 17, 2025', city: 'New Delhi, India',       incidents: 68,  size: 18 },
      { lat: 14.599, lng: 120.984, type: 'Firearm Discharged',  date: 'Aug 2, 2025',  city: 'Manila, Philippines',    incidents: 94,  size: 20 },
      { lat: 13.756, lng: 100.502, type: 'Suspicious Activity', date: 'Jul 15, 2025', city: 'Bangkok, Thailand',      incidents: 42,  size: 14 },
    ];

    incidents.forEach(inc => {
      const s = inc.size;
      const icon = L.divIcon({
        className: '',
        html: `<div class="pulse-marker" style="width:${s}px;height:${s}px;"></div>`,
        iconSize: [s, s],
        iconAnchor: [s / 2, s / 2],
      });

      L.marker([inc.lat, inc.lng], { icon }).addTo(map).bindPopup(
        `<div class="threat-popup-title">${inc.type}</div>` +
        `<div class="threat-popup-meta">` +
        `<b>${inc.city}</b><br>` +
        `Incidents: ${inc.incidents}<br>` +
        `Last: ${inc.date}` +
        `</div>`
      );
    });

    // Resize fix
    setTimeout(() => map.invalidateSize(), 300);
  }

  return { init };
})();

/* ================================================================
   ALERTS TABLE — Server-rendered from backend logs
   ================================================================ */
const AlertsTable = (() => {
  function init() {
    return;
  }

  return { init };
})();

/* ================================================================
   ANALYTICS DASHBOARD — Chart.js
   ================================================================ */
const AnalyticsDashboard = (() => {
  function init() {
    if (typeof Chart === 'undefined') return;

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor  = isDark ? '#cbd5e1' : '#374151';
    const gridColor  = isDark ? 'rgba(148,163,184,0.08)' : 'rgba(0,0,0,0.06)';
    const primary    = isDark ? '#06b6d4' : '#0891b2';
    const secondary  = isDark ? '#22d3ee' : '#06b6d4';
    const accent     = isDark ? '#8b5cf6' : '#7C3AED';
    const alert      = isDark ? '#ef4444' : '#DC2626';

    Chart.defaults.color       = textColor;
    Chart.defaults.borderColor = gridColor;

    const data = window.analyticsData;
    if (!data) return;

    // 1. Detection Trends
    const ctxTrends = document.getElementById('chartTrends');
    if (ctxTrends) {
      new Chart(ctxTrends, {
        type: 'line',
        data: {
          labels: data.trends.labels,
          datasets: [{
            label: 'Detections',
            data: data.trends.detections,
            borderColor: primary, backgroundColor: primary + '20',
            fill: true, tension: 0.4, pointRadius: 4,
          }, {
            label: 'Threats',
            data: data.trends.threats,
            borderColor: alert, backgroundColor: alert + '20',
            fill: true, tension: 0.4, pointRadius: 4,
          }]
        },
        options: { responsive: true, plugins: { legend: { position: 'top' } } }
      });
    }

    // 2. Threat Distribution
    const ctxThreats = document.getElementById('chartThreats');
    if (ctxThreats) {
      new Chart(ctxThreats, {
        type: 'doughnut',
        data: {
          labels: data.threatDist.labels,
          datasets: [{
            data: data.threatDist.data,
            backgroundColor: [alert, accent, primary],
            borderWidth: 0,
          }]
        },
        options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
      });
    }

    // 3. Usage by Source
    const ctxUsage = document.getElementById('chartUsage');
    if (ctxUsage) {
      new Chart(ctxUsage, {
        type: 'bar',
        data: {
          labels: data.sourceUsage.labels,
          datasets: [{
            label: 'Scans',
            data: data.sourceUsage.data,
            backgroundColor: [primary + '80', accent + '80', secondary + '80'],
            borderColor: [primary, accent, secondary],
            borderWidth: 1,
          }]
        },
        options: { responsive: true, plugins: { legend: { display: false } } }
      });
    }

    // 4. Hourly Activity
    const ctxHourly = document.getElementById('chartHourly');
    if (ctxHourly) {
      const hourLabels = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, '0') + ':00');
      new Chart(ctxHourly, {
        type: 'bar',
        data: {
          labels: hourLabels,
          datasets: [{
            label: 'Events',
            data: data.hourly,
            backgroundColor: secondary + '60',
            borderColor: secondary, borderWidth: 1,
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: { x: { ticks: { maxRotation: 90, font: { size: 9 } } } }
        }
      });
    }
  }

  return { init };
})();
