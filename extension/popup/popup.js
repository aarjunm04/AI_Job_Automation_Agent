/* ══════════════════════════════════════════════════════════════════
   popup.js — AI Job Agent Chrome Extension (BATCH-6)
   6-state DOM machine. All state transitions via setState().
   No direct DOM manipulation outside setState() / renderState().
   ══════════════════════════════════════════════════════════════════ */

'use strict';

// ══════════════════════════════════════════════════════════════════
// STATE MACHINE DEFINITION
// ══════════════════════════════════════════════════════════════════

const STATES = {
  IDLE:        'idle',
  SCANNING:    'scanning',
  SCORED:      'scored',
  AUTOFILLING: 'autofilling',
  APPLYING:    'applying',
  ERROR:       'error',
};

let currentState = STATES.IDLE;
let jobData = null; // { title, company, url, platform, fit_score,
                    //   resume_id, talking_points, route, job_id }

function setState(newState, payload = {}) {
  currentState = newState;
  if (payload.jobData) jobData = { ...(jobData || {}), ...payload.jobData };
  renderState(newState, payload);
}

// ══════════════════════════════════════════════════════════════════
// RENDER FUNCTION
// ══════════════════════════════════════════════════════════════════

function renderState(state, payload) {
  // Hide ALL state sections first
  document.querySelectorAll('.state-view').forEach(el => { el.hidden = true; });

  // Show only the active state section
  const activeEl = document.getElementById(state + '-view');
  if (activeEl) activeEl.hidden = false;

  switch (state) {
    case STATES.SCORED: {
      const pct = Math.round(((jobData && jobData.fit_score) || 0) * 100);
      const scoreValueEl = document.getElementById('score-value');
      if (scoreValueEl) scoreValueEl.textContent = pct + '%';

      const scoreBarEl = document.getElementById('score-bar');
      if (scoreBarEl) {
        scoreBarEl.style.width = pct + '%';
        scoreBarEl.className = pct >= 60 ? 'bar green' : pct >= 45 ? 'bar amber' : 'bar red';
      }

      const jobTitleEl = document.getElementById('job-title');
      if (jobTitleEl) jobTitleEl.textContent = (jobData && jobData.title) || '';

      const jobCompanyEl = document.getElementById('job-company');
      if (jobCompanyEl) jobCompanyEl.textContent = (jobData && jobData.company) || '';

      const resumeUsedEl = document.getElementById('resume-used');
      if (resumeUsedEl) resumeUsedEl.textContent = (jobData && jobData.resume_id) || '';

      // Populate talking points list
      const tpList = document.getElementById('talking-points');
      if (tpList) {
        tpList.innerHTML = '';
        const points = (jobData && jobData.talking_points) || [];
        points.forEach(tp => {
          const li = document.createElement('li');
          li.textContent = tp;
          tpList.appendChild(li);
        });
      }

      // Show/hide auto-apply button based on route
      const btnAutoApply = document.getElementById('btn-auto-apply');
      if (btnAutoApply) {
        btnAutoApply.hidden = !(jobData && jobData.route === 'auto');
      }
      break;
    }

    case STATES.ERROR: {
      const errEl = document.getElementById('error-message');
      if (errEl) errEl.textContent = payload.error || 'An error occurred.';
      break;
    }
  }
}

// ══════════════════════════════════════════════════════════════════
// STARTUP SEQUENCE
// ══════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
  // 1. Read apiBaseUrl from storage.sync (Batch 4B fix — preserved)
  chrome.storage.sync.get(['apiBaseUrl'], (data) => {
    if (chrome.runtime.lastError) {
      console.warn('[Popup] storage.sync.get error:', chrome.runtime.lastError.message);
    }
    // BASE_URL is managed by the service worker; popup reads it for display only
  });

  // 2. Query current active tab
  let tab = null;
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    tab = tabs[0] || null;
  } catch (err) {
    console.warn('[Popup] tabs.query error:', err);
  }

  if (!tab) {
    setState(STATES.IDLE);
    return;
  }

  // 3. Check chrome.storage.session for existing job data on this tab
  try {
    const session = await new Promise((resolve) => {
      chrome.storage.session.get([String(tab.id)], (result) => {
        if (chrome.runtime.lastError) {
          console.warn('[Popup] session.get error:', chrome.runtime.lastError.message);
          resolve({});
          return;
        }
        resolve(result);
      });
    });

    if (session[tab.id]) {
      setState(STATES.SCORED, { jobData: session[tab.id] });
    } else {
      const isJobSite = ['linkedin.com', 'indeed.com', 'glassdoor.com',
        'wellfound.com', 'arc.dev'].some(h => tab.url && tab.url.includes(h));
      setState(isJobSite ? STATES.SCANNING : STATES.IDLE);
    }
  } catch (err) {
    console.warn('[Popup] session lookup error:', err);
    setState(STATES.IDLE);
  }

  // 4. Load queue count into badge area
  chrome.runtime.sendMessage({ type: 'GET_QUEUE_COUNT' }, (res) => {
    if (chrome.runtime.lastError) {
      console.warn('[Popup] GET_QUEUE_COUNT error:', chrome.runtime.lastError.message);
      return;
    }
    if (res && res.count !== undefined) {
      const queueCountEl = document.getElementById('queue-count');
      if (queueCountEl) queueCountEl.textContent = res.count;
    }
  });

  // ── WIRE ALL BUTTON EVENTS ──────────────────────────────────────

  // btn-autofill
  const btnAutofillEl = document.getElementById('btn-autofill');
  if (btnAutofillEl) {
    btnAutofillEl.addEventListener('click', () => {
      setState(STATES.AUTOFILLING);
      chrome.runtime.sendMessage({
        type: 'REQUEST_AUTOFILL',
        payload: {
          job_id: jobData && jobData.job_id,
          platform: jobData && jobData.platform,
          url: jobData && jobData.url,
        },
      }, (res) => {
        if (chrome.runtime.lastError) {
          setState(STATES.ERROR, { error: chrome.runtime.lastError.message });
          return;
        }
        if (res && res.type === 'AUTOFILL_ERROR') {
          setState(STATES.ERROR, { error: res.error });
        } else {
          // Return to scored — content script performs the DOM fill
          setState(STATES.SCORED);
        }
      });
    });
  }

  // btn-auto-apply
  const btnAutoApplyEl = document.getElementById('btn-auto-apply');
  if (btnAutoApplyEl) {
    btnAutoApplyEl.addEventListener('click', () => {
      setState(STATES.APPLYING);
      chrome.runtime.sendMessage({
        type: 'TRIGGER_AUTO_APPLY',
        payload: {
          url: jobData && jobData.url,
          platform: jobData && jobData.platform,
        },
      }, (res) => {
        if (chrome.runtime.lastError) {
          setState(STATES.ERROR, { error: chrome.runtime.lastError.message });
          return;
        }
        if (res && res.error) {
          setState(STATES.ERROR, { error: res.error });
        } else {
          setState(STATES.SCORED);
        }
      });
    });
  }

  // btn-track-job
  const btnTrackJobEl = document.getElementById('btn-track-job');
  if (btnTrackJobEl) {
    btnTrackJobEl.addEventListener('click', () => {
      chrome.runtime.sendMessage({
        type: 'LOG_APPLICATION',
        payload: {
          job_id: jobData && jobData.job_id,
          platform: jobData && jobData.platform,
          url: jobData && jobData.url,
          status: 'manual',
          resume_id: jobData && jobData.resume_id,
          user_id: '', // Server reads from env — pass empty
        },
      }, (res) => {
        if (chrome.runtime.lastError) {
          console.warn('[Popup] LOG_APPLICATION error:', chrome.runtime.lastError.message);
        }
      });
    });
  }

  // btn-retry
  const btnRetryEl = document.getElementById('btn-retry');
  if (btnRetryEl) {
    btnRetryEl.addEventListener('click', () => {
      setState(STATES.IDLE);
    });
  }
});

// ══════════════════════════════════════════════════════════════════
// INCOMING MESSAGE LISTENER
// ══════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((message) => {
  switch (message.type) {
    case 'MATCH_RESULT':
      setState(STATES.SCORED, { jobData: message.payload });
      break;

    case 'AUTOFILL_ERROR':
      setState(STATES.ERROR, { error: message.error });
      break;

    case 'QUEUE_COUNT': {
      const qCountEl = document.getElementById('queue-count');
      if (qCountEl) qCountEl.textContent = message.count;
      break;
    }

    case 'LOG_DONE': {
      // Flash brief confirmation text — no state change
      const confirmEl = document.getElementById('track-confirm');
      if (confirmEl) {
        confirmEl.hidden = false;
        setTimeout(() => { confirmEl.hidden = true; }, 2000);
      }
      break;
    }

    default:
      // Ignore unhandled messages in popup
      break;
  }
});
