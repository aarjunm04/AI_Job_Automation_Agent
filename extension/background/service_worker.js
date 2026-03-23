/* ══════════════════════════════════════════════════════════════════
   service_worker.js — AI Job Agent Chrome Extension (BATCH-6)
   MV3 Background Service Worker.
   Implements: SPA navigation handler, session storage, badge updates,
   full message routing, and api_client.js integration.
   ══════════════════════════════════════════════════════════════════ */

'use strict';

// ══════════════════════════════════════════════════════════════════
// SECTION F — IMPORTS
// ══════════════════════════════════════════════════════════════════

importScripts('../utils/api_client.js');

// ══════════════════════════════════════════════════════════════════
// SECTION A — CONSTANTS AND STORAGE INIT
// ══════════════════════════════════════════════════════════════════

const PLATFORM_HOSTNAMES = [
  'linkedin.com',
  'indeed.com',
  'glassdoor.com',
  'wellfound.com',
  'arc.dev',
];

// Module-level BASE_URL — never a const, always updated from storage.sync
let BASE_URL = 'http://localhost:8000';
let API_KEY = '';

// Read apiBaseUrl and apiKey from chrome.storage.sync on SW startup
chrome.storage.sync.get(['apiBaseUrl', 'apiKey'], (data) => {
  if (chrome.runtime.lastError) {
    console.warn('[SW] storage.sync.get error:', chrome.runtime.lastError.message);
    return;
  }
  if (data.apiBaseUrl) BASE_URL = data.apiBaseUrl;
  if (data.apiKey) API_KEY = data.apiKey;
});

// Update BASE_URL and API_KEY whenever they change in storage.sync
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'sync') return;
  if (changes.apiBaseUrl) {
    BASE_URL = changes.apiBaseUrl.newValue || 'http://localhost:8000';
    console.log('[SW] BASE_URL updated:', BASE_URL);
  }
  if (changes.apiKey) {
    API_KEY = changes.apiKey.newValue || '';
  }
});

// ══════════════════════════════════════════════════════════════════
// SECTION D — setBadge HELPER
// ══════════════════════════════════════════════════════════════════

function setBadge(tabId, text, color) {
  chrome.action.setBadgeText({ text: String(text), tabId }, () => {
    if (chrome.runtime.lastError) {
      console.warn('[SW] setBadgeText error:', chrome.runtime.lastError.message);
    }
  });
  chrome.action.setBadgeBackgroundColor({ color, tabId }, () => {
    if (chrome.runtime.lastError) {
      console.warn('[SW] setBadgeBackgroundColor error:', chrome.runtime.lastError.message);
    }
  });
}

// ══════════════════════════════════════════════════════════════════
// SECTION B — chrome.tabs.onUpdated SPA NAVIGATION HANDLER
// ══════════════════════════════════════════════════════════════════

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (!tab.url) return;
  const isJobSite = PLATFORM_HOSTNAMES.some(h => tab.url.includes(h));
  if (!isJobSite) return;

  if (changeInfo.status === 'loading' || changeInfo.status === 'complete') {
    // Inject content script — fail-soft if tab disallows injection
    chrome.scripting.executeScript({
      target: { tabId },
      files: ['content_scripts/content.js'],
    }).catch(() => {});

    // Update badge to SCANNING state
    setBadge(tabId, '...', '#FFA500');
  }
});

// ══════════════════════════════════════════════════════════════════
// SECTION C — chrome.runtime.onMessage HANDLER
// ══════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const tabId = sender.tab ? sender.tab.id : null;

  switch (message.type) {

    // ── JOB_DETECTED ──────────────────────────────────────────────
    case 'JOB_DETECTED': {
      const payload = message.payload || {};
      const { title, company, url, platform } = payload;

      // 1. Store job data in session keyed by tabId
      if (tabId !== null) {
        chrome.storage.session.set({ [tabId]: payload }, () => {
          if (chrome.runtime.lastError) {
            console.warn('[SW] session.set error:', chrome.runtime.lastError.message);
          }
        });
        setBadge(tabId, '...', '#FFA500');
      }

      // 2. Call POST /match via api_client
      JobAgentAPI.callMatch(BASE_URL, API_KEY, url, '')
        .then((result) => {
          if (result.error) {
            console.error('[SW] /match error:', result.message);
            if (tabId !== null) setBadge(tabId, '!', '#CC0000');
            sendResponse({ type: 'MATCH_ERROR', error: result.message });
            return;
          }

          const data = result.data || {};
          const score = data.fit_score || 0;

          // 3. Store result in session and update badge
          if (tabId !== null) {
            const merged = { ...payload, ...data };
            chrome.storage.session.set({ [tabId]: merged }, () => {
              if (chrome.runtime.lastError) {
                console.warn('[SW] session.set (match) error:', chrome.runtime.lastError.message);
              }
            });

            const badgeText = Math.round(score * 100) + '%';
            const badgeColor = score >= 0.60
              ? '#00C851'
              : score >= 0.45
                ? '#FF8800'
                : '#CC0000';
            setBadge(tabId, badgeText, badgeColor);
          }

          // 5. Send MATCH_RESULT back to popup
          sendResponse({ type: 'MATCH_RESULT', payload: data });
        })
        .catch((err) => {
          console.error('[SW] /match threw:', err);
          if (tabId !== null) setBadge(tabId, '!', '#CC0000');
          sendResponse({ type: 'MATCH_ERROR', error: String(err) });
        });

      return true; // async response
    }

    // ── REQUEST_AUTOFILL ──────────────────────────────────────────
    case 'REQUEST_AUTOFILL': {
      const { job_id, platform, url } = message.payload || {};

      JobAgentAPI.callAutofill(BASE_URL, API_KEY, url, [])
        .then((result) => {
          if (result.error) {
            console.error('[SW] /autofill error:', result.message);
            sendResponse({ type: 'AUTOFILL_ERROR', error: result.message });
            return;
          }

          const fields = result.data || {};
          // Send autofill result to content script on the tab
          if (tabId !== null) {
            chrome.tabs.sendMessage(tabId, { type: 'AUTOFILL_RESULT', payload: fields }, () => {
              if (chrome.runtime.lastError) {
                console.warn('[SW] sendMessage to content script error:', chrome.runtime.lastError.message);
              }
            });
          }
          sendResponse({ type: 'AUTOFILL_RESULT', payload: fields });
        })
        .catch((err) => {
          console.error('[SW] /autofill threw:', err);
          sendResponse({ type: 'AUTOFILL_ERROR', error: String(err) });
        });

      return true; // async response
    }

    // ── TRIGGER_AUTO_APPLY ────────────────────────────────────────
    case 'TRIGGER_AUTO_APPLY': {
      const { url, platform } = message.payload || {};
      const endpoint = BASE_URL.replace(/\/+$/, '') + '/trigger-auto-apply';
      const headers = { 'Content-Type': 'application/json' };
      if (API_KEY) headers['X-API-Key'] = API_KEY;

      fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({ url, platform }),
      })
        .then(async (res) => {
          if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new Error('HTTP ' + res.status + (text ? ': ' + text : ''));
          }
          return res.json();
        })
        .then((data) => {
          // Badge success — restore after 3 seconds
          if (tabId !== null) {
            const prevScore = null; // restored via session lookup below
            setBadge(tabId, '✓', '#00C851');
            setTimeout(() => {
              chrome.storage.session.get([String(tabId)], (session) => {
                if (chrome.runtime.lastError) return;
                const stored = session[tabId];
                if (stored && stored.fit_score !== undefined) {
                  const pct = Math.round(stored.fit_score * 100) + '%';
                  const color = stored.fit_score >= 0.60
                    ? '#00C851'
                    : stored.fit_score >= 0.45
                      ? '#FF8800'
                      : '#CC0000';
                  setBadge(tabId, pct, color);
                } else {
                  setBadge(tabId, '', '#888888');
                }
              });
            }, 3000);
          }
          sendResponse({ type: 'AUTO_APPLY_DONE', payload: data });
        })
        .catch((err) => {
          console.error('[SW] /trigger-auto-apply threw:', err);
          sendResponse({ error: err.message });
        });

      return true; // async response
    }

    // ── LOG_APPLICATION ───────────────────────────────────────────
    case 'LOG_APPLICATION': {
      const logPayload = message.payload || {};

      JobAgentAPI.callLogApplication(BASE_URL, API_KEY, logPayload)
        .then((result) => {
          if (result.error) {
            console.error('[SW] /log-application error:', result.message);
            // Do not crash — log only
            return;
          }
          // Notify popup
          chrome.runtime.sendMessage({ type: 'LOG_DONE' }, () => {
            if (chrome.runtime.lastError) {
              // Popup may be closed — ignore
            }
          });
          sendResponse({ type: 'LOG_DONE' });
        })
        .catch((err) => {
          console.error('[SW] /log-application threw:', err);
          // Do not crash
        });

      return true; // async response
    }

    // ── GET_QUEUE_COUNT ───────────────────────────────────────────
    case 'GET_QUEUE_COUNT': {
      JobAgentAPI.callQueueCount(BASE_URL, API_KEY)
        .then((result) => {
          const count = (result.error ? null : (result.data && result.data.count !== undefined ? result.data.count : 0));
          sendResponse({ type: 'QUEUE_COUNT', count });
        })
        .catch((err) => {
          console.error('[SW] /queue-count threw:', err);
          sendResponse({ type: 'QUEUE_COUNT', count: null });
        });

      return true; // async response
    }

    // ── FIELDS_UPDATED ────────────────────────────────────────────
    case 'FIELDS_UPDATED':
      console.log('[SW] FIELDS_UPDATED', message.payload);
      break;

    // ── CONTENT_READY ─────────────────────────────────────────────
    case 'CONTENT_READY':
      console.log('[SW] CONTENT_READY', message.payload);
      break;

    // ── DEFAULT ───────────────────────────────────────────────────
    default:
      console.warn('[SW] Unhandled message type:', message.type);
  }
});

// ══════════════════════════════════════════════════════════════════
// SECTION E — chrome.runtime.onInstalled HANDLER
// ══════════════════════════════════════════════════════════════════

chrome.runtime.onInstalled.addListener((details) => {
  // Preserve existing install logic: set default api_base_url
  if (details.reason === 'install') {
    chrome.storage.local.set({ api_base_url: 'http://localhost:8000' }, () => {
      if (chrome.runtime.lastError) {
        console.warn('[SW] local.set error:', chrome.runtime.lastError.message);
      }
    });
  }

  // BATCH-6 addition: set default apiBaseUrl in storage.sync if not set
  chrome.storage.sync.get(['apiBaseUrl'], ({ apiBaseUrl }) => {
    if (chrome.runtime.lastError) {
      console.warn('[SW] sync.get onInstalled error:', chrome.runtime.lastError.message);
      return;
    }
    if (!apiBaseUrl) {
      chrome.storage.sync.set({ apiBaseUrl: 'http://localhost:8000' }, () => {
        if (chrome.runtime.lastError) {
          console.warn('[SW] sync.set onInstalled error:', chrome.runtime.lastError.message);
        }
      });
    }
  });
});
