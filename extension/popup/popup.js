/* ══════════════════════════════════════════════════════════════════
   popup.js — AI Job Agent Chrome Extension v1
   Sidebar panel logic. All API calls go directly to the FastAPI
   backend via fetch(). Zero MCP references. Zero chrome.runtime
   message relay for API calls — direct fetch() only.
   ══════════════════════════════════════════════════════════════════ */

'use strict';

// ══════════════════════════════════════════════════════════════════
// CONFIG — read from chrome.storage.local, written by service worker
// ══════════════════════════════════════════════════════════════════

/**
 * Resolves the FastAPI base URL from chrome.storage.local, falling
 * back to http://localhost:8000 if not set.
 * @returns {Promise<string>} The API base URL (no trailing slash).
 */
async function getApiBase() {
    try {
        const result = await chrome.storage.local.get('api_base_url');
        const url = result.api_base_url;
        return (url && typeof url === 'string' && url.trim()) ? url.trim().replace(/\/$/, '') : 'http://localhost:8000';
    } catch (_) {
        return 'http://localhost:8000';
    }
}

// ══════════════════════════════════════════════════════════════════
// STATE LOG — max 5 entries, prepended newest-first
// ══════════════════════════════════════════════════════════════════

/**
 * Prepends a timestamped entry to the #status-log div.
 * Keeps at most 5 entries; removes oldest when exceeded.
 * @param {string} message - Human-readable event message.
 * @param {'ok'|'warn'|'err'|''} [level=''] - Optional CSS level class.
 */
function addToStatusLog(message, level) {
    const log = document.getElementById('status-log');
    if (!log) return;

    const now = new Date();
    const ts = now.getHours().toString().padStart(2, '0') + ':' +
        now.getMinutes().toString().padStart(2, '0') + ':' +
        now.getSeconds().toString().padStart(2, '0');

    const entry = document.createElement('div');
    entry.className = 'log-entry' + (level ? ' log-' + level : '');
    entry.innerHTML = '<span class="log-ts">[' + ts + ']</span>' +
        document.createTextNode(message).textContent;

    // Prepend so newest is at top
    if (log.firstChild) {
        log.insertBefore(entry, log.firstChild);
    } else {
        log.appendChild(entry);
    }

    // Trim to max 5 entries
    while (log.children.length > 5) {
        log.removeChild(log.lastChild);
    }
}

// ══════════════════════════════════════════════════════════════════
// CONNECTION CHECK — runs every 30 seconds
// ══════════════════════════════════════════════════════════════════

/**
 * GETs {API_BASE}/health and updates the connection status dot.
 * Sets #conn-dot to class "connected" (green) or "disconnected" (red).
 * @returns {Promise<void>}
 */
async function checkConnection() {
    const dot = document.getElementById('conn-dot');
    try {
        const base = await getApiBase();
        const res = await fetch(base + '/health', { method: 'GET', signal: AbortSignal.timeout(5000) });
        if (res.ok) {
            if (dot) { dot.className = 'conn-dot connected'; dot.title = 'FastAPI connected'; }
        } else {
            throw new Error('HTTP ' + res.status);
        }
    } catch (err) {
        if (dot) { dot.className = 'conn-dot disconnected'; dot.title = 'FastAPI unreachable: ' + err.message; }
    }
}

// ══════════════════════════════════════════════════════════════════
// JOB MATCH — POST /match
// ══════════════════════════════════════════════════════════════════

/**
 * Reads the active tab URL and POSTs to {API_BASE}/match.
 * Populates the Score Card, Resume Suggestion, and Talking Points
 * sections from the response.
 *
 * Response shape: {fit_score, title, company, match_reasoning,
 *   resume_suggested, similarity_score, talking_points[]}
 *
 * On any failure shows "Unable to load match data" in the score card.
 * @returns {Promise<void>}
 */
async function loadJobMatch() {
    let currentUrl = '';
    try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        currentUrl = (tabs[0] && tabs[0].url) ? tabs[0].url : '';
    } catch (_) { }

    if (!currentUrl || currentUrl.startsWith('chrome://') || currentUrl.startsWith('chrome-extension://')) {
        _renderMatchError('Open a job page first.');
        return;
    }

    try {
        const base = await getApiBase();
        const res = await fetch(base + '/match', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_url: currentUrl }),
            signal: AbortSignal.timeout(15000),
        });

        if (!res.ok) {
            throw new Error('HTTP ' + res.status);
        }

        const data = await res.json();
        _renderMatch(data);
        addToStatusLog('Match loaded for ' + _truncate(currentUrl, 50), 'ok');
    } catch (err) {
        _renderMatchError('Unable to load match data: ' + err.message);
        addToStatusLog('Match failed: ' + err.message, 'err');
    }
}

/**
 * Renders a successful /match API response into the sidebar UI.
 * @param {Object} data - /match response payload.
 */
function _renderMatch(data) {
    // ─ Score badge ─
    const badge = document.getElementById('score-badge');
    if (badge) {
        const pct = data.fit_score !== undefined && data.fit_score !== null
            ? Math.round(data.fit_score * 100) + '%'
            : '—';
        badge.textContent = pct;
        badge.className = 'score-badge';
        if (data.fit_score >= 0.70) {
            badge.classList.add('high');
        } else if (data.fit_score >= 0.50) {
            badge.classList.add('amber');
        } else if (data.fit_score !== undefined && data.fit_score !== null) {
            badge.classList.add('low');
        }
    }

    // ─ Title / company / reasoning ─
    const titleEl = document.getElementById('score-title');
    if (titleEl) titleEl.textContent = data.title || 'Unknown Role';

    const compEl = document.getElementById('score-company');
    if (compEl) compEl.textContent = data.company || '';

    const reasonEl = document.getElementById('score-reasoning');
    if (reasonEl) reasonEl.textContent = data.match_reasoning || '';

    // ─ Resume suggestion ─
    const resumeEl = document.getElementById('resume-name');
    if (resumeEl) resumeEl.textContent = data.resume_suggested || '—';

    const simEl = document.getElementById('similarity-badge');
    if (simEl) {
        simEl.textContent = data.similarity_score !== undefined
            ? (data.similarity_score * 100).toFixed(1) + '% sim'
            : '';
    }

    // ─ Talking points ─
    _renderTalkingPoints(data.talking_points || []);
}

/**
 * Shows an error message in the score card area.
 * @param {string} msg - Error message to display.
 */
function _renderMatchError(msg) {
    const titleEl = document.getElementById('score-title');
    if (titleEl) titleEl.textContent = msg;
    const badge = document.getElementById('score-badge');
    if (badge) { badge.textContent = '—'; badge.className = 'score-badge'; }
}

/**
 * Populates the #tp-list element with talking point items.
 * @param {string[]} points - Array of talking point strings.
 */
function _renderTalkingPoints(points) {
    const list = document.getElementById('tp-list');
    if (!list) return;
    list.innerHTML = '';

    if (!points || points.length === 0) {
        const li = document.createElement('li');
        li.className = 'tp-item tp-placeholder';
        li.textContent = 'No talking points available.';
        list.appendChild(li);
        return;
    }

    const max = Math.min(points.length, 5);
    for (let i = 0; i < max; i++) {
        const li = document.createElement('li');
        li.className = 'tp-item';
        li.textContent = points[i];
        list.appendChild(li);
    }
}

// ══════════════════════════════════════════════════════════════════
// AUTOFILL — GET /autofill → send to content script
// ══════════════════════════════════════════════════════════════════

/**
 * Fetches user profile from {API_BASE}/autofill and sends it to the
 * active tab's content script via chrome.tabs.sendMessage with
 * action "autofill". Logs outcome to the status log div.
 *
 * Response shape: {USERNAME, USER_EMAIL, USER_PHONE,
 *   USER_LINKEDIN_URL, USER_PORTFOLIO_URL, USER_LOCATION,
 *   USER_YEARS_EXPERIENCE, default_resume, field_mappings, ...}
 * @returns {Promise<void>}
 */
async function autofillForm() {
    try {
        const base = await getApiBase();
        const res = await fetch(base + '/autofill', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_url: await _getActiveTabUrl(), detected_fields: [] }),
            signal: AbortSignal.timeout(10000),
        });

        if (!res.ok) {
            throw new Error('HTTP ' + res.status);
        }

        const data = await res.json();

        // Send profile data to active content script
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tabs[0] && tabs[0].id) {
            try {
                await chrome.tabs.sendMessage(tabs[0].id, { action: 'autofill', data: data });
            } catch (tabErr) {
                addToStatusLog('Content script not available on this page', 'warn');
                return;
            }
        }

        addToStatusLog('Autofill triggered ✓', 'ok');
    } catch (err) {
        addToStatusLog('Autofill failed: ' + err.message, 'err');
    }
}

// ══════════════════════════════════════════════════════════════════
// QUEUE TO NOTION — POST /apply/manual
// ══════════════════════════════════════════════════════════════════

/**
 * POSTs the current tab URL to {API_BASE}/apply/manual to queue the
 * job. Logs success or failure to the status log div.
 * @returns {Promise<void>}
 */
async function queueToNotion() {
    try {
        const currentUrl = await _getActiveTabUrl();
        if (!currentUrl) {
            addToStatusLog('No active job URL found', 'warn');
            return;
        }

        const base = await getApiBase();
        const res = await fetch(base + '/apply/manual', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_post_id: _pseudoId(currentUrl),
                user_id: 'extension_user',
                resume_filename: '',
                notes: 'Queued via Chrome Extension',
                source: 'extension',
            }),
            signal: AbortSignal.timeout(10000),
        });

        if (!res.ok) {
            throw new Error('HTTP ' + res.status);
        }

        addToStatusLog('Queued to Notion ✓', 'ok');
    } catch (err) {
        addToStatusLog('Queue failed — check API connection: ' + err.message, 'err');
    }
}

// ══════════════════════════════════════════════════════════════════
// MARK APPLIED — POST /apply/manual
// ══════════════════════════════════════════════════════════════════

/**
 * POSTs to {API_BASE}/apply/manual marking the current job as
 * manually applied. Logs outcome to status log.
 * @returns {Promise<void>}
 */
async function markApplied() {
    try {
        const currentUrl = await _getActiveTabUrl();
        if (!currentUrl) {
            addToStatusLog('No active job URL found', 'warn');
            return;
        }

        const base = await getApiBase();
        const res = await fetch(base + '/apply/manual', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_post_id: _pseudoId(currentUrl),
                user_id: 'extension_user',
                resume_filename: '',
                notes: 'Marked applied via Chrome Extension',
                status: 'applied',
                source: 'extension_manual',
            }),
            signal: AbortSignal.timeout(10000),
        });

        if (!res.ok) {
            throw new Error('HTTP ' + res.status);
        }

        addToStatusLog('Marked as Applied ✓', 'ok');
    } catch (err) {
        addToStatusLog('Mark applied failed: ' + err.message, 'err');
    }
}

// ══════════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════════

/**
 * Returns the URL of the currently active tab, or empty string on error.
 * @returns {Promise<string>}
 */
async function _getActiveTabUrl() {
    try {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        return (tabs[0] && tabs[0].url) ? tabs[0].url : '';
    } catch (_) {
        return '';
    }
}

/**
 * Deterministically derives a UUID-shaped string from a URL for use
 * as a job_post_id placeholder when the real DB ID is unknown.
 * NOTE: this is not a real UUID — it is only used for the manual
 * queue endpoint which accepts any string as job_post_id.
 * @param {string} url - Source URL.
 * @returns {string} A hex-string token based on the URL.
 */
function _pseudoId(url) {
    let h = 0;
    for (let i = 0; i < url.length; i++) {
        h = (Math.imul(31, h) + url.charCodeAt(i)) >>> 0;
    }
    return 'ext-' + h.toString(16).padStart(8, '0');
}

/**
 * Truncates a string to maxLen, appending "…" if needed.
 * @param {string} str
 * @param {number} maxLen
 * @returns {string}
 */
function _truncate(str, maxLen) {
    if (!str) return '';
    return str.length > maxLen ? str.substring(0, maxLen) + '…' : str;
}

// ══════════════════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', function () {
    // Initial connection probe and job match load
    checkConnection();
    loadJobMatch();

    // Re-check connection every 30 seconds
    setInterval(checkConnection, 30000);

    // ─ Button handlers ─
    const btnAutofill = document.getElementById('btn-autofill');
    const btnQueue = document.getElementById('btn-queue');
    const btnApplied = document.getElementById('btn-applied');

    if (btnAutofill) {
        btnAutofill.addEventListener('click', function () {
            btnAutofill.disabled = true;
            autofillForm().finally(function () {
                btnAutofill.disabled = false;
            });
        });
    }

    if (btnQueue) {
        btnQueue.addEventListener('click', function () {
            btnQueue.disabled = true;
            queueToNotion().finally(function () {
                btnQueue.disabled = false;
            });
        });
    }

    if (btnApplied) {
        btnApplied.addEventListener('click', function () {
            btnApplied.disabled = true;
            markApplied().finally(function () {
                // Keep disabled briefly to prevent double-submit
                setTimeout(function () { btnApplied.disabled = false; }, 2000);
            });
        });
    }
});
