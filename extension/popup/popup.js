/* ══════════════════════════════════════════════════════════════════
   popup.js — Job Agent Chrome Extension v1
   Main popup panel logic. Drives 6 UI states, communicates
   exclusively via chrome.runtime.sendMessage to background
   service worker. Zero direct fetch() calls.
   ══════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── Message Type Constants ──
    var MESSAGE_TYPES = {
        CALL_MATCH: 'CALL_MATCH',
        CALL_AUTOFILL: 'CALL_AUTOFILL',
        CALL_LOG_APPLICATION: 'CALL_LOG_APPLICATION',
        CALL_QUEUE_COUNT: 'CALL_QUEUE_COUNT',
        GET_FIELDS_FROM_TAB: 'GET_FIELDS_FROM_TAB',
        GET_JD_TEXT_FROM_TAB: 'GET_JD_TEXT_FROM_TAB',
        INJECT_FIELDS_INTO_TAB: 'INJECT_FIELDS_INTO_TAB',
        FORCE_RESCAN_TAB: 'FORCE_RESCAN_TAB',
        GET_PAGE_META_FROM_TAB: 'GET_PAGE_META_FROM_TAB',
        REFRESH_BADGE: 'REFRESH_BADGE',
    };

    // ── State IDs (matching HTML div IDs) ──
    var STATE_IDS = [
        'state-loading',
        'state-not-job',
        'state-error',
        'state-match',
        'state-applied',
    ];

    // ── Application State ──
    var appState = {
        currentTab: null,
        matchData: null,
        detectedFields: [],
        jdText: '',
        pageMeta: null,
        autofillDone: false,
        applied: false,
        lastError: null,
    };

    // ══════════════════════════════════════════════════════════════════
    // HELPERS
    // ══════════════════════════════════════════════════════════════════

    /**
     * Sends a message to the background service worker.
     * @param {string} type - Message type
     * @param {Object} [payload] - Additional payload
     * @returns {Promise<Object>}
     */
    function sendToBackground(type, payload) {
        return new Promise(function (resolve, reject) {
            var message = Object.assign({ type: type }, payload || {});
            chrome.runtime.sendMessage(message, function (response) {
                if (chrome.runtime.lastError) {
                    reject(new Error(chrome.runtime.lastError.message));
                    return;
                }
                if (response && response.error) {
                    reject(new Error(response.message || 'Unknown error'));
                    return;
                }
                resolve(response);
            });
        });
    }

    /**
     * Sends a message to the background that will be relayed to the active tab.
     * Injects tab_id from appState.currentTab.
     * @param {string} type - Message type (e.g. GET_FIELDS_FROM_TAB)
     * @param {Object} [payload] - Additional payload
     * @returns {Promise<Object>}
     */
    function sendToTab(type, payload) {
        if (!appState.currentTab || !appState.currentTab.id) {
            return Promise.reject(new Error('No active tab'));
        }
        var msg = Object.assign({ tab_id: appState.currentTab.id }, payload || {});
        return sendToBackground(type, msg);
    }

    /**
     * Shows exactly one state div, hides all others.
     * Footer is visible in all states except loading.
     * @param {string} stateId - ID of the state div to show
     */
    function showState(stateId) {
        for (var i = 0; i < STATE_IDS.length; i++) {
            var el = document.getElementById(STATE_IDS[i]);
            if (el) el.hidden = (STATE_IDS[i] !== stateId);
        }
        var footer = document.getElementById('popup-footer');
        if (footer) footer.hidden = (stateId === 'state-loading');
    }

    /**
     * Sets the score colour class on #score-number.
     * @param {number} score - Fit score 0.0–1.0
     */
    function renderScoreColour(score) {
        var el = document.getElementById('score-number');
        if (!el) return;
        el.classList.remove('score-high', 'score-mid', 'score-low');
        if (score >= 0.75) {
            el.classList.add('score-high');
        } else if (score >= 0.50) {
            el.classList.add('score-mid');
        } else {
            el.classList.add('score-low');
        }
    }

    /**
     * Populates the talking points accordion body.
     * @param {string[]} points
     */
    function renderTalkingPoints(points) {
        var container = document.getElementById('tp-body');
        if (!container) return;
        container.innerHTML = '';

        if (!points || points.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'tp-item';
            empty.textContent = 'No talking points available.';
            container.appendChild(empty);
            return;
        }

        for (var i = 0; i < points.length; i++) {
            var item = document.createElement('div');
            item.className = 'tp-item';
            item.textContent = points[i];
            container.appendChild(item);
        }
    }

    /**
     * Updates the queue badge in the popup header.
     * Non-blocking — fires and forgets.
     */
    function refreshQueueBadge() {
        sendToBackground(MESSAGE_TYPES.CALL_QUEUE_COUNT)
            .then(function (res) {
                var badge = document.getElementById('queue-badge');
                if (badge && res && res.data) {
                    var count = res.data.count !== undefined ? res.data.count : res.data.queue_count;
                    badge.textContent = count !== undefined ? count : '—';
                }
            })
            .catch(function () {
                // Silently fail — badge stays as-is
            });
    }

    /**
     * Truncates a string to maxLen characters, appending "…" if truncated.
     * @param {string} str
     * @param {number} maxLen
     * @returns {string}
     */
    function truncate(str, maxLen) {
        if (!str) return '';
        return str.length > maxLen ? str.substring(0, maxLen) + '…' : str;
    }

    // ══════════════════════════════════════════════════════════════════
    // RENDER
    // ══════════════════════════════════════════════════════════════════

    /**
     * Populates all elements in #state-match with data from /match response.
     * @param {MatchResponse} matchData
     * @param {chrome.tabs.Tab} tab
     * @param {Object} pageMeta
     */
    function renderMatchPanel(matchData, tab, pageMeta) {
        // Job info
        document.getElementById('job-url').textContent = truncate(tab.url, 60);
        document.getElementById('job-url').title = tab.url;
        document.getElementById('job-platform').textContent =
            (pageMeta && pageMeta.platform) ? pageMeta.platform.toUpperCase() : 'UNKNOWN';

        // Fit score
        var scoreNum = document.getElementById('score-number');
        if (matchData.fit_score !== undefined && matchData.fit_score !== null) {
            scoreNum.textContent = (matchData.fit_score * 100).toFixed(0) + '%';
            renderScoreColour(matchData.fit_score);
        } else {
            scoreNum.textContent = '—';
        }

        // Reasoning
        document.getElementById('score-reasoning').textContent =
            matchData.match_reasoning || '—';

        // Resume suggestion
        document.getElementById('resume-name').textContent =
            matchData.resume_suggested || '—';
        if (matchData.similarity_score !== undefined) {
            document.getElementById('resume-score').textContent =
                (matchData.similarity_score * 100).toFixed(1) + '% similarity';
        } else {
            document.getElementById('resume-score').textContent = '—';
        }

        // Talking points
        renderTalkingPoints(matchData.talking_points);

        // Footer timestamp
        document.getElementById('footer-timestamp').textContent =
            'Matched at ' + new Date().toLocaleTimeString();
    }

    // ══════════════════════════════════════════════════════════════════
    // MAIN INIT
    // ══════════════════════════════════════════════════════════════════

    /**
     * Main initialisation — runs on DOMContentLoaded.
     * Orchestrates the entire popup lifecycle.
     */
    async function init() {
        try {
            // 1. Get active tab
            var tabs = await chrome.tabs.query({ active: true, currentWindow: true });
            var tab = tabs[0];
            if (!tab) {
                showState('state-error');
                document.getElementById('error-title').textContent = 'No active tab';
                document.getElementById('error-detail').textContent = 'Could not detect an active browser tab.';
                return;
            }
            appState.currentTab = tab;

            // Check for browser internal pages
            var url = tab.url || '';
            if (url.startsWith('chrome://') || url.startsWith('edge://') ||
                url.startsWith('about:') || url.startsWith('chrome-extension://')) {
                showState('state-not-job');
                document.getElementById('not-job-detail').textContent =
                    'Cannot run on browser internal pages.';
                return;
            }

            // 2. Show loading
            showState('state-loading');

            // 3. Refresh badge (non-blocking)
            sendToBackground(MESSAGE_TYPES.REFRESH_BADGE).catch(function () { });
            refreshQueueBadge();

            // 4. Get page meta
            try {
                var pageMeta = await sendToTab(MESSAGE_TYPES.GET_PAGE_META_FROM_TAB);
                appState.pageMeta = pageMeta;
            } catch (e) {
                appState.pageMeta = {
                    url: tab.url,
                    title: tab.title,
                    platform: 'unknown',
                    time_on_page_seconds: 0,
                };
            }

            // 5. Get fields from tab
            try {
                var fieldsResponse = await sendToTab(MESSAGE_TYPES.GET_FIELDS_FROM_TAB);
                appState.detectedFields = fieldsResponse.fields || [];
            } catch (e) {
                appState.detectedFields = [];
            }

            // 6. Get JD text from tab
            try {
                var jdResponse = await sendToTab(MESSAGE_TYPES.GET_JD_TEXT_FROM_TAB);
                appState.jdText = jdResponse.jd_text || '';
            } catch (e) {
                appState.jdText = '';
            }

            // 7. Check if this is a job page
            var isJobPage = appState.detectedFields.length > 0 || appState.jdText.length > 100;
            if (!isJobPage) {
                showState('state-not-job');
                return;
            }

            // 8. Enable autofill button if fields detected
            if (appState.detectedFields.length > 0) {
                document.getElementById('btn-autofill').disabled = false;
            }

            // 9. Call /match via background
            var matchResponse = await sendToBackground(MESSAGE_TYPES.CALL_MATCH, {
                job_url: tab.url,
                jd_text: appState.jdText,
            });
            appState.matchData = matchResponse.data;

            // 10. Render match panel
            if (appState.matchData) {
                renderMatchPanel(appState.matchData, tab, appState.pageMeta);
                showState('state-match');
            } else {
                showState('state-error');
                document.getElementById('error-title').textContent = 'No match data';
                document.getElementById('error-detail').textContent =
                    'Server returned an empty response. Try again.';
                return;
            }

            // 11. Time-based enable for Mark as Applied (30s fallback)
            setTimeout(function () {
                if (!appState.applied) {
                    document.getElementById('btn-applied').disabled = false;
                }
            }, 30000);

        } catch (err) {
            appState.lastError = {
                title: 'Something went wrong',
                detail: err.message || 'Unknown error',
            };
            showState('state-error');
            document.getElementById('error-title').textContent = appState.lastError.title;
            document.getElementById('error-detail').textContent = appState.lastError.detail;
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // AUTOFILL HANDLER
    // ══════════════════════════════════════════════════════════════════

    /**
     * Handles the Autofill button click:
     * 1. Calls /autofill to get field mappings
     * 2. Injects values into the active tab via content script
     * 3. Updates status and enables Mark as Applied
     */
    async function handleAutofill() {
        var btn = document.getElementById('btn-autofill');
        var statusEl = document.getElementById('autofill-status');
        var originalText = btn.textContent;

        try {
            // 1. Disable and show progress
            btn.disabled = true;
            btn.textContent = '⚡ Filling…';
            statusEl.textContent = '';

            // 2. Call autofill
            var autofillResponse = await sendToBackground(MESSAGE_TYPES.CALL_AUTOFILL, {
                job_url: appState.currentTab.url,
                detected_fields: appState.detectedFields,
            });
            var fieldMappings = autofillResponse.data.field_mappings;

            // 3. Inject into tab
            var injectResponse = await sendToBackground(MESSAGE_TYPES.INJECT_FIELDS_INTO_TAB, {
                tab_id: appState.currentTab.id,
                field_mappings: fieldMappings,
            });

            // 4. Update status
            var filled = injectResponse.filled_count || 0;
            var total = injectResponse.total_attempted || 0;
            var failed = (injectResponse.failed_fields || []).length;

            var statusText = '✓ Filled ' + filled + '/' + total + ' fields';
            if (failed > 0) {
                statusText += ' · ' + failed + ' skipped';
            }
            statusEl.textContent = statusText;

            // 5. Update state
            appState.autofillDone = true;

            // 6. Enable Mark as Applied
            document.getElementById('btn-applied').disabled = false;

            // 7. Allow re-fill
            btn.textContent = '⚡ Re-fill Fields';
            btn.disabled = false;

        } catch (err) {
            statusEl.textContent = '✗ Autofill failed: ' + (err.message || 'Unknown error');
            btn.textContent = originalText;
            btn.disabled = false;
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // MARK AS APPLIED HANDLER
    // ══════════════════════════════════════════════════════════════════

    /**
     * Handles the Mark as Applied button click:
     * 1. Sends log-application to background → FastAPI → Postgres + Notion
     * 2. Shows success banner on completion
     */
    async function handleMarkApplied() {
        var btn = document.getElementById('btn-applied');
        var originalText = btn.textContent;

        try {
            // 1. Disable and show progress
            btn.disabled = true;
            btn.textContent = 'Logging…';

            // 2. Build payload
            var payload = {
                job_url: appState.currentTab.url,
                resume_used: (appState.matchData && appState.matchData.resume_suggested) || 'unknown',
                platform: (appState.pageMeta && appState.pageMeta.platform) || 'unknown',
                applied_at: new Date().toISOString(),
                notes: document.getElementById('notes-input').value.trim() || null,
            };

            // 3. Call log-application
            var logResponse = await sendToBackground(MESSAGE_TYPES.CALL_LOG_APPLICATION, payload);

            // 4. Success
            appState.applied = true;

            var appId = (logResponse.data && logResponse.data.application_id) || '';
            var subText = appId
                ? 'ID: ' + appId.substring(0, 8) + '… · Notion synced'
                : 'Synced to Notion Job Tracker';
            document.getElementById('success-sub').textContent = subText;

            showState('state-applied');

            // Refresh badge (non-blocking)
            sendToBackground(MESSAGE_TYPES.REFRESH_BADGE).catch(function () { });
            refreshQueueBadge();

        } catch (err) {
            btn.textContent = originalText;
            btn.disabled = false;

            // Show error below button
            var errorDiv = document.getElementById('apply-error');
            if (!errorDiv) {
                errorDiv = document.createElement('div');
                errorDiv.id = 'apply-error';
                errorDiv.style.cssText = 'color:var(--accent-red);font-size:11px;margin-top:6px';
                btn.parentNode.insertBefore(errorDiv, btn.nextSibling);
            }
            errorDiv.textContent = '✗ Failed to log: ' + (err.message || 'Unknown error');
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // RESCAN HANDLER
    // ══════════════════════════════════════════════════════════════════

    /**
     * Handles the Rescan button: forces a DOM re-scan and re-initialises.
     */
    async function handleRescan() {
        showState('state-loading');

        try {
            await sendToBackground(MESSAGE_TYPES.FORCE_RESCAN_TAB, {
                tab_id: appState.currentTab.id,
            });
        } catch (e) {
            // Continue with init anyway
        }

        // Reset state (preserve currentTab)
        var tab = appState.currentTab;
        appState.matchData = null;
        appState.detectedFields = [];
        appState.jdText = '';
        appState.pageMeta = null;
        appState.autofillDone = false;
        appState.applied = false;
        appState.lastError = null;
        appState.currentTab = tab;

        await init();
    }

    // ══════════════════════════════════════════════════════════════════
    // EVENT LISTENERS
    // ══════════════════════════════════════════════════════════════════

    document.addEventListener('DOMContentLoaded', function () {
        // Button handlers
        document.getElementById('btn-autofill').addEventListener('click', handleAutofill);
        document.getElementById('btn-applied').addEventListener('click', handleMarkApplied);
        document.getElementById('btn-rescan').addEventListener('click', handleRescan);

        document.getElementById('btn-retry').addEventListener('click', function () {
            appState.lastError = null;
            appState.matchData = null;
            appState.detectedFields = [];
            appState.jdText = '';
            appState.pageMeta = null;
            appState.autofillDone = false;
            appState.applied = false;
            init();
        });

        // Talking points accordion toggle
        document.getElementById('tp-toggle').addEventListener('click', function () {
            document.getElementById('tp-body').classList.toggle('open');
            document.getElementById('tp-chevron').classList.toggle('open');
        });

        // Footer settings link
        document.getElementById('footer-settings').addEventListener('click', function (e) {
            e.preventDefault();
            chrome.runtime.openOptionsPage();
        });

        // Start
        init();
    });
})();
