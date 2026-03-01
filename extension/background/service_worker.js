/* ══════════════════════════════════════════════════════════════════
   service_worker.js — Job Agent Chrome Extension v1
   MV3 Background Service Worker.
   Message router, badge management, session cache, tab relay.
   Reads config fresh from chrome.storage.sync on every call.
   ══════════════════════════════════════════════════════════════════ */

importScripts('../utils/api_client.js');

// ── Cache TTL (5 minutes) ──
var MATCH_CACHE_TTL_MS = 300000;

// ══════════════════════════════════════════════════════════════════
// CONFIG LOADER
// ══════════════════════════════════════════════════════════════════

/**
 * Reads FASTAPI_HOST and FASTAPI_API_KEY from chrome.storage.sync.
 * MUST be called fresh on every message handler — never cache in module scope.
 * @returns {Promise<{host: string, apiKey: string}>}
 */
async function getConfig() {
    try {
        var result = await chrome.storage.sync.get(['FASTAPI_HOST', 'FASTAPI_API_KEY']);
        return {
            host: result.FASTAPI_HOST || 'http://localhost:8000',
            apiKey: result.FASTAPI_API_KEY || '',
        };
    } catch (err) {
        console.error('[JobAgent SW] Failed to read config:', err);
        return { host: 'http://localhost:8000', apiKey: '' };
    }
}

// ══════════════════════════════════════════════════════════════════
// SESSION CACHE (/match response cache)
// ══════════════════════════════════════════════════════════════════

/**
 * Generates the session cache key for a job URL.
 * @param {string} jobUrl
 * @returns {string}
 */
function _cacheKey(jobUrl) {
    return 'match_cache_' + btoa(unescape(encodeURIComponent(jobUrl)));
}

/**
 * Retrieves a cached /match response from chrome.storage.session.
 * Returns null if expired or not found.
 * @param {string} jobUrl
 * @returns {Promise<MatchResponse|null>}
 */
async function getCachedMatch(jobUrl) {
    try {
        var key = _cacheKey(jobUrl);
        var result = await chrome.storage.session.get(key);
        var entry = result[key];

        if (!entry) return null;

        // Check TTL
        if (Date.now() - entry.cached_at > MATCH_CACHE_TTL_MS) {
            // Expired — remove silently
            chrome.storage.session.remove(key).catch(function () { });
            return null;
        }

        return entry.data;
    } catch (err) {
        return null;
    }
}

/**
 * Stores a /match response in chrome.storage.session with timestamp.
 * @param {string} jobUrl
 * @param {MatchResponse} data
 * @returns {Promise<void>}
 */
async function setCachedMatch(jobUrl, data) {
    try {
        var key = _cacheKey(jobUrl);
        var entry = {};
        entry[key] = { data: data, cached_at: Date.now() };
        await chrome.storage.session.set(entry);
    } catch (err) {
        console.error('[JobAgent SW] Cache write error:', err);
    }
}

// ══════════════════════════════════════════════════════════════════
// BADGE MANAGEMENT
// ══════════════════════════════════════════════════════════════════

/**
 * Updates the extension badge with the queue count.
 * @param {number} count
 */
async function updateBadge(count) {
    try {
        if (count > 0) {
            await chrome.action.setBadgeText({ text: String(count) });
            await chrome.action.setBadgeBackgroundColor({ color: '#4ade80' });
        } else {
            await chrome.action.setBadgeText({ text: '' });
        }
    } catch (err) {
        console.error('[JobAgent SW] Badge update error:', err);
    }
}

/**
 * Fetches queue count from FastAPI and updates the badge.
 * Fails silently — badge resets to empty on error.
 */
async function refreshBadge() {
    try {
        var config = await getConfig();
        var result = await globalThis.JobAgentAPI.callQueueCount(config.host, config.apiKey);
        if (!result.error && result.data) {
            var count = result.data.count !== undefined ? result.data.count : (result.data.queue_count || 0);
            await updateBadge(count);
            console.log('[JobAgent SW] Badge refreshed:', count);
        } else {
            await updateBadge(0);
        }
    } catch (err) {
        await updateBadge(0);
    }
}

// ══════════════════════════════════════════════════════════════════
// INSTALLATION / STARTUP
// ══════════════════════════════════════════════════════════════════

chrome.runtime.onInstalled.addListener(async function (details) {
    console.log('[JobAgent SW] Installed — reason:', details.reason);

    if (details.reason === 'install') {
        // Set default config
        try {
            await chrome.storage.sync.set({
                FASTAPI_HOST: 'http://localhost:8000',
                FASTAPI_API_KEY: '',
            });
            console.log('[JobAgent SW] Default config written to storage.sync');
        } catch (err) {
            console.error('[JobAgent SW] Failed to write default config:', err);
        }

        // Open options page for first-time setup
        try {
            chrome.runtime.openOptionsPage();
        } catch (err) {
            console.error('[JobAgent SW] Failed to open options page:', err);
        }
    }

    await refreshBadge();
});

chrome.runtime.onStartup.addListener(function () {
    console.log('[JobAgent SW] Browser startup — refreshing badge');
    refreshBadge();
});

// ══════════════════════════════════════════════════════════════════
// MESSAGE ROUTER
// ══════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
    console.log('[JobAgent SW] Message received:', message.type);

    (async function () {
        try {
            var result = await handleMessage(message, sender);
            sendResponse(result);
        } catch (err) {
            console.error('[JobAgent SW] Unhandled error in message handler:', err);
            sendResponse({ error: true, message: err.message || 'Internal service worker error' });
        }
    })();

    return true; // Always return true for async sendResponse
});

// ══════════════════════════════════════════════════════════════════
// MESSAGE HANDLER
// ══════════════════════════════════════════════════════════════════

/**
 * Routes incoming messages to the appropriate handler.
 * @param {Object} message - Message with type and payload
 * @param {Object} sender - Chrome sender info
 * @returns {Promise<Object>} Response object
 */
async function handleMessage(message, sender) {
    switch (message.type) {
        case 'CALL_MATCH':
            return await handleCallMatch(message);

        case 'CALL_AUTOFILL':
            return await handleCallAutofill(message);

        case 'CALL_LOG_APPLICATION':
            return await handleCallLogApplication(message);

        case 'CALL_QUEUE_COUNT':
            return await handleCallQueueCount();

        case 'GET_FIELDS_FROM_TAB':
            return await handleTabRelay(message.tab_id, { type: 'GET_FIELDS' });

        case 'GET_JD_TEXT_FROM_TAB':
            return await handleTabRelay(message.tab_id, { type: 'GET_JD_TEXT' });

        case 'INJECT_FIELDS_INTO_TAB':
            return await handleTabRelay(message.tab_id, {
                type: 'INJECT_FIELDS',
                field_mappings: message.field_mappings,
            });

        case 'FORCE_RESCAN_TAB':
            return await handleTabRelay(message.tab_id, { type: 'FORCE_RESCAN' });

        case 'GET_PAGE_META_FROM_TAB':
            return await handleTabRelay(message.tab_id, { type: 'GET_PAGE_META' });

        case 'REFRESH_BADGE':
            await refreshBadge();
            return { success: true };

        case 'CONTENT_READY':
            console.log(
                '[JobAgent SW] Content script ready —',
                'ATS:', message.ats_hint,
                '| Fields:', message.field_count
            );
            return { acknowledged: true };

        case 'FIELDS_UPDATED':
            console.log('[JobAgent SW] Fields updated — count:', message.field_count);
            return { acknowledged: true };

        default:
            return { error: true, message: 'Unknown message type: ' + message.type };
    }
}

// ══════════════════════════════════════════════════════════════════
// HANDLER IMPLEMENTATIONS
// ══════════════════════════════════════════════════════════════════

/**
 * Handles CALL_MATCH — checks session cache first, then calls API.
 * @param {Object} message - {job_url, jd_text}
 * @returns {Promise<Object>}
 */
async function handleCallMatch(message) {
    var jobUrl = message.job_url;
    var jdText = message.jd_text || '';

    // 1. Check cache
    try {
        var cached = await getCachedMatch(jobUrl);
        if (cached) {
            console.log('[JobAgent SW] Cache HIT for:', jobUrl);
            return { error: false, data: cached, cached: true };
        }
        console.log('[JobAgent SW] Cache MISS for:', jobUrl);
    } catch (err) {
        // Cache read failed — proceed to API call
    }

    // 2. Fresh API call
    var config = await getConfig();
    var result = await globalThis.JobAgentAPI.callMatch(config.host, config.apiKey, jobUrl, jdText);

    // 3. Cache on success
    if (!result.error && result.data) {
        try {
            await setCachedMatch(jobUrl, result.data);
        } catch (err) {
            // Cache write failed — non-critical
        }
        refreshBadge().catch(function () { }); // Fire-and-forget badge refresh
    }

    result.cached = false;
    return result;
}

/**
 * Handles CALL_AUTOFILL — calls POST /autofill.
 * @param {Object} message - {job_url, detected_fields}
 * @returns {Promise<Object>}
 */
async function handleCallAutofill(message) {
    var config = await getConfig();
    return await globalThis.JobAgentAPI.callAutofill(
        config.host,
        config.apiKey,
        message.job_url,
        message.detected_fields || []
    );
}

/**
 * Handles CALL_LOG_APPLICATION — logs application, invalidates cache, refreshes badge.
 * @param {Object} message - {job_url, resume_used, platform, applied_at, notes}
 * @returns {Promise<Object>}
 */
async function handleCallLogApplication(message) {
    var config = await getConfig();

    var payload = {
        job_url: message.job_url,
        resume_used: message.resume_used,
        platform: message.platform,
        applied_at: message.applied_at || new Date().toISOString(),
        notes: message.notes || null,
    };

    var result = await globalThis.JobAgentAPI.callLogApplication(config.host, config.apiKey, payload);

    if (!result.error) {
        // Invalidate match cache for this job URL
        try {
            var key = _cacheKey(message.job_url);
            await chrome.storage.session.remove(key);
            console.log('[JobAgent SW] Cache invalidated for:', message.job_url);
        } catch (err) {
            // Non-critical
        }

        // Refresh badge — queue count should have decremented
        refreshBadge().catch(function () { });
    }

    return result;
}

/**
 * Handles CALL_QUEUE_COUNT — fetches count and updates badge.
 * @returns {Promise<Object>}
 */
async function handleCallQueueCount() {
    var config = await getConfig();
    var result = await globalThis.JobAgentAPI.callQueueCount(config.host, config.apiKey);

    if (!result.error && result.data) {
        var count = result.data.count !== undefined ? result.data.count : (result.data.queue_count || 0);
        await updateBadge(count);
    }

    return result;
}

/**
 * Relays a message to a content script in a specific tab.
 * Handles errors gracefully when tab has no content script.
 * @param {number} tabId - Target tab ID
 * @param {Object} tabMessage - Message to send to content script
 * @returns {Promise<Object>}
 */
async function handleTabRelay(tabId, tabMessage) {
    if (!tabId && tabId !== 0) {
        return { error: true, message: 'No tab_id provided' };
    }

    try {
        var response = await chrome.tabs.sendMessage(tabId, tabMessage);
        return response;
    } catch (err) {
        console.error('[JobAgent SW] Tab relay failed (tab ' + tabId + '):', err.message);
        return {
            error: true,
            message: 'Content script not available on this page',
        };
    }
}
