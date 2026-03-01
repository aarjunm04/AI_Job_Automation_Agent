/* ══════════════════════════════════════════════════════════════════
   api_client.js — Job Agent Chrome Extension v1
   Shared HTTP client with retry, timeout, and error normalisation.
   Loaded via importScripts() in service_worker.js.
   All functions exposed on globalThis.JobAgentAPI.
   ══════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── Retry / Timeout Constants ──
    var MAX_RETRIES = 2;
    var DEFAULT_TIMEOUT_MS = 10000;
    var HEALTH_TIMEOUT_MS = 5000;
    var BACKOFF_MS = [1000, 2000];

    // ══════════════════════════════════════════════════════════════════
    // INTERNAL HELPERS
    // ══════════════════════════════════════════════════════════════════

    /**
     * Builds request headers with Content-Type and optional Authorization.
     * @param {string|null} apiKey - Bearer token. Omitted if empty/null.
     * @returns {Object}
     */
    function _buildHeaders(apiKey) {
        var headers = { 'Content-Type': 'application/json' };
        if (apiKey && apiKey.length > 0) {
            headers['Authorization'] = 'Bearer ' + apiKey;
        }
        return headers;
    }

    /**
     * Performs a fetch with retry logic, timeout via AbortController,
     * and normalised response format.
     *
     * Retry policy:
     *   - Retries on: network errors (TypeError) and HTTP 429
     *   - No retry on: other 4xx, 5xx — fail fast
     *   - Max attempts: 1 original + maxRetries
     *   - Backoff: 1000ms after 1st fail, 2000ms after 2nd fail
     *
     * @param {string} url - Full endpoint URL
     * @param {Object} options - fetch options (method, headers, body)
     * @param {number} maxRetries - Max retry count (default 2)
     * @param {number} timeoutMs - Per-attempt timeout (default 10000)
     * @returns {Promise<{error: boolean, data?: any, status?: number, message?: string, attempt?: number}>}
     */
    async function _fetchWithRetry(url, options, maxRetries, timeoutMs) {
        var totalAttempts = (maxRetries || MAX_RETRIES) + 1;
        var timeout = timeoutMs || DEFAULT_TIMEOUT_MS;
        var lastError = null;

        for (var attempt = 1; attempt <= totalAttempts; attempt++) {
            var controller = new AbortController();
            var timer = setTimeout(function () { controller.abort(); }, timeout);

            try {
                var fetchOptions = Object.assign({}, options, { signal: controller.signal });
                var response = await fetch(url, fetchOptions);
                clearTimeout(timer);

                // Parse response body
                var data = null;
                var contentType = response.headers.get('content-type') || '';
                if (contentType.includes('application/json')) {
                    try {
                        data = await response.json();
                    } catch (parseErr) {
                        data = null;
                    }
                }

                // Success
                if (response.ok) {
                    return { error: false, data: data, status: response.status };
                }

                // Rate limited — retryable
                if (response.status === 429) {
                    lastError = {
                        error: true,
                        message: 'Rate limited (429)',
                        status: response.status,
                        attempt: attempt,
                    };
                    if (attempt < totalAttempts) {
                        await _sleep(BACKOFF_MS[attempt - 1] || 2000);
                        continue;
                    }
                    return lastError;
                }

                // Other HTTP errors — fail fast, no retry
                var errorMsg = 'HTTP ' + response.status;
                if (data && data.detail) {
                    errorMsg += ': ' + (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
                } else if (data && data.message) {
                    errorMsg += ': ' + data.message;
                }
                return {
                    error: true,
                    message: errorMsg,
                    status: response.status,
                    attempt: attempt,
                };
            } catch (err) {
                clearTimeout(timer);

                // Network error or abort — retryable
                var isNetworkError = err instanceof TypeError;
                var isAbort = err.name === 'AbortError';

                lastError = {
                    error: true,
                    message: isAbort ? 'Request timeout (' + timeout + 'ms)' : (err.message || 'Network error'),
                    status: null,
                    attempt: attempt,
                };

                if ((isNetworkError || isAbort) && attempt < totalAttempts) {
                    await _sleep(BACKOFF_MS[attempt - 1] || 2000);
                    continue;
                }

                return lastError;
            }
        }

        // Should not reach here, but safety fallback
        return lastError || { error: true, message: 'All retry attempts exhausted', status: null, attempt: totalAttempts };
    }

    /**
     * Promise-based sleep utility.
     * @param {number} ms
     * @returns {Promise<void>}
     */
    function _sleep(ms) {
        return new Promise(function (resolve) { setTimeout(resolve, ms); });
    }

    /**
     * Strips trailing slashes from a URL.
     * @param {string} host
     * @returns {string}
     */
    function _cleanHost(host) {
        return (host || '').replace(/\/+$/, '');
    }

    // ══════════════════════════════════════════════════════════════════
    // PUBLIC API
    // ══════════════════════════════════════════════════════════════════

    var API = {};

    /**
     * Calls POST /match — RAG resume match for current job page.
     * @param {string} host - FastAPI base URL e.g. "http://localhost:8000"
     * @param {string} apiKey - Bearer token
     * @param {string} jobUrl - Current job page URL
     * @param {string} jdText - Job description text (max 3000 chars)
     * @returns {Promise<{error: boolean, data?: MatchResponse, message?: string}>}
     *
     * MatchResponse shape:
     * {
     *   resume_suggested: string,
     *   similarity_score: number,
     *   fit_score: number,
     *   match_reasoning: string,
     *   talking_points: string[],
     *   autofill_ready: boolean
     * }
     */
    API.callMatch = async function (host, apiKey, jobUrl, jdText) {
        var url = _cleanHost(host) + '/match';
        return _fetchWithRetry(url, {
            method: 'POST',
            headers: _buildHeaders(apiKey),
            body: JSON.stringify({
                job_url: jobUrl,
                jd_text: (jdText || '').slice(0, 3000),
            }),
        }, MAX_RETRIES, DEFAULT_TIMEOUT_MS);
    };

    /**
     * Calls POST /autofill — returns field-value map for detected form fields.
     * @param {string} host
     * @param {string} apiKey
     * @param {string} jobUrl
     * @param {Array<DetectedField>} detectedFields - from content.js GET_FIELDS response
     * @returns {Promise<{error: boolean, data?: AutofillResponse, message?: string}>}
     *
     * AutofillResponse shape:
     * {
     *   field_mappings: {[fieldIdOrName: string]: string},
     *   unmapped_fields: string[],
     *   mapped_count: number
     * }
     */
    API.callAutofill = async function (host, apiKey, jobUrl, detectedFields) {
        var url = _cleanHost(host) + '/autofill';
        return _fetchWithRetry(url, {
            method: 'POST',
            headers: _buildHeaders(apiKey),
            body: JSON.stringify({
                job_url: jobUrl,
                detected_fields: detectedFields || [],
            }),
        }, MAX_RETRIES, DEFAULT_TIMEOUT_MS);
    };

    /**
     * Calls POST /log-application — logs manual application to Postgres + Notion.
     * @param {string} host
     * @param {string} apiKey
     * @param {LogApplicationPayload} payload
     * @returns {Promise<{error: boolean, data?: LogResponse, message?: string}>}
     *
     * LogApplicationPayload shape:
     * {
     *   job_url: string,
     *   resume_used: string,
     *   platform: string,
     *   applied_at: string,   // ISO8601
     *   notes: string|null
     * }
     *
     * LogResponse shape:
     * {
     *   application_id: string,
     *   notion_page_id: string,
     *   status: "success"
     * }
     */
    API.callLogApplication = async function (host, apiKey, payload) {
        var url = _cleanHost(host) + '/log-application';
        return _fetchWithRetry(url, {
            method: 'POST',
            headers: _buildHeaders(apiKey),
            body: JSON.stringify(payload),
        }, MAX_RETRIES, DEFAULT_TIMEOUT_MS);
    };

    /**
     * Calls GET /queue-count — returns current manual queue depth.
     * @param {string} host
     * @param {string} apiKey
     * @returns {Promise<{error: boolean, data?: {count: number}, message?: string}>}
     */
    API.callQueueCount = async function (host, apiKey) {
        var url = _cleanHost(host) + '/queue-count';
        return _fetchWithRetry(url, {
            method: 'GET',
            headers: _buildHeaders(apiKey),
        }, MAX_RETRIES, DEFAULT_TIMEOUT_MS);
    };

    /**
     * Calls GET /health — used by options page test connection.
     * No auth required (health endpoint is public).
     * Timeout: 5 seconds only, no retries.
     * @param {string} host
     * @returns {Promise<{error: boolean, data?: {status: string}, message?: string}>}
     */
    API.callHealth = async function (host) {
        var url = _cleanHost(host) + '/health';
        return _fetchWithRetry(url, {
            method: 'GET',
            headers: _buildHeaders(null),
        }, 0, HEALTH_TIMEOUT_MS);
    };

    // ── Expose on globalThis ──
    globalThis.JobAgentAPI = API;
})();
