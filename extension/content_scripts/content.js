/* ══════════════════════════════════════════════════════════════════
   content.js — Job Agent Chrome Extension v1
   Content script injected into every page (document_idle).
   Uses window.JobAgentDetector from dom_detector.js (loaded first).
   Manages DOM scanning, MutationObserver, iframe handling,
   and message bridge to background service worker.
   ══════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    try {
        // ── Guard: dom_detector must be loaded first ──
        if (!window.JobAgentDetector) {
            console.error('[JobAgent] dom_detector.js not loaded — aborting content script');
            return;
        }

        const D = window.JobAgentDetector;

        // ════════════════════════════════════════════════════════
        // 1. INITIALISATION
        // ════════════════════════════════════════════════════════

        /** Live element reference store — NOT JSON-safe */
        window.__jobAgentFields = [];

        /** Serialisable state object */
        window.__jobAgentState = {
            detected_fields: [],
            jd_text: '',
            ats_hint: '',
            last_scan_at: null,
            tab_load_time: Date.now(),
        };

        /** Reference to the MutationObserver control object */
        let scannerControl = null;

        // ════════════════════════════════════════════════════════
        // 2. FULL SCAN
        // ════════════════════════════════════════════════════════

        /**
         * Performs a complete scan of the page for form fields,
         * shadow DOM fields, ATS detection, and JD extraction.
         * Updates global state and element reference store.
         */
        function fullScan() {
            try {
                // Clear live element store
                window.__jobAgentFields = [];

                // Scan document fields
                const docFields = D.scanFormFields(document);

                // Scan shadow DOM fields
                const shadowFields = D.scanShadowDomFields(document);

                // Merge results
                const allFields = docFields.concat(shadowFields);

                // Scan iframes (same-origin only)
                const iframeFields = scanIframes();
                for (let i = 0; i < iframeFields.length; i++) {
                    allFields.push(iframeFields[i]);
                }

                // Deduplicate by (id + name + placeholder) combo
                const seen = new Set();
                const dedupedFields = [];
                for (let i = 0; i < allFields.length; i++) {
                    const f = allFields[i];
                    const key = (f.id || '') + '|' + (f.name || '') + '|' + (f.placeholder || '');
                    if (!seen.has(key)) {
                        seen.add(key);
                        f.index = dedupedFields.length;
                        dedupedFields.push(f);
                    }
                }

                // Detect ATS
                const atsHint = D.detectAtsHint(location.href, document);

                // Extract job description
                const jdText = D.extractJobDescription(document);

                // Update global state
                window.__jobAgentState = {
                    detected_fields: dedupedFields,
                    jd_text: jdText,
                    ats_hint: atsHint,
                    last_scan_at: new Date().toISOString(),
                    tab_load_time: window.__jobAgentState.tab_load_time,
                };

                console.log(
                    '[JobAgent] Scan complete:',
                    dedupedFields.length, 'fields |',
                    'ATS:', atsHint, '|',
                    'JD:', jdText.length, 'chars'
                );
            } catch (e) {
                console.error('[JobAgent] fullScan error:', e);
            }
        }

        // ════════════════════════════════════════════════════════
        // 3. IFRAME HANDLING
        // ════════════════════════════════════════════════════════

        /**
         * Scans same-origin iframes for form fields.
         * Cross-origin iframes are silently skipped.
         * @returns {DetectedField[]}
         */
        function scanIframes() {
            const fields = [];
            try {
                const iframes = document.querySelectorAll('iframe');
                for (let i = 0; i < iframes.length; i++) {
                    try {
                        const iframeDoc = iframes[i].contentDocument;
                        if (!iframeDoc) continue;

                        const iframeFields = D.scanFormFields(iframeDoc);
                        for (let j = 0; j < iframeFields.length; j++) {
                            iframeFields[j].label_text = '[iframe] ' + iframeFields[j].label_text;
                            fields.push(iframeFields[j]);
                        }
                    } catch (e) {
                        // Cross-origin iframe — skip silently
                        continue;
                    }
                }
            } catch (e) {
                // Skip all iframe scanning on error
            }
            return fields;
        }

        // ════════════════════════════════════════════════════════
        // 4. MUTATION OBSERVER CALLBACK
        // ════════════════════════════════════════════════════════

        /**
         * Called by the debounced MutationObserver when new form fields
         * are detected in the DOM (SPA navigation, multi-step forms).
         * @param {DetectedField[]} _newFields - Unused, we re-scan fully
         */
        function onFieldsChanged(_newFields) {
            try {
                fullScan();

                // Notify background service worker of field changes
                chrome.runtime.sendMessage({
                    type: 'FIELDS_UPDATED',
                    field_count: window.__jobAgentState.detected_fields.length,
                }).catch(function () {
                    // Background may not be ready — ignore
                });
            } catch (e) {
                console.error('[JobAgent] onFieldsChanged error:', e);
            }
        }

        // ════════════════════════════════════════════════════════
        // 5. MESSAGE LISTENER
        // ════════════════════════════════════════════════════════

        chrome.runtime.onMessage.addListener(function (message, _sender, sendResponse) {
            try {
                switch (message.type) {
                    // ── GET_FIELDS ──
                    case 'GET_FIELDS':
                        sendResponse({
                            fields: window.__jobAgentState.detected_fields,
                            ats_hint: window.__jobAgentState.ats_hint,
                            field_count: window.__jobAgentState.detected_fields.length,
                            last_scan_at: window.__jobAgentState.last_scan_at,
                        });
                        return false;

                    // ── GET_JD_TEXT ──
                    case 'GET_JD_TEXT':
                        sendResponse({
                            jd_text: window.__jobAgentState.jd_text,
                            length: window.__jobAgentState.jd_text.length,
                        });
                        return false;

                    // ── FORCE_RESCAN ──
                    case 'FORCE_RESCAN':
                        fullScan();
                        sendResponse({
                            field_count: window.__jobAgentState.detected_fields.length,
                            ats_hint: window.__jobAgentState.ats_hint,
                        });
                        return false;

                    // ── INJECT_FIELDS ──
                    case 'INJECT_FIELDS':
                        handleInjectFields(message.field_mappings || {}, sendResponse);
                        return false;

                    // ── GET_PAGE_META ──
                    case 'GET_PAGE_META':
                        sendResponse({
                            url: location.href,
                            title: document.title,
                            platform: window.__jobAgentState.ats_hint,
                            time_on_page_seconds: Math.floor(
                                (Date.now() - window.__jobAgentState.tab_load_time) / 1000
                            ),
                        });
                        return false;

                    default:
                        sendResponse({ error: true, message: 'Unknown message type: ' + message.type });
                        return false;
                }
            } catch (e) {
                console.error('[JobAgent] Message handler error:', e);
                sendResponse({ error: true, message: e.message || 'Internal error' });
                return false;
            }
        });

        // ════════════════════════════════════════════════════════
        // 6. INJECT FIELDS HANDLER
        // ════════════════════════════════════════════════════════

        /**
         * Processes an INJECT_FIELDS request: matches field mappings to
         * detected fields and injects values via dom_detector.
         * @param {Object<string, string>} fieldMappings - key→value pairs
         * @param {function} sendResponse - Chrome message response callback
         */
        function handleInjectFields(fieldMappings, sendResponse) {
            try {
                let filledCount = 0;
                const failedFields = [];
                const keys = Object.keys(fieldMappings);

                for (let i = 0; i < keys.length; i++) {
                    const key = keys[i];
                    const value = fieldMappings[key];

                    try {
                        // Find matching DetectedField
                        const match = findFieldMatch(key);

                        if (!match) {
                            failedFields.push({ field: key, error: 'No matching field found' });
                            continue;
                        }

                        // Get live element reference
                        const element = window.__jobAgentFields[match.element_ref_index];
                        if (!element) {
                            failedFields.push({ field: key, error: 'Live element reference lost' });
                            continue;
                        }

                        // Inject value
                        const result = D.injectFieldValue(element, value, match.react_controlled);
                        if (result.success) {
                            filledCount++;
                        } else {
                            failedFields.push({ field: key, error: result.error });
                        }
                    } catch (fieldErr) {
                        failedFields.push({ field: key, error: fieldErr.message || 'Injection error' });
                    }
                }

                sendResponse({
                    filled_count: filledCount,
                    failed_fields: failedFields,
                    total_attempted: keys.length,
                });
            } catch (e) {
                sendResponse({
                    filled_count: 0,
                    failed_fields: [{ field: '*', error: e.message }],
                    total_attempted: Object.keys(fieldMappings).length,
                });
            }
        }

        /**
         * Finds a DetectedField matching the given key.
         * Match order: exact id → exact name → fuzzy label_text match.
         * @param {string} key
         * @returns {DetectedField|null}
         */
        function findFieldMatch(key) {
            const fields = window.__jobAgentState.detected_fields;
            const keyLower = key.toLowerCase();

            // Exact id match
            for (let i = 0; i < fields.length; i++) {
                if (fields[i].id === key) return fields[i];
            }

            // Exact name match
            for (let i = 0; i < fields.length; i++) {
                if (fields[i].name === key) return fields[i];
            }

            // Fuzzy label_text match
            for (let i = 0; i < fields.length; i++) {
                if (fields[i].label_text.toLowerCase().includes(keyLower)) return fields[i];
            }

            return null;
        }

        // ════════════════════════════════════════════════════════
        // 7. BOOT SEQUENCE
        // ════════════════════════════════════════════════════════

        // Run initial scan
        fullScan();

        // Start MutationObserver for SPA re-scanning
        scannerControl = D.createFieldScanner(onFieldsChanged, 500, document);

        // Notify background service worker that content script is ready
        chrome.runtime.sendMessage({
            type: 'CONTENT_READY',
            ats_hint: window.__jobAgentState.ats_hint,
            field_count: window.__jobAgentState.detected_fields.length,
        }).catch(function () {
            // Background service worker may not be ready yet — ignore
        });

        console.log(
            '[JobAgent] Content script initialised |',
            window.__jobAgentState.detected_fields.length, 'fields |',
            'ATS:', window.__jobAgentState.ats_hint
        );
    } catch (initError) {
        console.error('[JobAgent] Init error:', initError);
    }
})();
