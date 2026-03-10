/* ══════════════════════════════════════════════════════════════════
   service_worker.js — AI Job Agent Chrome Extension v1
   MV3 Background Service Worker.
   Minimal: only message relay between popup and content scripts.
   Zero MCP references. Zero direct API calls.
   All FastAPI communication is performed directly in popup.js.
   ══════════════════════════════════════════════════════════════════ */

'use strict';

// ══════════════════════════════════════════════════════════════════
// INSTALLATION — write default api_base_url to local storage
// ══════════════════════════════════════════════════════════════════

chrome.runtime.onInstalled.addListener(function (details) {
    if (details.reason === 'install') {
        chrome.storage.local.set({ api_base_url: 'http://localhost:8000' });
    }
});

// ══════════════════════════════════════════════════════════════════
// MESSAGE RELAY — between popup and content scripts only
// ══════════════════════════════════════════════════════════════════

/**
 * Listens for messages from the popup and relays them to the content
 * script in the target tab, or vice versa. Supports only the
 * "relay_to_tab" action for routing to a specific tabId.
 *
 * All FastAPI HTTP calls are made directly by popup.js, not here.
 */
chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
    if (message && message.action === 'relay_to_tab' && message.tab_id) {
        chrome.tabs.sendMessage(message.tab_id, message.payload || message, function (response) {
            if (chrome.runtime.lastError) {
                sendResponse({ error: true, message: chrome.runtime.lastError.message });
                return;
            }
            sendResponse(response);
        });
        return true; // keep channel open for async sendResponse
    }

    // Unrecognised message — acknowledge silently
    sendResponse({ acknowledged: true });
    return false;
});
