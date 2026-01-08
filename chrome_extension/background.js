/**
 * background.js
 * AI Job Automation Agent - Background Service Worker (Manifest V3)
 *
 * Production-grade refactor:
 *  - Defensively creates context menus
 *  - MCP bridge with session lifecycle & retries
 *  - Notion logging integration via NotionAPI (if available)
 *  - Message routing for content <-> sidebar
 *  - Storage and state persistence
 *  - Safe networking helpers
 *
 * Keep in sync with content.js and sidebar.js message types.
 *
 * Author: AI Job Automation Team (refactor 2025)
 */

'use strict';

const BG = (() => {
  // -----------------------
  // CONFIG
  // -----------------------
  const CONFIG = {
    DEBUG: false,
    API_BASE_URL: 'http://localhost:8080', // override via storage if needed
    MCP_ENDPOINT: 'http://localhost:3001',
    EXTENSION_VERSION: '1.0.0',
    STORAGE_KEYS: {
      USER_SETTINGS: 'user_settings',
      JOB_CACHE: 'job_cache',
      ANALYSIS_HISTORY: 'analysis_history',
      EXTENSION_STATE: 'extension_state',
      API_CONFIG: 'api_config'
    },
    CONTEXT_MENUS: [
      { id: 'ai_analyze', title: '🧠 Analyze Job with AI' },
      { id: 'ai_auto_fill', title: '📝 Auto-Fill Application Form' },
      { id: 'ai_log_apply', title: '📌 Log Application to Notion' }
    ],
    DEFAULT_SETTINGS: {
      auto_analysis: true,
      notifications: true,
      debug_mode: false,
      preferred_platforms: ['linkedin', 'indeed', 'naukri'],
      match_threshold: 70
    },
    FETCH_TIMEOUT_MS: 10_000,
    MCP_RETRY: 3,
    NOTION_WEBHOOK_KEY: 'notion_webhook_url'
  };

  // -----------------------
  // UTILITIES
  // -----------------------
  const log = (...args) => { if (CONFIG.DEBUG) console.log('[BG]', ...args); };
  const info = (...args) => console.info('[BG]', ...args);
  const warn = (...args) => console.warn('[BG]', ...args);
  const error = (...args) => console.error('[BG]', ...args);

  async function storageLocalGet(keys) {
    return new Promise((resolve) => chrome.storage.local.get(keys, (res) => resolve(res || {})));
  }
  async function storageLocalSet(obj) {
    return new Promise((resolve, reject) => {
      chrome.storage.local.set(obj, () => {
        const e = chrome.runtime.lastError;
        if (e) reject(e);
        else resolve();
      });
    });
  }
  async function storageSyncGet(keys) {
    return new Promise((resolve) => chrome.storage.sync.get(keys, (res) => resolve(res || {})));
  }
  async function storageSyncSet(obj) {
    return new Promise((resolve, reject) => {
      chrome.storage.sync.set(obj, () => {
        const e = chrome.runtime.lastError;
        if (e) reject(e);
        else resolve();
      });
    });
  }

  function timeoutPromise(ms, promise) {
    return Promise.race([
      promise,
      new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), ms))
    ]);
  }

  async function safeFetch(url, opts = {}, timeoutMs = CONFIG.FETCH_TIMEOUT_MS) {
    try {
      const res = await timeoutPromise(timeoutMs, fetch(url, opts));
      return res;
    } catch (err) {
      throw err;
    }
  }

  // -----------------------
  // GLOBAL STATE
  // -----------------------
  let STATE = {
    apiBaseUrl: CONFIG.API_BASE_URL,
    apiConfig: null,
    extensionState: { isActive: true, apiStatus: 'disconnected' },
    mcp: { sessionId: null, connected: false },
    notionApi: null,
    ports: new Set()
  };

  // -----------------------
  // NOTION API WRAPPER (if NotionAPI exposed)
  // -----------------------
  function initNotionApi() {
    try {
      if (typeof self.NotionAPI === 'function') {
        const webhook = (STATE.apiConfig && STATE.apiConfig[CONFIG.NOTION_WEBHOOK_KEY]) || null;
        if (!webhook) {
          log('Notion webhook not configured; NotionAPI not initialized');
          return;
        }
        STATE.notionApi = new self.NotionAPI({ webhookUrl: webhook });
        log('NotionAPI initialized');
      } else {
        log('NotionAPI not found in global scope; skipping initialization');
      }
    } catch (e) {
      warn('Failed to init NotionAPI', e);
    }
  }

  // -----------------------
  // MCP BRIDGE
  // -----------------------
  class MCPBridge {
    constructor(baseUrl) {
      this.baseUrl = baseUrl || STATE.apiBaseUrl;
      this.sessionId = null;
      this.connected = false;
      this.retry = CONFIG.MCP_RETRY;
    }

    async _post(path, body = {}, headers = {}) {
      const url = `${this.baseUrl}${path}`;
      try {
        const res = await safeFetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...headers },
          body: JSON.stringify(body)
        });
        if (!res.ok) {
          const text = await res.text().catch(() => '');
          throw new Error(`HTTP ${res.status} ${text}`);
        }
        return await res.json().catch(() => null);
      } catch (e) {
        throw e;
      }
    }

    async startSession() {
      for (let attempt = 0; attempt < this.retry; attempt++) {
        try {
          const resp = await this._post('/mcp/session/start', {
            client_type: 'chrome_extension',
            user_agent: navigator.userAgent
          });
          if (resp && resp.session_id) {
            this.sessionId = resp.session_id;
            this.connected = true;
            STATE.mcp.sessionId = this.sessionId;
            STATE.mcp.connected = true;
            await persistExtensionState();
            info('MCP session started', this.sessionId);
            return this.sessionId;
          }
        } catch (err) {
          warn('MCP startSession attempt failed', attempt, err.message);
          await new Promise(r => setTimeout(r, 500 * (attempt + 1)));
        }
      }
      this.connected = false;
      throw new Error('MCP startSession failed');
    }

    async endSession() {
      if (!this.sessionId) return;
      try {
        await this._post('/mcp/session/end', { session_id: this.sessionId });
      } catch (e) { /* ignore gracefully */ }
      this.sessionId = null;
      this.connected = false;
      STATE.mcp = { sessionId: null, connected: false };
      await persistExtensionState();
      info('MCP session ended');
    }

    async postItem(item) {
      if (!this.sessionId) {
        await this.startSession().catch(e => { throw e; });
      }
      try {
        const resp = await this._post('/v1/sessions/' + this.sessionId + '/items', item, { 'X-Session-ID': this.sessionId });
        return resp;
      } catch (err) {
        warn('postItem failed', err);
        throw err;
      }
    }
  }

  // create one bridge instance
  const mcpBridge = new MCPBridge(STATE.apiBaseUrl);

  // -----------------------
  // CONTEXT MENUS (de-duplicated)
  // -----------------------
  async function initContextMenus() {
    try {
      // Remove all (defensive) then recreate. This avoids duplicate creation errors on service worker restart.
      chrome.contextMenus.removeAll(() => {
        CONFIG.CONTEXT_MENUS.forEach(menu => {
          try {
            chrome.contextMenus.create({
              id: menu.id,
              title: menu.title,
              contexts: ['page']
            }, () => {
              if (chrome.runtime.lastError) {
                // do not spam console; log if debug
                log('contextMenus.create lastError', chrome.runtime.lastError.message);
              } else {
                log('Created context menu', menu.id);
              }
            });
          } catch (e) {
            warn('Failed to create context menu', menu.id, e);
          }
        });
      });
    } catch (e) {
      warn('initContextMenus failed', e);
    }
  }

  // -----------------------
  // MESSAGE BROADCAST
  // -----------------------
  function broadcastMessage(message) {
    // notify all connected ports (sidebar/popups)
    STATE.ports.forEach(port => {
      try {
        port.postMessage(message);
      } catch (e) {
        try { STATE.ports.delete(port); } catch (_) {}
      }
    });
  }

  // -----------------------
  // PERSIST EXT STATE
  // -----------------------
  async function persistExtensionState() {
    try {
      const obj = {};
      obj[CONFIG.STORAGE_KEYS.EXTENSION_STATE] = STATE.extensionState;
      await storageLocalSet(obj);
    } catch (e) {
      warn('persistExtensionState failed', e);
    }
  }

  // -----------------------
  // NOTION LOGGING
  // -----------------------
  async function logApplicationToNotion(jobData = {}) {
    try {
      if (STATE.notionApi && typeof STATE.notionApi.logApplication === 'function') {
        return await STATE.notionApi.logApplication(jobData);
      }
      // fallback: try webhook POST if url available in apiConfig
      const webhook = (STATE.apiConfig && STATE.apiConfig[CONFIG.NOTION_WEBHOOK_KEY]) || null;
      if (!webhook) {
        warn('Notion webhook not configured');
        return { success: false, error: 'no_noton_webhook' };
      }
      // do a single POST with timeout
      const res = await safeFetch(webhook, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jobData)
      }, CONFIG.FETCH_TIMEOUT_MS);
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        return { success: false, error: `HTTP ${res.status}: ${text}` };
      }
      return { success: true, raw: await res.json().catch(() => null) };
    } catch (e) {
      warn('logApplicationToNotion failed', e);
      return { success: false, error: String(e) };
    }
  }

  // -----------------------
  // MESSAGE HANDLERS
  // -----------------------
  async function handleMessage(message, sender, sendResponse) {
    try {
      if (!message || !message.type) return sendResponse({ success: false, error: 'no_message_type' });

      switch (message.type) {
        // Called by content script when a job detail is detected
        case 'JOB_DETECTED':
          {
            // cache job data for 24 hours
            const jobData = message.jobData || {};
            const cacheKey = `job_${(jobData.url || '').slice(0, 200)}`;
            const jobCacheObj = (await storageLocalGet([CONFIG.STORAGE_KEYS.JOB_CACHE]))[CONFIG.STORAGE_KEYS.JOB_CACHE] || {};
            jobCacheObj[cacheKey] = { jobData, timestamp: Date.now() };
            await storageLocalSet({ [CONFIG.STORAGE_KEYS.JOB_CACHE]: jobCacheObj });

            // broadcast to UI (sidebar)
            broadcastMessage({ type: 'JOB_DETECTED', jobData });
            return sendResponse({ success: true });
          }

        // request: extract profile from storage
        case 'GET_USER_PROFILE':
          {
            const sync = await storageSyncGet([CONFIG.STORAGE_KEYS.USER_SETTINGS]);
            const profile = (sync && sync[CONFIG.STORAGE_KEYS.USER_SETTINGS]) || CONFIG.DEFAULT_SETTINGS;
            return sendResponse({ success: true, profile });
          }

        // update profile
        case 'UPDATE_PROFILE':
          {
            const payload = message.payload || message.settings || {};
            try {
              await storageSyncSet({ [CONFIG.STORAGE_KEYS.USER_SETTINGS]: payload });
            } catch (e) {
              warn('UPDATE_PROFILE: storageSyncSet failed, falling back to local', e);
              await storageLocalSet({ [CONFIG.STORAGE_KEYS.USER_SETTINGS]: payload });
            }
            return sendResponse({ success: true });
          }

        // request to start dynamic resume generation (RAG)
        case 'REQUEST_DYNAMIC_RESUME':
          {
            // proxy request to API_BASE_URL / rag endpoint
            const jobTitle = (message.payload && message.payload.jobTitle) || '';
            try {
              const url = `${STATE.apiBaseUrl}/rag/dynamic_resume?job_title=${encodeURIComponent(jobTitle || '')}`;
              const res = await safeFetch(url, { method: 'GET' }, CONFIG.FETCH_TIMEOUT_MS);
              if (!res.ok) {
                const text = await res.text().catch(() => '');
                return sendResponse({ success: false, error: `HTTP ${res.status}: ${text}` });
              }
              const contentType = res.headers.get('content-type') || '';
              let resumeMeta = null;
              if (contentType.includes('application/json')) {
                const json = await res.json().catch(() => null);
                // expected: { resume_id, name, score, base64?, mime_type }
                resumeMeta = json;
              } else {
                // binary response -> make available through GET_RESUME_BLOB handler (background stores base64)
                const blob = await res.blob();
                const arr = new Uint8Array(await blob.arrayBuffer());
                const b64 = btoa(String.fromCharCode(...arr));
                resumeMeta = { resume_id: `r_${Date.now()}`, base64: b64, mime_type: blob.type || 'application/pdf', name: `resume_${Date.now()}.pdf`, cached_at: new Date().toISOString() };
              }
              // store resume meta in sync/local for sidebar to read
              const saved = (await storageSyncGet([CONFIG.STORAGE_KEYS.RESUMES_META]))[CONFIG.STORAGE_KEYS.RESUMES_META] || {};
              saved[resumeMeta.resume_id] = resumeMeta;
              await storageSyncSet({ [CONFIG.STORAGE_KEYS.RESUMES_META]: saved });
              // also store blob in local storage (large) as base64
              const blobs = (await storageLocalGet([CONFIG.STORAGE_KEYS.RESUME_BLOBS]))[CONFIG.STORAGE_KEYS.RESUME_BLOBS] || {};
              if (resumeMeta.base64) blobs[resumeMeta.resume_id] = { base64: resumeMeta.base64, mime: resumeMeta.mime_type || 'application/pdf', fileName: resumeMeta.name };
              await storageLocalSet({ [CONFIG.STORAGE_KEYS.RESUME_BLOBS]: blobs });

              // respond success
              return sendResponse({ success: true, resume: resumeMeta });
            } catch (e) {
              warn('REQUEST_DYNAMIC_RESUME failed', e);
              return sendResponse({ success: false, error: String(e) });
            }
          }

        // request to get resumes meta
        case 'GET_RESUMES_META':
          {
            const meta = (await storageSyncGet([CONFIG.STORAGE_KEYS.RESUMES_META]))[CONFIG.STORAGE_KEYS.RESUMES_META] || {};
            return sendResponse({ success: true, meta });
          }

        // request to get resume blob (base64) by id
        case 'GET_RESUME_BLOB':
          {
            const resumeId = message.payload && message.payload.resumeId;
            if (!resumeId) return sendResponse({ success: false, error: 'missing_resumeId' });
            const blobs = (await storageLocalGet([CONFIG.STORAGE_KEYS.RESUME_BLOBS]))[CONFIG.STORAGE_KEYS.RESUME_BLOBS] || {};
            const blob = blobs[resumeId];
            if (!blob) return sendResponse({ success: false, error: 'not_found' });
            return sendResponse({ success: true, base64: blob.base64, meta: { mime: blob.mime, fileName: blob.fileName } });
          }

        // apply events: content or sidebar will call this when an application is detected
        case 'APPLY_STARTED':
          {
            // store temporary apply start record or broadcast
            broadcastMessage({ type: 'APPLY_STARTED', metadata: message.metadata || {} });
            return sendResponse({ success: true });
          }

        case 'APPLY_COMPLETED':
          {
            const payload = message.metadata || {};
            // persist to activity log (local)
            const logEntry = {
              job_title: payload.jobTitle || payload.job_title || 'Applied',
              company: payload.company || '',
              job_url: payload.jobUrl || payload.job_url || payload.jobUrl || '',
              timestamp: payload.timestamp || new Date().toISOString(),
              resume_id: payload.resume_id || payload.resumeId || null
            };
            const existing = (await storageLocalGet([CONFIG.STORAGE_KEYS.ACTIVITY_LOG]))[CONFIG.STORAGE_KEYS.ACTIVITY_LOG] || [];
            existing.unshift(logEntry);
            if (existing.length > 500) existing.length = 500;
            await storageLocalSet({ [CONFIG.STORAGE_KEYS.ACTIVITY_LOG]: existing });

            // send to MCP
            try {
              await mcpBridge.postItem({ source: 'chrome_extension', job_url: logEntry.job_url, timestamp: logEntry.timestamp, status: 'applied' }).catch(e => { warn('MCP postItem failed', e); });
            } catch (e) {
              warn('APPLY_COMPLETED MCP error', e);
            }

            // send to Notion
            try {
              const notionResp = await logApplicationToNotion({
                job_url: logEntry.job_url, job_title: logEntry.job_title, company: logEntry.company, timestamp: logEntry.timestamp, resume_id: logEntry.resume_id
              });
              if (!notionResp || !notionResp.success) {
                warn('Notion logging failed', notionResp && notionResp.error);
              }
            } catch (e) { warn('Notion logging error', e); }

            // broadcast to connected UI
            broadcastMessage({ type: 'APPLICATION_COMPLETED', payload: logEntry });
            return sendResponse({ success: true });
          }

        case 'GET_EXTENSION_STATUS':
          {
            const status = {
              active: STATE.extensionState.isActive,
              apiStatus: STATE.extensionState.apiStatus,
              mcpConnected: !!(STATE.mcp && STATE.mcp.connected),
              version: CONFIG.EXTENSION_VERSION,
              apiEndpoints: { rag: !!STATE.apiConfig?.rag, notion: !!STATE.apiConfig?.notion }
            };
            return sendResponse({ success: true, status });
          }

        case 'REQUEST_INJECT_AND_AUTOFILL':
          {
            // attempt to ensure content script exists and forward autofill
            const tabs = await new Promise((res) => chrome.tabs.query({ active: true, currentWindow: true }, res));
            const tab = tabs && tabs[0];
            if (!tab || !tab.id) return sendResponse({ success: false, error: 'no_active_tab' });
            try {
              // ensure content script is injected via scripting API (manifest must declare)
              try {
                chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content/content.js'] }, () => { /* ignore */ });
              } catch (_) { /* ignore injection failure; maybe already injected */ }
              // send auto-fill message
              chrome.tabs.sendMessage(tab.id, { type: 'AUTO_FILL_FORM', resumeMode: message.payload && message.payload.resumeMode }, (resp) => {
                // we don't strictly wait; return success if message posted
              });
              return sendResponse({ success: true });
            } catch (e) {
              return sendResponse({ success: false, error: String(e) });
            }
          }

        default:
          warn('Unknown message type', message.type);
          return sendResponse({ success: false, error: 'unknown_message' });
      }
    } catch (err) {
      error('handleMessage top-level error', err);
      try { sendResponse({ success: false, error: String(err) }); } catch (_) {}
    }
  }

  // -----------------------
  // CONTEXT MENU CLICK HANDLER
  // -----------------------
  async function onContextMenuClick(info, tab) {
    try {
      if (!info.menuItemId) return;
      switch (info.menuItemId) {
        case 'ai_analyze':
          try {
            // ask content to extract job and send to MCP/analyze
            if (tab && tab.id) {
              chrome.tabs.sendMessage(tab.id, { type: 'EXTRACT_JOB_DATA' }, (resp) => {
                // If content returns jobData, trigger analysis pipeline in backend (MCP)
                if (resp && resp.jobData) {
                  // forward to MCP or other service via background
                  safeFetch(`${STATE.apiBaseUrl}/mcp/analyze-job`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ job_data: resp.jobData })
                  }).catch(e => warn('mcp analyze POST failed', e));
                } else {
                  warn('EXTRACT_JOB_DATA not returned or empty');
                }
              });
            }
          } catch (e) { warn('ai_analyze click failed', e); }
          break;

        case 'ai_auto_fill':
          if (tab && tab.id) {
            chrome.tabs.sendMessage(tab.id, { type: 'AUTO_FILL_FORM' }, (resp) => { /* noop */ });
          }
          break;

        case 'ai_log_apply':
          if (tab && tab.id) {
            chrome.tabs.sendMessage(tab.id, { type: 'EXTRACT_JOB_DATA_AND_LOG' }, (resp) => { /* noop */ });
          }
          break;

        default:
          break;
      }
    } catch (e) {
      warn('onContextMenuClick error', e);
    }
  }

  // -----------------------
  // ON INSTALLED / STARTUP
  // -----------------------
  async function onInstalled(details) {
    try {
      CONFIG.DEBUG = (await storageSyncGet([CONFIG.STORAGE_KEYS.USER_SETTINGS]))[CONFIG.STORAGE_KEYS.USER_SETTINGS]?.debug_mode || CONFIG.DEBUG;
      info('Extension installed/updated', details);
      // set default settings on first install
      if (details.reason === 'install') {
        await storageSyncSet({ [CONFIG.STORAGE_KEYS.USER_SETTINGS]: CONFIG.DEFAULT_SETTINGS });
      }
      // load apiConfig if present
      const apiConf = (await storageLocalGet([CONFIG.STORAGE_KEYS.API_CONFIG]))[CONFIG.STORAGE_KEYS.API_CONFIG] || null;
      STATE.apiConfig = apiConf;
      // initialize notion api if webhook present
      initNotionApi();
      // init context menus
      await initContextMenus();
      // initialize MCP session if desired
      try { await mcpBridge.startSession().catch(()=>{}); } catch (_) {}
      // persist state
      await persistExtensionState();
    } catch (e) { warn('onInstalled failed', e); }
  }

  // -----------------------
  // TABS & NAV ALERTS
  // -----------------------
  function onTabUpdated(tabId, changeInfo, tab) {
    try {
      if (changeInfo.status === 'complete' && tab && tab.url) {
        // notify content script if this looks like job site
        const supportedPlatforms = [
          'linkedin.com', 'indeed.com', 'naukri.com', 'flexjobs.com',
          'weworkremotely.com', 'remotive.io', 'wellfound.com', 'angel.co',
          'justremote.co', 'remoteok.io', 'glassdoor.com'
        ];
        const isJobSite = supportedPlatforms.some(p => tab.url.includes(p));
        if (isJobSite) {
          // try to send a lightweight message; content script will extract
          try {
            chrome.tabs.sendMessage(tabId, { type: 'JOB_SITE_DETECTED', url: tab.url }, (resp) => { /* noop */ });
          } catch (e) { /* ignore */ }
        }
      }
    } catch (e) { warn('onTabUpdated error', e); }
  }

  // -----------------------
  // PORT CONNECTIONS (sidebar popup)
  // -----------------------
  function onPortConnected(port) {
    try {
      STATE.ports.add(port);
      log('port connected', port.name);
      port.onMessage.addListener((msg) => {
        // port message handling (if any)
        if (!msg || !msg.type) return;
        switch (msg.type) {
          case 'PING':
            port.postMessage({ type: 'PONG' });
            break;
          default:
            break;
        }
      });
      port.onDisconnect.addListener(() => {
        try { STATE.ports.delete(port); } catch (e) { /* ignore */ }
      });
    } catch (e) { warn('onPortConnected failed', e); }
  }

  // -----------------------
  // KEEP SERVICE WORKER ALIVE (best-effort)
  // -----------------------
  function keepAlive() {
    try {
      // call a benign API to keep service worker active in dev; no-op in prod
      setInterval(() => {
        // use a low-cost call
        if (typeof chrome.runtime.getPlatformInfo === 'function') {
          chrome.runtime.getPlatformInfo(() => { /* noop */ });
        }
      }, 20_000);
    } catch (e) { /* ignore */ }
  }

  // -----------------------
  // BINDING & BOOT
  // -----------------------
  // runtime message listener
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    // keep message channel open for async responses
    try {
      handleMessage(message, sender, sendResponse);
    } catch (e) {
      error('runtime.onMessage error', e);
      try { sendResponse({ success: false, error: e.message }); } catch (_) {}
    }
    return true;
  });

  chrome.contextMenus.onClicked.addListener(onContextMenuClick);
  chrome.runtime.onConnect.addListener(onPortConnected);
  chrome.tabs.onUpdated.addListener(onTabUpdated);

  chrome.runtime.onInstalled.addListener(onInstalled);

  chrome.commands.onCommand.addListener(async (command, tab) => {
    try {
      switch (command) {
        case 'analyze-job':
        case 'analyze_job':
        case 'ai_analyze':
          if (tab && tab.id) chrome.tabs.sendMessage(tab.id, { type: 'EXTRACT_JOB_DATA' });
          break;
        case 'auto-fill':
        case 'auto_fill_now':
        case 'ai_auto_fill':
          if (tab && tab.id) chrome.tabs.sendMessage(tab.id, { type: 'AUTO_FILL_FORM' });
          break;
        case 'toggle-sidebar':
          if (tab && tab.id) chrome.tabs.sendMessage(tab.id, { type: 'TOGGLE_SIDEBAR' });
          break;
        default:
          log('unknown command', command);
      }
    } catch (e) { warn('onCommand error', e); }
  });

  // graceful suspend handling
  chrome.runtime.onSuspend.addListener(async () => {
    try {
      log('service worker suspending — cleanup tasks');
      // end MCP session gracefully
      await mcpBridge.endSession().catch(()=>{});
      // flush queued notion logs if possible
      if (STATE.notionApi && STATE.notionApi.processQueue) {
        try { await STATE.notionApi.processQueue(); } catch (_) {}
      }
      await persistExtensionState();
    } catch (e) { warn('onSuspend error', e); }
  });

  // startup (immediate)
  (async function bootstrap() {
    try {
      // load API config
      const local = await storageLocalGet([CONFIG.STORAGE_KEYS.API_CONFIG]);
      STATE.apiConfig = local[CONFIG.STORAGE_KEYS.API_CONFIG] || null;
      if (STATE.apiConfig && STATE.apiConfig.api_base_url) STATE.apiBaseUrl = STATE.apiConfig.api_base_url;
      // init Notion API if available
      initNotionApi();
      // create context menus
      await initContextMenus();
      // try to start MCP session in background (non-blocking)
      mcpBridge.startSession().catch((e) => { log('MCP startSession non-blocking failure', e && e.message); });
      // set up keep alive
      keepAlive();
      info('Background bootstrapped');
    } catch (e) {
      warn('bootstrap failed', e);
    }
  })();

  // Expose for testing/debug
  return {
    _internal: {
      CONFIG, STATE, mcpBridge, initContextMenus, logApplicationToNotion
    }
  };

})();