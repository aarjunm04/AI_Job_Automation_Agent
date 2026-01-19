/**
 * =============================================================================
 * AI JOB AUTOMATION AGENT - BACKGROUND SERVICE WORKER (COMPLETE)
 * =============================================================================
 * 
 * Manifest V3 service worker - central orchestrator for the extension.
 * 
 * Responsibilities:
 * - MCP client management and message routing
 * - Job detection and caching
 * - Resume generation via RAG endpoint
 * - Notion logging integration
 * - Context menus and keyboard commands
 * - Storage management (sync/local)
 * - Message broker between content scripts and sidebar
 * 
 * Author: AI Job Automation Team
 * Version: 1.0.0
 * =============================================================================
 */

'use strict';

// Import dependencies
import { 
  MCP_CONFIG,
  STORAGE_KEYS, 
  DEFAULT_USER_SETTINGS,
  TIMING,
  FEATURES,
  UI_IDS,
  MCP_TASK_TYPES,
  loadRuntimeConfig
} from './extension_config.js';

import mcpClient from './mcp_client.js';

/**
 * =============================================================================
 * GLOBAL STATE
 * =============================================================================
 */
const STATE = {
  runtimeConfig: null,
  notionApi: null,
  connectedPorts: new Set(),
  isInitialized: false,
  mcpEnabled: true,
  debugMode: false
};

/**
 * =============================================================================
 * LOGGING UTILITIES
 * =============================================================================
 */
const log = (...args) => {
  if (STATE.debugMode) console.log('[BG]', ...args);
};

const info = (...args) => console.info('[BG]', ...args);
const warn = (...args) => console.warn('[BG]', ...args);
const error = (...args) => console.error('[BG]', ...args);

/**
 * =============================================================================
 * STORAGE HELPERS
 * =============================================================================
 */
const Storage = {
  async getSync(keys) {
    return new Promise(resolve => {
      chrome.storage.sync.get(keys, result => resolve(result || {}));
    });
  },

  async setSync(obj) {
    return new Promise((resolve, reject) => {
      chrome.storage.sync.set(obj, () => {
        if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
        else resolve();
      });
    });
  },

  async getLocal(keys) {
    return new Promise(resolve => {
      chrome.storage.local.get(keys, result => resolve(result || {}));
    });
  },

  async setLocal(obj) {
    return new Promise((resolve, reject) => {
      chrome.storage.local.set(obj, () => {
        if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
        else resolve();
      });
    });
  }
};

/**
 * =============================================================================
 * HTTP UTILITIES
 * =============================================================================
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timeoutId);
    return response;
  } catch (err) {
    clearTimeout(timeoutId);
    throw err;
  }
}

/**
 * =============================================================================
 * NOTION API INTEGRATION
 * =============================================================================
 */
async function initNotionApi() {
  try {
    if (!FEATURES.ENABLE_NOTION_LOGGING) {
      log('Notion logging disabled by feature flag');
      return;
    }

    if (typeof NotionAPI !== 'function') {
      warn('NotionAPI class not available, skipping initialization');
      return;
    }

    const webhookUrl = STATE.runtimeConfig?.notion?.webhook_url;
    if (!webhookUrl) {
      log('Notion webhook URL not configured');
      return;
    }

    STATE.notionApi = new NotionAPI({ webhookUrl });
    info('Notion API initialized');
  } catch (err) {
    warn('Failed to initialize Notion API', err);
  }
}

async function logToNotion(jobData) {
  try {
    if (!FEATURES.ENABLE_NOTION_LOGGING) return { success: false, error: 'disabled' };

    if (STATE.notionApi && typeof STATE.notionApi.logApplication === 'function') {
      return await STATE.notionApi.logApplication(jobData);
    }

    // Fallback to direct webhook POST
    const webhookUrl = STATE.runtimeConfig?.notion?.webhook_url;
    if (!webhookUrl) {
      return { success: false, error: 'no_webhook_configured' };
    }

    const response = await fetchWithTimeout(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...jobData,
        timestamp: new Date().toISOString(),
        source: 'chrome_extension'
      })
    }, 10000);

    if (!response.ok) {
      throw new Error(`Webhook returned ${response.status}`);
    }

    return { success: true };
  } catch (err) {
    warn('Notion logging failed', err);
    return { success: false, error: err.message };
  }
}

/**
 * =============================================================================
 * ACTIVITY LOG
 * =============================================================================
 */
async function logActivity(type, data = {}) {
  try {
    const activityLog = (await Storage.getLocal([STORAGE_KEYS.AI_ACTIVITY_LOG]))[STORAGE_KEYS.AI_ACTIVITY_LOG] || [];
    
    activityLog.unshift({
      type,
      timestamp: new Date().toISOString(),
      ...data
    });

    // Keep only last N entries
    if (activityLog.length > TIMING.ACTIVITY_LOG_MAX_ENTRIES) {
      activityLog.length = TIMING.ACTIVITY_LOG_MAX_ENTRIES;
    }

    await Storage.setLocal({ [STORAGE_KEYS.AI_ACTIVITY_LOG]: activityLog });
  } catch (err) {
    warn('Failed to log activity', err);
  }
}

/**
 * =============================================================================
 * MESSAGE HANDLERS
 * =============================================================================
 */
const MessageHandlers = {
  /**
   * MCP completion request from sidebar or content
   */
  async MCP_COMPLETE(payload, sender) {
    try {
      if (!FEATURES.ENABLE_MCP_CHAT) {
        return { success: false, error: 'MCP chat is disabled' };
      }

      const { sessionName, taskType, prompt, meta } = payload;

      if (!taskType || !prompt) {
        return { success: false, error: 'taskType and prompt are required' };
      }

      // Call MCP client
      const result = await mcpClient.complete({
        sessionName: sessionName || 'default',
        taskType,
        prompt,
        meta: {
          ...meta,
          tab_url: sender.tab?.url,
          sender_origin: sender.tab ? 'content' : 'sidebar'
        }
      });

      // Log activity
      if (result.success) {
        await logActivity('mcp_completion', {
          taskType,
          promptLength: prompt.length,
          completionLength: result.completion?.length || 0
        });
      }

      return result;

    } catch (err) {
      error('MCP_COMPLETE handler failed', err);
      return { success: false, error: err.message };
    }
  },

  /**
   * Job detected on page
   */
  async JOB_DETECTED(payload, sender) {
    try {
      const jobData = payload.jobData || {};
      const jobUrl = jobData.url || sender.tab?.url || '';

      // Cache job data
      const cacheKey = `job_${jobUrl.slice(0, 200)}`;
      const jobCache = (await Storage.getLocal([STORAGE_KEYS.JOB_CACHE]))[STORAGE_KEYS.JOB_CACHE] || {};
      
      jobCache[cacheKey] = {
        jobData,
        timestamp: Date.now()
      };

      await Storage.setLocal({ [STORAGE_KEYS.JOB_CACHE]: jobCache });

      // Broadcast to all connected ports (sidebar)
      broadcastToSidebar({ type: 'JOB_DETECTED', jobData });

      // Log activity
      await logActivity('job_detected', {
        job_title: jobData.title,
        company: jobData.company,
        url: jobUrl
      });

      return { success: true };
    } catch (err) {
      error('JOB_DETECTED handler failed', err);
      return { success: false, error: err.message };
    }
  },

  /**
   * Get user profile
   */
  async GET_USER_PROFILE(payload, sender) {
    try {
      const settings = (await Storage.getSync([STORAGE_KEYS.USER_SETTINGS]))[STORAGE_KEYS.USER_SETTINGS];
      const profile = settings || DEFAULT_USER_SETTINGS;
      return { success: true, profile };
    } catch (err) {
      error('GET_USER_PROFILE failed', err);
      return { success: false, error: err.message };
    }
  },

  /**
   * Update user profile
   */
  async UPDATE_PROFILE(payload, sender) {
    try {
      const newProfile = payload.profile || payload.settings || {};
      await Storage.setSync({ [STORAGE_KEYS.USER_SETTINGS]: newProfile });
      
      await logActivity('profile_updated', {
        fields: Object.keys(newProfile)
      });

      return { success: true };
    } catch (err) {
      error('UPDATE_PROFILE failed', err);
      return { success: false, error: err.message };
    }
  },

  /**
   * Request dynamic resume generation
   */
  async REQUEST_DYNAMIC_RESUME(payload, sender) {
    try {
      if (!FEATURES.ENABLE_DYNAMIC_RESUME) {
        return { success: false, error: 'Dynamic resume generation is disabled' };
      }

      const jobTitle = payload.jobTitle || '';
      const ragUrl = `${STATE.runtimeConfig.mcp.base_url}/rag/dynamic_resume?job_title=${encodeURIComponent(jobTitle)}`;

      const response = await fetchWithTimeout(ragUrl, { method: 'GET' }, 20000);

      if (!response.ok) {
        throw new Error(`RAG endpoint returned ${response.status}`);
      }

      const contentType = response.headers.get('content-type') || '';
      let resumeMeta;

      if (contentType.includes('application/json')) {
        resumeMeta = await response.json();
      } else {
        // Binary response - convert to base64
        const blob = await response.blob();
        const arrayBuffer = await blob.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        const base64 = btoa(String.fromCharCode(...uint8Array));

        resumeMeta = {
          resume_id: `resume_${Date.now()}`,
          name: `resume_${jobTitle.slice(0, 30)}.pdf`,
          base64,
          mime_type: blob.type || 'application/pdf',
          cached_at: new Date().toISOString()
        };
      }

      // Store resume metadata
      const resumesMeta = (await Storage.getSync([STORAGE_KEYS.RESUMES_META]))[STORAGE_KEYS.RESUMES_META] || {};
      resumesMeta[resumeMeta.resume_id] = resumeMeta;
      await Storage.setSync({ [STORAGE_KEYS.RESUMES_META]: resumesMeta });

      // Store blob in local storage
      if (resumeMeta.base64) {
        const resumeBlobs = (await Storage.getLocal([STORAGE_KEYS.RESUME_BLOBS]))[STORAGE_KEYS.RESUME_BLOBS] || {};
        resumeBlobs[resumeMeta.resume_id] = {
          base64: resumeMeta.base64,
          mime: resumeMeta.mime_type,
          fileName: resumeMeta.name
        };
        await Storage.setLocal({ [STORAGE_KEYS.RESUME_BLOBS]: resumeBlobs });
      }

      await logActivity('resume_generated', {
        job_title: jobTitle,
        resume_id: resumeMeta.resume_id
      });

      return { success: true, resume: resumeMeta };

    } catch (err) {
      error('REQUEST_DYNAMIC_RESUME failed', err);
      return { success: false, error: err.message };
    }
  },

  /**
   * Get resumes metadata
   */
  async GET_RESUMES_META(payload, sender) {
    try {
      const meta = (await Storage.getSync([STORAGE_KEYS.RESUMES_META]))[STORAGE_KEYS.RESUMES_META] || {};
      return { success: true, meta };
    } catch (err) {
      return { success: false, error: err.message };
    }
  },

  /**
   * Get resume blob by ID
   */
  async GET_RESUME_BLOB(payload, sender) {
    try {
      const resumeId = payload.resumeId;
      if (!resumeId) {
        return { success: false, error: 'resumeId is required' };
      }

      const resumeBlobs = (await Storage.getLocal([STORAGE_KEYS.RESUME_BLOBS]))[STORAGE_KEYS.RESUME_BLOBS] || {};
      const blob = resumeBlobs[resumeId];

      if (!blob) {
        return { success: false, error: 'Resume not found' };
      }

      return {
        success: true,
        base64: blob.base64,
        meta: {
          mime: blob.mime,
          fileName: blob.fileName
        }
      };
    } catch (err) {
      return { success: false, error: err.message };
    }
  },

  /**
   * Log application to Notion
   */
  async LOG_APPLICATION(payload, sender) {
    try {
      const jobData = payload.jobData || {};
      const result = await logToNotion(jobData);

      if (result.success) {
        await logActivity('application_logged', {
          job_title: jobData.job_title,
          company: jobData.company
        });
      }

      return result;
    } catch (err) {
      return { success: false, error: err.message };
    }
  },

  /**
   * Get activity log
   */
  async GET_ACTIVITY_LOG(payload, sender) {
    try {
      const activityLog = (await Storage.getLocal([STORAGE_KEYS.AI_ACTIVITY_LOG]))[STORAGE_KEYS.AI_ACTIVITY_LOG] || [];
      return { success: true, log: activityLog };
    } catch (err) {
      return { success: false, error: err.message };
    }
  },

  /**
   * Test MCP connection
   */
  async TEST_MCP_CONNECTION(payload, sender) {
    try {
      const result = await mcpClient.testConnection();
      return result;
    } catch (err) {
      return { success: false, message: err.message };
    }
  },

  /**
   * Update MCP API key
   */
  async UPDATE_MCP_API_KEY(payload, sender) {
    try {
      const apiKey = payload.apiKey;
      if (!apiKey) {
        return { success: false, error: 'API key is required' };
      }

      await mcpClient.updateApiKey(apiKey);
      return { success: true };
    } catch (err) {
      return { success: false, error: err.message };
    }
  },

  /**
   * Clear MCP session
   */
  async CLEAR_MCP_SESSION(payload, sender) {
    try {
      const sessionName = payload.sessionName || 'default';
      mcpClient.clearSession(sessionName);
      return { success: true };
    } catch (err) {
      return { success: false, error: err.message };
    }
  }
};

/**
 * =============================================================================
 * MESSAGE ROUTER
 * =============================================================================
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    try {
      if (!message || !message.type) {
        sendResponse({ success: false, error: 'Invalid message format' });
        return;
      }

      const handler = MessageHandlers[message.type];
      if (!handler) {
        warn(`No handler for message type: ${message.type}`);
        sendResponse({ success: false, error: `Unknown message type: ${message.type}` });
        return;
      }

      const result = await handler(message.payload || {}, sender);
      sendResponse(result);

    } catch (err) {
      error('Message handler error', err);
      sendResponse({ success: false, error: err.message });
    }
  })();

  return true; // Keep channel open for async response
});

/**
 * =============================================================================
 * PORT CONNECTIONS (Sidebar)
 * =============================================================================
 */
chrome.runtime.onConnect.addListener((port) => {
  STATE.connectedPorts.add(port);
  log('Port connected:', port.name);

  port.onMessage.addListener((msg) => {
    log('Port message:', msg);
  });

  port.onDisconnect.addListener(() => {
    STATE.connectedPorts.delete(port);
    log('Port disconnected');
  });
});

function broadcastToSidebar(message) {
  STATE.connectedPorts.forEach(port => {
    try {
      port.postMessage(message);
    } catch (err) {
      STATE.connectedPorts.delete(port);
    }
  });
}

/**
 * =============================================================================
 * CONTEXT MENUS
 * =============================================================================
 */
async function initContextMenus() {
  try {
    chrome.contextMenus.removeAll(() => {
      const menus = [
        { id: UI_IDS.CONTEXT_ANALYZE_JOB, title: '🧠 Analyze Job with AI' },
        { id: UI_IDS.CONTEXT_AUTO_FILL, title: '📝 Auto-Fill Application Form' },
        { id: UI_IDS.CONTEXT_LOG_NOTION, title: '📌 Log Application to Notion' }
      ];

      menus.forEach(menu => {
        chrome.contextMenus.create({
          id: menu.id,
          title: menu.title,
          contexts: ['page']
        }, () => {
          if (chrome.runtime.lastError) {
            log('Context menu creation error:', chrome.runtime.lastError.message);
          }
        });
      });
    });

    info('Context menus initialized');
  } catch (err) {
    warn('Failed to initialize context menus', err);
  }
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
  try {
    switch (info.menuItemId) {
      case UI_IDS.CONTEXT_ANALYZE_JOB:
        chrome.tabs.sendMessage(tab.id, { type: 'ANALYZE_JOB' });
        break;
      case UI_IDS.CONTEXT_AUTO_FILL:
        chrome.tabs.sendMessage(tab.id, { type: 'AUTO_FILL_FORM' });
        break;
      case UI_IDS.CONTEXT_LOG_NOTION:
        chrome.tabs.sendMessage(tab.id, { type: 'LOG_APPLICATION_NOW' });
        break;
    }
  } catch (err) {
    warn('Context menu click handler error', err);
  }
});

/**
 * =============================================================================
 * KEYBOARD COMMANDS
 * =============================================================================
 */
chrome.commands.onCommand.addListener((command, tab) => {
  try {
    switch (command) {
      case UI_IDS.COMMAND_ANALYZE:
        chrome.tabs.sendMessage(tab.id, { type: 'ANALYZE_JOB' });
        break;
      case UI_IDS.COMMAND_AUTO_FILL:
        chrome.tabs.sendMessage(tab.id, { type: 'AUTO_FILL_FORM' });
        break;
      case UI_IDS.COMMAND_TOGGLE_SIDEBAR:
        chrome.tabs.sendMessage(tab.id, { type: 'TOGGLE_SIDEBAR' });
        break;
    }
  } catch (err) {
    warn('Command handler error', err);
  }
});

/**
 * =============================================================================
 * TAB EVENTS
 * =============================================================================
 */
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    // Notify content script of navigation
    try {
      chrome.tabs.sendMessage(tabId, {
        type: 'TAB_UPDATED',
        url: tab.url
      }, () => {
        if (chrome.runtime.lastError) {
          // Tab not ready or no listener, ignore
        }
      });
    } catch (err) {
      // Ignore
    }
  }
});

/**
 * =============================================================================
 * INITIALIZATION
 * =============================================================================
 */
async function initialize() {
  try {
    info('Initializing background service worker...');

    // Load runtime config
    STATE.runtimeConfig = await loadRuntimeConfig();
    STATE.debugMode = STATE.runtimeConfig.debug;
    STATE.mcpEnabled = STATE.runtimeConfig.mcp.enabled;

    // Initialize MCP client
    if (STATE.mcpEnabled) {
      await mcpClient.init();
      info('MCP client ready');
    }

    // Initialize Notion API
    await initNotionApi();

    // Set up context menus
    await initContextMenus();

    // Check if user settings exist, if not create defaults
    const settings = await Storage.getSync([STORAGE_KEYS.USER_SETTINGS]);
    if (!settings[STORAGE_KEYS.USER_SETTINGS]) {
      await Storage.setSync({ [STORAGE_KEYS.USER_SETTINGS]: DEFAULT_USER_SETTINGS });
      info('Default user settings created');
    }

    STATE.isInitialized = true;
    info('Background initialization complete');

  } catch (err) {
    error('Initialization failed', err);
  }
}

/**
 * =============================================================================
 * LIFECYCLE EVENTS
 * =============================================================================
 */
chrome.runtime.onInstalled.addListener((details) => {
  info('Extension installed/updated:', details.reason);
  initialize();
});

chrome.runtime.onStartup.addListener(() => {
  info('Extension startup');
  initialize();
});

chrome.runtime.onSuspend.addListener(() => {
  info('Service worker suspending...');
});

// Initialize immediately
initialize();
