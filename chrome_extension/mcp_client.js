/**
 * =============================================================================
 * AI JOB AUTOMATION AGENT - MCP CLIENT (COMPLETE & WORKING)
 * =============================================================================
 * 
 * Handles communication with MCP server for LLM completions.
 * 
 * Features:
 * - Session management with server-generated UUIDs
 * - Query parameter based API calls
 * - Automatic session creation and caching
 * - Connection testing
 * - Timeout handling
 * - API key management
 * 
 * Author: AI Job Automation Team
 * Version: 1.0.0
 * =============================================================================
 */

'use strict';

import { MCP_CONFIG } from './extension_config.js';

class MCPClient {
  constructor(config = {}) {
    this.config = {
      baseUrl: config.baseUrl || MCP_CONFIG.BASE_URL,
      completeEndpoint: config.completeEndpoint || MCP_CONFIG.COMPLETE_ENDPOINT,
      apiKey: config.apiKey || null,
      timeoutMs: config.timeoutMs || MCP_CONFIG.TIMEOUT_MS,
      maxRetries: config.maxRetries || MCP_CONFIG.MAX_RETRIES,
      retryBaseDelayMs: config.retryBaseDelayMs || MCP_CONFIG.RETRY_BASE_DELAY_MS
    };

    // Session management
    this.sessions = new Map(); // sessionName -> server-generated sessionId
    this.createdSessions = new Set(); // Track which session names we've created
    this.debugMode = false;
    
    this.init();
  }

  /**
   * Initialize client and load stored API key
   */
  async init() {
    try {
      const stored = await chrome.storage.sync.get([MCP_CONFIG.API_KEY_STORAGE_KEY]);
      if (stored[MCP_CONFIG.API_KEY_STORAGE_KEY]) {
        this.config.apiKey = stored[MCP_CONFIG.API_KEY_STORAGE_KEY];
      }
      this.log('MCP client initialized');
    } catch (err) {
      this.warn('Failed to load stored API key:', err);
    }
  }

  /**
   * Logging utilities
   */
  log(...args) {
    if (this.debugMode) console.log('[MCP]', ...args);
  }

  warn(...args) {
    console.warn('[MCP]', ...args);
  }

  error(...args) {
    console.error('[MCP]', ...args);
  }

  /**
   * Create session on server and get back the server-generated ID
   * @param {string} sessionName - Friendly name for the session
   * @param {string} owner - Owner identifier (default: chrome_extension)
   * @returns {Promise<string|null>} Server-generated session ID or null on failure
   */
  async ensureSession(sessionName, owner = 'chrome_extension') {
    // Check if we already have a server-side session ID for this name
    if (this.createdSessions.has(sessionName)) {
      return this.sessions.get(sessionName); // Return the server-generated ID
    }

    try {
      this.log(`Creating session: ${sessionName}`);

      const response = await this.fetchWithTimeout(`${this.config.baseUrl}/v1/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-MCP-API-Key': this.config.apiKey || ''
        },
        body: JSON.stringify({
          owner,
          metadata: { source: 'chrome_extension', session_name: sessionName },
          ttl_hours: 24
        })
      });

      if (response.ok) {
        const data = await response.json();
        const serverSessionId = data.session_id; // Get server-generated UUID
        
        // Store the mapping: sessionName -> serverSessionId
        this.sessions.set(sessionName, serverSessionId);
        this.createdSessions.add(sessionName);
        
        this.log(`✅ Session created: ${sessionName} -> ${serverSessionId}`);
        return serverSessionId;
      } else if (response.status === 409) {
        // Session might already exist (conflict) - try to use it anyway
        this.warn(`Session ${sessionName} already exists (409), will attempt to use`);
        // You might want to call GET /v1/sessions/{sessionName} here if your API supports it
        return null;
      } else {
        const errorText = await response.text();
        this.warn(`Failed to create session: ${response.status} ${errorText}`);
        return null;
      }
    } catch (err) {
      this.error('Session creation error:', err);
      return null;
    }
  }

  /**
   * Complete a prompt via MCP server
   * @param {Object} options
   * @param {string} options.sessionName - Session name (will be created if doesn't exist)
   * @param {string} options.taskType - Task type (general_assistant, job_analysis, etc)
   * @param {string} options.prompt - User prompt
   * @param {Object} options.meta - Additional metadata
   * @returns {Promise<Object>} Response with success, completion, usage, sessionId
   */
  async complete({ sessionName = 'default', taskType = 'general_assistant', prompt, meta = {} }) {
    if (!prompt) {
      return { success: false, error: 'Prompt is required' };
    }

    try {
      // Ensure session exists on server and get the server-generated ID
      const sessionId = await this.ensureSession(sessionName);
      
      if (!sessionId) {
        return { success: false, error: 'Failed to create or retrieve session' };
      }

      // Build URL with query parameters (your server expects query params, not body)
      const url = new URL(`${this.config.baseUrl}${this.config.completeEndpoint}`);
      url.searchParams.append('session_id', sessionId);
      url.searchParams.append('prompt', prompt);
      url.searchParams.append('task_type', taskType);

      this.log(`Calling MCP: ${url.toString()}`);

      const response = await this.fetchWithTimeout(url.toString(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-MCP-API-Key': this.config.apiKey || ''
        }
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`MCP returned ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      
      this.log('MCP response received:', data);

      return {
        success: true,
        completion: data.completion || data.response || data.text || '',
        usage: data.usage || {},
        sessionId
      };

    } catch (err) {
      this.error('MCP completion failed:', err);
      return {
        success: false,
        error: err.message
      };
    }
  }

  /**
   * Test MCP server connectivity
   * @returns {Promise<Object>} Test result with success, message, status
   */
  async testConnection() {
    try {
      this.log('Testing MCP connection...');

      // Create a test session and get server-generated ID
      const sessionId = await this.ensureSession('test_connection', 'test_user');

      if (!sessionId) {
        return {
          success: false,
          message: 'Failed to create test session'
        };
      }

      // Now test the complete endpoint with the server-generated ID
      const url = new URL(`${this.config.baseUrl}${this.config.completeEndpoint}`);
      url.searchParams.append('session_id', sessionId);
      url.searchParams.append('prompt', 'test');
      url.searchParams.append('task_type', 'general_assistant');
      
      this.log(`Testing connection with session: ${sessionId}`);

      const response = await this.fetchWithTimeout(url.toString(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-MCP-API-Key': this.config.apiKey || ''
        }
      });

      if (response.ok) {
        this.log('✅ MCP connection test successful');
        return { 
          success: true, 
          message: 'Connected to MCP server',
          status: response.status 
        };
      } else {
        const errorText = await response.text();
        this.warn('❌ MCP connection test failed:', response.status, errorText);
        return { 
          success: false, 
          message: `HTTP ${response.status}: ${errorText.slice(0, 200)}`,
          status: response.status
        };
      }
    } catch (err) {
      this.error('❌ MCP connection error:', err);
      return { 
        success: false, 
        message: `Connection error: ${err.message}` 
      };
    }
  }

  /**
   * Fetch with timeout
   * @param {string} url - URL to fetch
   * @param {Object} options - Fetch options
   * @returns {Promise<Response>}
   */
  async fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      return response;
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      throw err;
    }
  }

  /**
   * Clear a session (removes from local cache, optionally deletes on server)
   * @param {string} sessionName - Session name to clear
   * @param {boolean} deleteOnServer - Whether to delete session on server
   * @returns {Promise<boolean>}
   */
  async clearSession(sessionName, deleteOnServer = false) {
    const sessionId = this.sessions.get(sessionName);
    
    if (!sessionId) {
      this.warn(`Session ${sessionName} not found`);
      return false;
    }

    // Remove from local cache
    this.sessions.delete(sessionName);
    this.createdSessions.delete(sessionName);
    
    this.log(`Cleared session: ${sessionName}`);

    // Optionally delete on server
    if (deleteOnServer) {
      try {
        const response = await this.fetchWithTimeout(
          `${this.config.baseUrl}/v1/sessions/${sessionId}`,
          {
            method: 'DELETE',
            headers: {
              'X-MCP-API-Key': this.config.apiKey || ''
            }
          }
        );

        if (response.ok) {
          this.log(`✅ Deleted session ${sessionId} on server`);
          return true;
        } else {
          this.warn(`Failed to delete session on server: ${response.status}`);
          return false;
        }
      } catch (err) {
        this.error('Session deletion error:', err);
        return false;
      }
    }

    return true;
  }

  /**
   * Get session ID by name
   * @param {string} sessionName - Session name
   * @returns {string|null} Session ID or null
   */
  getSessionId(sessionName) {
    return this.sessions.get(sessionName) || null;
  }

  /**
   * Update API key
   * @param {string} apiKey - New API key
   * @returns {Promise<Object>} Result with success status
   */
  async updateApiKey(apiKey) {
    try {
      this.config.apiKey = apiKey;
      await chrome.storage.sync.set({ [MCP_CONFIG.API_KEY_STORAGE_KEY]: apiKey });
      this.log('API key updated');
      return { success: true };
    } catch (err) {
      this.error('Failed to update API key:', err);
      return { success: false, error: err.message };
    }
  }

  /**
   * Get current configuration
   * @returns {Object} Current config
   */
  getConfig() {
    return {
      baseUrl: this.config.baseUrl,
      completeEndpoint: this.config.completeEndpoint,
      hasApiKey: !!this.config.apiKey,
      timeoutMs: this.config.timeoutMs,
      activeSessions: this.sessions.size
    };
  }

  /**
   * Clear all sessions from local cache
   */
  clearAllSessions() {
    const count = this.sessions.size;
    this.sessions.clear();
    this.createdSessions.clear();
    this.log(`Cleared ${count} sessions from cache`);
  }
}

// Create singleton instance
const mcpClient = new MCPClient();

// Export for ES6 modules
export default mcpClient;



