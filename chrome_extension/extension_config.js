/**
 * =============================================================================
 * AI JOB AUTOMATION AGENT - EXTENSION CONFIGURATION
 * =============================================================================
 */

'use strict';

// ==================== MCP / LLM CONFIG ====================
export const MCP_CONFIG = {
  BASE_URL: 'http://localhost:8080',
  COMPLETE_ENDPOINT: '/llm/complete',
  TIMEOUT_MS: 15000,
  MAX_RETRIES: 2,
  RETRY_BASE_DELAY_MS: 500,
  API_KEY_STORAGE_KEY: 'mcp_api_key'
};

// ==================== BACKEND / RAG CONFIG ====================
export const API_CONFIG = {
  RAG_RESUME_ENDPOINT: '/rag/dynamic_resume'
};

// ==================== STORAGE KEYS ====================
export const STORAGE_KEYS = {
  USER_SETTINGS: 'user_settings',
  RESUMES_META: 'resumes_meta',
  JOB_CACHE: 'job_cache',
  ANALYSIS_HISTORY: 'analysis_history',
  RESUME_BLOBS: 'resume_blobs',
  EXTENSION_STATE: 'extension_state',
  AI_ACTIVITY_LOG: 'ai_activity_log',
  API_CONFIG: 'api_config',
  MCP_SESSIONS: 'mcp_sessions'
};

// ==================== DEFAULT USER SETTINGS ====================
export const DEFAULT_USER_SETTINGS = {
  full_name: '',
  first_name: '',
  last_name: '',
  email: '',
  phone: '',
  linkedin: '',
  location: '',
  github: '',
  cover_letter: '',
  auto_analysis: true,
  notifications: true,
  debug_mode: false,
  match_threshold: 85,
  daily_limit: 15,
  preferred_platforms: ['linkedin', 'indeed', 'naukri'],
  mcp_base_url: 'http://localhost:8080',
  mcp_enabled: true,
  notion_webhook_url: ''
};

// ==================== TIMING & BEHAVIOR ====================
export const TIMING = {
  SUBMIT_DETECTION_TIMEOUT_MS: 25000,
  SUBMIT_DETECTION_POLL_MS: 800,
  JOB_CACHE_TTL_MS: 24 * 60 * 60 * 1000,
  ACTIVITY_LOG_MAX_ENTRIES: 200,
  RESUME_BLOB_MAX_SIZE_MB: 10
};

// ==================== FEATURE FLAGS ====================
export const FEATURES = {
  ENABLE_MCP_CHAT: true,
  ENABLE_DYNAMIC_RESUME: true,
  ENABLE_AUTO_FILL: true,
  ENABLE_NOTION_LOGGING: true,
  ENABLE_TELEMETRY: false
};

// ==================== CONTEXT MENU & COMMAND IDs ====================
export const UI_IDS = {
  CONTEXT_ANALYZE_JOB: 'ai_analyze_job',
  CONTEXT_AUTO_FILL: 'ai_auto_fill',
  CONTEXT_LOG_NOTION: 'ai_log_notion',
  COMMAND_ANALYZE: 'analyze_job',
  COMMAND_AUTO_FILL: 'auto_fill_now',
  COMMAND_TOGGLE_SIDEBAR: 'toggle_sidebar'
};

// ==================== PLATFORM CONFIG ====================
export const SUPPORTED_PLATFORMS = [
  'linkedin.com',
  'indeed.com', 
  'naukri.com',
  'flexjobs.com',
  'weworkremotely.com',
  'remotive.io',
  'wellfound.com',
  'glassdoor.com',
  'simplyhired.com'
];

// ==================== TASK TYPES FOR MCP ====================
export const MCP_TASK_TYPES = {
  JOB_ANALYSIS: 'job_analysis',
  RESUME_HELP: 'resume_help',
  AUTO_FILL_HELP: 'autofill_help',
  GENERAL_ASSISTANT: 'general_assistant'
};

// ==================== MCP REQUEST SCHEMAS ====================
export const MCP_REQUEST_SCHEMA_VERSION = 'mcp-llm-complete-1';
export const EXTENSION_VERSION = '1.0.0';

// ==================== STORAGE HELPER ====================
export const ConfigStorage = {
  async get(key) {
    return new Promise(resolve => {
      chrome.storage.sync.get([key], result => resolve(result[key]));
    });
  },
  
  async set(obj) {
    return new Promise((resolve, reject) => {
      chrome.storage.sync.set(obj, () => {
        if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
        else resolve();
      });
    });
  }
};

// ==================== ENVIRONMENT LOADER ====================
export async function loadRuntimeConfig() {
  const [apiKey, apiConfig] = await Promise.all([
    ConfigStorage.get(MCP_CONFIG.API_KEY_STORAGE_KEY),
    ConfigStorage.get(STORAGE_KEYS.API_CONFIG)
  ]);
  
  return {
    mcp: {
      base_url: apiConfig?.mcp_base_url || MCP_CONFIG.BASE_URL,
      api_key: apiKey || null,
      enabled: FEATURES.ENABLE_MCP_CHAT
    },
    rag: {
      enabled: FEATURES.ENABLE_DYNAMIC_RESUME
    },
    notion: {
      webhook_url: apiConfig?.notion_webhook_url || null
    },
    debug: DEFAULT_USER_SETTINGS.debug_mode
  };
}
