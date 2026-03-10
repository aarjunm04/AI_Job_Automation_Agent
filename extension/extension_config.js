// Merged from chrome_extension/ — 2026-03-03
/**
 * =============================================================================
 * AI JOB AUTOMATION AGENT - EXTENSION CONFIGURATION
 * =============================================================================
 */

'use strict';

// ==================== FASTAPI CONFIG ====================
export const API_CONFIG = {
  BASE_URL_STORAGE_KEY: 'api_base_url',
  DEFAULT_BASE_URL: 'http://localhost:8000',
  API_KEY_STORAGE_KEY: 'api_key',
  ENDPOINTS: {
    HEALTH: '/health',
    MATCH: '/match',
    AUTOFILL: '/autofill',
    APPLY_MANUAL: '/apply/manual'
  }
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
  API_CONFIG: 'api_config'
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
  ENABLE_AUTO_MATCH: true,
  ENABLE_AUTOFILL: true,
  ENABLE_NOTION_QUEUE: true
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
    ConfigStorage.get(API_CONFIG.API_KEY_STORAGE_KEY),
    ConfigStorage.get(STORAGE_KEYS.API_CONFIG)
  ]);

  return {
    api: {
      base_url: apiConfig?.api_base_url || API_CONFIG.DEFAULT_BASE_URL,
      api_key: apiKey || null
    },
    features: {
      auto_match: FEATURES.ENABLE_AUTO_MATCH,
      autofill: FEATURES.ENABLE_AUTOFILL,
      notion_queue: FEATURES.ENABLE_NOTION_QUEUE
    },
    notion: {
      webhook_url: apiConfig?.notion_webhook_url || null
    },
    debug: DEFAULT_USER_SETTINGS.debug_mode
  };
}
