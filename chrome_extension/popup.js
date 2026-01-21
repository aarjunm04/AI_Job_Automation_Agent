/**
 * =============================================================================
 * AI JOB AUTOMATION AGENT - POPUP SCRIPT
 * =============================================================================
 * 
 * Manifest V3 compliant popup with 6 modular systems:
 * A. State Manager - Application state persistence
 * B. Automation Controller - Start/stop automation
 * C. Settings Manager - User preferences
 * D. Stats Display - Real-time metrics
 * E. MCP Monitor - Connection indicator
 * F. Message Handler - Background communication
 * 
 * Performance: <50ms init, debounced saves, minimal DOM manipulation
 * 
 * Author: AI Job Automation Team
 * Version: 1.0.0
 * =============================================================================
 */

'use strict';

/**
 * =============================================================================
 * A. STATE MANAGER
 * =============================================================================
 * Centralized application state with chrome.storage persistence
 */
const AppState = {
  sessionStats: {
    matchScore: 0,
    jobsApplied: 0,
    analysisResult: { status: 'pending', message: '', timestamp: 0 }
  },
  automationSettings: {
    autoApplyEnabled: true,
    minMatchScore: 70,
    maxJobsPerSession: 50,
    platforms: { linkedin: true, indeed: true, naukri: true }
  },
  mcpStatus: { connected: false, lastPing: 0, reconnecting: false },
  isRunning: false,

  /**
   * Persist entire state to chrome.storage.local
   * @async
   */
  async save() {
    try {
      await chrome.storage.local.set({ appState: this });
      console.log('[Popup] State saved');
    } catch (err) {
      console.error('[Popup] Failed to save state:', err);
    }
  },

  /**
   * Load state from chrome.storage.local
   * @async
   */
  async load() {
    try {
      const result = await chrome.storage.local.get('appState');
      if (result.appState) {
        Object.assign(this, result.appState);
        console.log('[Popup] State loaded');
      }
    } catch (err) {
      console.error('[Popup] Failed to load state:', err);
    }
  },

  /**
   * Reset to defaults
   */
  reset() {
    this.sessionStats = {
      matchScore: 0,
      jobsApplied: 0,
      analysisResult: { status: 'pending', message: '', timestamp: 0 }
    };
    this.isRunning = false;
    console.log('[Popup] State reset');
  }
};

/**
 * =============================================================================
 * B. AUTOMATION CONTROLLER
 * =============================================================================
 * Handles start/stop automation with error handling and UI updates
 */
const AutomationController = {
  /**
   * Start automation with current settings
   * @async
   */
  async start() {
    try {
      const config = {
        autoApply: AppState.automationSettings.autoApplyEnabled,
        minScore: AppState.automationSettings.minMatchScore,
        maxJobs: AppState.automationSettings.maxJobsPerSession,
        platforms: AppState.automationSettings.platforms
      };

      // Send START_AUTOMATION message to background
      const response = await chrome.runtime.sendMessage({
        type: 'START_AUTOMATION',
        config: config
      });

      if (response?.success) {
        AppState.isRunning = true;
        await AppState.save();
        this.updateUI();
        console.log('[Popup] Automation started:', config);
      } else {
        throw new Error(response?.error || 'Failed to start automation');
      }
    } catch (err) {
      console.error('[Popup] Start automation error:', err);
      this.showError(`Failed to start: ${err.message}`);
    }
  },

  /**
   * Stop automation
   * @async
   */
  async stop() {
    try {
      const response = await chrome.runtime.sendMessage({
        type: 'STOP_AUTOMATION'
      });

      if (response?.success) {
        AppState.isRunning = false;
        await AppState.save();
        this.updateUI();
        console.log('[Popup] Automation stopped');
      } else {
        throw new Error(response?.error || 'Failed to stop automation');
      }
    } catch (err) {
      console.error('[Popup] Stop automation error:', err);
      this.showError(`Failed to stop: ${err.message}`);
    }
  },

  /**
   * Update button state and appearance based on running status
   */
  updateUI() {
    const btn = document.getElementById('automationBtn');
    if (!btn) return;

    if (AppState.isRunning) {
      btn.textContent = 'Stop Automation';
      btn.style.background = '#dc2626';
    } else {
      btn.textContent = 'Start Automation';
      btn.style.background = '';
    }
  },

  /**
   * Show error message to user
   * @param {string} message
   */
  showError(message) {
    const errorEl = document.getElementById('automationError');
    if (!errorEl) return;

    errorEl.textContent = message;
    errorEl.classList.add('show');
    setTimeout(() => {
      errorEl.classList.remove('show');
    }, 4000);
  }
};

/**
 * =============================================================================
 * C. SETTINGS MANAGER
 * =============================================================================
 * Bind controls, save/load preferences, validate user input
 */
const SettingsManager = {
  debounceTimer: null,

  /**
   * Wire up all form controls with event listeners
   */
  bindControls() {
    // Auto-apply checkbox
    const autoApplyCheckbox = document.getElementById('autoApplyCheckbox');
    if (autoApplyCheckbox) {
      autoApplyCheckbox.addEventListener('change', (e) => {
        AppState.automationSettings.autoApplyEnabled = e.target.checked;
        this.debouncedSave();
      });
    }

    // Min score slider
    const minScoreSlider = document.getElementById('minScoreSlider');
    if (minScoreSlider) {
      minScoreSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        AppState.automationSettings.minMatchScore = value;
        document.getElementById('minScoreValue').textContent = `${value}%`;
        this.debouncedSave();
      });
    }

    // Max jobs input
    const maxJobsInput = document.getElementById('maxJobsInput');
    if (maxJobsInput) {
      maxJobsInput.addEventListener('change', (e) => {
        const value = parseInt(e.target.value) || 50;
        if (this.validate({ maxJobs: value })) {
          AppState.automationSettings.maxJobsPerSession = value;
          this.debouncedSave();
        } else {
          e.target.value = AppState.automationSettings.maxJobsPerSession;
        }
      });
    }

    console.log('[Popup] Settings controls bound');
  },

  /**
   * Debounced save operation (500ms delay)
   */
  debouncedSave() {
    clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.save();
    }, 500);
  },

  /**
   * Save settings to storage and notify background
   * @async
   */
  async save() {
    try {
      await chrome.storage.local.set({
        automationSettings: AppState.automationSettings
      });

      // Notify background of settings update
      chrome.runtime.sendMessage({
        type: 'UPDATE_SETTINGS',
        settings: AppState.automationSettings
      }).catch(() => {
        // Background may not be ready yet
      });

      console.log('[Popup] Settings saved');
    } catch (err) {
      console.error('[Popup] Settings save error:', err);
    }
  },

  /**
   * Load settings from storage
   * @async
   */
  async load() {
    try {
      const result = await chrome.storage.local.get('automationSettings');
      if (result.automationSettings) {
        AppState.automationSettings = {
          ...AppState.automationSettings,
          ...result.automationSettings
        };
        this.updateUI();
        console.log('[Popup] Settings loaded');
      }
    } catch (err) {
      console.error('[Popup] Settings load error:', err);
    }
  },

  /**
   * Update UI to reflect current settings
   */
  updateUI() {
    const autoApplyCheckbox = document.getElementById('autoApplyCheckbox');
    if (autoApplyCheckbox) {
      autoApplyCheckbox.checked = AppState.automationSettings.autoApplyEnabled;
    }

    const minScoreSlider = document.getElementById('minScoreSlider');
    if (minScoreSlider) {
      minScoreSlider.value = AppState.automationSettings.minMatchScore;
      document.getElementById('minScoreValue').textContent = 
        `${AppState.automationSettings.minMatchScore}%`;
    }

    const maxJobsInput = document.getElementById('maxJobsInput');
    if (maxJobsInput) {
      maxJobsInput.value = AppState.automationSettings.maxJobsPerSession;
    }
  },

  /**
   * Validate settings values
   * @param {object} settings
   * @returns {boolean}
   */
  validate(settings) {
    if (settings.minMatchScore !== undefined) {
      const score = settings.minMatchScore;
      if (score < 0 || score > 100) {
        console.warn('[Popup] Invalid min match score:', score);
        return false;
      }
    }

    if (settings.maxJobs !== undefined) {
      const jobs = settings.maxJobs;
      if (jobs < 1 || jobs > 500) {
        console.warn('[Popup] Invalid max jobs:', jobs);
        return false;
      }
    }

    return true;
  }
};

/**
 * =============================================================================
 * D. STATS DISPLAY
 * =============================================================================
 * Real-time metrics display with formatting
 */
const StatsDisplay = {
  /**
   * Update all 3 stats in DOM
   */
  update() {
    this.updateMatchScore();
    this.updateJobsApplied();
    this.updateAnalysisResult();
  },

  /**
   * Update match score stat
   */
  updateMatchScore() {
    const el = document.getElementById('matchScoreStat');
    if (!el) return;
    el.textContent = `${AppState.sessionStats.matchScore}%`;
  },

  /**
   * Update jobs applied stat
   */
  updateJobsApplied() {
    const el = document.getElementById('jobsAppliedStat');
    if (!el) return;
    el.textContent = String(AppState.sessionStats.jobsApplied);
  },

  /**
   * Update analysis result stat with status styling
   */
  updateAnalysisResult() {
    const el = document.getElementById('analysisStat');
    if (!el) return;

    const status = AppState.sessionStats.analysisResult.status;
    const message = AppState.sessionStats.analysisResult.message;

    // Remove all status classes
    el.classList.remove('pending', 'success', 'error');

    // Set display text and status class
    switch (status) {
      case 'success':
        el.textContent = message || 'Success';
        el.classList.add('success');
        break;
      case 'error':
        el.textContent = message || 'Error';
        el.classList.add('error');
        break;
      case 'pending':
      default:
        el.textContent = 'Pending';
        el.classList.add('pending');
        break;
    }
  }
};

/**
 * =============================================================================
 * E. MCP MONITOR
 * =============================================================================
 * Connection status indicator with periodic health check
 */
const MCPMonitor = {
  pingIntervalId: null,

  /**
   * Initialize MCP monitoring
   */
  init() {
    this.checkStatus();
    // Ping MCP status every 10 seconds
    this.pingIntervalId = setInterval(() => this.checkStatus(), 10000);
  },

  /**
   * Update MCP indicator dot color and text
   * @param {boolean} connected
   */
  updateIndicator(connected) {
    const dot = document.getElementById('mcpDot');
    if (!dot) return;

    AppState.mcpStatus.connected = connected;
    if (connected) {
      dot.classList.add('connected');
    } else {
      dot.classList.remove('connected');
    }
  },

  /**
   * Check MCP connection status from background
   * @async
   */
  async checkStatus() {
    try {
      const response = await chrome.runtime.sendMessage({
        type: 'GET_MCP_STATUS'
      });

      if (response?.mcpStatus) {
        this.updateIndicator(response.mcpStatus.connected);
        AppState.mcpStatus.lastPing = Date.now();
      }
    } catch (err) {
      // Background not ready, assume disconnected
      this.updateIndicator(false);
    }
  },

  /**
   * Cleanup on popup close
   */
  cleanup() {
    if (this.pingIntervalId) {
      clearInterval(this.pingIntervalId);
    }
  }
};

/**
 * =============================================================================
 * F. MESSAGE HANDLER
 * =============================================================================
 * Listen for background messages and handle automation status updates
 */
const MessageHandler = {
  /**
   * Initialize message listener from background
   */
  init() {
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (!message?.type) return;

      try {
        switch (message.type) {
          case 'STATS_UPDATE':
            this.handleStatsUpdate(message.stats);
            break;
          case 'AUTOMATION_STATUS':
            this.handleAutomationStatus(message.running, message.message);
            break;
          case 'MCP_STATUS':
            this.handleMCPStatus(message.connected);
            break;
          case 'ANALYSIS_COMPLETE':
            this.handleAnalysisComplete(message.result);
            break;
          default:
            console.log('[Popup] Unknown message type:', message.type);
        }
      } catch (err) {
        console.error('[Popup] Message handler error:', err);
      }
    });

    console.log('[Popup] Message handler initialized');
  },

  /**
   * Handle stats update from background
   * @param {object} stats
   */
  handleStatsUpdate(stats) {
    if (!stats) return;
    AppState.sessionStats = {
      ...AppState.sessionStats,
      ...stats
    };
    StatsDisplay.update();
  },

  /**
   * Handle automation status change
   * @param {boolean} isRunning
   * @param {string} message
   */
  handleAutomationStatus(isRunning, message) {
    AppState.isRunning = isRunning;
    AutomationController.updateUI();
    if (message) {
      console.log('[Popup] Automation status:', message);
    }
  },

  /**
   * Handle MCP connection status
   * @param {boolean} connected
   */
  handleMCPStatus(connected) {
    MCPMonitor.updateIndicator(connected);
  },

  /**
   * Handle analysis completion
   * @param {object} result
   */
  handleAnalysisComplete(result) {
    if (result) {
      AppState.sessionStats.analysisResult = result;
      StatsDisplay.updateAnalysisResult();
    }
  }
};

/**
 * =============================================================================
 * INITIALIZATION
 * =============================================================================
 */

/**
 * Main initialization function
 * @async
 */
async function initializePopup() {
  try {
    console.log('[Popup] Initializing...');
    const startTime = performance.now();

    // Load state
    await AppState.load();

    // Initialize all systems
    SettingsManager.bindControls();
    await SettingsManager.load();
    MessageHandler.init();
    MCPMonitor.init();

    // Bind UI event listeners
    const automationBtn = document.getElementById('automationBtn');
    if (automationBtn) {
      automationBtn.addEventListener('click', () => {
        if (AppState.isRunning) {
          AutomationController.stop();
        } else {
          AutomationController.start();
        }
      });
    }

    const viewDashboardBtn = document.getElementById('viewDashboardBtn');
    if (viewDashboardBtn) {
      viewDashboardBtn.addEventListener('click', () => {
        chrome.runtime.sendMessage({
          type: 'OPEN_DASHBOARD'
        }).catch(() => {
          console.log('[Popup] Dashboard not available');
        });
      });
    }

    // Initial UI update
    AutomationController.updateUI();
    StatsDisplay.update();

    // Request initial stats from background
    chrome.runtime.sendMessage({
      type: 'GET_SESSION_STATS'
    }).then(response => {
      if (response?.stats) {
        AppState.sessionStats = response.stats;
        StatsDisplay.update();
      }
    }).catch(() => {
      console.log('[Popup] Background not ready for initial stats');
    });

    const initTime = performance.now() - startTime;
    console.log(`[Popup] Initialized in ${initTime.toFixed(2)}ms`);

    // Log memory usage after init
    if (performance.memory) {
      console.log('[Popup] Memory usage:', {
        usedJSHeapSize: `${(performance.memory.usedJSHeapSize / 1048576).toFixed(2)}MB`,
        jsHeapSizeLimit: `${(performance.memory.jsHeapSizeLimit / 1048576).toFixed(2)}MB`
      });
    }
  } catch (err) {
    console.error('[Popup] Initialization error:', err);
  }
}

/**
 * Cleanup on popup close
 */
window.addEventListener('beforeunload', () => {
  try {
    AppState.save();
    MCPMonitor.cleanup();
    console.log('[Popup] Cleanup completed');
  } catch (err) {
    console.error('[Popup] Cleanup error:', err);
  }
});

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializePopup);
} else {
  initializePopup();
}
